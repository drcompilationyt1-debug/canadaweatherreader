"""The research additions: the complexity timer, the TabPFN head (with a stand-in model), the qlib TRA block, Chronos-2 inputs."""
import numpy as np
import pandas as pd

import stockbot.agent  # noqa: F401
from stockbot.agent.timing import ComplexityTimer
from stockbot.data.loader import synthetic_universe
from stockbot.env.dataset import MarketDataset
from stockbot.signals.base import SignalContext
from stockbot.signals.registry import build_context, build_layout, build_providers


def test_complexity_timer_is_causal_and_finds_a_real_signal():
    rng = np.random.default_rng(0)
    n = 1500
    F = rng.normal(size=(n, 10)).astype(np.float32)
    y = 0.02 * F[:, 0] - 0.01 * F[:, 1] ** 2 + rng.normal(0, 0.03, n)          # a nonlinear but learnable relation
    t = ComplexityTimer(n_features=800, gamma=0.5, ridge=5.0)
    pred = t.walk_forward(F, y, min_train=400, refit_every=21)
    ok = np.isfinite(pred)
    assert ok.sum() > 900 and np.corrcoef(pred[ok], y[ok])[0, 1] > 0.3         # real out-of-sample skill
    noise = ComplexityTimer(n_features=800, gamma=0.5, ridge=5.0).walk_forward(F, rng.normal(0, 0.03, n), min_train=400)
    okn = np.isfinite(noise)
    assert abs(np.corrcoef(noise[okn], y[okn])[0, 1]) < 0.15                     # nothing to find: nothing found
    y2 = y.copy()
    y2[900:] += 10.0                                                             # the future must not leak into earlier rows
    pred2 = ComplexityTimer(n_features=800, gamma=0.5, ridge=5.0).walk_forward(F, y2, min_train=400, refit_every=21)
    assert np.allclose(pred[400:880], pred2[400:880], atol=1e-4)
    d = ComplexityTimer.from_dict(t.to_dict())
    assert np.allclose(d.predict(F[-5:]), t.predict(F[-5:]), atol=1e-4)


class _FakeReg:
    def __init__(self, **kw):
        self.w = None

    def fit(self, X, y):
        X = np.c_[np.ones(len(X)), X]
        self.w = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, X):
        return np.c_[np.ones(len(X)), X] @ self.w


def test_tabpfn_head_runs_with_a_stand_in_and_only_recent_refits(cfg, monkeypatch):
    from stockbot.signals.xs_tabpfn import XSTabPFNSignal

    monkeypatch.setattr(XSTabPFNSignal, "model_cls", _FakeReg)
    tickers = [f"T{i}" for i in range(8)]
    frames = synthetic_universe(tickers, n=2200, seed=5)
    cfg.set_path("universe", tickers)
    cfg.set_path("signals.xs_tabpfn.years_back", 2)
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend", "xs_tabpfn")]
    assert providers[-1].name == "xs_tabpfn" and providers[-1].availability()[0]
    ds = MarketDataset.build(frames, providers, build_layout(providers), ctx, fit=True, train_end=None)
    b = ds.layout.block("xs_tabpfn")
    flags = ds.data["T0"].signals[:, b.offset]
    assert flags[-60:].all() and not flags[:600].any()                           # scored in the last years only
    assert (ctx.state_dir() / "xs_tabpfn.joblib").exists()


def test_qlib_tra_block_reads_its_own_table(cfg, tmp_path):
    from stockbot.signals.thirdparty.qlib_tra_signal import QlibTRASignal

    idx = pd.bdate_range("2026-01-01", periods=30)
    rows = [{"datetime": d, "instrument": t, "score": s} for d in idx for t, s in (("AAA", 0.01), ("BBB", -0.01))]
    pd.DataFrame(rows).to_parquet(tmp_path / "pred_tra.parquet")
    cfg.set_path("signals.qlib_tra.predictions", (tmp_path / "pred_tra.parquet").as_posix())
    ctx = SignalContext(cfg=cfg, models_dir=tmp_path)
    p = QlibTRASignal(cfg, ctx)
    assert p.name == "qlib_tra" and p.feature_names == ["tra_score", "tra_rank"] and p.availability()[0]
    df = pd.DataFrame({"close": np.linspace(10, 11, 30), "volume": 1e6}, index=idx)
    a = p.compute_history("AAA", df)
    assert a.shape == (30, 2) and a[-1, 0] > 0 and a[-1, 1] > 0
    assert np.isnan(p.compute_history("ZZZ", df)).all()


def test_chronos2_inputs_carry_covariates(cfg, tmp_path):
    from stockbot.signals.thirdparty.chronos2_signal import Chronos2Signal

    ctx = SignalContext(cfg=cfg, models_dir=tmp_path)
    p = Chronos2Signal(cfg, ctx)
    n = 400
    idx = pd.bdate_range("2025-01-01", periods=n)
    df = pd.DataFrame({"close": 100 + np.arange(n) * 0.1, "volume": 1e6 + np.arange(n)}, index=idx)
    ctx.extra["frames"] = {"SPY": pd.DataFrame({"close": 500 + np.arange(n) * 0.2, "volume": 1e7}, index=idx)}
    logc = np.log(df["close"].to_numpy())
    inputs = p._forecast_inputs(logc, p._covariate_series(df), np.array([n - 1, 300]))
    assert len(inputs) == 2 and inputs[0]["target"].shape == (p.context,) and set(inputs[0]["past_covariates"]) == {"log_volume", "log_index"}
    assert inputs[1]["past_covariates"]["log_index"].shape == (p.context,)
    assert p.cache.folder.name == "chronos2" and "chronos-2" in p.cache.version


class _Heavy(_FakeReg):
    def __init__(self, **kw):
        super().__init__()
        self.blob = np.zeros(2_000_000, dtype=np.float32)         # 8 MB of "weights" that must not reach the state branch


def test_tabpfn_wrapper_never_pickles_the_model_and_refits_from_its_context():
    import pickle

    from stockbot.signals.xs_tabpfn import _TabPFN

    Heavy = _Heavy
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 6)).astype(np.float32)
    y = X[:, 0] * 2 + rng.normal(size=5000).astype(np.float32) * 0.1
    w = _TabPFN(max_rows=1000, chunk=500, model_cls=Heavy, stride=1).fit(X, y)
    before = w.predict(X[:20])
    blob = pickle.dumps(w)
    assert len(blob) < 200_000 and w.X_ctx.shape == (1000, 6)      # the context travels, the weights do not
    w2 = pickle.loads(blob)
    assert w2.model is None
    assert np.allclose(w2.predict(X[:20]), before, atol=1e-4)      # rebuilt from the context on first use


def test_state_save_leaves_out_files_github_would_reject(tmp_path):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    import ci_state

    small = tmp_path / "small.bin"
    small.write_bytes(b"x" * 1024)
    assert not ci_state._oversized(small)
    assert ci_state._oversized(small, limit_mb=0.0005)
    assert not ci_state._oversized(tmp_path / "missing.bin")

