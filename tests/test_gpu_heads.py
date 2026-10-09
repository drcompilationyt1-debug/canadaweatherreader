"""GPU-trained heads: the torch ensemble works on a CPU and pickles to CPU weights; a runner reuses a fresh pretrained state and
only scores the new rows; a stale or mismatched one is refitted; gpu-train rebuilds the heads' inputs from the dataset."""
import pickle

import numpy as np
import pandas as pd

import stockbot.agent  # noqa: F401
from stockbot.data.loader import synthetic_universe
from stockbot.env.dataset import MarketDataset
from stockbot.signals.registry import build_context, build_layout, build_providers
from stockbot.signals.xs_nn import _TorchEnsemble


def test_torch_ensemble_learns_and_pickles_to_cpu():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6000, 12)).astype(np.float32)
    y = (0.3 * X[:, 0] - 0.2 * X[:, 3] + rng.normal(0, 0.1, 6000)).astype(np.float32)
    m = _TorchEnsemble(hidden=(16, 8), seeds=2, epochs=8, device="cpu").fit(X, y)
    p = m.predict(X[:500])
    assert np.corrcoef(p, y[:500])[0, 1] > 0.7
    m2 = pickle.loads(pickle.dumps(m))
    assert m2._nets is None and m2.device == "cpu" and np.allclose(m2.predict(X[:50]), p[:50], atol=1e-5)


def _build(cfg, tickers, n):
    frames = synthetic_universe(tickers, n=n, seed=4)
    cfg.set_path("universe", tickers)
    cfg.set_path("signals.xs_nn.seeds_cpu", 1)
    cfg.set_path("signals.xs_nn.epochs_cpu", 3)
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend", "xs_nn")]
    return frames, ctx, providers


def test_runner_reuses_a_fresh_pretrained_state_and_scores_only_new_rows(cfg):
    tickers = [f"T{i}" for i in range(8)]
    frames, ctx, providers = _build(cfg, tickers, 1400)
    head = providers[-1]
    head.pretrained = False
    old = {t: f.iloc[:-30] for t, f in frames.items()}                     # trained "on the GPU" a month ago
    MarketDataset.build(old, providers, build_layout(providers), ctx, fit=True, train_end=None)
    stored = pd.read_parquet(head.state_path(head.PREDS_FILE))
    assert len(stored) > 0
    head.pretrained, head.pretrained_max_age = True, 45
    calls = {"fit": 0}
    real = head._lgbm

    def counting():
        calls["fit"] += 1
        return real()

    head._lgbm = counting
    ds = MarketDataset.build(frames, providers, build_layout(providers), ctx, fit=True, train_end=None)
    assert calls["fit"] == 0                                                # nothing refitted on the "runner"
    b = ds.layout.block("xs_nn")
    assert (ds.data["T0"].signals[-25:, b.offset] > 0.5).all()              # the new month is scored by the saved model
    head.pretrained_max_age = 5                                             # a stale state is refitted instead
    MarketDataset.build(frames, providers, build_layout(providers), ctx, fit=True, train_end=None)
    assert calls["fit"] > 0


def test_gpu_train_rebuilds_the_heads_inputs_from_the_dataset(cfg, tmp_path):
    from stockbot.agent.gpu_train import frames_from_dataset, gpu_train, per_block_from_dataset

    tickers = [f"T{i}" for i in range(8)]
    frames, ctx, providers = _build(cfg, tickers, 1400)
    ds = MarketDataset.build(frames, providers, build_layout(providers), ctx, fit=True, train_end=None)
    ds.save(cfg.path("models_dir", "models") / "dataset")
    pb = per_block_from_dataset(ds, "xs_nn")
    assert set(pb) == {"technical", "trend"} and pb["technical"]["T0"].shape[0] == len(ds.data["T0"].dates)
    assert len(frames_from_dataset(ds)["T0"]) == len(ds.data["T0"].dates)
    res = gpu_train(cfg, heads=("xs_nn",), ic_start="2012-01-01")
    assert res["xs_nn"]["rows"] > 0 and res["xs_nn"]["device"] in ("cpu",) or res["xs_nn"]["device"].startswith("cuda")
