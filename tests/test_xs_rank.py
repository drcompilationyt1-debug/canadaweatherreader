"""Cross-sectional ranking head: walk-forward out-of-sample ranks in the dataset build, live ranks in the runner."""
from __future__ import annotations

import numpy as np
import pytest

from stockbot.agent.policy import PolicyBundle
from stockbot.data.loader import synthetic_universe
from stockbot.env.dataset import MarketDataset, compute_signal_arrays
from stockbot.execution.runner import TradingRunner
from stockbot.signals.registry import build_context, build_layout, build_providers

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def _setup(cfg):
    pytest.importorskip("lightgbm")
    frames = synthetic_universe(TICKERS, n=2200, seed=11)          # ~8.7 years x 6 names: enough rows to fit
    cfg.set_path("universe", TICKERS)
    cfg.set_path("signals.xs_rank.min_train_years", 2)
    cfg.set_path("signals.xs_rank.n_estimators", 40)
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend", "market_regime", "xs_rank")]
    assert providers[-1].name == "xs_rank"                        # it must run last
    return frames, ctx, providers


def test_xs_rank_walk_forward_and_live(cfg, tmp_path):
    frames, ctx, providers = _setup(cfg)
    layout = build_layout(providers)
    ds = MarketDataset.build(frames, providers, layout, ctx, fit=True, train_end="2020-12-31")
    b = layout.block("xs_rank")
    xs = providers[-1]
    assert xs.model is not None and xs.preds is not None and len(xs.preds) > 0
    assert [name for name, _ in xs.spec] == ["technical", "trend", "market_regime"]
    assert xs.state_path("xs_rank.joblib").exists() and xs.state_path("xs_rank_preds.parquet").exists()
    # the first refit year is masked (no model yet), later years carry out-of-sample ranks in -1..1
    sig = ds.data["AAA"].signals
    flags = sig[:, b.offset]
    dates = ds.data["AAA"].dates.astype("datetime64[Y]").astype(int) + 1970
    assert flags[dates < dates.min() + 2].max() == 0.0 and flags[dates >= dates.max() - 2].mean() > 0.9
    on = flags > 0.5
    ranks = sig[on, b.start + 1]
    assert ranks.min() >= -1.0 - 1e-6 and ranks.max() <= 1.0 + 1e-6
    # on one date the six names get six distinct ranks, with one top and one bottom decile flag at most
    last = {t: ds.data[t].signals[-1] for t in TICKERS}
    assert all(v[b.offset] > 0.5 for v in last.values())
    assert len({round(float(v[b.start + 1]), 4) for v in last.values()}) == len(TICKERS)
    assert sum(v[b.start + 2] for v in last.values()) <= 1 and sum(v[b.start + 3] for v in last.values()) <= 1
    # no look-ahead: the prediction for a date only uses models fitted on earlier years
    preds = xs.preds
    assert preds["date"].min().year >= dates.min() + 2
    # live path: the runner exposes the other blocks' latest vectors and gets one rank per ticker
    class M:
        num_timesteps = 0

        def predict(self, obs, deterministic=True):
            return np.array([0.5], dtype=np.float32), None

    bundle = PolicyBundle(M(), layout, {"algo": "ppo"})
    runner = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False)
    runner.frames = frames
    vectors, _ = runner.latest_vectors()
    live = {t: vectors[t]["xs_rank"] for t in TICKERS}
    assert all(v is not None and v.shape == (4,) for v in live.values())
    assert len({round(float(v[1]), 4) for v in live.values()}) == len(TICKERS)
    # without the other blocks in the context the head masks itself instead of failing
    ctx.extra.pop("per_block", None)
    ctx.extra.pop("latest_vectors", None)
    assert all(a is None for a in xs.compute_history_all(frames).values())
