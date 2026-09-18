"""The neural ranking head runs the same walk-forward as the LightGBM head and keeps its own state files."""
import numpy as np

import stockbot.agent  # noqa: F401
from stockbot.data.loader import synthetic_universe
from stockbot.env.dataset import MarketDataset
from stockbot.signals.registry import build_context, build_layout, build_providers


def test_neural_ranking_head_scores_the_universe(cfg):
    tickers = [f"T{i}" for i in range(8)]
    frames = synthetic_universe(tickers, n=1400, seed=3)
    cfg.set_path("universe", tickers)
    cfg.set_path("signals.xs_nn.max_iter", 8)
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend", "xs_nn")]
    assert [p.name for p in providers][-1] == "xs_nn" and providers[-1].STATE_FILE == "xs_nn.joblib"
    ds = MarketDataset.build(frames, providers, build_layout(providers), ctx, fit=True, train_end=None)
    b = ds.layout.block("xs_nn")
    sig = ds.data["T0"].signals
    assert b.size == 4 and (sig[-100:, b.offset] > 0.5).all()                     # scored in the final years
    scores = sig[-100:, b.start]
    assert np.isfinite(scores).all() and np.abs(scores).max() <= 3.0
    ranks = np.stack([ds.data[t].signals[-1, b.start + 1] for t in tickers])
    assert ranks.min() >= -1.0 and ranks.max() <= 1.0 and len(set(np.round(ranks, 3))) > 1   # a real ordering across names
    assert (ctx.state_dir() / "xs_nn.joblib").exists() and not (ctx.state_dir() / "xs_rank.joblib").exists()
