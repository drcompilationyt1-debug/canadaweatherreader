import numpy as np
import pandas as pd

from stockbot.signals.base import SignalContext
from stockbot.signals.factors import FEATURES, FactorSignal, ranked_factors, raw_factors


def _frames(n=400, names=("A", "B", "C", "D", "E", "SPY"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    out = {}
    for i, t in enumerate(names):
        drift = 0.004 if t == "A" else 0.0
        c = 100 * np.cumprod(1 + rng.normal(drift, 0.012, n))
        out[t] = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                               "volume": rng.integers(100_000, 1_000_000, n).astype(float) * (i + 1)}, index=idx)
    return out


def test_factor_block_shapes_ranks_and_sector_relative(cfg, tmp_path):
    frames = _frames()
    ctx = SignalContext(cfg=cfg, models_dir=tmp_path)
    ctx.extra["offline"] = True
    ctx.extra["sectors"] = {"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy", "E": "Health", "SPY": "Index"}
    p = FactorSignal(cfg, ctx)
    assert p.enabled and p.needs_universe and list(p.feature_names) == FEATURES
    arrs = p.compute_history_all(frames)
    a = arrs["A"]
    assert a.shape == (400, len(FEATURES)) and a.dtype == np.float32
    assert np.isnan(a[:250, 0]).all()                                  # 12-1 momentum needs a year of bars
    tail = a[300:]
    assert np.isfinite(tail).all()
    assert (tail[:, :8] >= -1.0).all() and (tail[:, :8] <= 1.0).all()  # percentiles across the universe
    assert tail[:, 0].mean() > 0.3                                     # the drifting name ranks near the top on momentum
    assert (a[300:, 9] <= 0.0).all()                                   # raw distance from the 52-week high is never positive
    raw = raw_factors(pd.DataFrame({t: f["close"] for t, f in frames.items()}), pd.DataFrame({t: f["volume"] for t, f in frames.items()}),
                      ctx.extra["sectors"], None)
    rel = raw["sec_rel_3m"].iloc[-1]
    assert abs(rel["A"] + rel["B"]) < 1e-9 and abs(rel["C"] + rel["D"]) < 1e-9   # sector-relative returns net to zero within a sector
    assert np.isfinite(rel["E"])                                                  # a lone name is measured against the universe
    r = ranked_factors(raw)
    assert r["dvol"].iloc[-1].idxmax() == "SPY"                                   # the biggest dollar volume ranks first


def test_factor_block_single_ticker_fallback(cfg, tmp_path):
    frames = _frames(names=("A", "B", "C", "D", "E"))
    ctx = SignalContext(cfg=cfg, models_dir=tmp_path)
    ctx.extra["offline"] = True
    ctx.extra["sectors"] = {}
    p = FactorSignal(cfg, ctx)
    ctx.extra["frames"] = frames
    assert p.compute_history("A", frames["A"]).shape == (400, len(FEATURES))
    assert np.isnan(p.compute_history_all({"A": frames["A"]})["A"][300:, 0]).all()   # one name has no cross-section: masked
