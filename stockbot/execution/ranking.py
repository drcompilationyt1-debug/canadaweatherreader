"""The rank-core decision layer: hold the top-K names by a blended cross-sectional score.

Diagnosis on the cached 50-name dataset (2026-09-15): the policy on its own tracks buy-and-hold (Sharpe 0.76
vs 0.73 out of sample, 83% invested, no edge) while the cross-sectional ranking head has real, stable alpha
(20-day information coefficient +0.07, t-statistic 9 over three years).  Harvesting it the standard way -
top-K, periodic rebalance, hysteresis so the book does not churn on small ranking moves - gave +28% in the
out-of-sample year against SPY's +20% and +140% over three years against +76%, with about 15 turnovers a year.

So the runner ranks every name by a blend of percentile ranks of a few inputs (``execution.rank.inputs``:
the ranking head first, TimesFM second), keeps the top ``top_k`` (a held name keeps its slot until it falls
more than ``hysteresis`` places below the cut), rebalances every ``every_bars`` trading days, and gives each
selected name ``1/top_k`` of the book scaled by the policy's conviction (``policy_floor`` .. 1); the policy
can veto a name (``policy_veto``).  The fee per round trip decides the cadence: two weeks for a $100k book,
monthly for $10k.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)

CADENCE_BARS = {"daily": 1, "weekly": 5, "biweekly": 10, "monthly": 21}
DEFAULT_INPUTS = {"xs_rank.xs_score": 1.0, "timesfm.tfm_ret_20": 0.5}


def rank_scores(layout, obs_by: dict[str, np.ndarray], inputs: dict[str, float] | None = None) -> dict[str, float]:
    """Blended percentile-rank score per ticker from the live observation vectors (masked blocks are skipped
    and the remaining weights renormalised; a name with no usable input gets NaN)."""
    inputs = inputs or DEFAULT_INPUTS
    cols: dict[str, tuple[int, int, float]] = {}
    for key, w in inputs.items():
        block, feat = key.split(".", 1)
        try:
            b = layout.block(block)
            j = b.start + list(b.feature_names).index(feat)
        except (KeyError, ValueError):
            log.debug("rank input %s not in the layout - ignored", key)
            continue
        cols[key] = (b.offset, j, float(w))
    if not cols or not obs_by:
        return {t: float("nan") for t in obs_by}
    tickers = list(obs_by)
    frame = {}
    for key, (off, j, _w) in cols.items():
        vals = []
        for t in tickers:
            o = np.asarray(obs_by[t], dtype=float)
            vals.append(o[j] if o[off] > 0.5 and np.isfinite(o[j]) else np.nan)
        frame[key] = vals
    df = pd.DataFrame(frame, index=tickers)
    ranks = df.rank(pct=True)
    weights = pd.Series({k: cols[k][2] for k in cols})
    num = (ranks * weights).sum(axis=1, min_count=1)
    den = ranks.notna().mul(weights, axis=1).sum(axis=1)
    score = num / den.replace(0.0, np.nan)
    return {t: float(score.get(t, np.nan)) for t in tickers}


def select_top(scores: dict[str, float], held: list[str], k: int, hysteresis: int = 3) -> list[str]:
    """Top-``k`` names by score; a held name keeps its slot while it ranks within ``k + hysteresis``."""
    ranked = [t for t, s in sorted(scores.items(), key=lambda kv: -kv[1]) if np.isfinite(s)]
    rank = {t: i for i, t in enumerate(ranked)}
    keep = sorted([t for t in held if t in rank and rank[t] < k + hysteresis], key=lambda t: rank[t])[:k]
    for t in ranked:
        if len(keep) >= k:
            break
        if t not in keep:
            keep.append(t)
    return keep


def bars_since(index, last_date: str | None, as_of: str) -> int | None:
    """Trading days from ``last_date`` (exclusive) to ``as_of`` (inclusive) on a bar index; None without a last date."""
    if not last_date:
        return None
    idx = pd.DatetimeIndex(index)
    a, b = pd.Timestamp(last_date), pd.Timestamp(as_of)
    return int(((idx > a) & (idx <= b)).sum())


def rebalance_due(index, last_date: str | None, as_of: str, every_bars: int) -> bool:
    n = bars_since(index, last_date, as_of)
    return n is None or n >= max(1, int(every_bars))


def every_bars_of(value) -> int:
    """``every_bars`` from a number or a cadence word (daily / weekly / biweekly / monthly)."""
    if value is None:
        return 1
    if isinstance(value, str):
        return CADENCE_BARS.get(value.lower(), 1)
    return max(1, int(value))


def slot_weights(chosen: list[str], k: int, ppo_frac: dict[str, float], floor: float = 0.5, veto: float = 0.05) -> dict[str, float]:
    """Each selected name gets ``1/k`` of the book scaled by the policy's conviction (``floor`` .. 1), or nothing
    when the policy wants (almost) nothing in it."""
    out = {}
    for t in chosen:
        f = float(np.clip(ppo_frac.get(t, 1.0), 0.0, 1.0))
        out[t] = 0.0 if f < veto else (1.0 / max(k, 1)) * (floor + (1.0 - floor) * f)
    return out
