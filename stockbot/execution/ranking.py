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
# inputs the weekend tuner may add to the blend, weighted by their trailing information coefficient
CANDIDATE_INPUTS = ["xs_rank.xs_score", "timesfm.tfm_ret_20", "es_agent.es_action", "qlib.qlib_score", "kronos.kr_ret_5",
                    "chronos.chr_ret_20", "dl_forecast.dl_pred", "alpha_factors.af_pred", "technical.ret_20", "fundamentals.f_ey",
                    "dqn_agent.dqn_buy_pref", "trend.slope_30", "factors.mom_12_1", "factors.hi_52w", "factors.sec_rel_3m", "xs_nn.nn_score",
                    "xs_tabpfn.pfn_score", "qlib_tra.tra_score", "chronos2.c2_ret_20"]
ANCHOR = "xs_rank.xs_score"


def load_tuned_inputs(models_dir, fallback: dict[str, float] | None = None) -> dict[str, float]:
    """The weekend-tuned blend (``models/rank_weights.json``) when it was accepted, else ``fallback``."""
    import json
    from pathlib import Path

    f = Path(models_dir) / "rank_weights.json"
    if f.exists():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("accepted") and d.get("inputs"):
                return {str(k): float(v) for k, v in d["inputs"].items()}
        except Exception as e:  # noqa: BLE001
            log.warning("rank weights file %s unreadable: %s", f, e)
    return dict(fallback or DEFAULT_INPUTS)


def load_tuned_profile(models_dir, account: str = "main", config_base: dict | None = None) -> dict | None:
    """The structure in force for an account (``models/rank_profile_<account>.json``: top_k, every_bars, hysteresis, core_share):
    what the weekend tuner adopted after two consecutive weekly passes, kept until it adopts something else.  A file tuned
    from a different config than ``config_base`` is ignored: an edit to the config wins until the tuner runs again."""
    import json
    from pathlib import Path

    f = Path(models_dir) / f"rank_profile_{account}.json"
    if f.exists():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            prof = d.get("profile")
            in_force = bool(prof) and ("guard_passed" in d or d.get("accepted"))
            base = d.get("config_base")
            if in_force and config_base is not None and isinstance(base, dict):
                for k, v in config_base.items():
                    if k in base and (base[k] != v if not isinstance(v, (int, float)) or isinstance(v, bool) else abs(float(base[k]) - float(v)) > 1e-9):
                        log.info("rank profile %s tuned from a different config (%s: %s -> %s) - the config wins until the next tune",
                                 account, k, base[k], v)
                        return None
            if in_force:
                return dict(prof)
        except Exception as e:  # noqa: BLE001
            log.warning("rank profile file %s unreadable: %s", f, e)
    return None


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


def select_top(scores: dict[str, float], held: list[str], k: int, hysteresis: int = 3, sectors: dict[str, str] | None = None,
               max_per_sector: int = 0) -> list[str]:
    """Top-``k`` names by score; a held name keeps its slot while it ranks within ``k + hysteresis``; at most
    ``max_per_sector`` names from one sector (0 = no cap; names without a sector are never capped)."""
    ranked = [t for t, s in sorted(scores.items(), key=lambda kv: -kv[1]) if np.isfinite(s)]
    rank = {t: i for i, t in enumerate(ranked)}
    counts: dict[str, int] = {}

    def fits(t: str) -> bool:
        if not max_per_sector or not sectors:
            return True
        sec = sectors.get(t)
        return not sec or sec in ("Unknown", "Index") or counts.get(sec, 0) < max_per_sector

    def take(t: str) -> None:
        sec = (sectors or {}).get(t)
        if sec:
            counts[sec] = counts.get(sec, 0) + 1

    keep: list[str] = []
    for t in sorted([t for t in held if t in rank and rank[t] < k + hysteresis], key=lambda t: rank[t]):
        if len(keep) < k and fits(t):
            keep.append(t)
            take(t)
    for t in ranked:
        if len(keep) >= k:
            break
        if t not in keep and fits(t):
            keep.append(t)
            take(t)
    return keep


def vol_scale(daily_returns, target_vol: float, window: int = 20, floor: float = 0.4, cap: float = 1.0) -> float:
    """Volatility targeting (Moreira & Muir): the multiplier on gross exposure that brings the book's recent realised
    volatility to ``target_vol`` (annualised), between ``floor`` and ``cap``.  1.0 while there is no history yet."""
    r = np.asarray([x for x in daily_returns if x is not None and np.isfinite(x)], dtype=float)
    if target_vol <= 0 or len(r) < max(5, window // 2):
        return 1.0
    vol = float(np.std(r[-window:], ddof=1) * np.sqrt(252))
    if vol <= 1e-9:
        return 1.0
    return float(np.clip(target_vol / vol, floor, cap))


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
