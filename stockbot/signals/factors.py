"""Classic cross-sectional factors from prices and volume, ranked across the universe every day.

The best-documented return predictors that need nothing but bars: 12-1 momentum (Jegadeesh and Titman 1993), 6-1
momentum, one-month reversal, nearness to the 52-week high (George and Hwang 2004), dollar volume (size / liquidity),
idiosyncratic volatility (Ang et al. 2006), market beta (Frazzini and Pedersen's betting against beta) and the 3-month
return relative to the name's own sector (a sector-neutral momentum).  Every value is the name's percentile across the
universe that day, scaled to [-1, 1] (comparable across names and time, bounded for the policy); the raw 12-1 momentum
and the raw distance from the 52-week high are kept as well.  Sectors come from ``models/sectors.json`` (Yahoo,
refreshed weekly) - today's sector applied through history, the usual concession with free data.  Nothing here looks
ahead: every input is a trailing window ending at the bar itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .base import SignalProvider

log = get_logger(__name__)

FEATURES = ["mom_12_1", "mom_6_1", "rev_1", "hi_52w", "dvol", "ivol", "beta", "sec_rel_3m", "mom_12_1_raw", "hi_52w_raw"]
RANKED = FEATURES[:8]
MIN_NAMES = 5                      # a percentile across fewer names than this is noise: masked


def raw_factors(close: pd.DataFrame, volume: pd.DataFrame, sectors: dict[str, str], market: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    """Each factor as a (dates x tickers) frame from union-indexed closes and volumes (NaN where a name has no bar)."""
    c = close.ffill(limit=5)
    r = c.pct_change(fill_method=None)
    out = {"mom_12_1": c.shift(21) / c.shift(252) - 1.0, "mom_6_1": c.shift(21) / c.shift(126) - 1.0, "rev_1": c / c.shift(21) - 1.0,
           "hi_52w": c / c.rolling(252, min_periods=200).max() - 1.0,
           "dvol": np.log((c * volume.fillna(0.0)).rolling(20, min_periods=15).mean().clip(lower=1.0))}
    mkt = market.reindex(c.index) if market is not None else r.mean(axis=1)
    var = mkt.rolling(60, min_periods=40).var().replace(0.0, np.nan)
    beta = pd.DataFrame({t: r[t].rolling(60, min_periods=40).cov(mkt) for t in r.columns}, index=r.index).div(var, axis=0)
    resid = r - beta.mul(mkt, axis=0)
    out["ivol"] = resid.rolling(60, min_periods=40).std()
    out["beta"] = beta
    ret_63 = c / c.shift(63) - 1.0
    rel = ret_63.copy()
    groups: dict[str, list[str]] = {}
    for t in c.columns:
        sec = sectors.get(t) or "Unknown"
        groups.setdefault(sec if sec not in ("Unknown", "Index") else "_all", []).append(t)
    universe_mean = ret_63.mean(axis=1)
    for sec, names in groups.items():
        peer = ret_63[names].mean(axis=1) if sec != "_all" and len(names) >= 2 else universe_mean
        rel[names] = ret_63[names].sub(peer, axis=0)
    out["sec_rel_3m"] = rel
    return out


def ranked_factors(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Percentile across the universe per day, scaled to [-1, 1]; rows with fewer than MIN_NAMES values are NaN."""
    out = {}
    for name in RANKED:
        f = raw[name]
        n = f.notna().sum(axis=1)
        rk = f.rank(axis=1, pct=True) * 2.0 - 1.0
        rk[n < MIN_NAMES] = np.nan
        out[name] = rk
    out["mom_12_1_raw"] = raw["mom_12_1"].clip(-1.0, 3.0)
    out["hi_52w_raw"] = raw["hi_52w"].clip(-1.0, 0.0)
    return out


class FactorSignal(SignalProvider):
    name = "factors"
    feature_names = list(FEATURES)
    needs_universe = True
    tier = "A"

    def _sectors(self, tickers: list[str]) -> dict[str, str]:
        given = self.ctx.extra.get("sectors")
        if isinstance(given, dict) and given:
            return {t: str(given.get(t, "Unknown")) for t in tickers}
        try:
            from ..data.sectors import load_sectors

            return load_sectors(self.root_cfg, tickers, refresh=not bool(self.ctx.extra.get("offline", False)))
        except Exception as e:  # noqa: BLE001
            log.debug("factors: sectors unavailable (%s) - the sector-relative return falls back to the universe", e)
            return {t: "Unknown" for t in tickers}

    def compute_history_all(self, frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray | None]:
        if not frames:
            return {}
        close = pd.DataFrame({t: df["close"].astype(float) for t, df in frames.items()})
        volume = pd.DataFrame({t: df["volume"].astype(float) for t, df in frames.items()}).reindex(close.index)
        market = close["SPY"].ffill(limit=5).pct_change(fill_method=None) if "SPY" in close.columns else None
        feats = ranked_factors(raw_factors(close, volume, self._sectors(list(frames)), market))
        out = {}
        for t, df in frames.items():
            arr = np.column_stack([feats[name][t].reindex(df.index).to_numpy(np.float32) for name in FEATURES])
            out[t] = arr.astype(np.float32)
        return out

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        frames = self.ctx.extra.get("frames") or {}
        if ticker in frames:
            return self.compute_history_all(frames).get(ticker)
        return self.compute_history_all({ticker: df}).get(ticker)
