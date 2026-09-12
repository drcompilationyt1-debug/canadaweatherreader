"""Market regime block: FinRL-style turbulence index + market return / volatility / breadth.

The turbulence index (Kritzman & Li, used by FinRL's ``FeatureEngineer``) is the Mahalanobis
distance of today's cross-section of returns from the trailing-window distribution: high values
flag crisis regimes where FinRL's agents stop buying.  Computed natively here; when the FinRL
submodule is importable its ``FeatureEngineer`` implementation is reported in ``doctor``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..paths import import_optional
from .base import SignalProvider

log = get_logger(__name__)


def turbulence_index(returns: pd.DataFrame, window: int = 252) -> pd.Series:
    R = returns.to_numpy(float)
    n_t, _ = R.shape
    out = np.full(n_t, np.nan)
    for i in range(window, n_t):
        cur = R[i]
        win = R[i - window:i]
        cols = ~np.isnan(cur) & (np.isnan(win).mean(axis=0) < 0.2)
        if cols.sum() < 2:
            continue
        w = win[:, cols]
        w = w[~np.isnan(w).any(axis=1)]
        if len(w) < window // 2:
            continue
        mu = w.mean(axis=0)
        cov = np.cov(w, rowvar=False) + 1e-8 * np.eye(cols.sum())
        d = cur[cols] - mu
        try:
            out[i] = float(d @ np.linalg.solve(cov, d)) / cols.sum()
        except np.linalg.LinAlgError:
            continue
    return pd.Series(out, index=returns.index)


class MarketRegimeSignal(SignalProvider):
    name = "market_regime"
    feature_names = ["turbulence", "turbulence_pct", "mkt_ret_20", "mkt_vol_20", "breadth"]
    needs_universe = True
    tier = "A"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.window = int(self.cfg.get("window", 252))
        self._cache: dict[int, pd.DataFrame] = {}

    def availability(self) -> tuple[bool, str]:
        frames = self.ctx.extra.get("frames")
        if not frames or len(frames) < 2:
            return False, "needs >= 2 tickers in context (universe frames)"
        finrl = import_optional("finrl.meta.preprocessor.preprocessors", "FinRL")
        return True, "native turbulence" + (" (FinRL FeatureEngineer importable)" if finrl else "")

    def _market_table(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        key = id(frames)
        if key in self._cache:
            return self._cache[key]
        rets = pd.concat({t: np.log(df["close"].astype(float)).diff() for t, df in frames.items()}, axis=1)
        rets = rets.dropna(how="all").sort_index()
        turb = turbulence_index(rets, self.window)
        mkt = rets.mean(axis=1)
        above = pd.concat({t: (df["close"] > df["close"].rolling(50).mean()).astype(float).where(df["close"].rolling(50).mean().notna())
                           for t, df in frames.items()}, axis=1).reindex(rets.index)
        table = pd.DataFrame(index=rets.index)
        table["turbulence"] = np.log1p(turb) - 1.0
        table["turbulence_pct"] = turb.rolling(252, min_periods=60).rank(pct=True) * 2.0 - 1.0
        table["mkt_ret_20"] = mkt.rolling(20).sum() * 5.0
        table["mkt_vol_20"] = mkt.rolling(20).std() * np.sqrt(252) * 3.0 - 1.0
        table["breadth"] = above.mean(axis=1) * 2.0 - 1.0
        table = table.clip(-5, 5)
        self._cache = {key: table}
        return table

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        frames = self.ctx.extra.get("frames")
        if not frames:
            return None
        table = self._market_table(frames).reindex(df.index)
        return table.to_numpy(np.float32)

    def compute_history_all(self, frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray | None]:
        self.ctx.extra["frames"] = frames
        table = self._market_table(frames)
        return {t: table.reindex(df.index).to_numpy(np.float32) for t, df in frames.items()}
