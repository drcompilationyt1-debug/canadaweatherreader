"""Volatility-targeted position sizing, shared by the simulator, the live runner and backtrader.

The policy outputs conviction in [-1, 1]; the size of the position is scaled so that a full
conviction position runs at roughly ``vol_target`` annualised volatility.  Calm stocks get more
capital, wild ones less - the single most reliable Sharpe improvement in systematic trading.
"""
from __future__ import annotations

import numpy as np


def realized_vol(close: np.ndarray, window: int = 20) -> float:
    """Annualised standard deviation of the last ``window`` daily log returns (NaN if too short)."""
    c = np.asarray(close, dtype=float)
    if len(c) < window + 1:
        return float("nan")
    r = np.diff(np.log(np.maximum(c[-window - 1:], 1e-9)))
    return float(np.std(r, ddof=1) * np.sqrt(252.0))


def realized_vol_series(close: np.ndarray, window: int = 20) -> np.ndarray:
    """Per-bar annualised volatility of the trailing ``window`` log returns (NaN during warm-up)."""
    import pandas as pd

    c = pd.Series(np.asarray(close, dtype=float)).clip(lower=1e-9)
    return (np.log(c).diff().rolling(window).std(ddof=1) * np.sqrt(252.0)).to_numpy()


def conviction_to_exposure(action: float, allow_short: bool) -> float:
    """Map the policy's action in [-1, 1] to a raw exposure.

    Long-only: the whole range is used, ``-1`` = flat, ``0`` = half invested, ``+1`` = fully
    invested (an untrained policy therefore starts half invested instead of flat).
    With shorting: the action is the signed exposure itself.
    """
    a = float(np.clip(action, -1.0, 1.0))
    return a if allow_short else (a + 1.0) / 2.0


def size_exposure(conviction: float, vol: float, vol_target: float, max_leverage: float = 1.0,
                  max_scale: float = 1.5) -> float:
    """Exposure = conviction * min(max_scale, vol_target / vol), capped at +-max_leverage."""
    conviction = float(np.clip(conviction, -1.0, 1.0))
    if vol_target <= 0 or not np.isfinite(vol) or vol <= 1e-6:
        return float(np.clip(conviction, -max_leverage, max_leverage))
    scale = min(float(max_scale), float(vol_target) / float(vol))
    return float(np.clip(conviction * scale, -max_leverage, max_leverage))
