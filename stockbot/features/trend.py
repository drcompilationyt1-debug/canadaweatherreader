"""Trend features: regression slopes, R^2, ADX, moving-average regime, trend age."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9

TREND_FEATURES = [
    "slope_10", "slope_30", "slope_90", "r2_30", "r2_90",
    "adx_14", "di_diff", "above_sma200", "golden_cross", "trend_age",
]


def rolling_slope(y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling OLS slope of y on t=0..k-1 and R^2, vectorised with a convolution."""
    n = len(y)
    slope = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    if n < k:
        return slope, r2
    t = np.arange(k, dtype=float)
    tc = t - t.mean()
    denom = float((tc ** 2).sum())
    w = tc / denom  # slope = sum(w_i * y_{t-k+1+i})
    conv = np.convolve(y, w[::-1], mode="valid")  # length n-k+1, aligned to window end
    slope[k - 1:] = conv
    ys = pd.Series(y)
    var_y = ys.rolling(k).var(ddof=0).to_numpy()
    var_t = float(tc.var())
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_full = (slope ** 2) * var_t / np.where(var_y > EPS, var_y, np.nan)
    r2[:] = np.clip(np.nan_to_num(r2_full, nan=0.0), 0.0, 1.0)
    r2[: k - 1] = np.nan
    return slope, r2


def adx(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    down = -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / (atr + EPS)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / (atr + EPS)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    return dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean(), plus_di, minus_di


def compute_trend(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].astype(float)
    logc = np.log(c.clip(lower=EPS)).to_numpy()
    out = pd.DataFrame(index=df.index)
    for k in (10, 30, 90):
        s, r2 = rolling_slope(logc, k)
        out[f"slope_{k}"] = np.tanh(s * 252.0 / 2.0)  # annualised log-slope, squashed
        if k in (30, 90):
            out[f"r2_{k}"] = r2 * 2.0 - 1.0
    a, pdi, mdi = adx(df, 14)
    out["adx_14"] = a / 50.0 - 1.0
    out["di_diff"] = (pdi - mdi) / 50.0
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    out["above_sma200"] = np.where(sma200.isna(), np.nan, np.where(c > sma200, 1.0, -1.0))
    out["golden_cross"] = np.where(sma200.isna(), np.nan, np.where(sma50 > sma200, 1.0, -1.0))
    sign = np.sign(np.nan_to_num(out["slope_30"].to_numpy(), nan=0.0))
    age = np.zeros(len(sign))
    for i in range(1, len(sign)):
        age[i] = age[i - 1] + 1 if sign[i] == sign[i - 1] and sign[i] != 0 else 0
    out["trend_age"] = np.clip(age / 60.0, 0.0, 2.0) * sign
    return out[TREND_FEATURES].replace([np.inf, -np.inf], np.nan).clip(-5.0, 5.0)
