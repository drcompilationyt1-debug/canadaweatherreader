"""Classic technical indicators, pre-scaled to roughly unit range so the policy needs no normalizer."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9
CLIP = 5.0

TECH_FEATURES = [
    "ret_1", "ret_5", "ret_20", "ret_60",
    "vol_20", "vol_60",
    "rsi_14", "macd_hist", "bb_pctb", "bb_width", "atr_pct",
    "sma20_r", "sma50_r", "sma200_r", "ema_cross",
    "vol_z", "hi52", "lo52", "stoch_k", "obv_slope", "gap", "range_pct",
]


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_up / (avg_down + EPS)
    return 100.0 - 100.0 / (1.0 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def compute_technical(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame (same index) with the ``TECH_FEATURES`` columns; warm-up rows are NaN."""
    c = df["close"].astype(float)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)
    logc = np.log(c.clip(lower=EPS))
    r1 = logc.diff()

    out = pd.DataFrame(index=df.index)
    out["ret_1"] = r1 * 20.0
    out["ret_5"] = logc.diff(5) * 10.0
    out["ret_20"] = logc.diff(20) * 5.0
    out["ret_60"] = logc.diff(60) * 3.0
    out["vol_20"] = r1.rolling(20).std() * np.sqrt(252) * 3.0 - 1.0
    out["vol_60"] = r1.rolling(60).std() * np.sqrt(252) * 3.0 - 1.0
    out["rsi_14"] = (rsi(c, 14) - 50.0) / 25.0

    ema12, ema26 = _ema(c, 12), _ema(c, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    out["macd_hist"] = (macd - signal) / c * 100.0

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    out["bb_pctb"] = ((c - (sma20 - 2 * std20)) / (4 * std20 + EPS) - 0.5) * 2.0
    out["bb_width"] = 4 * std20 / (sma20 + EPS) * 10.0 - 1.0
    out["atr_pct"] = atr(df, 14) / c * 50.0 - 1.0

    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    out["sma20_r"] = (c / sma20 - 1.0) * 20.0
    out["sma50_r"] = (c / sma50 - 1.0) * 10.0
    out["sma200_r"] = (c / sma200 - 1.0) * 5.0
    out["ema_cross"] = (ema12 / ema26 - 1.0) * 50.0

    lv = np.log(v + 1.0)
    out["vol_z"] = (lv - lv.rolling(20).mean()) / (lv.rolling(20).std() + EPS)
    out["hi52"] = (c / h.rolling(252).max() - 1.0) * 5.0
    out["lo52"] = (c / l.rolling(252).min() - 1.0) * 2.0

    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    out["stoch_k"] = ((c - lo14) / (hi14 - lo14 + EPS) - 0.5) * 2.0

    obv = (np.sign(c.diff().fillna(0.0)) * v).cumsum()
    out["obv_slope"] = (obv - obv.shift(20)) / (v.rolling(20).mean() * 20.0 + EPS) * 2.0
    out["gap"] = (o / c.shift(1) - 1.0) * 50.0
    out["range_pct"] = (h - l) / c * 30.0 - 1.0

    out = out[TECH_FEATURES].replace([np.inf, -np.inf], np.nan).clip(-CLIP, CLIP)
    return out
