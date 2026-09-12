"""Strategy zoo: what classic rule-based strategies would do today, as features.

Every column is a position / indication in {-1, 0, +1}, computed causally.  Sources:

* QuantConnect/Lean ``Algorithm.Python`` - MovingAverageCrossAlgorithm (EMA 15/30, tolerance
  0.00015), MACDTrendAlgorithm (12/26/9, tolerance 0.0025 of the fast EMA), RsiAlphaModel (30/70),
  EmaCrossAlphaModel (12/26), plus Bollinger mean reversion and 12-1 momentum from the framework
  alpha models.
* huseinzol05/Stock-Prediction-Models ``agent/`` - turtle agent, moving-average agent, signal
  rolling agent.
* akurgat/automating-technical-analysis ``app/indicator_analysis.py`` - engulfing, MACD, RSI,
  stochastic+RSI, MA-cross and support/resistance *indications* (their encoding: 2 bullish,
  0 bearish, 1 neutral -> mapped to +1/-1/0) and the EWM-combined action with the pivot filter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .technical import _ema, rsi

EPS = 1e-9

STRATEGY_FEATURES = [
    "lean_ema_cross", "lean_macd_trend", "lean_rsi", "lean_ema_alpha", "lean_bollinger", "lean_momentum",
    "hz_turtle", "hz_ma_agent", "hz_signal_rolling",
    "ak_engulfing", "ak_macd", "ak_rsi", "ak_stochastic", "ak_ma_cross", "ak_support", "ak_action",
    "vote",
]


def _stateful(enter_long: np.ndarray, exit_long: np.ndarray, enter_short: np.ndarray | None = None,
              exit_short: np.ndarray | None = None) -> np.ndarray:
    """Position state machine: +1 after enter_long until exit_long, -1 likewise for shorts."""
    n = len(enter_long)
    pos = np.zeros(n)
    cur = 0.0
    el, xl = np.nan_to_num(enter_long.astype(float)), np.nan_to_num(exit_long.astype(float))
    es = np.nan_to_num(enter_short.astype(float)) if enter_short is not None else np.zeros(n)
    xs = np.nan_to_num(exit_short.astype(float)) if exit_short is not None else np.zeros(n)
    for i in range(n):
        if cur > 0 and xl[i]:
            cur = 0.0
        elif cur < 0 and xs[i]:
            cur = 0.0
        if cur == 0.0:
            if el[i]:
                cur = 1.0
            elif es[i]:
                cur = -1.0
        pos[i] = cur
    return pos


def compute_strategies(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].astype(float)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    cv = c.to_numpy()
    out = pd.DataFrame(index=df.index)

    # ---- Lean ---------------------------------------------------------------------------
    fast, slow = _ema(c, 15), _ema(c, 30)
    out["lean_ema_cross"] = _stateful((fast > slow * (1 + 0.00015)).to_numpy(), (fast < slow).to_numpy())
    macd = _ema(c, 12) - _ema(c, 26)
    signal = _ema(macd, 9)
    delta = (macd - signal) / (_ema(c, 12) + EPS)
    out["lean_macd_trend"] = _stateful((delta > 0.0025).to_numpy(), (delta < -0.0025).to_numpy())
    r = rsi(c, 14)
    out["lean_rsi"] = _stateful((r < 30).to_numpy(), (r > 50).to_numpy(), (r > 70).to_numpy(), (r < 50).to_numpy())
    e12, e26 = _ema(c, 12), _ema(c, 26)
    out["lean_ema_alpha"] = np.where(e12.isna() | e26.isna(), np.nan, np.where(e12 > e26, 1.0, -1.0))
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    out["lean_bollinger"] = _stateful((c < mid - 2 * sd).to_numpy(), (c >= mid).to_numpy(),
                                      (c > mid + 2 * sd).to_numpy(), (c <= mid).to_numpy())
    mom = c.shift(21) / c.shift(252) - 1.0
    out["lean_momentum"] = np.where(mom.isna(), np.nan, np.sign(mom))

    # ---- huseinzol05 agents -------------------------------------------------------------
    lo20, hi20 = l.rolling(20).min(), h.rolling(20).max()
    out["hz_turtle"] = np.where(lo20.isna(), np.nan, np.where(c <= lo20, 1.0, np.where(c >= hi20, -1.0, 0.0)))
    sma = c.rolling(20).mean()
    out["hz_ma_agent"] = np.where(sma.isna(), np.nan, np.where(c > sma, 1.0, -1.0))
    ret = c.diff()
    down4 = (ret <= 0).rolling(4).sum() == 4
    up4 = (ret >= 0).rolling(4).sum() == 4
    out["hz_signal_rolling"] = np.where(down4, 1.0, np.where(up4, -1.0, 0.0))

    # ---- akurgat indications (2 -> +1 bullish, 0 -> -1 bearish, 1 -> 0) --------------------
    bull_engulf = (c > o) & (c.shift(1) < o.shift(1)) & (c > o.shift(1)) & (o < c.shift(1))
    bear_engulf = (c < o) & (c.shift(1) > o.shift(1)) & (c < o.shift(1)) & (o > c.shift(1))
    ak = pd.DataFrame(index=df.index)
    ak["ak_engulfing"] = np.where(bull_engulf, 1.0, np.where(bear_engulf, -1.0, 0.0))
    ak["ak_macd"] = np.where(macd < signal, 1.0, np.where(macd > signal, -1.0, 0.0))
    ak["ak_rsi"] = np.where(r <= 30, 1.0, np.where(r >= 70, -1.0, 0.0))
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    k = 100 * (c - lo14) / (hi14 - lo14 + EPS)
    d = k.rolling(3).mean()
    ak["ak_stochastic"] = np.where((k < d) & (k <= 20) & (r <= 30), 1.0, np.where((k > d) & (k >= 80) & (r >= 70), -1.0, 0.0))
    sma_s, sma_l = c.rolling(20).mean(), c.rolling(50).mean()
    ak["ak_ma_cross"] = np.where((sma_s > sma_l) & (sma_s.shift(20) < sma_l.shift(50)), 1.0,
                                 np.where((sma_s < sma_l) & (sma_s.shift(20) > sma_l.shift(50)), -1.0, 0.0))
    ak["ak_support"] = np.where((sma_s > c) & (sma_s.shift(20) < c.shift(20)), 1.0,
                                np.where((sma_s < c) & (sma_s.shift(20) > c.shift(20)), -1.0, 0.0))
    indication = (ak.mean(axis=1) + 1.0).ewm(com=2).mean()  # back to their 0..2 scale, smoothed
    pivot = ((h + l + c) / 3.0).shift(1)
    ak["ak_action"] = np.where((indication >= 1.25) & (c <= pivot), 1.0, np.where((indication <= 0.75) & (c >= pivot), -1.0, 0.0))
    for col in ak.columns:
        out[col] = ak[col]

    out["vote"] = out.drop(columns=["vote"], errors="ignore").mean(axis=1)
    out.iloc[:60] = np.nan
    return out[STRATEGY_FEATURES]
