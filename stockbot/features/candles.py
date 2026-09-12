"""Candlestick pattern detection + empirical "what usually happens next" statistics.

``detect_patterns`` flags the most widely used single-, double- and triple-candle patterns using
the textbook definitions (body / shadow ratios and the trend leading into the pattern) - no
external library needed.  TA-Lib's 61 recognisers are available as a second block through
``stockbot.signals.talib_candles`` when TA-Lib is installed.

``PatternStats`` measures, on historical data, the forward return distribution after each
pattern (win rate and mean return over several horizons, relative to the unconditional
baseline).  The policy therefore receives both *which* patterns just fired and *how much edge*
those patterns historically carried.  It works for any set of 0/1 pattern columns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12

# pattern -> direction (+1 bullish, -1 bearish, 0 neutral / indecision)
PATTERN_DIRECTION: dict[str, int] = {
    "doji": 0,
    "hammer": 1,
    "inverted_hammer": 1,
    "hanging_man": -1,
    "shooting_star": -1,
    "bullish_engulfing": 1,
    "bearish_engulfing": -1,
    "bullish_harami": 1,
    "bearish_harami": -1,
    "piercing_line": 1,
    "dark_cloud_cover": -1,
    "morning_star": 1,
    "evening_star": -1,
    "three_white_soldiers": 1,
    "three_black_crows": -1,
    "bullish_marubozu": 1,
    "bearish_marubozu": -1,
    "tweezer_bottom": 1,
    "tweezer_top": -1,
}
PATTERNS = list(PATTERN_DIRECTION)


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """a[t-k] with NaN padding (k > 0 looks back)."""
    out = np.full_like(a, np.nan, dtype=float)
    if k == 0:
        return a.astype(float)
    out[k:] = a[:-k]
    return out


def detect_patterns(df: pd.DataFrame, trend_lookback: int = 5, avg_window: int = 14) -> pd.DataFrame:
    """Return an int8 DataFrame (one column per pattern, 1 = pattern fired on that bar)."""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(c)

    body = c - o
    abody = np.abs(body)
    rng = np.maximum(h - l, EPS)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    bull = body > 0
    bear = body < 0
    avg_body = pd.Series(abody).rolling(avg_window, min_periods=3).mean().to_numpy()
    avg_body = np.where(np.isnan(avg_body), np.nanmean(abody[: max(3, avg_window)]) if n else 0.0, avg_body)
    long_body = abody > 1.0 * avg_body
    # prior trend measured over the bars *before* the pattern's first candle
    prior = pd.Series(c).pct_change(trend_lookback).to_numpy()
    down_before_1 = _shift(prior, 1) < 0  # trend ending at t-1
    up_before_1 = _shift(prior, 1) > 0
    down_before_2 = _shift(prior, 2) < 0
    up_before_2 = _shift(prior, 2) > 0
    down_before_3 = _shift(prior, 3) < 0
    up_before_3 = _shift(prior, 3) > 0

    o1, c1, h1, l1 = _shift(o, 1), _shift(c, 1), _shift(h, 1), _shift(l, 1)
    o2, c2 = _shift(o, 2), _shift(c, 2)
    body1, body2 = c1 - o1, c2 - o2
    abody1, abody2 = np.abs(body1), np.abs(body2)
    avg1 = _shift(avg_body, 1)
    avg2 = _shift(avg_body, 2)

    P: dict[str, np.ndarray] = {}
    P["doji"] = abody <= 0.1 * rng

    hammer_shape = (lower >= 2.0 * abody) & (upper <= 0.15 * rng) & (abody > 0.05 * rng)
    inv_shape = (upper >= 2.0 * abody) & (lower <= 0.15 * rng) & (abody > 0.05 * rng)
    P["hammer"] = hammer_shape & down_before_1
    P["hanging_man"] = hammer_shape & up_before_1
    P["inverted_hammer"] = inv_shape & down_before_1
    P["shooting_star"] = inv_shape & up_before_1

    P["bullish_engulfing"] = bull & (body1 < 0) & (o <= c1) & (c >= o1) & (abody > abody1) & (abody > 0.5 * avg_body)
    P["bearish_engulfing"] = bear & (body1 > 0) & (o >= c1) & (c <= o1) & (abody > abody1) & (abody > 0.5 * avg_body)

    P["bullish_harami"] = (body1 < 0) & (abody1 > avg1) & bull & (o > c1) & (c < o1) & (abody < 0.6 * abody1)
    P["bearish_harami"] = (body1 > 0) & (abody1 > avg1) & bear & (o < c1) & (c > o1) & (abody < 0.6 * abody1)

    mid1 = (o1 + c1) / 2.0
    P["piercing_line"] = (body1 < 0) & (abody1 > 0.5 * avg1) & bull & (o < c1) & (c > mid1) & (c < o1) & down_before_2
    P["dark_cloud_cover"] = (body1 > 0) & (abody1 > 0.5 * avg1) & bear & (o > c1) & (c < mid1) & (c > o1) & up_before_2

    mid2 = (o2 + c2) / 2.0
    P["morning_star"] = (body2 < 0) & (abody2 > avg2) & (abody1 < 0.5 * avg1) & bull & (c > mid2) & down_before_3
    P["evening_star"] = (body2 > 0) & (abody2 > avg2) & (abody1 < 0.5 * avg1) & bear & (c < mid2) & up_before_3

    bull1, bull2 = body1 > 0, body2 > 0
    bear1, bear2 = body1 < 0, body2 < 0
    P["three_white_soldiers"] = (
        bull & bull1 & bull2 & (c > c1) & (c1 > c2) & (o > o1) & (o1 > o2)
        & (o < c1) & (o1 < c2) & (upper < 0.3 * abody) & (abody > 0.5 * avg_body)
    )
    P["three_black_crows"] = (
        bear & bear1 & bear2 & (c < c1) & (c1 < c2) & (o < o1) & (o1 < o2)
        & (o > c1) & (o1 > c2) & (lower < 0.3 * abody) & (abody > 0.5 * avg_body)
    )

    P["bullish_marubozu"] = bull & (abody >= 0.9 * rng) & long_body
    P["bearish_marubozu"] = bear & (abody >= 0.9 * rng) & long_body

    close_lows = np.abs(l - l1) <= 0.002 * np.maximum(l, EPS)
    close_highs = np.abs(h - h1) <= 0.002 * np.maximum(h, EPS)
    P["tweezer_bottom"] = close_lows & (body1 < 0) & bull & down_before_2
    P["tweezer_top"] = close_highs & (body1 > 0) & bear & up_before_2

    out = pd.DataFrame({k: np.nan_to_num(P[k].astype(float), nan=0.0).astype(np.int8) for k in PATTERNS}, index=df.index)
    out.iloc[: max(3, avg_window)] = 0
    return out


@dataclass
class PatternStats:
    """Forward-return statistics conditioned on each pattern (pooled over the fitted tickers).

    Works for any 0/1 pattern columns: pass ``patterns`` / ``directions`` for a custom set
    (e.g. TA-Lib's recognisers); the defaults are the native ``PATTERNS``.
    """

    horizons: list[int] = field(default_factory=lambda: [1, 5, 20])
    patterns: list[str] = field(default_factory=lambda: list(PATTERNS))
    directions: dict[str, int] = field(default_factory=lambda: dict(PATTERN_DIRECTION))
    counts: dict[str, int] = field(default_factory=dict)
    mean_ret: dict[str, dict[int, float]] = field(default_factory=dict)      # pattern -> h -> mean fwd log ret
    win_rate: dict[str, dict[int, float]] = field(default_factory=dict)
    base_mean: dict[int, float] = field(default_factory=dict)
    base_win: dict[int, float] = field(default_factory=dict)
    n_bars: int = 0
    fitted: bool = False

    # ------------------------------------------------------------------ fit
    def fit(self, frames: dict[str, pd.DataFrame], patterns: dict[str, pd.DataFrame] | None = None,
            end: str | pd.Timestamp | None = None) -> "PatternStats":
        end_ts = pd.Timestamp(end) if end is not None else None
        names = self.patterns
        sums = {p: {h: 0.0 for h in self.horizons} for p in names}
        wins = {p: {h: 0 for h in self.horizons} for p in names}
        cnt = {p: {h: 0 for h in self.horizons} for p in names}
        bsum = {h: 0.0 for h in self.horizons}
        bwin = {h: 0 for h in self.horizons}
        bcnt = {h: 0 for h in self.horizons}
        for t, df in frames.items():
            if end_ts is not None:
                df = df[df.index <= end_ts]
            if len(df) < 50:
                continue
            pat = patterns[t].loc[df.index] if patterns and t in patterns else detect_patterns(df)
            logc = np.log(df["close"].to_numpy(float))
            for h in self.horizons:
                fwd = np.full(len(logc), np.nan)
                fwd[:-h] = logc[h:] - logc[:-h]
                ok = ~np.isnan(fwd)
                bsum[h] += float(np.nansum(fwd))
                bwin[h] += int(np.sum(fwd[ok] > 0))
                bcnt[h] += int(ok.sum())
                for p in names:
                    if p not in pat:
                        continue
                    fired = (pat[p].to_numpy() > 0) & ok
                    k = int(fired.sum())
                    if k:
                        sums[p][h] += float(fwd[fired].sum())
                        wins[p][h] += int((fwd[fired] > 0).sum())
                        cnt[p][h] += k
        self.base_mean = {h: bsum[h] / max(bcnt[h], 1) for h in self.horizons}
        self.base_win = {h: bwin[h] / max(bcnt[h], 1) for h in self.horizons}
        self.n_bars = int(bcnt[self.horizons[0]]) if self.horizons else 0
        self.counts = {p: cnt[p][self.horizons[0]] for p in names}
        self.mean_ret = {p: {h: (sums[p][h] / cnt[p][h]) if cnt[p][h] else self.base_mean[h] for h in self.horizons}
                         for p in names}
        self.win_rate = {p: {h: (wins[p][h] / cnt[p][h]) if cnt[p][h] else self.base_win[h] for h in self.horizons}
                         for p in names}
        self.fitted = True
        return self

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "horizons": self.horizons, "patterns": self.patterns, "directions": self.directions,
            "counts": self.counts, "n_bars": self.n_bars, "fitted": self.fitted,
            "mean_ret": {p: {str(h): v for h, v in d.items()} for p, d in self.mean_ret.items()},
            "win_rate": {p: {str(h): v for h, v in d.items()} for p, d in self.win_rate.items()},
            "base_mean": {str(h): v for h, v in self.base_mean.items()},
            "base_win": {str(h): v for h, v in self.base_win.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PatternStats":
        ps = cls(horizons=[int(h) for h in d["horizons"]], patterns=list(d.get("patterns", PATTERNS)),
                 directions={k: int(v) for k, v in d.get("directions", PATTERN_DIRECTION).items()})
        ps.counts = {k: int(v) for k, v in d.get("counts", {}).items()}
        ps.n_bars = int(d.get("n_bars", 0))
        ps.mean_ret = {p: {int(h): float(v) for h, v in dd.items()} for p, dd in d["mean_ret"].items()}
        ps.win_rate = {p: {int(h): float(v) for h, v in dd.items()} for p, dd in d["win_rate"].items()}
        ps.base_mean = {int(h): float(v) for h, v in d["base_mean"].items()}
        ps.base_win = {int(h): float(v) for h, v in d["base_win"].items()}
        ps.fitted = bool(d.get("fitted", True))
        return ps

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PatternStats":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ reporting
    def table(self) -> pd.DataFrame:
        rows = []
        for p in self.patterns:
            row = {"pattern": p, "direction": self.directions.get(p, 0), "n": self.counts.get(p, 0)}
            for h in self.horizons:
                row[f"ret_{h}d_%"] = 100 * self.mean_ret.get(p, {}).get(h, np.nan)
                row[f"edge_{h}d_%"] = 100 * (self.mean_ret.get(p, {}).get(h, np.nan) - self.base_mean.get(h, 0.0))
                row[f"win_{h}d_%"] = 100 * self.win_rate.get(p, {}).get(h, np.nan)
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ features
    def edge_arrays(self, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-pattern vectors: return edge, win-rate edge, and sample-size support in [0, 1]."""
        edge = np.array([self.mean_ret.get(p, {}).get(h, 0.0) - self.base_mean.get(h, 0.0) for p in self.patterns])
        wedge = np.array([self.win_rate.get(p, {}).get(h, 0.5) - self.base_win.get(h, 0.5) for p in self.patterns])
        support = np.array([min(1.0, self.counts.get(p, 0) / 100.0) for p in self.patterns])
        return edge, wedge, support


STAT_FEATURES = ["n_recent", "direction_score", "edge_5", "edge_20", "win_edge_5", "win_edge_20"]


def pattern_features(pat: pd.DataFrame, stats: PatternStats | None, decay: list[float] | None = None,
                     patterns: list[str] | None = None, directions: dict[str, int] | None = None,
                     include_flags: bool = True) -> pd.DataFrame:
    """Combine raw pattern flags (current bar) with decayed, statistics-weighted summaries.

    Columns: ``STAT_FEATURES`` + (optionally) one column per pattern.
    """
    decay = decay or [1.0, 0.6, 0.3]
    use_stats = stats is not None and stats.fitted
    patterns = list(stats.patterns) if use_stats else list(patterns or PATTERNS)
    directions = stats.directions if use_stats else (directions or PATTERN_DIRECTION)
    flags = pat.reindex(columns=patterns).fillna(0).to_numpy(float)
    n, k = flags.shape
    weighted = np.zeros_like(flags)
    for lag, w in enumerate(decay):
        weighted[lag:] += w * flags[: n - lag] if lag else w * flags
    direction = np.array([directions.get(p, 0) for p in patterns], dtype=float)
    out = pd.DataFrame(index=pat.index)
    out["n_recent"] = weighted.sum(axis=1) / 2.0
    out["direction_score"] = weighted @ direction
    if use_stats:
        h5 = 5 if 5 in stats.horizons else stats.horizons[min(1, len(stats.horizons) - 1)]
        h20 = 20 if 20 in stats.horizons else stats.horizons[-1]
        e5, w5, s5 = stats.edge_arrays(h5)
        e20, w20, s20 = stats.edge_arrays(h20)
        out["edge_5"] = (weighted @ (e5 * s5)) * 100.0
        out["edge_20"] = (weighted @ (e20 * s20)) * 50.0
        out["win_edge_5"] = (weighted @ (w5 * s5)) * 10.0
        out["win_edge_20"] = (weighted @ (w20 * s20)) * 10.0
    else:
        for col in ("edge_5", "edge_20", "win_edge_5", "win_edge_20"):
            out[col] = 0.0
    if include_flags:
        for i, p in enumerate(patterns):
            out[p] = flags[:, i]
    return out.clip(-5.0, 5.0)


CANDLE_FEATURES = STAT_FEATURES + PATTERNS
