import numpy as np
import pandas as pd

from stockbot.data.loader import standardize_ohlcv, synthetic_ohlcv
from stockbot.features import PATTERNS, TECH_FEATURES, TREND_FEATURES, PatternStats, compute_technical, compute_trend, detect_patterns
from stockbot.features.candles import CANDLE_FEATURES, pattern_features


def _bars(rows):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 1000.0
    df.index = pd.bdate_range("2020-01-01", periods=len(df))
    return standardize_ohlcv(df)


def test_technical_shape_and_ranges():
    df = synthetic_ohlcv(600, seed=3)
    f = compute_technical(df)
    assert list(f.columns) == TECH_FEATURES
    assert f.shape == (600, len(TECH_FEATURES))
    tail = f.iloc[300:]
    assert not tail.isna().any().any()
    assert float(tail.abs().max().max()) <= 5.0


def test_trend_shape():
    df = synthetic_ohlcv(400, seed=4)
    f = compute_trend(df)
    assert list(f.columns) == TREND_FEATURES
    assert not f.iloc[250:].isna().any().any()


def test_hammer_and_engulfing_detected():
    # 10 falling bars then a hammer (long lower shadow, small body near the top)
    rows = [[100 - i, 100.5 - i, 98.5 - i, 99 - i] for i in range(12)]
    rows.append([88.0, 88.5, 84.0, 88.4])            # hammer after a downtrend (long lower shadow, small body)
    rows.append([88.0, 91.0, 87.9, 90.8])
    df = _bars(rows)
    pat = detect_patterns(df, avg_window=5)
    assert pat["hammer"].iloc[12] == 1
    # explicit engulfing pair: bearish then bullish that engulfs
    rows2 = [[100, 101, 99, 100.2]] * 8 + [[100, 100.5, 98, 98.5], [98.2, 102, 98.1, 101.5]]
    pat2 = detect_patterns(_bars(rows2), avg_window=5)
    assert pat2["bullish_engulfing"].iloc[-1] == 1
    assert set(pat.columns) == set(PATTERNS)


def test_pattern_stats_roundtrip(tmp_path):
    frames = {f"T{i}": synthetic_ohlcv(700, seed=i) for i in range(3)}
    stats = PatternStats(horizons=[1, 5, 20]).fit(frames)
    assert stats.fitted and stats.n_bars > 0
    p = tmp_path / "stats.json"
    stats.save(p)
    loaded = PatternStats.load(p)
    assert loaded.mean_ret["doji"][5] == stats.mean_ret["doji"][5]
    feats = pattern_features(detect_patterns(frames["T0"]), loaded)
    assert list(feats.columns) == CANDLE_FEATURES
    assert np.isfinite(feats.to_numpy()).all()
    assert stats.table().shape[0] == len(PATTERNS)
