"""OHLCV data loading with a parquet cache, plus a synthetic generator for tests / offline runs."""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..paths import resolve

log = get_logger(__name__)

COLUMNS = ["open", "high", "low", "close", "volume"]


def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, tz-naive DatetimeIndex named ``date``, float64 values, no NaN closes."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=COLUMNS)
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    if "adj_close" in out.columns and "close" not in out.columns:
        out["close"] = out["adj_close"]
    missing = [c for c in COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV frame is missing columns {missing}; has {list(out.columns)}")
    out = out[COLUMNS].astype("float64")
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["close"])
    out["volume"] = out["volume"].fillna(0.0)
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    return out


def _cache_file(cache_dir: Path, ticker: str, interval: str) -> Path:
    safe = ticker.replace("/", "_").replace("^", "idx_")
    return cache_dir / f"{safe}_{interval}.parquet"


def _download(ticker: str, start: str, end: str | None, interval: str) -> pd.DataFrame:
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is not None and len(df) > 0:
                return standardize_ohlcv(df)
            last_err = RuntimeError(f"empty frame for {ticker}")
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"could not download {ticker}: {last_err}")


def fetch_ohlcv(
    ticker: str,
    start: str = "2008-01-01",
    end: str | None = None,
    interval: str = "1d",
    cache_dir: str | Path = "data/cache",
    refresh_days: float = 1.0,
    refresh: bool = False,
    offline: bool = False,
) -> pd.DataFrame:
    """Return a standardized OHLCV frame for ``ticker`` (cached as parquet).

    * The cache is re-downloaded when it is older than ``refresh_days`` (and ``end`` is open-ended),
      or when ``refresh=True``.
    * ``offline=True`` never touches the network (raises if the ticker is not cached).
    """
    cache_dir = resolve(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = _cache_file(cache_dir, ticker, interval)
    cached: pd.DataFrame | None = None
    if f.exists():
        try:
            cached = standardize_ohlcv(pd.read_parquet(f))
        except Exception as e:  # noqa: BLE001
            log.warning("cache for %s unreadable (%s) - re-downloading", ticker, e)
            cached = None
    stale = True
    if cached is not None and len(cached):
        age_days = (time.time() - f.stat().st_mtime) / 86400.0
        covers_start = cached.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=7)
        # daily bars are stale as soon as a trading day has closed that the cache does not have yet
        missing_session = interval == "1d" and end is None and cached.index[-1].date() < _last_completed_session()
        stale = (end is None and age_days > refresh_days) or not covers_start or missing_session
    if cached is not None and not stale and not refresh:
        return _slice(cached, start, end)
    if offline:
        if cached is None:
            raise FileNotFoundError(f"{ticker}: not cached and offline=True")
        return _slice(cached, start, end)
    try:
        fresh = _download(ticker, start, end, interval)
    except Exception as e:  # noqa: BLE001
        if cached is not None:
            log.warning("%s: download failed (%s) - using stale cache", ticker, e)
            return _slice(cached, start, end)
        raise
    if interval == "1d":
        fresh = drop_unfinished_bar(fresh)
    if cached is not None:
        fresh = standardize_ohlcv(pd.concat([cached, fresh]))
    fresh.to_parquet(f)
    return _slice(fresh, start, end)


def _last_completed_session():
    from ..execution.market_hours import last_completed_session

    return last_completed_session()


def drop_unfinished_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo returns today's *in-progress* daily bar while the market is open; never cache or trade on it.

    Every signal and the simulator assume complete bars, so a bar dated after the last completed
    session (today, before the close) is removed.
    """
    if df is None or len(df) == 0:
        return df
    last_done = pd.Timestamp(_last_completed_session())
    if df.index[-1] > last_done:
        log.info("dropping the unfinished bar of %s (last completed session %s)", df.index[-1].date(), last_done.date())
        return df[df.index <= last_done]
    return df


def _slice(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def load_universe(
    tickers: list[str],
    start: str,
    end: str | None,
    interval: str = "1d",
    cache_dir: str | Path = "data/cache",
    refresh_days: float = 1.0,
    min_rows: int = 300,
    offline: bool = False,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load every ticker that can be loaded; failures are logged and skipped."""
    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = fetch_ohlcv(t, start, end, interval, cache_dir, refresh_days, refresh=refresh, offline=offline)
        except Exception as e:  # noqa: BLE001
            log.warning("skipping %s: %s", t, e)
            continue
        if len(df) < min_rows:
            log.warning("skipping %s: only %d rows (< %d)", t, len(df), min_rows)
            continue
        frames[t] = df
    if not frames:
        raise RuntimeError("no ticker could be loaded - check network / tickers / cache")
    return frames


RESAMPLE_RULES = {"D": None, "W": "W-FRI", "M": "ME", "Q": "QE"}


def resample_ohlcv(df: pd.DataFrame, rule: str = "W") -> pd.DataFrame:
    """Turn any OHLCV bars into candles of a coarser timeframe (``W`` weekly, ``M`` monthly, or any
    pandas offset alias such as ``2W`` / ``4h`` for intraday data)."""
    rule = RESAMPLE_RULES.get(rule.upper(), rule) if len(rule) == 1 else rule
    if rule is None:
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    try:
        out = df.resample(rule).agg(agg)
    except ValueError:  # older pandas spells month-end "M"
        out = df.resample(rule.replace("ME", "M").replace("QE", "Q")).agg(agg)
    return out.dropna(subset=["close"])


def synthetic_ohlcv(
    n: int = 1500,
    seed: int = 0,
    start: str = "2015-01-01",
    s0: float = 100.0,
    regime_len: int = 120,
) -> pd.DataFrame:
    """Geometric-Brownian-ish price path with drifting regimes and realistic OHLC / volume.

    Used by the unit tests and by ``--offline`` runs when no cached data exists.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    n_reg = n // regime_len + 1
    drifts = rng.normal(0.0004, 0.0012, n_reg)
    vols = rng.uniform(0.008, 0.03, n_reg)
    r = np.concatenate([rng.normal(drifts[i], vols[i], regime_len) for i in range(n_reg)])[:n]
    close = s0 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    open_[1:] = close[:-1] * np.exp(rng.normal(0, 0.004, n - 1))
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, n)))
    vol = rng.lognormal(15, 0.4, n) * (1 + 5 * np.abs(r))
    df = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": vol}, index=dates)
    return standardize_ohlcv(df)


def synthetic_universe(tickers: list[str], n: int = 1500, seed: int = 0) -> dict[str, pd.DataFrame]:
    return {t: synthetic_ohlcv(n, seed + i) for i, t in enumerate(tickers)}


def business_days_ago(days: int) -> str:
    return (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
