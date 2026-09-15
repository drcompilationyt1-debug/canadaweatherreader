"""What Alpaca already keeps for us: the account's equity curve, every fill, and intraday prices.

The paper dashboard on alpaca.markets is built from three REST feeds, all available with the same
keys the broker uses (``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY``):

* ``/v2/account/portfolio/history``  - equity and profit / loss per day (or per 15 minutes intraday);
* ``/v2/account/activities/FILL``    - every execution: symbol, side, quantity, price, time;
* ``/v2/stocks/bars`` (data API)     - 15-minute bars for the whole session, not just our watch window.

So instead of trusting only our own snapshots, the reviews read the broker's record of what happened:
the day review replays the full 09:30-16:00 path of every name, the account command prints the
per-day ups and downs, and the intraday exit model learns from the bars.  Everything is cached under
``data/paper/alpaca`` (part of the saved state) so the weekend jobs can work offline.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .alpaca import AlpacaBroker, alpaca_keys
from .market_hours import NY

log = get_logger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


class AlpacaHistory:
    def __init__(self, cfg=None, paper: bool | None = None, cache_dir: str | Path | None = None, session=None):
        self.key, self.secret = alpaca_keys()
        if paper is None:
            paper = bool(cfg.get_path("execution.alpaca.paper", True)) if cfg is not None else True
        self.paper = bool(paper)
        self.base = PAPER_URL if self.paper else LIVE_URL
        self.feed = str(cfg.get_path("execution.alpaca.feed", "iex")) if cfg is not None else "iex"
        if cache_dir is None:
            cache_dir = cfg.path("execution.alpaca.history_dir", "data/paper/alpaca") if cfg is not None else Path("data/paper/alpaca")
        self.cache_dir = Path(cache_dir)
        self._session = session

    @staticmethod
    def available() -> bool:
        k, s = alpaca_keys()
        return bool(k and s)

    # ------------------------------------------------------------------ http
    def _get(self, url: str, params: dict | None = None) -> dict | list:
        import requests

        if not (self.key and self.secret):
            raise RuntimeError("set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        s = self._session or requests
        r = s.get(url, params=params or {}, headers={"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"alpaca {r.status_code}: {r.text[:200]}")
        return r.json()

    # ------------------------------------------------------------------ the dashboard's equity curve
    def portfolio_history(self, period: str = "1M", timeframe: str = "1D", intraday_reporting: str = "market_hours",
                          pnl_reset: str = "per_day") -> pd.DataFrame:
        """Equity and profit / loss per bar, as the dashboard shows them (``period`` 1D/1W/1M/3M/1A/all,
        ``timeframe`` 1Min/5Min/15Min/1H/1D)."""
        params = {"period": period, "timeframe": timeframe, "extended_hours": "false"}
        if timeframe != "1D":
            params.update({"intraday_reporting": intraday_reporting, "pnl_reset": pnl_reset})
        raw = self._get(f"{self.base}/v2/account/portfolio/history", params)
        ts = pd.to_datetime(pd.Series(raw.get("timestamp", [])), unit="s", utc=True).dt.tz_convert(NY)
        df = pd.DataFrame({"timestamp": ts, "equity": pd.to_numeric(pd.Series(raw.get("equity", [])), errors="coerce"),
                           "profit_loss": pd.to_numeric(pd.Series(raw.get("profit_loss", [])), errors="coerce"),
                           "profit_loss_pct": pd.to_numeric(pd.Series(raw.get("profit_loss_pct", [])), errors="coerce")})
        df = df.dropna(subset=["equity"]).reset_index(drop=True)
        df.attrs["base_value"] = raw.get("base_value")
        self._save_json(f"history_{period}_{timeframe}.json", {"fetched": datetime.now(timezone.utc).isoformat(), **raw})
        return df

    def daily(self, period: str = "1M") -> pd.DataFrame:
        """One row per trading day: equity, profit / loss, percentage, and whether the day was up or down."""
        df = self.portfolio_history(period, "1D")
        if len(df) == 0:
            return df
        df["date"] = df["timestamp"].dt.date
        df["direction"] = np.where(df["profit_loss"] > 0, "up", np.where(df["profit_loss"] < 0, "down", "flat"))
        return df[["date", "equity", "profit_loss", "profit_loss_pct", "direction"]]

    # ------------------------------------------------------------------ transactions
    def fills(self, after: date | datetime | None = None, until: date | datetime | None = None, page_size: int = 100) -> pd.DataFrame:
        """Every execution in the window (oldest first): ts, ticker, side, qty, price, order_id."""
        rows: list[dict] = []
        params: dict = {"direction": "asc", "page_size": int(page_size)}
        if after is not None:
            params["after"] = _iso(after)
        if until is not None:
            params["until"] = _iso(until)
        token = None
        for _ in range(200):
            if token:
                params["page_token"] = token
            page = self._get(f"{self.base}/v2/account/activities/FILL", params)
            if not page:
                break
            for a in page:
                rows.append({"id": a.get("id"), "ts": a.get("transaction_time"), "ticker": AlpacaBroker._ticker(a.get("symbol", "")),
                             "side": "buy" if str(a.get("side", "")).startswith("buy") else "sell", "qty": float(a.get("qty") or 0.0),
                             "price": float(a.get("price") or 0.0), "order_id": a.get("order_id"), "type": a.get("type"),
                             "cum_qty": float(a.get("cum_qty") or 0.0), "leaves_qty": float(a.get("leaves_qty") or 0.0)})
            if len(page) < page_size:
                break
            token = page[-1].get("id")
        df = pd.DataFrame(rows)
        if len(df):
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
            df["notional"] = df["qty"] * df["price"]
        return df

    def sync_fills(self) -> pd.DataFrame:
        """Pull the fills newer than the cache and keep them all in ``fills.jsonl`` (state)."""
        cached = self.cached_fills()
        after = None
        if len(cached):
            after = pd.Timestamp(cached["ts"].max()).to_pydatetime() + timedelta(seconds=1)
        new = self.fills(after=after)
        if len(new):
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_dir / "fills.jsonl", "a", encoding="utf-8") as f:
                for _, r in new.iterrows():
                    rec = {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in r.items()}
                    f.write(json.dumps(rec, default=str) + "\n")
        return self.cached_fills()

    def cached_fills(self) -> pd.DataFrame:
        f = self.cache_dir / "fills.jsonl"
        if not f.exists():
            return pd.DataFrame(columns=["id", "ts", "ticker", "side", "qty", "price", "order_id", "notional"])
        rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(rows)
        if len(df):
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
            df = df.drop_duplicates("id").sort_values("ts").reset_index(drop=True)
        return df

    def orders(self, after: date | datetime | None = None, status: str = "all", limit: int = 500) -> pd.DataFrame:
        params: dict = {"status": status, "limit": int(limit), "direction": "asc", "nested": "false"}
        if after is not None:
            params["after"] = _iso(after)
        raw = self._get(f"{self.base}/v2/orders", params)
        rows = [{"id": o.get("id"), "ticker": AlpacaBroker._ticker(o.get("symbol", "")), "side": o.get("side"), "qty": float(o.get("qty") or 0.0),
                 "filled_qty": float(o.get("filled_qty") or 0.0), "filled_avg_price": float(o.get("filled_avg_price") or 0.0),
                 "status": o.get("status"), "submitted_at": o.get("submitted_at"), "filled_at": o.get("filled_at"), "type": o.get("type")}
                for o in raw]
        return pd.DataFrame(rows)

    def positions(self) -> pd.DataFrame:
        raw = self._get(f"{self.base}/v2/positions")
        rows = [{"ticker": AlpacaBroker._ticker(p.get("symbol", "")), "qty": float(p.get("qty") or 0.0), "avg_entry": float(p.get("avg_entry_price") or 0.0),
                 "price": float(p.get("current_price") or 0.0), "market_value": float(p.get("market_value") or 0.0),
                 "unrealized_pl": float(p.get("unrealized_pl") or 0.0), "unrealized_plpc": float(p.get("unrealized_plpc") or 0.0),
                 "today_pl": float(p.get("unrealized_intraday_pl") or 0.0)} for p in raw]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ intraday bars
    def bars(self, tickers: list[str], start: date | datetime, end: date | datetime, timeframe: str = "15Min",
             limit: int = 10000) -> dict[str, pd.DataFrame]:
        """Intraday bars per ticker (index = bar start in New York time; open/high/low/close/volume)."""
        out: dict[str, list[dict]] = {t: [] for t in tickers}
        by_symbol = {AlpacaBroker._symbol(t): t for t in tickers}
        symbols = list(by_symbol)
        for i in range(0, len(symbols), 50):
            chunk = symbols[i: i + 50]
            params = {"symbols": ",".join(chunk), "timeframe": timeframe, "start": _iso(start), "end": _iso(end, end_of_day=True),
                      "limit": int(limit), "feed": self.feed, "adjustment": "raw", "sort": "asc"}
            token = None
            for _ in range(200):
                if token:
                    params["page_token"] = token
                raw = self._get(f"{DATA_URL}/v2/stocks/bars", params)
                for sym, rows in (raw.get("bars") or {}).items():
                    t = by_symbol.get(sym, AlpacaBroker._ticker(sym))
                    out.setdefault(t, []).extend(rows)
                token = raw.get("next_page_token")
                if not token:
                    break
        frames = {}
        for t, rows in out.items():
            if not rows:
                continue
            df = pd.DataFrame(rows).rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
            frames[t] = df.set_index("ts")[["open", "high", "low", "close", "volume"]].sort_index()
        return frames

    def bars_cached(self, tickers: list[str], start: date, end: date, timeframe: str = "15Min") -> dict[str, pd.DataFrame]:
        """Bars from the parquet cache, fetching only the days that are missing (per ticker)."""
        folder = self.cache_dir / f"bars_{timeframe}"
        folder.mkdir(parents=True, exist_ok=True)
        have: dict[str, pd.DataFrame] = {}
        need: list[str] = []
        for t in tickers:
            f = folder / f"{t}.parquet"
            if f.exists():
                try:
                    df = _ny_index(pd.read_parquet(f))
                    have[t] = df
                    if len(df) and df.index.max().date() >= end and df.index.min().date() <= start:
                        continue
                except Exception:  # noqa: BLE001
                    pass
            need.append(t)
        if need:
            try:
                fresh = self.bars(need, start, end, timeframe)
            except Exception as e:  # noqa: BLE001
                log.warning("alpaca bars unavailable: %s", e)
                fresh = {}
            for t, df in fresh.items():
                old = have.get(t)
                merged = pd.concat([_ny_index(old), _ny_index(df)]) if old is not None and len(old) else _ny_index(df)
                merged = _ny_index(merged[~merged.index.duplicated(keep="last")].sort_index())
                have[t] = merged
                try:
                    merged.to_parquet(folder / f"{t}.parquet")
                except Exception as e:  # noqa: BLE001
                    log.debug("bars cache %s: %s", t, e)
        out = {}
        for t, df in have.items():
            if len(df):
                df = _ny_index(df)
                out[t] = df[(df.index.date >= start) & (df.index.date <= end)]
        return out

    def day_paths(self, tickers: list[str], day: date, timeframe: str = "15Min") -> dict[str, pd.Series]:
        """Regular-hours close path of each ticker on ``day`` (09:30-16:00 New York)."""
        bars = self.bars_cached(tickers, day, day, timeframe)
        out = {}
        for t, df in bars.items():
            d = df[df.index.date == day]
            d = d[(d.index.hour * 60 + d.index.minute >= 9 * 60 + 30) & (d.index.hour * 60 + d.index.minute < 16 * 60)]
            if len(d) >= 3:
                out[t] = d["close"].astype(float)
        return out

    # ------------------------------------------------------------------ helpers
    def _save_json(self, name: str, obj: dict) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / name).write_text(json.dumps(obj, default=str), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log.debug("alpaca cache %s: %s", name, e)


def _ny_index(df: pd.DataFrame) -> pd.DataFrame:
    """One timezone object for every bar index: a cached (parquet) index and a fresh one can carry different
    tz implementations, and concatenating those silently turns the index into plain objects."""
    idx = pd.to_datetime(df.index, utc=True) if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None         else df.index.tz_convert("UTC")
    df = df.copy()
    df.index = pd.DatetimeIndex(idx).tz_convert(NY)
    return df


def _iso(d: date | datetime, end_of_day: bool = False) -> str:
    if isinstance(d, datetime):
        dt = d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    else:
        dt = datetime(d.year, d.month, d.day, 23, 59, 59 if end_of_day else 0, tzinfo=NY) if end_of_day \
            else datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=NY)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
