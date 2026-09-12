"""News fetching: yfinance headlines, Google News RSS, and local historical CSVs (for training).

Raw headlines are cached per (ticker, day) so re-running the bot on the same day is free.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd

from ..logging_utils import get_logger
from ..paths import resolve

log = get_logger(__name__)


@dataclass
class NewsItem:
    ticker: str
    title: str
    published: str          # ISO 8601 (UTC)
    source: str = ""
    summary: str = ""
    url: str = ""

    @property
    def ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.published)

    def text(self) -> str:
        s = self.title.strip()
        if self.summary and self.summary.strip() and self.summary.strip() != s:
            s += " - " + self.summary.strip()[:300]
        return s


def _iso(ts) -> str:
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t.tz_convert("UTC").isoformat()
    except Exception:  # noqa: BLE001
        return datetime.now(timezone.utc).isoformat()


def headlines_hash(items: list[NewsItem]) -> str:
    h = hashlib.sha1()
    for it in sorted(items, key=lambda x: (x.published, x.title)):
        h.update(it.title.encode("utf-8", "ignore"))
    return h.hexdigest()[:12]


class NewsFetcher:
    def __init__(self, cfg):
        news = cfg.section("news")
        self.sources = list(news.get("sources", ["yfinance", "google_rss", "local_csv"]))
        self.cache_dir = resolve(news.get("cache_dir", "data/news_cache"))
        self.local_dir = resolve(news.get("local_csv_dir", "data/news"))
        self.max_age_hours = float(news.get("max_age_hours", 12))
        self.company_names = dict(news.get("company_names") or {})
        self.gdelt_sleep = float(news.get("gdelt_sleep", 6.0))
        self._history_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ live
    def fetch(self, ticker: str, lookback_days: int = 3, use_cache: bool = True) -> list[NewsItem]:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cache = self.cache_dir / "raw" / f"{ticker}_{day}.json"
        if use_cache and cache.exists() and (time.time() - cache.stat().st_mtime) < self.max_age_hours * 3600:
            try:
                return [NewsItem(**d) for d in json.loads(cache.read_text(encoding="utf-8"))]
            except Exception:  # noqa: BLE001
                pass
        items: list[NewsItem] = []
        if "yfinance" in self.sources:
            items += self._yfinance(ticker)
        if "google_rss" in self.sources:
            items += self._google_rss(ticker)
        if "local_csv" in self.sources:
            items += self._local_recent(ticker, lookback_days)
        if "gdelt" in self.sources:
            items += self._gdelt(ticker, lookback_days)
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        seen: set[str] = set()
        out: list[NewsItem] = []
        for it in sorted(items, key=lambda x: x.published, reverse=True):
            key = it.title.strip().lower()
            if not key or key in seen:
                continue
            try:
                if it.ts.tz_convert("UTC") < cutoff:
                    continue
            except Exception:  # noqa: BLE001
                pass
            seen.add(key)
            out.append(it)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps([asdict(i) for i in out]), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        return out

    def _yfinance(self, ticker: str) -> list[NewsItem]:
        try:
            import yfinance as yf

            raw = yf.Ticker(ticker).news or []
        except Exception as e:  # noqa: BLE001
            log.debug("yfinance news failed for %s: %s", ticker, e)
            return []
        items = []
        for n in raw:
            try:
                c = n.get("content") if isinstance(n, dict) else None
                if c:  # yfinance >= 0.2.50 format
                    title = c.get("title") or ""
                    pub = c.get("pubDate") or c.get("displayTime") or ""
                    src = (c.get("provider") or {}).get("displayName", "")
                    url = ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url") or "")
                    summary = c.get("summary") or c.get("description") or ""
                else:  # legacy format
                    title = n.get("title") or ""
                    pub = n.get("providerPublishTime")
                    pub = datetime.fromtimestamp(pub, tz=timezone.utc).isoformat() if pub else ""
                    src = n.get("publisher", "")
                    url = n.get("link", "")
                    summary = ""
                if title:
                    items.append(NewsItem(ticker, title, _iso(pub) if pub else _iso(datetime.now(timezone.utc)), src, summary, url))
            except Exception:  # noqa: BLE001
                continue
        return items

    def _google_rss(self, ticker: str) -> list[NewsItem]:
        try:
            import feedparser
        except ImportError:
            return []
        q = quote_plus(f"{ticker} stock")
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        try:
            feed = feedparser.parse(url)
        except Exception as e:  # noqa: BLE001
            log.debug("google rss failed for %s: %s", ticker, e)
            return []
        items = []
        for e in feed.entries[:40]:
            try:
                pub = e.get("published_parsed")
                published = _iso(datetime(*pub[:6], tzinfo=timezone.utc)) if pub else _iso(datetime.now(timezone.utc))
                src = (e.get("source") or {}).get("title", "") if isinstance(e.get("source"), dict) else ""
                items.append(NewsItem(ticker, e.get("title", ""), published, src, "", e.get("link", "")))
            except Exception:  # noqa: BLE001
                continue
        return items

    def _gdelt(self, ticker: str, lookback_days: int) -> list[NewsItem]:
        from .gdelt import company_name, fetch_gdelt, gdelt_query

        try:
            name = company_name(ticker, self.company_names)
            end = datetime.utcnow()
            arts = fetch_gdelt(gdelt_query(name), end - timedelta(days=lookback_days), end, maxrecords=60)
            time.sleep(self.gdelt_sleep)
        except Exception as e:  # noqa: BLE001
            log.debug("gdelt live failed for %s: %s", ticker, e)
            return []
        items = []
        for a in arts:
            seen = a.get("seendate") or ""
            if not a.get("title") or len(seen) < 15:
                continue
            published = _iso(datetime.strptime(seen[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc))
            items.append(NewsItem(ticker, a["title"], published, a.get("domain", "gdelt"), "", a.get("url", "")))
        return items

    # ------------------------------------------------------------------ historical (local CSV)
    def history(self, ticker: str) -> pd.DataFrame:
        """Historical headlines from ``data/news/*.csv`` with columns: date, ticker, title[, summary].

        Any CSV in the folder is scanned; rows for ``ticker`` are returned sorted by date.
        Kaggle datasets such as "Daily Financial News for 6000+ Stocks" drop straight in.
        """
        if ticker in self._history_cache:
            return self._history_cache[ticker]
        frames = []
        if self.local_dir.is_dir():
            for f in sorted(self.local_dir.glob("*.csv")):
                try:
                    df = pd.read_csv(f)
                except Exception as e:  # noqa: BLE001
                    log.warning("could not read %s: %s", f, e)
                    continue
                df.columns = [c.lower().strip() for c in df.columns]
                col_t = next((c for c in ("ticker", "stock", "symbol", "tic") if c in df.columns), None)
                col_d = next((c for c in ("date", "datetime", "published", "time") if c in df.columns), None)
                col_h = next((c for c in ("title", "headline", "text") if c in df.columns), None)
                if not (col_t and col_d and col_h):
                    continue
                sub = df[df[col_t].astype(str).str.upper() == ticker.upper()]
                if len(sub) == 0:
                    continue
                out = pd.DataFrame({
                    "date": pd.to_datetime(sub[col_d], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize(),
                    "title": sub[col_h].astype(str),
                    "summary": sub["summary"].astype(str) if "summary" in sub.columns else "",
                })
                frames.append(out.dropna(subset=["date"]))
        res = pd.concat(frames).sort_values("date").reset_index(drop=True) if frames else pd.DataFrame(columns=["date", "title", "summary"])
        self._history_cache[ticker] = res
        return res

    def _local_recent(self, ticker: str, lookback_days: int) -> list[NewsItem]:
        h = self.history(ticker)
        if len(h) == 0:
            return []
        cutoff = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None) - pd.Timedelta(days=lookback_days)
        recent = h[h["date"] >= cutoff]
        return [NewsItem(ticker, r.title, _iso(r.date), "local", str(r.summary) if r.summary else "") for r in recent.itertuples()]
