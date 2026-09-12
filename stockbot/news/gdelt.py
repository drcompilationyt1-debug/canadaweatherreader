"""GDELT DOC 2.0 API - free, keyless, global news search with history back to 2017.

Used two ways:
* ``stockbot news-history`` downloads headlines month by month into ``data/news/gdelt_<TICKER>.csv``
  so the sentiment / LLM-news blocks get a training history;
* optionally as a live source (``news.sources: [..., gdelt]``).

GDELT asks for gentle use: one request every couple of seconds, 250 articles per call.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from ..logging_utils import get_logger

log = get_logger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
STOPWORDS = ("inc", "inc.", "corporation", "corp", "corp.", "co", "co.", "ltd", "plc", "holdings", "company", "class a", "class b", "the")


def clean_company_name(name: str) -> str:
    words = [w for w in name.replace(",", " ").split() if w.lower() not in STOPWORDS]
    return " ".join(words).strip() or name


def company_name(ticker: str, mapping: dict | None = None) -> str:
    """Config mapping first, then yfinance's short name, then the ticker itself."""
    if mapping and ticker in mapping and mapping[ticker]:
        return str(mapping[ticker])
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        name = info.get("shortName") or info.get("longName")
        if name:
            return clean_company_name(name)
    except Exception:  # noqa: BLE001
        pass
    return ticker


def gdelt_query(name: str) -> str:
    phrase = f'"{name}"' if " " in name else name
    return f"{phrase} stock sourcelang:english"


def fetch_gdelt(query: str, start: datetime, end: datetime, maxrecords: int = 250, timeout: float = 30.0,
                retries: int = 3) -> list[dict]:
    params = {
        "query": query, "mode": "artlist", "maxrecords": int(maxrecords), "format": "json", "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"), "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    last = ""
    for attempt in range(retries):
        try:
            r = requests.get(GDELT_URL, params=params, timeout=timeout, headers={"User-Agent": "StockBot/0.1 (research)"})
        except requests.RequestException as e:
            last = str(e)
            time.sleep(3.0 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:  # GDELT throttles bursts per IP for a while
            last = f"http {r.status_code}"
            log.info("gdelt throttled (%s) - waiting %ds", last, 60 * (attempt + 1))
            time.sleep(60.0 * (attempt + 1))
            continue
        if not r.ok:
            raise RuntimeError(f"gdelt http {r.status_code}: {r.text[:200]}")
        try:
            return r.json().get("articles", []) or []
        except ValueError:  # GDELT answers plain text for throttling / query problems
            text = r.text.strip()
            last = text[:200]
            if "limit" in text.lower() or "too many" in text.lower() or "rate" in text.lower():
                time.sleep(10.0 * (attempt + 1))
                continue
            if not text:
                return []
            raise RuntimeError(f"gdelt: {last}")
    log.warning("gdelt gave up after %d attempts: %s", retries, last)
    return []


def articles_to_frame(ticker: str, articles: list[dict]) -> pd.DataFrame:
    rows = []
    for a in articles:
        title = (a.get("title") or "").strip()
        seen = a.get("seendate") or ""
        if not title or len(seen) < 8:
            continue
        rows.append({"ticker": ticker, "date": pd.Timestamp(seen[:8]).strftime("%Y-%m-%d"), "title": title,
                     "summary": a.get("domain") or "", "url": a.get("url") or ""})
    df = pd.DataFrame(rows, columns=["ticker", "date", "title", "summary", "url"])
    return df.drop_duplicates(subset=["title"]).sort_values("date").reset_index(drop=True)


def period_chunks(start: datetime, end: datetime, months: int = 3):
    """Consecutive [start, end] windows of ``months`` calendar months (GDELT returns <= 250 articles each)."""
    cur = datetime(start.year, start.month, 1)
    while cur <= end:
        m = cur.month - 1 + months
        nxt = datetime(cur.year + m // 12, m % 12 + 1, 1)
        yield max(cur, start), min(nxt - timedelta(seconds=1), end)
        cur = nxt


def month_chunks(start: datetime, end: datetime):
    return period_chunks(start, end, months=1)


def fetch_gdelt_history(ticker: str, name: str, start: datetime, end: datetime, sleep: float = 6.0,
                        maxrecords: int = 250, months: int = 3) -> pd.DataFrame:
    frames = []
    chunks = list(period_chunks(start, end, months))
    for i, (a, b) in enumerate(chunks):
        try:
            arts = fetch_gdelt(gdelt_query(name), a, b, maxrecords=maxrecords)
        except Exception as e:  # noqa: BLE001
            log.warning("gdelt %s %s: %s", ticker, a.strftime("%Y-%m"), e)
            arts = []
        frames.append(articles_to_frame(ticker, arts))
        if (i + 1) % 8 == 0:
            log.info("gdelt %s: %d/%d periods, %d headlines so far", ticker, i + 1, len(chunks), sum(len(f) for f in frames))
        time.sleep(sleep)
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "title", "summary", "url"])
    return pd.concat(frames).drop_duplicates(subset=["title"]).sort_values("date").reset_index(drop=True)
