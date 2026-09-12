"""Historical headlines from Google News RSS - free, no key, its search accepts ``after:`` / ``before:``.

Each query returns at most 100 items, so a window that comes back full is split in half until it
fits (down to ``min_days``).  About 2 seconds between requests keeps Google happy; a 429 or a
consent page is retried after a pause.
"""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests

from ..logging_utils import get_logger

log = get_logger(__name__)

RSS = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
CAP = 100
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; StockBot/0.1; research)"}
COLUMNS = ["ticker", "date", "title", "summary", "url"]


def google_query(name: str, start: datetime, end: datetime) -> str:
    return f"{name} stock after:{start:%Y-%m-%d} before:{end:%Y-%m-%d}"


def fetch_window(name: str, start: datetime, end: datetime, timeout: float = 30.0, retries: int = 3) -> list[dict]:
    url = RSS.format(q=quote_plus(google_query(name, start, end)))
    last = ""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers=HEADERS)
        except requests.RequestException as e:
            last = str(e)
            time.sleep(10.0 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last = f"http {r.status_code}"
            log.info("google news throttled (%s) - waiting %ds", last, 60 * (attempt + 1))
            time.sleep(60.0 * (attempt + 1))
            continue
        if not r.ok:
            raise RuntimeError(f"google news http {r.status_code}")
        feed = feedparser.parse(r.text)
        if not feed.entries and "<rss" not in r.text[:500]:
            last = "no RSS in reply (consent / block page)"
            time.sleep(30.0 * (attempt + 1))
            continue
        out = []
        for e in feed.entries:
            pp = e.get("published_parsed")
            title = (e.get("title") or "").strip()
            if not pp or not title:
                continue
            src = e.get("source")
            src = (src.get("title") or "") if isinstance(src, dict) else ""
            if src and title.endswith(" - " + src):
                title = title[: -len(src) - 3].strip()
            out.append({"date": datetime(*pp[:3]).strftime("%Y-%m-%d"), "title": title, "summary": src, "url": e.get("link", "")})
        return out
    log.warning("google news gave up on %s..%s: %s", start.date(), end.date(), last)
    return []


def fetch_google_history(ticker: str, name: str, start: datetime, end: datetime, sleep: float = 2.0,
                         window_days: int = 30, min_days: int = 3, max_requests: int | None = None) -> pd.DataFrame:
    """Collect headlines for [start, end]; windows that hit the 100-item cap are split recursively."""
    queue: deque[tuple[datetime, datetime]] = deque()
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=window_days), end)
        queue.append((cur, nxt))
        cur = nxt
    rows: list[dict] = []
    n_req = 0
    total_windows = len(queue)
    while queue:
        a, b = queue.popleft()
        arts = fetch_window(name, a, b)
        n_req += 1
        span = (b - a).days
        if len(arts) >= CAP and span > min_days:
            mid = a + timedelta(days=span // 2)
            queue.appendleft((mid, b))
            queue.appendleft((a, mid))
        else:
            rows.extend({"ticker": ticker, **x} for x in arts)
        if n_req % 10 == 0:
            log.info("google news %s: %d requests, %d windows left, %d headlines", ticker, n_req, len(queue), len(rows))
        if max_requests and n_req >= max_requests:
            log.warning("google news %s: request budget reached (%d)", ticker, max_requests)
            break
        time.sleep(sleep)
    log.info("google news %s: %d headlines from %d requests (%d initial windows)", ticker, len(rows), n_req, total_windows)
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.drop_duplicates(subset=["title"]).sort_values("date").reset_index(drop=True)
