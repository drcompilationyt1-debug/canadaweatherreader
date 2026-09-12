"""When is the exchange open?

``MarketClock`` asks Alpaca's clock endpoint when Alpaca keys are available (it knows every
holiday and early close) and otherwise falls back to the built-in NYSE schedule: weekdays
09:30-16:00 New York time minus the exchange holidays listed below (13:00 close on the half
days).  Everything is expressed in New York time so cron jobs, Colab and Windows agree.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable

from ..logging_utils import get_logger

log = get_logger(__name__)

# NYSE full-day closes (weekday holidays and their observed dates) and half days (13:00 close).
NYSE_HOLIDAYS: set[str] = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
    "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29", "2028-06-19", "2028-07-04", "2028-09-04", "2028-11-23", "2028-12-25",
}
NYSE_HALF_DAYS: set[str] = {"2026-11-27", "2026-12-24", "2027-11-26", "2028-07-03", "2028-11-24"}
OPEN = time(9, 30)
CLOSE = time(16, 0)
HALF_CLOSE = time(13, 0)


def new_york_tz():
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("America/New_York")
    except Exception:  # noqa: BLE001 - no tzdata on this Windows: fall back to a fixed offset (EDT/EST guessed by month)
        log.warning("America/New_York timezone unavailable (pip install tzdata) - using a fixed offset")
        m = datetime.now(timezone.utc).month
        return timezone(timedelta(hours=-4 if 3 < m < 11 else -5))


NY = new_york_tz()


def now_ny() -> datetime:
    return datetime.now(NY)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d.strftime("%Y-%m-%d") not in NYSE_HOLIDAYS


def close_time(d: date) -> time:
    return HALF_CLOSE if d.strftime("%Y-%m-%d") in NYSE_HALF_DAYS else CLOSE


def last_completed_session(now: datetime | None = None) -> date:
    """Date of the most recent trading day whose daily bar is complete (today only after the close)."""
    now = now or now_ny()
    d = now.date()
    done_today = is_trading_day(d) and now.time() >= (datetime.combine(d, close_time(d)) + timedelta(minutes=5)).time()
    if not done_today:
        d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


@dataclass
class ClockStatus:
    now: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime
    source: str

    @property
    def minutes_to_open(self) -> float:
        return 0.0 if self.is_open else max(0.0, (self.next_open - self.now).total_seconds() / 60.0)

    @property
    def minutes_to_close(self) -> float:
        return max(0.0, (self.next_close - self.now).total_seconds() / 60.0) if self.is_open else 0.0

    @property
    def minutes_since_open(self) -> float:
        """Minutes since today's open while the market is open (0 when closed)."""
        if not self.is_open:
            return 0.0
        opened = self.now.replace(hour=OPEN.hour, minute=OPEN.minute, second=0, microsecond=0)
        return max(0.0, (self.now - opened).total_seconds() / 60.0)

    def to_dict(self) -> dict:
        return {"now_ny": self.now.isoformat(timespec="seconds"), "is_open": self.is_open,
                "next_open": self.next_open.isoformat(timespec="seconds"), "next_close": self.next_close.isoformat(timespec="seconds"),
                "minutes_to_open": round(self.minutes_to_open, 1), "minutes_to_close": round(self.minutes_to_close, 1), "source": self.source}


def builtin_status(now: datetime | None = None) -> ClockStatus:
    now = now or now_ny()
    if now.tzinfo is None:
        now = now.replace(tzinfo=NY)
    now = now.astimezone(NY)
    d = now.date()
    is_open = False
    if is_trading_day(d):
        is_open = datetime.combine(d, OPEN, tzinfo=NY) <= now < datetime.combine(d, close_time(d), tzinfo=NY)
    nd = d if (is_trading_day(d) and now < datetime.combine(d, OPEN, tzinfo=NY)) else d + timedelta(days=1)
    while not is_trading_day(nd):
        nd += timedelta(days=1)
    next_open = datetime.combine(nd, OPEN, tzinfo=NY)
    cd = d if (is_trading_day(d) and now < datetime.combine(d, close_time(d), tzinfo=NY)) else d + timedelta(days=1)
    while not is_trading_day(cd):
        cd += timedelta(days=1)
    next_close = datetime.combine(cd, close_time(cd), tzinfo=NY)
    return ClockStatus(now, is_open, next_open, next_close, "builtin")


class MarketClock:
    """``status()`` from Alpaca when possible, else the built-in NYSE calendar."""

    def __init__(self, source: str = "auto", now_fn: Callable[[], datetime] | None = None):
        self.source = source
        self._now = now_fn
        self._alpaca = None
        if source in ("auto", "alpaca") and now_fn is None:
            try:
                from .alpaca import alpaca_keys

                key, secret = alpaca_keys()
                if key and secret:
                    from alpaca.trading.client import TradingClient

                    self._alpaca = TradingClient(key, secret, paper=True)
            except Exception as e:  # noqa: BLE001
                log.debug("alpaca clock unavailable: %s", e)
        if source == "alpaca" and self._alpaca is None:
            raise RuntimeError("alpaca clock requested but ALPACA_API_KEY / ALPACA_SECRET_KEY are not set")

    def now(self) -> datetime:
        return (self._now() if self._now else now_ny()).astimezone(NY)

    def status(self) -> ClockStatus:
        if self._alpaca is not None:
            try:
                c = self._alpaca.get_clock()
                now = c.timestamp.astimezone(NY)
                return ClockStatus(now, bool(c.is_open), c.next_open.astimezone(NY), c.next_close.astimezone(NY), "alpaca")
            except Exception as e:  # noqa: BLE001
                log.warning("alpaca clock failed (%s) - using the built-in calendar", e)
        return builtin_status(self.now())

    def is_open(self) -> bool:
        return self.status().is_open

    def wait_for_open(self, max_wait_minutes: float, poll_seconds: float = 30.0,
                      sleep: Callable[[float], None] = _time.sleep) -> ClockStatus:
        """Block until the market opens; returns the final status (check ``is_open``)."""
        deadline = self.now() + timedelta(minutes=max_wait_minutes)
        st = self.status()
        while not st.is_open:
            if st.next_open > deadline:
                log.info("market opens %s - more than %.0f minutes away, not waiting",
                         st.next_open.strftime("%a %Y-%m-%d %H:%M"), max_wait_minutes)
                return st
            remaining = (st.next_open - self.now()).total_seconds()
            log.info("market closed - opens in %.0f minutes (%s)", remaining / 60.0, st.next_open.strftime("%H:%M %Z"))
            sleep(max(1.0, min(poll_seconds, remaining + 1.0)))
            st = self.status()
        return st
