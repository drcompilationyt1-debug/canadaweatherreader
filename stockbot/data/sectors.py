"""Sector of every name in the universe (Yahoo Finance), cached in ``models/sectors.json`` and refreshed weekly.

The rank layer caps how many of the top-K may come from one sector: a return ranking loves whatever sector is
running, and a top-20 that is a third semiconductors is not twenty bets.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..logging_utils import get_logger

log = get_logger(__name__)

ETF_SECTOR = "Index"


def load_sectors(cfg, tickers: list[str] | None = None, max_age_days: float = 7.0, refresh: bool = True) -> dict[str, str]:
    """{ticker: sector}; names Yahoo does not classify (ETFs) get ``Index``; unknown names get ``Unknown``."""
    tickers = list(tickers or cfg.get("universe", []))
    path = Path(cfg.path("models_dir", "models")) / "sectors.json"
    cached: dict = {}
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cached = {}
    fresh = time.time() - float(cached.get("_fetched", 0)) < max_age_days * 86400
    missing = [t for t in tickers if t not in cached]
    if refresh and (missing or not fresh):
        try:
            import yfinance as yf

            for t in (tickers if not fresh else missing):
                try:
                    info = yf.Ticker(t).info
                    sector = info.get("sector") or (ETF_SECTOR if str(info.get("quoteType", "")).upper() == "ETF" else None)
                    cached[t] = sector or "Unknown"
                except Exception as e:  # noqa: BLE001
                    log.debug("sector %s: %s", t, e)
                    cached.setdefault(t, "Unknown")
            cached["_fetched"] = time.time()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cached, indent=1), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log.warning("sector lookup failed: %s", e)
    return {t: str(cached.get(t, "Unknown")) for t in tickers}
