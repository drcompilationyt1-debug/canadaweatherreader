"""Which market a ticker trades on, and what that means for fees, currency and routing.

Yahoo-style symbols: ``RY.TO`` / ``XYZ.V`` are Canadian (TSX / TSX-V, CAD); everything else is a
US listing (USD) - including the China / Hong Kong ADRs (BABA, JD, PDD, ...), which is how a
moomoo Canada account gets that exposure (it trades US and Canadian markets only).
"""
from __future__ import annotations

MARKETS = {
    "us": {"currency": "USD", "suffixes": (), "fees": "moomoo", "label": "US (NYSE / Nasdaq, incl. ADRs)"},
    "ca": {"currency": "CAD", "suffixes": (".TO", ".V", ".NE", ".CN"), "fees": "moomoo_ca", "label": "Canada (TSX / TSX-V)"},
}


def market_of(ticker: str) -> str:
    t = str(ticker).upper()
    for m, spec in MARKETS.items():
        if any(t.endswith(s) for s in spec["suffixes"]):
            return m
    return "us"


def currency_of(ticker: str) -> str:
    return MARKETS[market_of(ticker)]["currency"]


def group_by_market(tickers) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for t in tickers:
        out.setdefault(market_of(t), []).append(t)
    return out
