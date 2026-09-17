"""A broker made of brokers: every ticker is routed to the sleeve that can trade its market.

moomoo Canada trades US and Canadian stocks, but the free paper venue (Alpaca) is US only, so the
Canadian names paper-trade on the built-in simulator (``PaperBroker``, moomoo Canada fees, CAD)
while the US names (including the China / Hong Kong ADRs) go to Alpaca paper.  Each sleeve has
its own equity and currency; the runner sizes positions against the sleeve's equity and the
allocator caps gross exposure per sleeve.
"""
from __future__ import annotations

from ..logging_utils import get_logger
from .base import Broker, Fill, Order, Position
from .markets import MARKETS, market_of

log = get_logger(__name__)


class RoutedBroker(Broker):
    name = "routed"

    def __init__(self, sleeves: dict[str, Broker], default: str = "us", seeds: dict[str, float] | None = None):
        self.sleeves = dict(sleeves)
        self.default = default if default in self.sleeves else next(iter(self.sleeves))
        # a simulated sleeve (a paper book for the names the real broker cannot trade) is funded from the same cash as
        # the real one - one moomoo account trades both markets - so its seed money is not counted a second time: it
        # contributes its profit and loss, its purchases consume the book's cash, and every name is sized against the
        # whole book.  ``seeds`` = {market: the simulated sleeve's starting cash}.
        self.seeds = {m: float(v) for m, v in (seeds or {}).items() if m in self.sleeves}
        self.name = "routed(" + ", ".join(f"{m}={b.name}" for m, b in self.sleeves.items()) + ")"
        self.supports_short = all(b.supports_short for b in self.sleeves.values())

    def sleeve_for(self, ticker: str) -> Broker:
        return self.sleeves.get(market_of(ticker), self.sleeves[self.default])

    def market_for(self, ticker: str) -> str:
        m = market_of(ticker)
        return m if m in self.sleeves else self.default

    # ------------------------------------------------------------------ aggregate view
    def _in_book(self, market: str) -> bool:
        """The real default sleeve and the simulated sleeves form one book; other real sleeves are their own accounts."""
        return bool(self.seeds) and (market == self.default or market in self.seeds)

    def equity(self) -> float:
        """The sleeves' equities (each in its own currency - a rough total); a simulated sleeve counts only its P&L."""
        return float(sum(b.equity() - self.seeds.get(m, 0.0) for m, b in self.sleeves.items()))

    def cash(self) -> float:
        """Cash still spendable: the real cash less what the simulated sleeves have invested."""
        return float(sum(b.cash() - self.seeds.get(m, 0.0) for m, b in self.sleeves.items()))

    def book_equity(self) -> float:
        return float(sum(b.equity() - self.seeds.get(m, 0.0) for m, b in self.sleeves.items() if self._in_book(m)))

    def book_cash(self) -> float:
        return float(sum(b.cash() - self.seeds.get(m, 0.0) for m, b in self.sleeves.items() if self._in_book(m)))

    def equity_for(self, ticker: str) -> float:
        m = self.market_for(ticker)
        return self.book_equity() if self._in_book(m) else float(self.sleeves[m].equity())

    def cash_for(self, ticker: str) -> float:
        m = self.market_for(ticker)
        return self.book_cash() if self._in_book(m) else float(self.sleeves[m].cash())

    def positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for b in self.sleeves.values():
            out.update(b.positions())
        return out

    def position(self, ticker: str) -> Position:
        return self.sleeve_for(ticker).position(ticker)

    def price(self, ticker: str) -> float:
        return self.sleeve_for(ticker).price(ticker)

    def submit(self, order: Order) -> Fill | None:
        return self.sleeve_for(order.ticker).submit(order)

    def save(self) -> None:
        for b in self.sleeves.values():
            b.save()

    def summary(self) -> dict:
        out = {"broker": self.name, "equity": self.equity(), "cash": self.cash(), "sleeves": {}, "positions": {}}
        for m, b in self.sleeves.items():
            s = b.summary()
            out["sleeves"][m] = {"broker": b.name, "currency": MARKETS.get(m, {}).get("currency", "?"), "equity": s["equity"], "cash": s["cash"],
                                 "positions": len(s["positions"])}
            if m in self.seeds:
                out["sleeves"][m].update({"seed": self.seeds[m], "pnl": float(s["equity"]) - self.seeds[m],
                                          "note": "simulated: funded from the book's cash, counts only its P&L"})
            out["positions"].update(s["positions"])
        return out
