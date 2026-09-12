"""Broker interface shared by the paper broker and the live (Alpaca) broker."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class Position:
    ticker: str
    shares: float = 0.0
    avg_price: float = 0.0

    def value(self, price: float) -> float:
        return self.shares * price


@dataclass
class Order:
    ticker: str
    side: str            # "buy" | "sell"
    qty: float           # shares (positive)
    note: str = ""


@dataclass
class Fill:
    ticker: str
    side: str
    qty: float
    price: float
    cost: float
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


class Broker:
    name = "base"
    supports_short = False

    def equity(self) -> float:
        raise NotImplementedError

    def cash(self) -> float:
        raise NotImplementedError

    def positions(self) -> dict[str, Position]:
        raise NotImplementedError

    def price(self, ticker: str) -> float:
        raise NotImplementedError

    def submit(self, order: Order) -> Fill | None:
        raise NotImplementedError

    def save(self) -> None:
        pass

    def position(self, ticker: str) -> Position:
        return self.positions().get(ticker, Position(ticker))

    def summary(self) -> dict:
        pos = self.positions()
        return {"broker": self.name, "equity": self.equity(), "cash": self.cash(),
                "positions": {t: {"shares": p.shares, "avg_price": p.avg_price} for t, p in pos.items() if abs(p.shares) > 1e-9}}
