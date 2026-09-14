"""Paper broker: fills at the latest close with commission + slippage, state persisted to JSON."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..logging_utils import get_logger
from .base import Broker, Fill, Order, Position

log = get_logger(__name__)


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, state_file: str | Path, price_source: Callable[[str], float], initial_cash: float = 100_000.0,
                 commission: float = 0.0005, slippage: float = 0.0005, allow_short: bool = True, fees=None):
        self.state_file = Path(state_file)
        self.price_source = price_source
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.fees = fees  # FeeSchedule (moomoo's real schedule); None = proportional `commission`
        self.supports_short = bool(allow_short)
        self._cash = float(initial_cash)
        self.initial_cash = float(initial_cash)
        self._positions: dict[str, Position] = {}
        self.fills: list[dict] = []
        self.equity_history: list[dict] = []
        self.created = datetime.now(timezone.utc).isoformat()
        self.peak_equity = float(initial_cash)
        self._prices: dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        if not self.state_file.exists():
            return
        try:
            d = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log.warning("paper state unreadable (%s) - starting fresh", e)
            return
        self._cash = float(d.get("cash", self._cash))
        self.initial_cash = float(d.get("initial_cash", self.initial_cash))
        self._positions = {t: Position(t, float(p["shares"]), float(p.get("avg_price", 0.0))) for t, p in d.get("positions", {}).items()}
        self.fills = list(d.get("fills", []))[-5000:]
        self.equity_history = list(d.get("equity_history", []))[-5000:]
        self.created = d.get("created", self.created)
        self.peak_equity = float(d.get("peak_equity", self.peak_equity))
        self._prices = {k: float(v) for k, v in d.get("last_prices", {}).items()}

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        d = {
            "cash": self._cash, "initial_cash": self.initial_cash, "created": self.created,
            "positions": {t: {"shares": p.shares, "avg_price": p.avg_price} for t, p in self._positions.items() if abs(p.shares) > 1e-9},
            "fills": self.fills[-5000:], "equity_history": self.equity_history[-5000:],
            "peak_equity": self.peak_equity, "last_prices": self._prices,
        }
        self.state_file.write_text(json.dumps(d, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ prices
    def price(self, ticker: str) -> float:
        try:
            p = float(self.price_source(ticker))
            if p > 0:
                self._prices[ticker] = p
                return p
        except Exception as e:  # noqa: BLE001
            log.warning("no fresh price for %s (%s) - using last known", ticker, e)
        return self._prices.get(ticker, 0.0)

    # ------------------------------------------------------------------ state
    def cash(self) -> float:
        return self._cash

    def positions(self) -> dict[str, Position]:
        return {t: p for t, p in self._positions.items() if abs(p.shares) > 1e-9}

    def equity(self) -> float:
        eq = self._cash
        for t, p in self.positions().items():
            eq += p.shares * self.price(t)  # price() falls back to the last known quote
        return eq

    def mark(self, as_of: str | None = None) -> float:
        eq = self.equity()
        self.peak_equity = max(self.peak_equity, eq)
        self.equity_history.append({"ts": as_of or datetime.now(timezone.utc).isoformat(), "equity": eq, "cash": self._cash})
        return eq

    # ------------------------------------------------------------------ orders
    def affordable_qty(self, ticker: str, price: float) -> float:
        """Largest buy (shares) the cash on hand covers, fees and slippage included."""
        if price <= 0 or self._cash <= 0:
            return 0.0
        qty = self._cash / (price * (1.0 + self.slippage + (0.0 if self.fees is not None else self.commission)))
        for _ in range(3):  # the per-order minimums make the fee non-linear: settle in a couple of passes
            cost = (self.fees.cost(qty, price, "buy") if self.fees is not None else qty * price * self.commission) + qty * price * self.slippage
            over = qty * price + cost - self._cash
            if over <= 1e-6:
                break
            qty = max(0.0, qty - over / price)
        return max(0.0, qty)

    def submit(self, order: Order) -> Fill | None:
        price = self.price(order.ticker)
        if price <= 0 or order.qty <= 0:
            return None
        signed = order.qty if order.side == "buy" else -order.qty
        pos = self._positions.get(order.ticker, Position(order.ticker))
        if signed > 0:  # a real broker rejects a buy that exceeds the cash on hand: cap it (fees included)
            affordable = self.affordable_qty(order.ticker, price)
            if affordable <= 0:
                log.warning("paper %s: no cash for %s (cash %.2f) - order skipped", self.state_file.stem, order.ticker, self._cash)
                return None
            if signed > affordable:
                log.info("paper %s: %s buy cut from %.3f to %.3f shares to stay within cash %.2f", self.state_file.stem, order.ticker,
                         signed, affordable, self._cash)
                signed = affordable
        new_shares = pos.shares + signed
        if new_shares < -1e-9 and not self.supports_short:
            signed = -pos.shares  # sell down to flat only
            new_shares = 0.0
            if abs(signed) < 1e-9:
                return None
        if self.fees is not None:
            cost = self.fees.cost(abs(signed), price, "buy" if signed > 0 else "sell") + abs(signed) * price * self.slippage
        else:
            cost = abs(signed) * price * (self.commission + self.slippage)
        self._cash -= signed * price + cost
        if abs(new_shares) < 1e-9:
            pos.avg_price = 0.0
        elif (pos.shares >= 0 and signed > 0) or (pos.shares <= 0 and signed < 0):
            pos.avg_price = (abs(pos.shares) * pos.avg_price + abs(signed) * price) / (abs(pos.shares) + abs(signed))
        elif (new_shares > 0 > pos.shares) or (new_shares < 0 < pos.shares):
            pos.avg_price = price  # flipped direction
        pos.shares = new_shares
        self._positions[order.ticker] = pos
        fill = Fill(order.ticker, "buy" if signed > 0 else "sell", abs(signed), price, cost)
        self.fills.append({**fill.to_dict(), "note": order.note})
        return fill
