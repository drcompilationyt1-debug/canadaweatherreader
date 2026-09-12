"""Live broker via Alpaca (``pip install alpaca-py``).  Keys: ALPACA_API_KEY / ALPACA_SECRET_KEY.

``execution.alpaca.paper: true`` (default) talks to Alpaca's paper-trading endpoint; set it to
false - and pass ``--i-understand-real-money`` on the CLI - for a real account.
"""
from __future__ import annotations

import os

from ..logging_utils import get_logger
from .base import Broker, Fill, Order, Position

log = get_logger(__name__)


def alpaca_keys() -> tuple[str | None, str | None]:
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    return key, secret


class AlpacaBroker(Broker):
    name = "alpaca"
    supports_short = True

    def __init__(self, paper: bool = True, fractional: bool = True, fees=None):
        key, secret = alpaca_keys()
        if not key or not secret:
            raise RuntimeError("set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        self.paper = bool(paper)
        self.fractional = bool(fractional)
        self.fees = fees  # Alpaca charges nothing; the moomoo schedule is booked virtually in every Fill.cost
        self.virtual_fees = 0.0
        self.client = TradingClient(key, secret, paper=self.paper)
        self.data = StockHistoricalDataClient(key, secret)
        self.name = "alpaca-paper" if self.paper else "alpaca-LIVE"

    @staticmethod
    def _symbol(ticker: str) -> str:
        return ticker.replace("-", ".")  # BRK-B -> BRK.B

    @staticmethod
    def _ticker(symbol: str) -> str:
        return str(symbol).replace(".", "-")

    def equity(self) -> float:
        return float(self.client.get_account().equity)

    def cash(self) -> float:
        return float(self.client.get_account().cash)

    def positions(self) -> dict[str, Position]:
        """Held positions plus the unfilled part of open orders (so a re-run never doubles an order)."""
        out: dict[str, Position] = {}
        for p in self.client.get_all_positions():
            qty = float(p.qty)
            if str(getattr(p, "side", "")).lower().endswith("short"):
                qty = -abs(qty)
            t = self._ticker(p.symbol)
            out[t] = Position(t, qty, float(p.avg_entry_price))
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            for o in self.client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)):
                pending = float(o.qty or 0) - float(o.filled_qty or 0)
                if pending <= 0:
                    continue
                signed = pending if str(o.side).lower().endswith("buy") else -pending
                t = self._ticker(o.symbol)
                cur = out.get(t, Position(t, 0.0, 0.0))
                out[t] = Position(t, cur.shares + signed, cur.avg_price)
        except Exception as e:  # noqa: BLE001
            log.warning("could not read open orders: %s", e)
        return {t: p for t, p in out.items() if abs(p.shares) > 1e-9}

    def price(self, ticker: str) -> float:
        from alpaca.data.requests import StockLatestTradeRequest

        sym = self._symbol(ticker)
        trades = self.data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=sym))
        return float(trades[sym].price)

    def submit(self, order: Order) -> Fill | None:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        pos = self.position(order.ticker)
        signed = order.qty if order.side == "buy" else -order.qty
        new_shares = pos.shares + signed
        # Alpaca does not allow fractional shorts: whole shares whenever the short side is involved
        if self.fractional and new_shares >= 0 and pos.shares >= 0:
            qty = round(order.qty, 3)
        else:
            qty = float(int(order.qty))
        if qty <= 0:
            return None
        req = MarketOrderRequest(symbol=self._symbol(order.ticker), qty=qty,
                                 side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL, time_in_force=TimeInForce.DAY)
        resp = self.client.submit_order(req)
        price = self.price(order.ticker)
        cost = float(self.fees.cost(qty, price, order.side)) if self.fees is not None else 0.0
        self.virtual_fees += cost
        log.info("alpaca order %s %s %.3f %s -> id %s (fees %.2f, booked virtually)", self.name, order.side, qty, order.ticker,
                 getattr(resp, "id", "?"), cost)
        return Fill(order.ticker, order.side, qty, price, cost)
