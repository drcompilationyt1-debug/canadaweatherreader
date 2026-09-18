"""Live broker via Alpaca (``pip install alpaca-py``).  Keys: ALPACA_API_KEY / ALPACA_SECRET_KEY (another
account: ``execution.alpaca.keys_env: ALPACA_10K`` -> ALPACA_10K_API_KEY / ALPACA_10K_SECRET_KEY).

``execution.alpaca.paper: true`` (default) talks to Alpaca's paper-trading endpoint; set it to
false - and pass ``--i-understand-real-money`` on the CLI - for a real account.

Alpaca cannot be told to charge another broker's commissions, so the moomoo environment is emulated on
top of it: every fill books the moomoo fee in a persisted virtual ledger that is deducted from the equity
and cash the strategy sees; whole shares only when ``fractional`` is off (moomoo Canada); buys use own
cash with a reserve (runner); no shorts; market hours only.  ``configure_like_moomoo`` also switches the
Alpaca account itself to no margin / no shorting / no fractional shares.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..logging_utils import get_logger
from .base import Broker, Fill, Order, Position

log = get_logger(__name__)


def alpaca_keys(prefix: str = "ALPACA") -> tuple[str | None, str | None]:
    """``<prefix>_API_KEY`` / ``<prefix>_SECRET_KEY`` from the environment (the default prefix also accepts Alpaca's own APCA_ names)."""
    prefix = (prefix or "ALPACA").rstrip("_")
    key = os.environ.get(f"{prefix}_API_KEY")
    secret = os.environ.get(f"{prefix}_SECRET_KEY")
    if prefix == "ALPACA":
        key = key or os.environ.get("APCA_API_KEY_ID")
        secret = secret or os.environ.get("APCA_API_SECRET_KEY")
    return key, secret


class VirtualLedger:
    """Costs the real broker would charge but Alpaca does not: the cumulative moomoo fees of every fill, persisted
    so they keep reducing the equity and cash the strategy works with across sessions."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.fees = 0.0
        self.n_fills = 0
        if self.path is not None and self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                self.fees = float(d.get("fees", 0.0))
                self.n_fills = int(d.get("n_fills", 0))
            except Exception as e:  # noqa: BLE001
                log.warning("virtual ledger %s unreadable (%s) - starting at zero", self.path, e)

    def add(self, cost: float) -> None:
        self.fees += float(cost)
        self.n_fills += 1

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"fees": self.fees, "n_fills": self.n_fills}, indent=1), encoding="utf-8")

    def to_dict(self) -> dict:
        return {"fees": self.fees, "n_fills": self.n_fills, "file": str(self.path) if self.path else None}


PROXIES: dict[str, str] = {}      # ticker -> the US listing Alpaca trades for it (RY.TO -> RY); set from execution.alpaca.proxies
_REVERSE: dict[str, str] = {}


def set_proxies(mapping: dict | None) -> None:
    """Names Alpaca cannot trade (TSX lines) are paper-traded through the same company's US listing; the universe, the
    data and the live moomoo line keep the original ticker."""
    PROXIES.clear()
    _REVERSE.clear()
    for t, sym in (mapping or {}).items():
        PROXIES[str(t)] = str(sym)
        _REVERSE[str(sym)] = str(t)


class AlpacaBroker(Broker):
    name = "alpaca"
    supports_short = True

    def __init__(self, paper: bool = True, fractional: bool = True, fees=None, keys_env: str = "ALPACA", ledger_file: str | Path | None = None,
                 fee_book=None):
        self.keys_env = (keys_env or "ALPACA").rstrip("_")
        key, secret = alpaca_keys(self.keys_env)
        if not key or not secret:
            raise RuntimeError(f"set {self.keys_env}_API_KEY and {self.keys_env}_SECRET_KEY")
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        self.paper = bool(paper)
        self.fractional = bool(fractional)
        self.fees = fees  # Alpaca charges nothing; the moomoo schedule is booked virtually in every Fill.cost
        self.fee_book = fee_book   # per-market schedules: a Canadian name traded through its US listing still pays moomoo Canada's fees
        self.ledger = VirtualLedger(ledger_file)   # ... and deducted from the equity / cash the strategy sees
        self.client = TradingClient(key, secret, paper=self.paper)
        self.data = StockHistoricalDataClient(key, secret)
        self.name = "alpaca-paper" if self.paper else "alpaca-LIVE"

    @staticmethod
    def _symbol(ticker: str) -> str:
        return PROXIES.get(ticker) or ticker.replace("-", ".")   # RY.TO -> RY (proxy), BRK-B -> BRK.B

    @staticmethod
    def _ticker(symbol: str) -> str:
        s = str(symbol)
        return _REVERSE.get(s) or s.replace(".", "-")

    @property
    def virtual_fees(self) -> float:
        return self.ledger.fees

    def equity(self) -> float:
        """Alpaca's equity minus the fees moomoo would have charged so far."""
        return float(self.client.get_account().equity) - (self.ledger.fees if self.fees is not None else 0.0)

    def cash(self) -> float:
        return float(self.client.get_account().cash) - (self.ledger.fees if self.fees is not None else 0.0)

    def save(self) -> None:
        self.ledger.save()

    def constraints(self) -> str:
        return (f"moomoo fees booked virtually ({self.ledger.fees:,.2f} so far over {self.ledger.n_fills} fills), "
                f"{'fractional shares' if self.fractional else 'whole shares only'}, own cash only, no shorts, market hours only")

    def configure_like_moomoo(self) -> dict:
        """Switch the Alpaca account itself as close to a moomoo Canada cash account as its settings allow: no margin
        (multiplier 1), no shorting, fractional trading only if this broker allows it.  Returns the resulting settings."""
        cfg = self.client.get_account_configurations()
        cfg.max_margin_multiplier = "1"
        cfg.no_shorting = True
        cfg.fractional_trading = bool(self.fractional)
        out = self.client.set_account_configurations(cfg)
        d = out.model_dump() if hasattr(out, "model_dump") else dict(out)
        log.info("alpaca account configured like moomoo: %s", d)
        return d

    def cancel_open(self, ticker: str) -> float:
        """Cancel the ticker's open orders; returns the quantity still unfilled (0 = everything filled or nothing open).  The
        moomoo fee of the filled part is booked here, at the order's limit price."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        sym = self._symbol(ticker)
        unfilled = 0.0
        for o in self.client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[sym])):
            qty, filled = float(o.qty or 0), float(o.filled_qty or 0)
            unfilled += max(0.0, qty - filled)
            if filled > 0:
                price = float(getattr(o, "filled_avg_price", None) or getattr(o, "limit_price", None) or 0.0)
                sched = (self.fee_book.for_ticker(ticker) if self.fee_book is not None else None) or self.fees
                if sched is not None and price > 0:
                    self.ledger.add(float(sched.cost(filled, price, "buy" if str(o.side).lower().endswith("buy") else "sell")))
            try:
                self.client.cancel_order_by_id(o.id)
            except Exception as e:  # noqa: BLE001
                log.warning("could not cancel order %s for %s: %s", getattr(o, "id", "?"), ticker, e)
        self.ledger.save()
        return unfilled

    def close_dust(self, max_value: float = 500.0) -> list[dict]:
        """Liquidate the fractional remnants (under one share and under ``max_value``) the whole-share rule leaves behind
        after the earlier fractional days; Alpaca closes a fractional position in full even with fractional trading off."""
        out = []
        for p in self.client.get_all_positions():
            qty, value = abs(float(p.qty)), abs(float(getattr(p, "market_value", 0.0) or 0.0))
            if qty >= 1.0 or value >= max_value:
                continue
            rec = {"ticker": self._ticker(p.symbol), "qty": qty, "value": value}
            try:
                self.client.close_position(p.symbol)
                rec["closed"] = True
                log.info("closed dust %s: %.4f shares (%.2f)", p.symbol, qty, value)
            except Exception as e:  # noqa: BLE001
                rec.update({"closed": False, "error": str(e)})
                log.warning("could not close dust %s (%.4f shares): %s", p.symbol, qty, e)
            out.append(rec)
        return out

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
        if order.limit_price:                                              # rests for the day; fees are booked when we learn what filled
            from alpaca.trading.requests import LimitOrderRequest

            req = LimitOrderRequest(symbol=self._symbol(order.ticker), qty=qty, side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
                                    time_in_force=TimeInForce.DAY, limit_price=round(float(order.limit_price), 2))
            resp = self.client.submit_order(req)
            log.info("alpaca limit order %s %s %.3f %s @ %.2f -> id %s", self.name, order.side, qty, order.ticker, float(order.limit_price),
                     getattr(resp, "id", "?"))
            return None
        req = MarketOrderRequest(symbol=self._symbol(order.ticker), qty=qty,
                                 side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL, time_in_force=TimeInForce.DAY)
        resp = self.client.submit_order(req)
        price = self.price(order.ticker)
        sched = (self.fee_book.for_ticker(order.ticker) if self.fee_book is not None else None) or self.fees
        cost = float(sched.cost(qty, price, order.side)) if sched is not None else 0.0
        self.ledger.add(cost)
        self.ledger.save()
        log.info("alpaca order %s %s %.3f %s -> id %s (fees %.2f, booked virtually)", self.name, order.side, qty, order.ticker,
                 getattr(resp, "id", "?"), cost)
        return Fill(order.ticker, order.side, qty, price, cost)
