"""moomoo (Futu) OpenAPI broker - paper ("simulate") and real accounts through the OpenD gateway.

Setup
-----
1. ``pip install moomoo-api`` (already in the project venv).
2. Install OpenD from https://www.moomoo.com/download/OpenAPI, log in with your moomoo account and
   leave it running (default 127.0.0.1:11111).
3. Paper trading: ``execution.moomoo.env: simulate`` (default) - uses your moomoo paper account, no
   password needed.  Real money: ``env: real`` plus the trade password in ``MOOMOO_TRADE_PASSWORD``
   and ``--i-understand-real-money`` on the command line.

Symbols are moomoo style (``US.AAPL``, ``US.BRK.B``); orders are whole-share market orders.
"""
from __future__ import annotations

import os
from typing import Any

from ..logging_utils import get_logger
from .base import Broker, Fill, Order, Position

log = get_logger(__name__)


def _sdk():
    try:
        import moomoo as sdk  # pip install moomoo-api
    except ImportError:
        import futu as sdk  # the older package name (pip install futu-api)
    return sdk


class MoomooBroker(Broker):
    name = "moomoo"

    def __init__(self, env: str = "simulate", host: str = "127.0.0.1", port: int = 11111, security_firm: str = "FUTUINC",
                 market: str = "US", allow_short: bool = False, trd_ctx: Any = None, quote_ctx: Any = None, fees=None):
        self.env_name = str(env).lower()
        self.market = market.upper()
        self.supports_short = bool(allow_short)
        self.fees = fees  # estimated moomoo fees per fill (the account statement has the exact figure)
        self.name = "moomoo-paper" if self.env_name == "simulate" else "moomoo-REAL"
        self._unlocked = False
        if trd_ctx is not None and quote_ctx is not None:  # injected fakes (tests)
            self.sdk = None
            self.trd, self.quote = trd_ctx, quote_ctx
            self.trd_env = "SIMULATE" if self.env_name == "simulate" else "REAL"
            self.acc_id = 0
            return
        sdk = _sdk()
        self.sdk = sdk
        self.trd_env = sdk.TrdEnv.SIMULATE if self.env_name == "simulate" else sdk.TrdEnv.REAL
        firm = getattr(sdk.SecurityFirm, security_firm, sdk.SecurityFirm.FUTUINC)
        mkt = getattr(sdk.TrdMarket, self.market, sdk.TrdMarket.US)
        self.trd = sdk.OpenSecTradeContext(filter_trdmarket=mkt, host=host, port=port, security_firm=firm)
        self.quote = sdk.OpenQuoteContext(host=host, port=port)
        self.acc_id = self._pick_account()

    # ------------------------------------------------------------------ helpers
    def _ok(self, ret, data, what: str):
        ok = getattr(self.sdk, "RET_OK", 0) if self.sdk is not None else 0
        if ret != ok:
            raise RuntimeError(f"moomoo {what} failed: {data}")
        return data

    def _pick_account(self) -> int:
        ret, accs = self.trd.get_acc_list()
        accs = self._ok(ret, accs, "get_acc_list")
        want = "SIMULATE" if self.env_name == "simulate" else "REAL"
        rows = accs[accs["trd_env"].astype(str).str.upper() == want] if len(accs) else accs
        if len(rows) == 0:
            raise RuntimeError(f"no moomoo {want} account found in OpenD (accounts: {accs['trd_env'].tolist() if len(accs) else []})")
        return int(rows.iloc[0]["acc_id"])

    def _code(self, ticker: str) -> str:
        return f"{self.market}.{ticker.replace('-', '.')}"

    def _ticker(self, code: str) -> str:
        code = str(code)
        if "." in code:
            code = code.split(".", 1)[1]
        return code.replace(".", "-")

    def _unlock(self) -> None:
        if self.env_name != "real" or self._unlocked:
            return
        pw = os.environ.get("MOOMOO_TRADE_PASSWORD")
        if not pw:
            raise RuntimeError("set MOOMOO_TRADE_PASSWORD to trade a real moomoo account")
        ret, data = self.trd.unlock_trade(password=pw)
        self._ok(ret, data, "unlock_trade")
        self._unlocked = True

    # ------------------------------------------------------------------ Broker API
    def _account(self):
        ret, data = self.trd.accinfo_query(trd_env=self.trd_env, acc_id=self.acc_id, currency="USD")
        return self._ok(ret, data, "accinfo_query").iloc[0]

    def equity(self) -> float:
        row = self._account()
        return float(row.get("total_assets", row.get("net_assets", 0.0)))

    def cash(self) -> float:
        row = self._account()
        for col in ("cash", "avl_withdrawal_cash"):   # never "power": that is margin buying power, not our money
            if col in row and row[col] == row[col]:
                return float(row[col])
        return 0.0

    def positions(self) -> dict[str, Position]:
        ret, data = self.trd.position_list_query(trd_env=self.trd_env, acc_id=self.acc_id, currency="USD")
        data = self._ok(ret, data, "position_list_query")
        out: dict[str, Position] = {}
        for _, r in data.iterrows():
            qty = float(r["qty"])
            if str(r.get("position_side", "LONG")).upper().endswith("SHORT"):
                qty = -abs(qty)
            if abs(qty) < 1e-9:
                continue
            out[self._ticker(r["code"])] = Position(self._ticker(r["code"]), qty, float(r.get("cost_price", 0.0) or 0.0))
        return out

    def price(self, ticker: str) -> float:
        ret, data = self.quote.get_market_snapshot([self._code(ticker)])
        data = self._ok(ret, data, "get_market_snapshot")
        return float(data.iloc[0]["last_price"])

    def submit(self, order: Order) -> Fill | None:
        qty = float(int(order.qty))  # whole shares only
        if qty <= 0:
            return None
        pos = self.position(order.ticker)
        if order.side == "sell" and not self.supports_short:
            qty = min(qty, float(int(max(pos.shares, 0))))
            if qty <= 0:
                return None
        self._unlock()
        sdk = self.sdk
        side = (sdk.TrdSide.BUY if order.side == "buy" else sdk.TrdSide.SELL) if sdk else order.side.upper()
        otype = sdk.OrderType.MARKET if sdk else "MARKET"
        ret, data = self.trd.place_order(price=0.0, qty=qty, code=self._code(order.ticker), trd_side=side, order_type=otype,
                                         trd_env=self.trd_env, acc_id=self.acc_id, remark=order.note[:30] if order.note else None)
        data = self._ok(ret, data, "place_order")
        try:
            price = self.price(order.ticker)
        except Exception:  # noqa: BLE001
            price = 0.0
        order_id = data.iloc[0].get("order_id", "?") if hasattr(data, "iloc") and len(data) else "?"
        log.info("moomoo %s order %s %.0f %s -> id %s", self.name, order.side, qty, order.ticker, order_id)
        cost = float(self.fees.cost(qty, price, order.side)) if (self.fees is not None and price > 0) else 0.0
        return Fill(order.ticker, order.side, qty, price, cost)

    def close(self) -> None:
        for ctx in (self.trd, self.quote):
            try:
                ctx.close()
            except Exception:  # noqa: BLE001
                pass
