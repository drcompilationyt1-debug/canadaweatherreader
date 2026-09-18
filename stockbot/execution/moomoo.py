"""moomoo (Futu) OpenAPI broker - paper ("simulate") and real accounts through the OpenD gateway.

Setup
-----
1. ``pip install moomoo-api`` (already in the project venv).
2. Install OpenD from https://www.moomoo.com/download/OpenAPI, log in with your moomoo account and
   leave it running (default 127.0.0.1:11111).
3. Paper trading: ``execution.moomoo.env: simulate`` (default) - uses your moomoo paper account, no
   password needed.  Real money: ``env: real`` plus the trade password in ``MOOMOO_TRADE_PASSWORD``
   and ``--i-understand-real-money`` on the command line.

Symbols are moomoo style: ``US.AAPL``, ``US.BRK.B`` for US listings and ``CA.RY`` for a TSX name the universe spells
``RY.TO`` - one moomoo Canada account trades both markets (``market: ALL``), each fill booked at its own market's fee
schedule; orders are whole-share market orders.
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
                 market: str = "ALL", allow_short: bool = False, trd_ctx: Any = None, quote_ctx: Any = None, fees=None, fee_book=None):
        self.env_name = str(env).lower()
        self.market = market.upper()              # ALL = every market the account is authorised for (US + CA on moomoo Canada)
        self.supports_short = bool(allow_short)
        self.fees = fees  # estimated moomoo fees per fill (the account statement has the exact figure)
        self.fee_book = fee_book                  # per-market schedules (moomoo US vs moomoo Canada)
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
        mkt = getattr(sdk.TrdMarket, "NONE" if self.market == "ALL" else self.market, sdk.TrdMarket.US)   # NONE = no market filter
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

    MARKET_CODES = {"us": "US", "ca": "CA"}         # our market keys -> moomoo market prefixes
    SUFFIX_OF = {"CA": ".TO"}                        # moomoo market prefix -> the universe's ticker suffix

    def _code(self, ticker: str) -> str:
        """``RY.TO`` -> ``CA.RY``, ``BRK-B`` -> ``US.BRK.B``; a fixed single market (``market: US``) prefixes everything with it."""
        from .markets import market_of

        m = self.MARKET_CODES.get(market_of(ticker), "US") if self.market == "ALL" else self.market
        base = ticker
        for suf in (".TO", ".V", ".NE", ".CN"):
            if m == "CA" and base.upper().endswith(suf):
                base = base[: -len(suf)]
                break
        return f"{m}.{base.replace('-', '.')}"

    def _ticker(self, code: str) -> str:
        """``CA.RY`` -> ``RY.TO``, ``US.BRK.B`` -> ``BRK-B``."""
        code = str(code)
        m = None
        if "." in code:
            m, code = code.split(".", 1)
        t = code.replace(".", "-")
        return t + self.SUFFIX_OF.get(str(m).upper(), "") if m else t

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
        if order.limit_price:                                              # a resting limit order: no fill to report yet
            otype = sdk.OrderType.NORMAL if sdk else "NORMAL"
            ret, data = self.trd.place_order(price=float(order.limit_price), qty=qty, code=self._code(order.ticker), trd_side=side, order_type=otype,
                                             trd_env=self.trd_env, acc_id=self.acc_id, remark=order.note[:30] if order.note else None)
            self._ok(ret, data, "place_order")
            log.info("moomoo %s limit order %s %.0f %s @ %.2f", self.name, order.side, qty, order.ticker, float(order.limit_price))
            return None
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
        sched = (self.fee_book.for_ticker(order.ticker) if self.fee_book is not None else None) or self.fees
        cost = float(sched.cost(qty, price, order.side)) if (sched is not None and price > 0) else 0.0
        return Fill(order.ticker, order.side, qty, price, cost)

    def cancel_open(self, ticker: str) -> float:
        """Cancel the ticker's open orders; returns the unfilled quantity."""
        sdk = self.sdk
        code = self._code(ticker)
        unfilled = 0.0
        try:
            ret, data = self.trd.order_list_query(code=code, trd_env=self.trd_env, acc_id=self.acc_id)
            data = self._ok(ret, data, "order_list_query")
            for _, o in data.iterrows():
                status = str(o.get("order_status", "")).upper()
                if any(s in status for s in ("FILLED_ALL", "CANCELLED", "FAILED", "DELETED", "DISABLED")):
                    continue
                unfilled += max(0.0, float(o.get("qty", 0)) - float(o.get("dealt_qty", 0) or 0))
                op = sdk.ModifyOrderOp.CANCEL if sdk else "CANCEL"
                self.trd.modify_order(op, o["order_id"], 0, 0, trd_env=self.trd_env, acc_id=self.acc_id)
        except Exception as e:  # noqa: BLE001
            log.warning("moomoo cancel_open %s: %s", ticker, e)
        return unfilled

    def close(self) -> None:
        for ctx in (self.trd, self.quote):
            try:
                ctx.close()
            except Exception:  # noqa: BLE001
                pass
