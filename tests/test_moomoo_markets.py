"""One moomoo Canada account for both markets: TSX names become CA.<code>, US names US.<code>, fees per market."""
import pandas as pd

import stockbot.agent  # noqa: F401
from stockbot.execution.base import Order
from stockbot.execution.fees import FeeBook
from stockbot.execution.moomoo import MoomooBroker


class FakeTrd:
    def __init__(self):
        self.orders = []

    def get_acc_list(self):
        return 0, pd.DataFrame({"acc_id": [1], "trd_env": ["SIMULATE"]})

    def accinfo_query(self, **kw):
        return 0, pd.DataFrame({"total_assets": [100_000.0], "cash": [40_000.0]})

    def position_list_query(self, **kw):
        return 0, pd.DataFrame({"code": ["US.AAPL", "CA.RY", "US.BRK.B"], "qty": [10, 20, 3], "cost_price": [180.0, 120.0, 400.0],
                                "position_side": ["LONG", "LONG", "LONG"]})

    def place_order(self, **kw):
        self.orders.append(kw)
        return 0, pd.DataFrame({"order_id": ["o1"]})


class FakeQuote:
    def get_market_snapshot(self, codes):
        return 0, pd.DataFrame({"code": codes, "last_price": [100.0] * len(codes)})


def test_moomoo_codes_positions_and_fees_span_both_markets(cfg):
    fb = FeeBook.from_config(cfg)
    b = MoomooBroker(env="simulate", trd_ctx=FakeTrd(), quote_ctx=FakeQuote(), fee_book=fb, market="ALL")
    assert b._code("RY.TO") == "CA.RY" and b._code("BRK-B") == "US.BRK.B" and b._code("AAPL") == "US.AAPL"
    assert b._ticker("CA.RY") == "RY.TO" and b._ticker("US.BRK.B") == "BRK-B" and b._ticker("US.AAPL") == "AAPL"
    pos = b.positions()
    assert set(pos) == {"AAPL", "RY.TO", "BRK-B"} and pos["RY.TO"].shares == 20
    assert b.equity() == 100_000.0 and b.cash() == 40_000.0
    f_ca = b.submit(Order("RY.TO", "buy", 10))
    f_us = b.submit(Order("AAPL", "buy", 10))
    assert b.trd.orders[0]["code"] == "CA.RY" and b.trd.orders[1]["code"] == "US.AAPL"
    assert f_ca.cost > 0 and f_us.cost > 0 and abs(f_ca.cost - f_us.cost) > 1e-9        # each market's own moomoo schedule
    single = MoomooBroker(env="simulate", trd_ctx=FakeTrd(), quote_ctx=FakeQuote(), market="US")
    assert single._code("BRK-B") == "US.BRK.B"                                          # a fixed single market still works
