"""One book across the real broker and the simulated sleeve, and the dust the whole-share rule leaves behind."""
import stockbot.agent  # noqa: F401
from stockbot.execution.base import Order
from stockbot.execution.paper import PaperBroker
from stockbot.execution.routed import RoutedBroker


def test_simulated_sleeve_counts_only_its_pnl(tmp_path):
    prices = {"AAPL": 100.0, "RY.TO": 50.0}
    us = PaperBroker(tmp_path / "us.json", lambda t: prices[t], 100_000.0, 0.0, 0.0, False)
    ca = PaperBroker(tmp_path / "ca.json", lambda t: prices[t], 100_000.0, 0.0, 0.0, False)
    rb = RoutedBroker({"us": us, "ca": ca}, default="us", seeds={"ca": 100_000.0})
    assert rb.equity() == 100_000.0 and rb.cash() == 100_000.0                      # one book, not two
    assert rb.equity_for("RY.TO") == 100_000.0 and rb.cash_for("RY.TO") == 100_000.0 and rb.equity_for("AAPL") == 100_000.0
    rb.submit(Order("RY.TO", "buy", 100))                                            # 5,000 of a Canadian name comes out of the book's cash
    assert abs(rb.cash() - 95_000.0) < 1e-6 and abs(rb.equity() - 100_000.0) < 1e-6
    prices["RY.TO"] = 60.0                                                            # its +1,000 shows in the book
    assert abs(rb.equity() - 101_000.0) < 1e-6 and abs(rb.equity_for("AAPL") - 101_000.0) < 1e-6
    s = rb.summary()
    assert abs(s["equity"] - 101_000.0) < 1e-6 and s["sleeves"]["ca"]["seed"] == 100_000.0 and abs(s["sleeves"]["ca"]["pnl"] - 1_000.0) < 1e-6
    plain = RoutedBroker({"us": us, "ca": ca}, default="us")                          # two real accounts: sleeves stay separate
    assert abs(plain.equity() - (us.equity() + ca.equity())) < 1e-6 and plain.equity_for("RY.TO") == ca.equity()


def test_close_dust_only_liquidates_fractional_remnants():
    from stockbot.execution.alpaca import AlpacaBroker

    class P:
        def __init__(self, sym, qty, mv):
            self.symbol, self.qty, self.market_value = sym, qty, mv

    closed = []

    class Client:
        def get_all_positions(self):
            return [P("AAPL", "0.673", "225.76"), P("BIDU", "31.067", "2802.7"), P("F", "30", "360.0"), P("BRK.B", "0.613", "312.1")]

        def close_position(self, sym):
            if sym == "BRK.B":
                raise RuntimeError("rejected")
            closed.append(sym)

    b = AlpacaBroker.__new__(AlpacaBroker)
    b.client = Client()
    out = b.close_dust(500.0)
    assert closed == ["AAPL"]                                                          # whole-share positions are never touched
    assert [(o["ticker"], o["closed"]) for o in out] == [("AAPL", True), ("BRK-B", False)] and "rejected" in out[1]["error"]
