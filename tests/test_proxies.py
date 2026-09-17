"""Canadian names paper-traded through their US listings: symbol mapping, routing to the main sleeve, one-time migration."""
import stockbot.agent  # noqa: F401
from stockbot.execution.alpaca import AlpacaBroker, set_proxies
from stockbot.execution.base import Order
from stockbot.execution.paper import PaperBroker
from stockbot.execution.routed import RoutedBroker, migrate_proxied


def test_proxy_symbols_round_trip():
    set_proxies({"RY.TO": "RY", "CNR.TO": "CNI"})
    try:
        assert AlpacaBroker._symbol("RY.TO") == "RY" and AlpacaBroker._symbol("CNR.TO") == "CNI" and AlpacaBroker._symbol("BRK-B") == "BRK.B"
        assert AlpacaBroker._ticker("CNI") == "CNR.TO" and AlpacaBroker._ticker("BRK.B") == "BRK-B" and AlpacaBroker._ticker("AAPL") == "AAPL"
    finally:
        set_proxies({})
    assert AlpacaBroker._symbol("RY.TO") == "RY.TO"


def test_proxied_names_route_to_the_main_sleeve_and_migrate(tmp_path):
    prices = {"RY.TO": 100.0, "ATD.TO": 50.0, "AAPL": 200.0}
    us = PaperBroker(tmp_path / "us.json", lambda t: prices[t], 100_000.0, 0.0, 0.0, False)
    ca = PaperBroker(tmp_path / "ca.json", lambda t: prices[t], 100_000.0, 0.0, 0.0, False)
    ca.submit(Order("RY.TO", "buy", 10))                                              # positions from the simulated days
    ca.submit(Order("ATD.TO", "buy", 10))
    rb = RoutedBroker({"us": us, "ca": ca}, default="us", seeds={"ca": 100_000.0}, proxied={"RY.TO"})
    assert rb.market_for("RY.TO") == "us" and rb.market_for("ATD.TO") == "ca" and rb.sleeve_for("RY.TO") is us
    rb.submit(Order("RY.TO", "buy", 5))
    assert us.positions()["RY.TO"].shares == 5                                          # the Alpaca stand-in takes the order
    assert migrate_proxied(rb) == ["RY.TO"]                                             # the simulator no longer carries it ...
    assert "RY.TO" not in ca.positions() or ca.positions()["RY.TO"].shares == 0
    assert ca.positions()["ATD.TO"].shares == 10                                        # ... while a name without a US listing stays
    assert abs(rb.equity() - (100_000.0 + 0.0)) < 1e-6                                  # one book, unchanged by the move
