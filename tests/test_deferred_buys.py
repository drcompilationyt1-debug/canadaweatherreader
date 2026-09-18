"""Sells at the decision cycle, buys later in the day; a resting limit order is cancelled and topped up at market."""
import numpy as np

import stockbot.agent  # noqa: F401
from stockbot.agent.policy import PolicyBundle
from stockbot.execution.runner import TradingRunner
from stockbot.signals.registry import build_context, build_layout, build_providers


class BuyEverything:
    num_timesteps = 0

    def predict(self, obs, deterministic=True):
        return np.array([1.0], dtype=np.float32), None


def _runner(cfg, frames, **ex):
    for k, v in ex.items():
        cfg.set_path(f"execution.{k}", v)
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    bundle = PolicyBundle(BuyEverything(), build_layout(providers), {"algo": "ppo"})
    return TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False)


def test_buys_wait_for_buy_at_and_then_fill(cfg, frames):
    r = _runner(cfg, frames, buy_at="13:15", require_market_open=False)
    decisions = r.cycle(dry_run=False, refresh=False)
    buys = [d for d in decisions if d.action == "BUY"]
    assert buys and all("scheduled for 13:15" in d.note for d in buys)
    assert all(r.broker.position(d.ticker).shares == 0 for d in buys)            # nothing bought yet
    assert len(r.pending_buys) == len(buys) and r.broker.cash() == r.broker.equity()
    fills = r.execute_pending_buys()
    assert len(fills) == len(buys) and all(f["deferred"] for f in fills)
    assert all(r.broker.position(d.ticker).shares > 0 for d in buys) and r.pending_buys == []
    assert r.state["fees_paid"] > 0 and r.execute_pending_buys() == []


def test_immediate_buys_when_no_buy_at(cfg, frames):
    r = _runner(cfg, frames, buy_at="", require_market_open=False)
    decisions = r.cycle(dry_run=False, refresh=False)
    buys = [d for d in decisions if d.action == "BUY"]
    assert buys and all(r.broker.position(d.ticker).shares > 0 for d in buys) and r.pending_buys == []


def test_alpaca_cancel_open_books_the_filled_part_and_returns_the_rest():
    from stockbot.execution.alpaca import AlpacaBroker, VirtualLedger
    from stockbot.execution.fees import FeeSchedule

    class O:
        def __init__(self, oid, qty, filled, side="buy", limit=100.0):
            self.id, self.qty, self.filled_qty, self.side, self.limit_price, self.filled_avg_price, self.symbol = oid, qty, filled, side, limit, limit, "AAPL"

    cancelled = []

    class Client:
        def get_orders(self, filter=None):
            return [O("a", "10", "4"), O("b", "5", "0")]

        def cancel_order_by_id(self, oid):
            cancelled.append(oid)

    b = AlpacaBroker.__new__(AlpacaBroker)
    b.client, b.fee_book, b.fees, b.ledger = Client(), None, FeeSchedule.from_config({"preset": "moomoo"}), VirtualLedger(None)
    assert b.cancel_open("AAPL") == 11.0 and cancelled == ["a", "b"]         # 6 + 5 shares still unfilled
    assert b.ledger.fees > 0 and b.ledger.n_fills == 1                        # the 4 filled shares paid their moomoo fee
