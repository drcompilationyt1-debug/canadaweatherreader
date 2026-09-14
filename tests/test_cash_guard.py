"""Buys never exceed the cash on hand: the paper broker caps them and the cycle plans around the sleeve's cash."""
from __future__ import annotations

import numpy as np

from stockbot.agent.policy import PolicyBundle
from stockbot.data.loader import synthetic_universe
from stockbot.execution.base import Order
from stockbot.execution.fees import FeeSchedule
from stockbot.execution.paper import PaperBroker
from stockbot.execution.runner import TradingRunner
from stockbot.signals.registry import build_context, build_layout, build_providers


def test_paper_broker_caps_buys_at_cash(tmp_path):
    fees = FeeSchedule.from_preset("moomoo")
    b = PaperBroker(tmp_path / "s.json", lambda t: 100.0, initial_cash=1000.0, commission=0.0, slippage=0.0, fees=fees)
    fill = b.submit(Order("AAA", "buy", 20))                     # wants $2000, has $1000
    assert fill is not None and fill.qty < 10.0 and fill.qty > 9.9
    assert b.cash() >= 0.0 and abs(b.cash()) < 0.02              # spent everything but the fees
    assert b.submit(Order("AAA", "buy", 1)) is None              # nothing left: skipped, not overdrawn
    assert b.affordable_qty("AAA", 100.0) < 1e-3
    sold = b.submit(Order("AAA", "sell", 5))
    assert sold is not None and b.cash() > 490.0


class BuyEverything:
    num_timesteps = 0

    def predict(self, obs, deterministic=True):
        return np.array([1.0], dtype=np.float32), None


def test_cycle_plans_buys_within_sleeve_cash(cfg, frames):
    cfg.set_path("env.initial_cash", 2500)
    cfg.set_path("execution.max_position", 1.0)
    cfg.set_path("execution.max_gross_exposure", 3.0)
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend", "market_regime")]
    bundle = PolicyBundle(BuyEverything(), build_layout(providers), {"algo": "ppo"})
    runner = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False)
    decisions = runner.cycle(dry_run=False, refresh=False)
    by = {d.ticker: d for d in decisions}
    bought = [d for d in decisions if d.action == "BUY"]
    held = [d for d in decisions if d.action == "HOLD"]
    assert len(bought) == 1 and len(held) == 2                   # $2,500 buys one full slice, not three
    assert all("insufficient cash" in d.note for d in held)
    assert "cut to cash" in bought[0].note                       # the one buy was trimmed to the 99% of cash available
    assert runner.broker.cash() >= 0.0
    assert runner.broker.cash() < 2500 * 0.02                    # nearly all cash deployed, none overdrawn
    # a second cycle with no cash left sends nothing and does not error
    decisions2 = runner.cycle(dry_run=False, refresh=False)
    assert all(d.action == "HOLD" for d in decisions2 if d.ticker != bought[0].ticker)
    assert runner.broker.cash() >= 0.0
