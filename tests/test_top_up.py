"""Idle cash is put to work between rebalances: held names are bought back up to their slot, never sold on drift."""
import numpy as np

import stockbot.agent  # noqa: F401
from stockbot.agent.policy import PolicyBundle
from stockbot.execution.runner import TradingRunner
from stockbot.signals.registry import build_context, build_layout, build_providers


class Const:
    num_timesteps = 0

    def predict(self, obs, deterministic=True):
        return np.array([1.0], dtype=np.float32), None


def _runner(cfg, frames, **ex):
    for k, v in ex.items():
        cfg.set_path(f"execution.{k}", v)
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    bundle = PolicyBundle(Const(), build_layout(providers), {"algo": "ppo"})
    return TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False)


def test_top_up_fills_slots_from_idle_cash_and_never_sells(cfg, frames):
    r = _runner(cfg, frames, require_market_open=False, min_trade_usd=10)
    r.rank_top_k, r.max_position, r.cash_reserve = 3, 0.5, 0.10
    r.top_up, r.top_up_min_cash, r.top_up_band, r.top_up_max_names = True, 0.02, 0.15, 5

    class Broker:
        def equity(self):
            return 100_000.0

        def cash(self):
            return 40_000.0                                   # 30% idle above the 10% reserve

    r.broker = Broker()
    universe = list(frames)[:3]
    satellite = 0.9
    target = satellite / 3                                     # 30% of equity per slot
    current = {universe[0]: 0.2, universe[1]: 0.9, universe[2]: 0.0}     # fractions of the capital slice (max_position 0.5)
    kept = {t: current[t] * r.max_position for t in universe}
    before = dict(kept)
    topped = r._top_up(kept, current, universe, satellite)
    assert topped == [universe[0]]                             # only the held name below its slot; the empty one is not opened here
    assert abs(kept[universe[0]] - target) < 1e-9 and kept[universe[1]] == before[universe[1]]   # a drifted-up name is never trimmed
    assert kept[universe[2]] == 0.0

    class Poor(Broker):
        def cash(self):
            return 10_500.0                                    # only 0.5% idle above the reserve

    r.broker = Poor()
    kept2 = dict(before)
    assert r._top_up(kept2, current, universe, satellite) == [] and kept2 == before

    r.broker, r.top_up = Broker(), False                       # switched off: nothing happens
    kept3 = dict(before)
    assert r._top_up(kept3, current, universe, satellite) == [] and kept3 == before


def test_top_up_respects_the_order_cap_and_the_minimum_trade(cfg, frames):
    r = _runner(cfg, frames, require_market_open=False, min_trade_usd=1_000)
    r.rank_top_k, r.max_position, r.cash_reserve = 10, 0.5, 0.10
    r.top_up, r.top_up_min_cash, r.top_up_band, r.top_up_max_names = True, 0.02, 0.15, 2

    class Broker:
        def equity(self):
            return 100_000.0

        def cash(self):
            return 50_000.0

    r.broker = Broker()
    universe = list(frames)[:3]
    current = {t: 0.1 for t in universe}
    kept = {t: current[t] * r.max_position for t in universe}
    topped = r._top_up(kept, current, universe, 0.9)
    assert len(topped) == 2 and len(universe) == 3             # the order cap holds: three candidates, two orders
    r.min_trade_usd = 100_000                                  # every top-up would be below the minimum trade
    kept2 = {t: current[t] * r.max_position for t in universe}
    assert r._top_up(kept2, current, universe, 0.9) == []
