"""The rank-core decision layer: blended rank scores, top-K with hysteresis, the rebalance cadence, the runner using
it, and the portfolio backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockbot.agent.policy import PolicyBundle
from stockbot.execution.ranking import bars_since, every_bars_of, rank_scores, rebalance_due, select_top, slot_weights
from stockbot.execution.runner import TradingRunner
from stockbot.signals.registry import build_context, build_layout, build_providers


def test_select_top_keeps_held_names_within_the_hysteresis_band():
    scores = {f"T{i}": 10 - i for i in range(10)}          # T0 best ... T9 worst
    assert select_top(scores, [], 3, 2) == ["T0", "T1", "T2"]
    assert select_top(scores, ["T4"], 3, 2) == ["T4", "T0", "T1"]       # rank 4 < 3 + 2: kept, takes a slot
    assert select_top(scores, ["T6"], 3, 2) == ["T0", "T1", "T2"]       # rank 6: dropped
    assert select_top({**scores, "X": float("nan")}, ["X"], 3, 2) == ["T0", "T1", "T2"]
    assert every_bars_of("biweekly") == 10 and every_bars_of(21) == 21 and every_bars_of(None) == 1


def test_rank_scores_blend_percentiles_and_skip_masked_blocks(cfg):
    ctx = build_context(cfg, with_llm=False, with_news=False)
    layout = build_layout([p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")])
    tech, trend = layout.block("technical"), layout.block("trend")
    j1 = tech.start + list(tech.feature_names).index("ret_20")
    j2 = trend.start + list(trend.feature_names).index("slope_30")
    obs = {}
    for i, t in enumerate(("A", "B", "C", "D")):
        o = np.zeros(layout.signal_dim, dtype=np.float32)
        o[tech.offset] = 1.0
        o[j1] = i                                                        # A lowest ... D highest
        o[trend.offset] = 1.0 if t != "D" else 0.0                       # D's trend block is masked
        o[j2] = -i                                                       # the opposite order
        obs[t] = o
    s = rank_scores(layout, obs, {"technical.ret_20": 1.0, "trend.slope_30": 1.0})
    assert s["D"] == pytest.approx(1.0)                                  # only the technical rank counts for D
    assert s["A"] == pytest.approx((0.25 + 1.0) / 2) and s["B"] == pytest.approx((0.5 + 2 / 3) / 2)   # trend ranks are over the 3 unmasked names
    assert all(np.isnan(v) for v in rank_scores(layout, obs, {"nope.x": 1.0}).values())
    idx = pd.date_range("2026-09-01", periods=12, freq="B")
    assert bars_since(idx, None, "2026-09-10") is None and bars_since(idx, "2026-09-03", "2026-09-10") == 5
    assert rebalance_due(idx, "2026-09-03", "2026-09-10", 5) and not rebalance_due(idx, "2026-09-03", "2026-09-10", 6)
    w = slot_weights(["A", "B", "C"], 4, {"A": 1.0, "B": 0.5, "C": 0.01}, floor=0.5, veto=0.05)
    assert w["A"] == pytest.approx(0.25) and w["B"] == pytest.approx(0.25 * 0.75) and w["C"] == 0.0


class BuyEverything:
    num_timesteps = 0

    def predict(self, obs, deterministic=True):
        return np.array([1.0], dtype=np.float32), None


def test_runner_holds_the_top_k_and_rebalances_on_schedule(cfg, frames):
    cfg.set_path("execution.rank", {"enabled": True, "top_k": 2, "every_bars": 3, "hysteresis": 1, "inputs": {"technical.ret_20": 1.0},
                                    "policy_floor": 0.5, "policy_veto": 0.05})
    cfg.set_path("execution.max_position", 0.5)
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    bundle = PolicyBundle(BuyEverything(), build_layout(providers), {"algo": "ppo"})
    r = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False)
    assert r.rank_enabled and r.rank_top_k == 2 and r.rank_every == 3
    decisions = r.cycle(dry_run=False, refresh=False)
    bought = [d for d in decisions if d.action == "BUY"]
    assert len(bought) == 2 and len(r.last_rank["chosen"]) == 2
    top2 = sorted(frames, key=lambda t: -float(frames[t]["close"].pct_change(20).iloc[-1]))[:2]
    assert set(r.last_rank["chosen"]) == set(top2)                      # the two names with the best 20-day return
    assert r.state["last_rebalance_date"] == str(frames["AAA"].index[-1].date())
    amounts = sorted((d.amount_usd for d in bought), reverse=True)        # 1/K of the book each (full conviction) ...
    assert amounts[0] == pytest.approx(0.5 * r.broker.equity(), rel=0.05)
    assert sum(amounts) == pytest.approx(0.9 * r.broker.equity(), rel=0.05)   # ... the second cut to keep the 10% cash reserve
    decisions2 = r.cycle(dry_run=False, refresh=False)                   # same day: not a rebalance day -> hold
    assert all(d.action == "HOLD" for d in decisions2) and "rank core" in r.last_cycle_note


def test_backtest_reports_the_rule_vs_benchmarks(cfg, frames):
    from stockbot.agent.backtest import format_report, round_trip_bps, run_backtests, simulate
    from stockbot.env.dataset import MarketDataset

    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    ds = MarketDataset.build(frames, providers, build_layout(providers), ctx, fit=True, train_end="2018-12-31")
    cfg.set_path("execution.rank", {"enabled": True, "top_k": 2, "every_bars": 5, "hysteresis": 1, "inputs": {"technical.ret_20": 1.0}})
    cfg.set_path("accounts", {})
    rep = run_backtests(cfg, ds, oos_start="2019-01-01", years=2, benchmark="AAA")
    res = rep["results"]["oos"]
    assert {"AAA", "equal_weight", "rank_100k"} <= set(res) and res["rank_100k"]["k"] == 2 and res["rank_100k"]["days"] > 200
    assert "excess_vs_benchmark" in res["rank_100k"] and res["equal_weight"]["turnover_per_year"] < 1.0
    assert round_trip_bps(cfg, 10_000, 10) > round_trip_bps(cfg, 100_000, 20) > 0
    txt = format_report(rep)
    assert "rank_100k" in txt and "equal_weight" in txt
    px = pd.DataFrame({"A": np.linspace(100, 200, 60), "B": np.linspace(100, 50, 60)}, index=pd.date_range("2026-01-01", periods=60, freq="B"))
    sc = pd.DataFrame({"A": 1.0, "B": 0.0}, index=px.index)
    r = simulate(px, sc, "2026-01-01", k=1, every=5, hysteresis=0, fee_bps=10.0)
    assert r["total"] > 0.9 and r["turnover_per_year"] < 30
