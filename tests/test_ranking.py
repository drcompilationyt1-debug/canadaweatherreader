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
    amounts = sorted((d.amount_usd for d in bought), reverse=True)        # (1 - 10% reserve) / K of the book each at full conviction
    assert amounts[0] == pytest.approx(0.45 * r.broker.equity(), rel=0.05)
    assert sum(amounts) == pytest.approx(0.9 * r.broker.equity(), rel=0.05)
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

def test_weekend_tuning_reweights_from_trailing_ic_with_a_guard(cfg, frames, tmp_path, monkeypatch):
    from stockbot.agent import backtest as bt
    from stockbot.env.dataset import MarketDataset
    from stockbot.execution.ranking import load_tuned_inputs

    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    ds = MarketDataset.build(frames, providers, build_layout(providers), ctx, fit=True, train_end="2018-12-31")
    cfg.set_path("execution.rank", {"enabled": True, "top_k": 2, "every_bars": 5, "hysteresis": 1, "inputs": {"technical.ret_20": 1.0}})
    monkeypatch.setattr(bt, "CANDIDATE_INPUTS", ["technical.ret_20", "trend.slope_30", "technical.ret_5"])
    monkeypatch.setattr(bt, "ANCHOR", "technical.ret_20")
    ics = bt.trailing_ic(ds, ["technical.ret_20", "trend.slope_30", "nope.x"], days=120, min_names=3)
    assert set(ics) <= {"technical.ret_20", "trend.slope_30"} and all("t" in v for v in ics.values())
    out = tmp_path / "rank_weights.json"
    rep = bt.tune_rank_weights(cfg, ds, out_path=out, days=120, min_t=0.0)
    assert out.exists() and rep["tuned"]["technical.ret_20"] == 1.0 and "backtest_last_year" in rep
    assert set(rep["inputs"]) == (set(rep["tuned"]) if rep["accepted"] else {"technical.ret_20"})
    loaded = load_tuned_inputs(tmp_path, {"x": 1.0})
    assert loaded == (rep["inputs"] if rep["accepted"] else {"x": 1.0})
    # a rejected tuning never reaches the runner: the file says accepted=false -> the fallback is used
    out.write_text('{"accepted": false, "inputs": {"trend.slope_30": 1.0}}', encoding="utf-8")
    assert load_tuned_inputs(tmp_path, {"x": 1.0}) == {"x": 1.0}
    out.write_text('{"accepted": true, "inputs": {"trend.slope_30": 1.0}}', encoding="utf-8")
    assert load_tuned_inputs(tmp_path, {"x": 1.0}) == {"trend.slope_30": 1.0}


def test_core_and_trend_filter_in_the_backtest():
    from stockbot.agent.backtest import simulate, trend_state
    from stockbot.execution.fees import FeeSchedule

    idx = pd.date_range("2026-01-01", periods=120, freq="B")
    spy = np.r_[np.linspace(100, 110, 60), np.linspace(110, 80, 60)]                      # rises, then falls through its average
    px = pd.DataFrame({"SPY": spy, "A": np.linspace(100, 130, 120), "B": np.linspace(100, 90, 120)}, index=idx)
    st = trend_state(px, "SPY", sma=20, band=0.0)
    assert st.iloc[30] == 1.0 and st.iloc[-1] == 0.0
    sc = pd.DataFrame({"A": 1.0, "B": 0.0}, index=idx)
    r = simulate(px, sc, "2026-01-01", k=1, every=5, hysteresis=0, fee_bps=0.0, core={"ticker": "SPY", "share": 0.5},
                 trend={"benchmark": "SPY", "sma": 20, "band": 0.0}, reserve=0.1)
    assert r["total"] < 0.2 and r["turnover_per_year"] > 0                                  # the satellite went to cash in the fall
    r2 = simulate(px, sc, "2026-01-01", k=1, every=5, hysteresis=0, fee_bps=0.0, core={"ticker": "SPY", "share": 0.5}, reserve=0.1)
    assert r2["total"] < r["total"] + 0.15                                                  # without the filter the satellite rode A up and SPY down
    assert FeeSchedule.from_preset("webull").cost(10, 100.0, "buy") == 0.0 and FeeSchedule.from_preset("webull").cost(10, 100.0, "sell") < 0.05


def test_runner_keeps_a_buy_only_core_and_obeys_the_trend_filter(cfg, frames):
    from stockbot.execution.base import Order

    cfg.set_path("execution.core", {"ticker": "AAA", "share": 0.5, "decide": "hold"})
    cfg.set_path("execution.rank", {"enabled": True, "top_k": 1, "every_bars": 3, "hysteresis": 0, "inputs": {"technical.ret_20": 1.0},
                                    "policy_floor": 1.0, "policy_veto": 0.05, "adaptive": False})
    cfg.set_path("execution.max_position", 0.5)
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    bundle = PolicyBundle(BuyEverything(), build_layout(providers), {"algo": "ppo"})
    r = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False)
    assert r.core_ticker == "AAA" and r.core_share == 0.5
    decisions = r.cycle(dry_run=False, refresh=False)
    by = {d.ticker: d for d in decisions}
    eq = r.broker.equity()
    assert by["AAA"].action == "BUY" and by["AAA"].amount_usd == pytest.approx(0.5 * eq, rel=0.05)        # the core: half the book
    sat = [d for d in decisions if d.ticker != "AAA" and d.action == "BUY"]
    assert len(sat) == 1 and sat[0].amount_usd == pytest.approx(0.4 * eq, rel=0.05)                        # one slot = 1 - 0.5 - 0.1 reserve
    assert "AAA" not in r.last_rank["chosen"]
    # decide: hold -> the core is never sold: even a policy that wants nothing leaves it alone, and the satellite holds off-schedule
    r.bundle = PolicyBundle(type("Flat", (), {"num_timesteps": 0, "predict": staticmethod(lambda obs, deterministic=True: (np.array([-1.0], dtype=np.float32), None))})(),
                            bundle.layout, {"algo": "ppo"})
    decisions2 = r.cycle(dry_run=False, refresh=False)
    assert all(d.action == "HOLD" for d in decisions2) and r.broker.position("AAA").shares > 0
    # the trend filter turns off when the benchmark closes below its average: the satellite is sold, the core stays
    cfg.set_path("execution.rank.trend_filter", {"enabled": True, "benchmark": "AAA", "sma": 20, "band": 0.0})
    low = {t: df.copy() for t, df in frames.items()}
    low["AAA"].loc[low["AAA"].index[-1], "close"] = float(low["AAA"]["close"].iloc[-25:].mean()) * 0.8
    r2 = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: low, with_llm=False)
    r2.state["last_rebalance_date"] = None
    decisions3 = r2.cycle(dry_run=False, refresh=False)
    assert r2.state["trend_on"] is False and r2.last_rank["trend_on"] is False
    sells = [d for d in decisions3 if d.action == "SELL"]
    assert sat[0].ticker in {d.ticker for d in sells} and r2.broker.position("AAA").shares > 0 and "trend filter off" in r2.last_cycle_note


def test_training_draws_the_fee_regime_per_episode(cfg, frames):
    from stockbot.config import env_settings
    from stockbot.env.dataset import MarketDataset
    from stockbot.env.trading_env import TradingEnv

    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    ds = MarketDataset.build(frames, providers, build_layout(providers), ctx, fit=True, train_end="2018-12-31")
    env_cfg = {**env_settings(cfg), "fees": "moomoo", "fee_scale": 0.1, "fee_choices": ["moomoo", "webull"]}
    env = TradingEnv(ds, env_cfg, seed=1)
    seen = {env.reset()[0] is not None and env.portfolio.fees.preset for _ in range(30)}
    assert seen == {"moomoo", "webull"}
    drags = {}
    for _ in range(30):
        env.reset(options={"cash": 10_000})
        drags.setdefault(env.portfolio.fees.preset, set()).add(round(float(env.fee_drag(100.0)), 4))
    assert max(drags["moomoo"]) > max(drags["webull"])                                       # the fee_drag input tells the regimes apart
    evl = TradingEnv(ds, env_cfg, seed=1, eval_mode=True)
    assert {evl.reset()[0] is not None and evl.portfolio.fees.preset for _ in range(5)} == {"moomoo"}


def test_model_timed_core_trims_high_and_rebuilds_low(cfg, frames):
    cfg.set_path("execution.core", {"ticker": "AAA", "share": 0.5, "decide": "model", "min_share": 0.25, "max_share": 0.65, "every_bars": 3, "band": 0.05})
    cfg.set_path("execution.rank", {"enabled": True, "top_k": 1, "every_bars": 3, "hysteresis": 0, "inputs": {"technical.ret_20": 1.0},
                                    "policy_floor": 1.0, "policy_veto": 0.05, "adaptive": False})
    cfg.set_path("execution.max_position", 0.5)
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    layout = build_layout(providers)
    full = PolicyBundle(BuyEverything(), layout, {"algo": "ppo"})
    flat = PolicyBundle(type("Flat", (), {"num_timesteps": 0, "predict": staticmethod(lambda obs, deterministic=True: (np.array([-1.0], dtype=np.float32), None))})(),
                        layout, {"algo": "ppo"})
    r = TradingRunner(cfg, mode="paper", bundle=full, frames_loader=lambda refresh: frames, with_llm=False)
    assert r.core_decide == "model" and r.core_min == 0.25 and r.core_max == 0.65
    decisions = r.cycle(dry_run=False, refresh=False)
    by = {d.ticker: d for d in decisions}
    eq = r.broker.equity()
    assert by["AAA"].action == "BUY" and by["AAA"].amount_usd == pytest.approx(0.65 * eq, rel=0.05)         # full conviction: the ceiling
    sat = [d for d in decisions if d.ticker != "AAA" and d.action == "BUY"]
    assert len(sat) == 1 and sat[0].amount_usd == pytest.approx((1 - 0.65 - 0.1) * eq, rel=0.1)          # the slots share what is left
    assert r.state["core_last_date"] == str(frames["AAA"].index[-1].date())
    # no conviction, but not a core decision day yet: the core is kept
    r.bundle = flat
    decisions2 = r.cycle(dry_run=False, refresh=False)
    assert {d.action for d in decisions2 if d.ticker == "AAA"} == {"HOLD"}
    # on a core decision day with no conviction it is trimmed to the floor (sold high), never below it
    r.state["core_last_date"] = None
    r.state["last_rebalance_date"] = None
    decisions3 = r.cycle(dry_run=False, refresh=False)
    core3 = next(d for d in decisions3 if d.ticker == "AAA")
    assert core3.action == "SELL"
    assert r.core_now("AAA", r.broker.equity()) == pytest.approx(0.25, abs=0.03)
    # and with conviction back it is rebuilt towards the ceiling with the cash on hand
    r.bundle = full
    r.state["core_last_date"] = None
    decisions4 = r.cycle(dry_run=False, refresh=False)
    core4 = next(d for d in decisions4 if d.ticker == "AAA")
    assert core4.action == "BUY" and r.core_now("AAA", r.broker.equity()) > 0.45


def test_backtest_core_series_and_policy_timing(cfg, frames):
    from stockbot.agent.backtest import core_series_from_policy, simulate
    from stockbot.env.dataset import MarketDataset

    idx = pd.date_range("2026-01-01", periods=40, freq="B")
    px = pd.DataFrame({"SPY": np.linspace(100, 120, 40), "A": np.linspace(100, 110, 40)}, index=idx)
    series = pd.Series(np.where(np.arange(40) < 20, 0.65, 0.25), index=idx)
    r = simulate(px, pd.DataFrame({"A": 1.0}, index=idx), "2026-01-01", k=1, every=5, hysteresis=0, fee_bps=0.0,
                 core={"ticker": "SPY", "share": 0.5, "series": series}, reserve=0.1)
    assert r["turnover_per_year"] > 0 and 0.0 < r["total"] < 0.2
    assert core_series_from_policy(cfg, MarketDataset.build(frames, [], build_layout([]), build_context(cfg, with_llm=False, with_news=False),
                                                            fit=False, train_end="2018-12-31"), "AAA", "2019-06-01", 0.25, 0.65) is None   # no policy trained


def test_structure_tuning_decides_slots_cadence_and_the_index_sleeve(cfg, frames, tmp_path, monkeypatch):
    from stockbot.agent import backtest as bt
    from stockbot.env.dataset import MarketDataset
    from stockbot.execution.ranking import load_tuned_profile

    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    ds = MarketDataset.build(frames, providers, build_layout(providers), ctx, fit=True, train_end="2018-12-31")
    cfg.set_path("execution.rank", {"enabled": True, "top_k": 2, "every_bars": 5, "hysteresis": 1, "inputs": {"technical.ret_20": 1.0}, "adaptive": True})
    cfg.set_path("execution.core", {"ticker": "AAA", "share": 0.0})
    cfg.set_path("env.initial_cash", 10_000)
    monkeypatch.setattr(bt, "PROFILE_GRID", {"small": {"top_k": (1, 2), "every_bars": (5, 21), "hysteresis": (1,), "core_share": (0.0, 0.5)},
                                             "main": {"top_k": (2,), "every_bars": (5,), "hysteresis": (1,), "core_share": (0.0,)}})
    out = tmp_path / "rank_profile_main.json"
    rep = bt.tune_profile(cfg, ds, "main", out_path=out, years=2)
    assert out.exists() and len(rep["candidates"]) >= 8 and set(rep["profile"]) >= {"top_k", "every_bars", "hysteresis", "core_share"}
    assert any(c["core_share"] == 0.5 for c in rep["candidates"])                 # the index sleeve is one of the candidates ...
    assert all("1y_sharpe" in c and "2y_total" in c for c in rep["candidates"])
    if rep["accepted"]:                                                          # ... and only a two-window improvement is adopted
        assert rep["best_result"]["2y"]["sharpe"] >= rep["current_result"]["2y"]["sharpe"] - 0.02
        assert load_tuned_profile(tmp_path, "main") == rep["profile"]
    else:
        assert load_tuned_profile(tmp_path, "main") is None
    # the runner picks a tuned structure up (and a tuned index sleeve becomes a model-timed core)
    (cfg.path("models_dir")).mkdir(parents=True, exist_ok=True)
    (cfg.path("models_dir") / "rank_profile_main.json").write_text(
        '{"accepted": true, "profile": {"top_k": 1, "every_bars": 21, "hysteresis": 2, "core_share": 0.4, "core_ticker": "AAA"}}', encoding="utf-8")
    bundle = PolicyBundle(BuyEverything(), build_layout(providers), {"algo": "ppo"})
    r = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False)
    assert (r.rank_top_k, r.rank_every, r.rank_hysteresis) == (1, 21, 2)
    assert r.core_ticker == "AAA" and r.core_share == 0.4 and r.core_min == 0.2 and r.core_max == 0.4 and r.core_decide == "model"
