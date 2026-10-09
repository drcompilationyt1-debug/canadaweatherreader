"""The daily book: look every day, trade rarely, let winners run - and every order worth its fee."""
import numpy as np
import pytest

import stockbot.agent  # noqa: F401
from stockbot.execution.daily_book import DailyRules, market_activity, name_stats, plan_day, shadow_step


def _walk(n=120, start=100.0, step=0.01):
    """A quiet zig-zag (daily moves of about 1%) ending at ``start``."""
    moves = np.array([step if i % 2 else -step for i in range(n)])
    c = start * np.exp(np.cumsum(moves[::-1]) - np.sum(moves))
    return c


def _stats(closes, bench=None):
    return name_stats(np.asarray(closes, float), bench)


RULES = DailyRules(k=3, buy_rank=3, hold_rank=6, min_hold_days=5, hurdle=2.0, max_new=2, max_swaps=1, starter=1.0)


def _plan(weights, book, pct, stats, age=None, level="normal", cash_w=0.0, cost=0.004, min_w=0.05, agree=None, sold_high=None,
          bad_news=None, rules=RULES):
    names = set(pct) | set(weights)
    return plan_day(weights=weights, book=book, age=age or {t: 20 for t in weights}, pct=pct, stats=stats, agree=agree or {},
                    level=level, slot=0.3, capacity=0.9, cash_w=cash_w, cost_rt={t: cost for t in names},
                    min_w={t: min_w for t in names}, ic=0.05, disp=0.08, bad_news=bad_news, sold_high=sold_high, max_w=0.45, rules=rules)


def _universe(n=20):
    """Percentiles for n names: N00 best."""
    names = [f"N{i:02d}" for i in range(n)]
    return names, {t: 1.0 - i / n for i, t in enumerate(names)}


def test_a_spike_is_sold_in_part_then_the_trail_and_a_cheaper_buy_back_compound():
    names, pct = _universe()
    base = _walk()
    spike = np.concatenate([base, base[-1] * np.exp(np.cumsum([0.05, 0.05, 0.05, 0.05]))])     # +22% in four days
    stats = {t: _stats(base) for t in names}
    stats["N01"] = _stats(spike)
    book = {"N01": {"entry": 100.0, "high": 100.0, "pct0": pct["N01"]}}
    sold = {}
    out, events, book = _plan({"N01": 0.3}, book, pct, stats, age={"N01": 4}, sold_high=sold)
    assert out["N01"] == pytest.approx(0.15)                                   # half sold into the spike
    assert book["N01"]["armed"] and book["N01"]["spiked"] and "N01" in sold
    assert {e[0] for e in events} >= {"armed", "spike"}
    # the price falls back through the trailing stop: the rest goes, still above the buy price
    high = spike[-1]
    fall = np.concatenate([spike, high * np.exp(np.cumsum([-0.04, -0.04, -0.04]))])
    stats["N01"] = _stats(fall)
    out, events, book = _plan({"N01": 0.15}, book, pct, stats, age={"N01": 7}, sold_high=sold)
    assert out["N01"] == 0.0 and events[0][0] == "trail" and events[0][2] > 0
    assert sold["N01"]["w"] == pytest.approx(0.3)                               # both sales remembered
    # cheaper again and still ranked: bought back with all the money the two sales raised (more shares than before)
    cheap = np.concatenate([fall, fall[-1] * np.exp(np.cumsum([-0.04, -0.04]))])
    stats["N01"] = _stats(cheap)
    out, events, book = _plan({}, {}, pct, stats, cash_w=0.5, sold_high=sold)
    assert out["N01"] == pytest.approx(0.3) and any(e[0] == "rebuy" for e in events) and "N01" not in sold
    assert book["N01"]["entry"] == pytest.approx(cheap[-1])


def test_never_sells_at_a_loss_unless_it_is_really_bad():
    names, pct = _universe()
    stats = {t: _stats(_walk()) for t in names}
    book = {"N08": {"entry": 120.0, "high": 120.0, "pct0": 0.9}}                 # under water, out of the hold band (#9), top half
    out, events, _ = _plan({"N08": 0.3}, book, pct, stats)
    assert "N08" not in out and events[0][0] == "kept_loss"
    pct2 = dict(pct, N08=0.2)                                                   # the ranking collapsed: really bad
    out, events, _ = _plan({"N08": 0.3}, book, pct2, stats)
    assert out["N08"] == 0.0 and events[0][0] == "bad"
    drop = np.concatenate([_walk(), 100 * np.exp(np.cumsum([-0.03] * 10))])     # a stock-specific crash
    stats2 = dict(stats, N08=_stats(drop, _walk(130)))
    out, events, _ = _plan({"N08": 0.3}, book, pct, stats2)
    assert out["N08"] == 0.0 and "stock-specific drop" in events[0][3]
    in_profit = {"N08": {"entry": 80.0, "high": 100.0, "pct0": 0.9}}           # out of the band in profit: sold
    out, events, _ = _plan({"N08": 0.3}, in_profit, pct, stats)
    assert out["N08"] == 0.0 and any(e[0] == "band" for e in events)


def test_bad_news_needs_the_price_to_confirm_it():
    names, pct = _universe()
    stats = {t: _stats(_walk()) for t in names}
    book = {"N00": {"entry": 100.0, "high": 100.0, "pct0": 1.0}}
    out, _, _ = _plan({"N00": 0.3}, book, pct, stats, bad_news={"N00"})
    assert "N00" not in out                                                     # bad news, no drop: nothing
    slide = np.concatenate([_walk(), 100 * np.exp(np.cumsum([-0.007] * 10))])          # ~2.2 volatilities: under the 2.5 alone
    out, events, _ = _plan({"N00": 0.3}, book, pct, dict(stats, N00=_stats(slide, _walk(130))), bad_news={"N00"})
    assert out["N00"] == 0.0 and "bad news" in events[0][3]


def test_a_buy_must_be_worth_its_fees_and_never_chases_a_spike():
    names, pct = _universe()
    stats = {t: _stats(_walk()) for t in names}
    out, events, _ = _plan({}, {}, pct, stats, cash_w=0.9)
    assert set(out) == {"N00", "N01"} and all(v == pytest.approx(0.3) for v in out.values())   # normal day: 2 new names
    out, _, _ = _plan({}, {}, pct, stats, cash_w=0.9, cost=0.02)                # fees larger than the expected gain: nothing
    assert out == {}
    jump = np.concatenate([_walk(), 100 * np.exp(np.cumsum([0.03] * 5))])
    out, events, _ = _plan({}, {}, pct, dict(stats, N00=_stats(jump)), cash_w=0.9)
    assert "N00" not in out and any(e[0] == "wait" and e[1] == "N00" for e in events)
    out, _, _ = _plan({}, {}, pct, stats, cash_w=0.9, min_w=0.5)                # an order under the minimum is never sent
    assert out == {}


def test_quiet_days_take_only_a_very_good_deal():
    names, pct = _universe()
    stats = {t: _stats(_walk()) for t in names}
    out, _, _ = _plan({}, {}, pct, stats, cash_w=0.9, level="quiet")
    assert out == {}
    out, events, _ = _plan({}, {}, pct, stats, cash_w=0.9, level="quiet", agree={"N00": 0.9})
    assert list(out) == ["N00"] and "very good deal" in events[-1][3]


def test_a_swap_needs_a_clear_gain_and_never_sells_at_a_loss():
    names, pct = _universe(40)
    stats = {t: _stats(_walk()) for t in names}
    held = {"N10": 0.3, "N11": 0.3, "N12": 0.3}                                 # full book of middling names (top 30%)
    book = {t: {"entry": 90.0, "high": 100.0, "pct0": 0.9} for t in held}
    rules = DailyRules(k=3, buy_rank=3, hold_rank=20, min_hold_days=5, hurdle=2.0, max_swaps=1)
    out, events, _ = _plan(held, book, pct, stats, rules=rules)
    assert out == {"N12": 0.0, "N00": 0.3} and events[-1][0] == "swap"          # one swap: the weakest for the best
    losing = {t: {"entry": 120.0, "high": 120.0, "pct0": 0.9} for t in held}
    out, _, _ = _plan(held, losing, pct, stats, rules=rules)
    assert out == {}                                                            # all under water: no swap
    out, _, _ = _plan(held, book, pct, stats, rules=rules, cost=0.01)          # the gain does not beat 2 round trips
    assert out == {}


def test_activity_level_follows_the_market_and_the_models():
    rng = np.random.default_rng(0)
    calm = 100 * np.exp(np.cumsum(np.r_[rng.normal(0.0005, 0.01, 700), rng.normal(0.001, 0.006, 40)]))
    assert market_activity(calm, 0.8, DailyRules())[0] == "active"
    assert market_activity(calm, 0.5, DailyRules())[0] == "normal"
    crash = 100 * np.exp(np.cumsum(np.r_[rng.normal(0.0005, 0.01, 700), np.tile([-0.05, 0.03], 15)]))
    level, info = market_activity(crash, 0.5, DailyRules())
    assert level == "quiet" and info["below_200d"] and info["vol_pct"] > 0.9


def test_the_old_rule_runs_as_a_shadow_with_fees():
    scores = {"A": 0.9, "B": 0.8, "C": 0.1}
    prices = {"A": 10.0, "B": 20.0, "C": 5.0}
    fee = lambda t, d: 2.0  # noqa: E731
    sh = shadow_step({"cash": 1000.0, "units": {"C": 100.0}}, scores, prices, None, 2, 10, 0, fee)
    assert set(sh["units"]) == {"A", "B"} and sh["rebalanced"]
    assert sh["equity"] == pytest.approx(1500.0 - 3 * 2.0, abs=1e-6)          # three orders paid for
    sh2 = shadow_step(sh, {"C": 1.0, "A": 0.0, "B": 0.0}, dict(prices, A=11.0), 3, 2, 10, 0, fee)
    assert not sh2["rebalanced"] and set(sh2["units"]) == {"A", "B"}           # frozen between rebalances


def test_runner_trades_the_daily_book_and_keeps_the_shadow(cfg, frames):
    from stockbot.agent.policy import PolicyBundle
    from stockbot.execution.runner import TradingRunner
    from stockbot.signals.registry import build_context, build_layout, build_providers

    class Buy:
        num_timesteps = 0

        def predict(self, obs, deterministic=True):
            return np.array([1.0], dtype=np.float32), None

    cfg.set_path("execution.rank", {"enabled": True, "top_k": 2, "every_bars": 3, "hysteresis": 1, "inputs": {"technical.ret_20": 1.0},
                                    "policy_floor": 1.0, "daily": {"enabled": True, "k": 2, "buy_rank": 2, "hold_rank": 3,
                                                                   "hurdle": 0.0, "chase_z": 99, "level": "normal"}})
    cfg.set_path("execution.max_position", 0.5)
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    bundle = PolicyBundle(Buy(), build_layout(providers), {"algo": "ppo"})
    r = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False)
    assert r.daily_rules is not None and r.daily_rules.k == 2
    decisions = r.cycle(dry_run=False, refresh=False)
    bought = [d for d in decisions if d.action == "BUY"]
    assert len(bought) == 2 and "daily book [normal" in r.last_cycle_note
    assert set(r.state["daily_book"]) == {d.ticker for d in bought}
    assert r.state["shadow"]["start"] == str(frames["AAA"].index[-1].date())
    r.cycle(dry_run=False, refresh=False)                                        # same day again: nothing new, nothing sold
    assert set(r.state["daily_book"]) == {d.ticker for d in bought}


def test_news_scorecard_settles_verdicts_against_what_followed(tmp_path):
    import json

    import pandas as pd

    from stockbot.feedback.news_score import score_news

    dates = pd.bdate_range("2026-01-01", periods=80)
    names = [f"T{i}" for i in range(12)]
    closes = pd.DataFrame({t: 100.0 * np.exp(np.arange(80) * 0.001) for t in names}, index=dates)
    closes["T0"] = 100.0 * np.exp(-np.arange(80) * 0.004)                     # the name the LLM keeps calling a full close
    f = tmp_path / "news.jsonl"
    with open(f, "w", encoding="utf-8") as fh:
        for d in dates[:50]:
            for t in names:
                bad = t == "T0"
                fh.write(json.dumps({"date": str(d.date()), "ticker": t, "nofx_direction": -1.0 if bad else 0.5,
                                     "nofx_confidence": 0.9, "nofx_close": 1.0 if bad else 0.0, "nofx_open": 0.0 if bad else 1.0,
                                     "fb_net": -0.5 if bad else 0.2, "fb_n": 5}) + "\n")
    rep = score_news(f, closes, min_calls=40)
    assert rep["llm_full_close"]["calls"] == 50 and rep["llm_full_close"]["excess_20d"] < 0
    assert rep["llm_close_proven"] and rep["rank_ic"]["finbert"]["ic_20d"] > 0.2
    assert score_news(tmp_path / "missing.jsonl", closes)["verdicts"] == 0


def test_whole_shares_only_a_name_whose_share_does_not_fit_the_slot_is_skipped():
    names, pct = _universe()
    stats = {t: _stats(_walk()) for t in names}
    stats["N00"] = _stats(_walk(start=5000.0))                                 # one share = $5,000
    out, _, _ = plan_day(weights={}, book={}, age={}, pct=pct, stats=stats, agree={}, level="normal", slot=0.3, capacity=0.9,
                         cash_w=0.9, cost_rt={t: 0.004 for t in names}, min_w={t: 0.05 for t in names}, ic=0.05, disp=0.08,
                         equity=10_000.0, rules=RULES)
    assert "N00" not in out and set(out) == {"N01", "N02"}                     # a $3,000 slot cannot hold it: the next names
