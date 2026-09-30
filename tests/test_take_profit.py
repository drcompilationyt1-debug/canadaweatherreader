"""The owner's take-profit re-evaluation: the shared step, the weekly verdict file, the runner applying the rule in force."""
import json

import numpy as np
import pandas as pd

import stockbot.agent  # noqa: F401
from stockbot.agent.backtest import load_tuned_take_profit, simulate, take_profit_step


def test_step_keeps_a_spike_that_still_ranks_and_sells_one_that_does_not():
    held = ["A", "B", "C"]
    entry = {"A": (100.0, 0), "B": (100.0, 0), "C": (100.0, 0)}
    price = {"A": 130.0, "B": 130.0, "C": 101.0, "D": 50.0}
    vol = {"A": 0.02, "B": 0.02, "C": 0.02, "D": 0.02}
    pct = {"A": 0.9, "B": 0.2, "C": 0.5, "D": 0.95}
    sold, stats = {}, {"triggers": 0, "kept": 0, "sold": 0, "rebought": 0}
    new, ev = take_profit_step(held, price, entry, vol, pct, {"z": 3.0, "keep_pct": 0.5}, 4, sold, 3, ranked=["D", "A", "C", "B"], stats=stats)
    assert new == ["A", "C"] and ("sold", "B", 0.30000000000000004) in ev or ("sold", "B", 0.3) in [(k, t, round(g, 2)) for k, t, g in ev]
    assert entry["A"][:2] == (130.0, 4) and "B" not in entry and sold["B"] == (130.0, 4)     # A kept with its reference reset, B sold
    assert stats == {"triggers": 2, "kept": 1, "sold": 1, "rebought": 0}                  # C's +1% is no spike
    no, _ = take_profit_step(["A", "C"], {"A": 131.0, "C": 101.0, "B": 120.0}, entry, vol, pct, {"z": 3.0, "keep_pct": 0.5, "rebuy_dip": 0.05},
                             5, sold, 3)
    assert "B" not in no and "B" in sold                                                 # on the dip but ranked in the bottom half: no
    pct2 = {**pct, "B": 0.6}
    new2, ev2 = take_profit_step(["A", "C"], {"A": 131.0, "C": 101.0, "B": 120.0}, entry, vol, pct2, {"z": 3.0, "keep_pct": 0.5, "rebuy_dip": 0.05},
                                 5, sold, 3, stats=stats)
    assert "B" in new2 and ("rebought", "B") == ev2[-1][:2] and "B" not in sold        # bought back 7.7% under the sale while it ranks
    new3, _ = take_profit_step(["A", "C"], price, {"A": (100.0, 0), "C": (100.0, 0)}, vol, pct, {"pct": 0.25, "keep_pct": 1.01, "replace": True},
                               4, {}, 2, ranked=["D", "A", "C"])
    assert new3 == ["C", "D"]                                                             # always sell at +25%, refill with the best name


def test_simulator_uses_the_shared_step():
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(1)
    px = pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(0.001, 0.03, 300)) for t in "ABCDEFGHIJKL"}, index=idx)
    sc = pd.DataFrame(rng.normal(size=(300, 12)), index=idx, columns=px.columns)
    base = simulate(px, sc, idx[0], k=4, every=21, hysteresis=2, fee_bps=10.0)
    tp = simulate(px, sc, idx[0], k=4, every=21, hysteresis=2, fee_bps=10.0, take_profit={"z": 2.0, "keep_pct": 1.01})
    assert "take_profit" not in base and tp["take_profit"]["triggers"] > 0 and tp["take_profit"]["sold"] == tp["take_profit"]["triggers"]
    assert tp["turnover_per_year"] > base["turnover_per_year"]


def test_verdict_file_semantics(tmp_path):
    assert load_tuned_take_profit(tmp_path, "main") is None
    (tmp_path / "take_profit_main.json").write_text(json.dumps({"accepted": False, "in_force": None}), encoding="utf-8")
    assert load_tuned_take_profit(tmp_path, "main") is None                               # plain holding
    (tmp_path / "take_profit_small.json").write_text(json.dumps({"accepted": True, "in_force": {"z": 3.0, "keep_pct": 0.5}}), encoding="utf-8")
    assert load_tuned_take_profit(tmp_path, "small") == {"z": 3.0, "keep_pct": 0.5}


def test_weekly_test_writes_a_verdict_and_never_adopts_on_the_first_pass(cfg, frames, tmp_path):
    from stockbot.agent.backtest import tune_take_profit
    from stockbot.env.dataset import MarketDataset
    from stockbot.signals.registry import build_context, build_layout, build_providers

    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    ds = MarketDataset.build(frames, providers, build_layout(providers), ctx, fit=True, train_end="2018-12-31")
    cfg.set_path("execution.rank", {"enabled": True, "top_k": 2, "every_bars": 10, "hysteresis": 1, "inputs": {"technical.ret_20": 1.0}, "adaptive": False})
    out = tmp_path / "take_profit_main.json"
    rep = tune_take_profit(cfg, ds, "main", out_path=out, years=2, recent_years=1, offsets=(0,))
    assert out.exists() and len(rep["results"]) >= 17 and all({"long", "recent", "rule"} <= set(r) for r in rep["results"])
    hold = next(r for r in rep["results"] if r["rule"] is None)
    assert (rep["in_force"] is None) == (not rep["better"])                              # on after ONE winning replay, else holding
    if rep["in_force"] is not None:
        assert rep["results"][[r["rule"] for r in rep["results"]].index(rep["in_force"])]["long"]["geo"] >= hold["long"]["geo"] + 0.002


def test_off_only_after_two_very_bad_weeks(cfg, frames, tmp_path, monkeypatch):
    """Once on, a take-profit rule stays on through one very bad week and is switched off by the second in a row."""
    from stockbot.agent import backtest as bt
    from stockbot.env.dataset import MarketDataset
    from stockbot.signals.registry import build_context, build_layout, build_providers

    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    ds = MarketDataset.build(frames, providers, build_layout(providers), ctx, fit=True, train_end="2018-12-31")
    cfg.set_path("execution.rank", {"enabled": True, "top_k": 2, "every_bars": 10, "hysteresis": 1, "inputs": {"technical.ret_20": 1.0}, "adaptive": False})
    rule = {"pct": 0.15, "keep_pct": 1.01}
    monkeypatch.setattr(bt, "TP_CANDIDATES", [None, rule])
    real = bt.simulate

    def rigged(*a, **kw):                                                    # the rule in force loses 5% a year against holding
        r = real(*a, **kw)
        if kw.get("take_profit"):
            r = {**r, "total": r["total"] - 0.5}
        return r

    monkeypatch.setattr(bt, "simulate", rigged)
    out = tmp_path / "take_profit_main.json"
    out.write_text(json.dumps({"tuned_at": "2026-09-19", "in_force": rule, "loss_streak": 0}), encoding="utf-8")
    week1 = bt.tune_take_profit(cfg, ds, "main", out_path=out, years=2, recent_years=1, offsets=(0,))
    assert week1["in_force"] == rule and week1["loss_streak"] == 1 and "stays on" in week1["reason"]     # one very bad week: still on
    again = bt.tune_take_profit(cfg, ds, "main", out_path=out, years=2, recent_years=1, offsets=(0,))
    assert again["in_force"] == rule and again["loss_streak"] == 1                                        # a same-day rerun is not a week
    data = json.loads(out.read_text(encoding="utf-8"))
    data["tuned_at"] = "2026-09-26"
    out.write_text(json.dumps(data), encoding="utf-8")
    week2 = bt.tune_take_profit(cfg, ds, "main", out_path=out, years=2, recent_years=1, offsets=(0,))
    assert week2["in_force"] is None and "turned off" in week2["reason"]                                   # the second in a row: off


def test_projection_peak_waits_for_the_price_to_beat_the_models_forecast():
    entry = {"A": (100.0, 0, 0.10), "B": (100.0, 0, 0.0)}                   # A was projected to gain 10% over 20 days, B nothing
    price = {"A": 112.0, "B": 112.0}
    vol = {"A": 0.01, "B": 0.01}
    pct = {"A": 0.1, "B": 0.1}
    new, ev = take_profit_step(["A", "B"], price, entry, vol, pct, {"fz": 1.5, "keep_pct": 1.01}, 20, {}, 2, proj={"A": 0.1, "B": 0.0})
    assert new == ["A"] and [e[:2] for e in ev] == [("sold", "B")]          # +12% is a peak for B, merely the forecast for A


def test_runner_applies_the_rule_in_force(cfg, frames):
    from stockbot.agent.policy import PolicyBundle
    from stockbot.execution.runner import TradingRunner
    from stockbot.signals.registry import build_context, build_layout, build_providers

    class Const:
        num_timesteps = 0

        def predict(self, obs, deterministic=True):
            return np.array([1.0], dtype=np.float32), None

    cfg.set_path("execution.rank.enabled", True)
    cfg.set_path("execution.rank.take_profit", {"enabled": True, "pct": 0.10, "keep_pct": 0.9})
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    r = TradingRunner(cfg, mode="paper", bundle=PolicyBundle(Const(), build_layout(providers), {"algo": "ppo"}),
                      frames_loader=lambda refresh: frames, with_llm=False)
    assert r.take_profit == {"pct": 0.10, "keep_pct": 0.9}
    r.frames = frames
    names = list(frames)
    last = {t: float(frames[t]["close"].iloc[-1]) for t in names}
    r.ctx.extra["positions"] = {t: {"avg_price": last[t] / 1.2, "price": last[t], "days": 5} for t in names[:2]}   # both up 20%
    current = {t: (1.0 if t in names[:2] else 0.0) for t in names}
    kept = {t: current[t] * r.max_position for t in names}
    scores = {names[0]: 3.0, names[1]: -3.0, **{t: 0.0 for t in names[2:]}}
    note = r._take_profit(kept, current, names, scores, 0.9, "2026-09-29")
    assert kept[names[1]] == 0.0 and kept[names[0]] > 0 and "sold" in note and "kept" in note     # the low-ranked spike goes
    assert names[1] in r.state["tp_sold"] and names[0] in r.state["tp_entry"]
    assert abs(r.state["tp_entry"][names[0]][0] - last[names[0]]) < 1e-6                         # the kept name's reference reset
