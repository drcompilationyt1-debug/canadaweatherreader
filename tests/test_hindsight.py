"""Regret attribution, the fee-aware path oracle, the parallel what-if search, the period review's extras,
and the hindsight fine-tune (learns from the paper decisions, guarded by the out-of-sample score)."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockbot.feedback.attribution import (attribute_regret, attribute_regret_daily, ic_by_voter, path_oracle,
                                           portfolio_path_oracle, regime, round_trips, what_if_day)


def test_attribution_adds_up_to_the_regret():
    moves = {"A": 0.02, "B": -0.03, "C": 0.01, "D": -0.01}
    held = {"A": 0.5, "B": 1.0, "C": 0.0, "D": 0.0}
    slice_frac = 0.1
    actual = sum(held[t] * moves[t] for t in moves) * slice_frac
    oracle = sum(max(m, 0.0) for m in moves.values()) * slice_frac
    a = attribute_regret(held, moves, slice_frac, cost=0.0004)
    assert abs(a["total"] - (oracle - actual + 0.0004)) < 1e-12
    assert a["missed"] == pytest.approx(0.001) and a["wrong_side"] == pytest.approx(0.003) and a["under_sized"] == pytest.approx(0.001)
    assert a["biggest"] == "wrong_side"
    W = pd.DataFrame({"A": [0.05, 0.10, 0.10], "B": [0.0, 0.0, 0.10]}, index=pd.date_range("2026-09-14", periods=3))
    R = pd.DataFrame({"A": [0.0, 0.02, -0.01], "B": [0.0, 0.01, 0.03]}, index=W.index)
    d = attribute_regret_daily(W, R, 0.10, cost=0.0)
    # day 2: A held half (0.05/0.10) up 2% -> under-sized 0.001; B not held up 1% -> missed 0.001; day 3: A full, down 1% -> wrong 0.001; B not held up 3% -> missed 0.003
    assert d["under_sized"] == pytest.approx(0.001) and d["missed"] == pytest.approx(0.004) and d["wrong_side"] == pytest.approx(0.001)


def test_path_oracle_holds_through_noise_when_fees_bite():
    r = np.array([0.01, -0.002, 0.01, -0.002, 0.01, -0.002, 0.01])
    free, path_free = path_oracle(r, fee_frac=0.0)
    assert free == pytest.approx(0.04)                                       # flips out of every down day for free
    assert path_free.tolist() == [1, 0, 1, 0, 1, 0, 1]
    paid, path_paid = path_oracle(r, fee_frac=0.005)
    assert path_paid.tolist() == [1, 1, 1, 1, 1, 1, 1]                        # with fees: stay in
    assert paid == pytest.approx(r.sum() - 0.005) and paid < free
    R = pd.DataFrame({"A": r, "B": -r, "C": r * 0.5}, index=pd.date_range("2026-09-01", periods=7))
    po = portfolio_path_oracle(R, {"A": 0.005, "B": 0.005, "C": 0.005}, max_position=0.5, gross_cap=1.0)
    assert po["names"] == ["A", "C"] and po["return"] == pytest.approx(0.5 * (paid + (r * 0.5).sum() - 0.005))


def test_what_if_search_splits_sizing_and_timing():
    fee = lambda notional, price, side: 0.001 * notional
    # AAA: up 1% then gives it back; BBB: straight up 2%; CCC: down 2%
    prices = {"AAA": [100, 101, 100], "BBB": [100, 101, 102], "CCC": [100, 99, 98]}
    held = {"AAA": 1.0, "BBB": 0.5, "CCC": 1.0}
    res = what_if_day(prices, held, slice_cap=10_000, equity=100_000, fee_fn=fee, labels=["open", "t+1", "t+2"], workers=2)
    by = {r["ticker"]: r for r in res["per_name"]}
    assert by["AAA"]["best_exit"] == "t+1" and by["AAA"]["best_level"] == 1.0 and by["AAA"]["timing_regret"] > 0
    assert by["AAA"]["sizing_regret"] == pytest.approx(0.0001)                # full slice held to a flat close: only the entry fee was avoidable
    assert by["BBB"]["best_exit"] == "hold" and by["BBB"]["sizing_regret"] > 0 and by["BBB"]["timing_regret"] == pytest.approx(0.0)
    assert by["CCC"]["best_level"] == 0.0 and by["CCC"]["regret"] > 0.0019
    assert res["best_names"] == ["BBB", "AAA"] and res["best_return"] > res["actual_return"]
    assert res["moves_searched"] > 0


def test_round_trips_ic_and_regime():
    fills = [{"ticker": "A", "side": "buy", "qty": 10, "price": 100, "ts": "2026-09-08T14:30:00+00:00"},
             {"ticker": "A", "side": "sell", "qty": 10, "price": 103, "ts": "2026-09-10T14:30:00+00:00"},
             {"ticker": "B", "side": "buy", "qty": 5, "price": 50, "ts": "2026-09-08T14:31:00+00:00"},
             {"ticker": "B", "side": "sell", "qty": 5, "price": 49, "ts": "2026-09-11T14:30:00+00:00"},
             {"ticker": "C", "side": "buy", "qty": 8, "price": 20, "ts": "2026-09-10T14:30:00+00:00"}]
    rt = round_trips(fills, equity=100_000.0)
    pytest.importorskip("pyfolio")
    assert rt["n"] == 2 and rt["win_rate"] == 0.5 and rt["best"]["symbol"] == "A" and rt["open_positions"] == 1
    assert rt["profit_factor"] == pytest.approx(30.0 / 5.0)
    idx = pd.date_range("2026-09-07", periods=4)
    R = pd.DataFrame({t: np.linspace(-0.02, 0.02, 4) * (k + 1) for k, t in enumerate("ABCDE")}, index=idx)
    votes = {}
    for d in idx[:-1]:
        for k, t in enumerate("ABCDE"):
            nxt = R.shift(-1).loc[d, t]
            votes[(d, t)] = {"good": {"vote": np.sign(nxt), "score": float(nxt)}, "bad": {"vote": -np.sign(nxt), "score": float(-nxt)}}
    ic = ic_by_voter(votes, R, min_names=5)
    assert ic["good"]["ic"] == pytest.approx(1.0) and ic["bad"]["ic"] == pytest.approx(-1.0) and ic["good"]["days"] == 3
    rg = regime(pd.Series([0.01, 0.02, -0.005, 0.015]))
    assert rg["label"].startswith("up-") and rg["benchmark_return"] > 0.03


def _train_small_bundle(cfg, frames, names=("technical", "trend")):
    from stockbot.agent.train import train
    from stockbot.env.dataset import MarketDataset
    from stockbot.signals.registry import build_context, build_layout, build_providers

    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in names]
    layout = build_layout(providers)
    ds = MarketDataset.build(frames, providers, layout, ctx, fit=True, train_end="2018-12-31")
    train(cfg, total_timesteps=128, dataset=ds, n_envs=2)
    return ds, layout


def _record_decisions(cfg, frames, layout, days, rng):
    """Paper decisions on real (synthetic) bars with observations of the right size and the true forward move."""
    from stockbot.feedback.experience import ExperienceStore

    store = ExperienceStore(cfg.path("feedback.experience_file"))
    for d in days:
        for t in frames:
            obs = rng.normal(size=layout.obs_dim).astype(np.float32)
            obs[-6:] = [0.5, 0.0, 0.0, 0.1, 1.0, 0.0]                           # half a slice held
            store.record(mode="paper", ticker=t, date=d.strftime("%Y-%m-%d"), obs=obs, action=0.5, target_exposure=0.5, weight=0.05,
                         decision="HOLD", price=float(frames[t].loc[d, "close"]), equity=1e5, availability={}, conviction=0.0)
    return store


def test_hindsight_learns_one_unit_at_a_time(cfg, frames, monkeypatch):
    from stockbot.agent.policy import PolicyBundle
    from stockbot.feedback import hindsight
    from stockbot.feedback.hindsight import build_samples, finetune, learn, learn_due, policy_mean

    cfg.set_path("feedback.hindsight.min_samples", 20)
    cfg.set_path("feedback.hindsight.epochs", 30)
    cfg.set_path("feedback.hindsight.lr", 0.003)
    cfg.set_path("feedback.hindsight.anchor_weight", 0.1)
    cfg.set_path("feedback.hindsight.ref_steps", 64)
    cfg.set_path("feedback.hindsight.eval_tickers", 2)
    cfg.set_path("feedback.hindsight.eval_bars", 60)
    ds, layout = _train_small_bundle(cfg, frames)
    rng = np.random.default_rng(1)
    days = [d for d in frames["AAA"].index if pd.Timestamp("2019-10-01") <= d <= pd.Timestamp("2019-11-15")]
    _record_decisions(cfg, frames, layout, days, rng)                          # half a slice held every day
    today = date(2019, 11, 16)

    # the day unit: yesterday's decisions only, each labelled by yesterday's close (fee-aware: small moves keep the size)
    s_day = build_samples(cfg, layout.obs_dim, period="day", frames=frames, today=today)
    assert s_day["start"] == s_day["end"] == "2019-11-15" and len(s_day["meta"]) == 3
    for m in s_day["meta"]:
        assert m["level_best"] in (0.0, 0.25, 0.5, 0.75, 1.0)
        assert (m["level_best"] >= 0.5) if m["r_day"] > 0 else (m["level_best"] <= 0.5)
    # the month unit: all of October as ONE unit, labelled by the fee-aware best path through the month
    s_month = build_samples(cfg, layout.obs_dim, period="month", end=date(2019, 10, 31), frames=frames, today=today)
    oct_days = [d for d in days if d.month == 10]
    n = len(s_month["meta"])
    assert n == 3 * len(oct_days) and s_month["start"] == "2019-10-01" and s_month["X"].shape == (n, layout.obs_dim)
    assert np.all(np.abs(s_month["y"]) <= 1.0) and np.all(s_month["w"] >= 0.1)
    lv = np.array([m["level_best"] for m in s_month["meta"]])
    rd = np.array([m["r_day"] for m in s_month["meta"]])
    assert np.corrcoef(lv, rd)[0, 1] > 0.15                                     # the path holds more on the days that went up ...
    for t in frames:                                                            # ... but does not flip on every wiggle (fees)
        rows = [m for m in s_month["meta"] if m["ticker"] == t]
        switches = int(np.sum(np.diff([m["level_best"] for m in rows]) != 0))
        flips = int(np.sum(np.diff(np.sign([m["r_day"] for m in rows])) != 0))
        assert switches <= flips
    # a unit whose bars have not settled cannot be learned; records of another layout are ignored
    assert len(build_samples(cfg, layout.obs_dim, period="week", end=date(2019, 11, 22), frames=frames, today=today)["meta"]) == 0
    from stockbot.feedback.experience import ExperienceStore

    ExperienceStore(cfg.path("feedback.experience_file")).record(mode="paper", ticker="AAA", date="2019-10-15", obs=np.zeros(5), action=0.0,
                                                                  target_exposure=0.0, weight=0.0, decision="HOLD", price=100.0, equity=1e5, availability={})
    assert len(build_samples(cfg, layout.obs_dim, period="month", end=date(2019, 10, 31), frames=frames, today=today)["meta"]) == n

    # the fine-tune moves the actor toward the labels, and stops at a deadline after at least one epoch
    bundle = PolicyBundle.load(cfg.path("train.checkpoint_dir"))
    before = policy_mean(bundle.model, s_month["X"])
    fit = finetune(bundle.model, s_month["X"], s_month["y"], s_month["w"], s_month["X"][:16], epochs=30, lr=3e-3, anchor_weight=0.0)
    after = policy_mean(bundle.model, s_month["X"])
    assert fit["loss_after"] < fit["loss_before"] and fit["epochs"] == 30
    assert np.mean(np.abs(np.clip(after, -1, 1) - s_month["y"])) < np.mean(np.abs(np.clip(before, -1, 1) - s_month["y"]))
    assert finetune(bundle.model, s_month["X"], s_month["y"], s_month["w"], s_month["X"][:16], epochs=5, deadline=time.time() - 1)["epochs"] == 1

    # learn(): the guard keeps an update only when the out-of-sample score holds
    path = cfg.path("train.checkpoint_dir") / "latest.zip"
    digest0 = hashlib.sha1(path.read_bytes()).hexdigest()
    scores = iter([0.50, 0.20])                                                  # before 0.50, after 0.20: a big drop -> rejected
    monkeypatch.setattr(hindsight, "_score", lambda *a, **k: next(scores))
    rep = learn(cfg, period="month", end=date(2019, 10, 31), frames=frames, dataset=ds, today=today)
    assert rep["samples"] == n and rep["accepted"] == 0 and rep["yardstick"]["tickers"] == 3   # no budget given -> the full yardstick (all 3 test names)
    member = next(iter(rep["members"].values()))
    assert member["accepted"] is False and member["score_before"] == 0.5
    assert hashlib.sha1(path.read_bytes()).hexdigest() == digest0               # the file was not touched
    state = json.loads((cfg.path("train.checkpoint_dir") / "hindsight.json").read_text())
    assert state["learned"]["month"] == "2019-10-31" and state["history"][-1]["samples"] == n
    # the same unit again -> skipped as already learned; forced with a score that holds -> accepted and saved
    assert "already learned" in learn(cfg, period="month", end=date(2019, 10, 31), frames=frames, dataset=ds, today=today)["skipped"]
    scores = iter([0.50, 0.49])
    rep3 = learn(cfg, period="month", end=date(2019, 10, 31), force=True, frames=frames, dataset=ds, today=today, budget_minutes=5)
    assert rep3["accepted"] == 1 and rep3["yardstick"]["bars"] == 60             # a short budget -> the quick yardstick
    assert hashlib.sha1(path.read_bytes()).hexdigest() != digest0
    reloaded = PolicyBundle.load(cfg.path("train.checkpoint_dir"))                # the fine-tuned weights are what got saved
    m = next(iter(rep3["members"].values()))
    loss_reloaded = float(np.average((np.clip(policy_mean(reloaded.model, s_month["X"]), -1, 1) - s_month["y"]) ** 2,
                                     weights=s_month["w"] / s_month["w"].mean()))
    assert abs(loss_reloaded - m["loss_after"]) < 1e-4 and loss_reloaded < m["loss_before"]
    # learn_due: yesterday (3 decisions) and last week (15) are too small, the month is done, the year has nothing
    scores = iter([0.5, 0.5] * 4)
    due = learn_due(cfg, today=today, frames=frames, dataset=ds)
    assert "month" not in due and "need 20" in due["day 2019-11-15"]["skipped"] and "need 20" in due["week"]["skipped"]
    assert due["week"]["end"] == "2019-11-15" and "year" in due and sum(k.startswith("day ") for k in due) == 5


def test_session_learns_before_the_open_alongside_the_warm_up(cfg, frames, monkeypatch):
    import sys
    from datetime import datetime, timedelta

    from stockbot.agent.policy import PolicyBundle
    from stockbot.execution.market_hours import NY, MarketClock
    from stockbot.execution.runner import TradingRunner
    from stockbot.execution.session import TradingSession
    from stockbot.signals.registry import build_context, build_layout, build_providers

    class Clock(MarketClock):
        def __init__(self, start):
            self.t = start
            super().__init__(source="builtin", now_fn=lambda: self.t)

        def sleep(self, s):
            self.t += timedelta(seconds=s)

    class M:
        num_timesteps = 0

        def predict(self, obs, deterministic=True):
            return np.array([1.0], dtype=np.float32), None

    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    bundle = PolicyBundle(M(), build_layout(providers), {"algo": "ppo"})
    clock = Clock(datetime(2026, 9, 14, 9, 0, tzinfo=NY))                                   # 30 min before the open
    runner = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False, clock=clock)
    seen = {}
    monkeypatch.setattr(TradingSession, "learn_command", lambda self, minutes: seen.setdefault("minutes", minutes) and
                        [sys.executable, "-c", "import sys; sys.exit(0)"])
    session = TradingSession(cfg, mode="paper", hours=1, train=False, clock=clock, sleep=clock.sleep, runner=runner, snapshot_minutes=30,
                             after_open_minutes=0, max_wait_minutes=60, review_after=False, learn_margin_minutes=8)
    summary = session.run()
    assert "skipped" not in summary and summary["learn"]["returncode"] == 0 and summary["learn"]["killed"] is False
    assert abs(seen["minutes"] - 22.0) < 0.5                                                # 30 min to the open minus the 8 min margin
    assert "reloaded" in summary["learn"]                                                  # no policy on disk here -> False, but the hook ran
    assert summary["learn"]["reloaded"] is False


def test_period_review_has_path_oracle_attribution_and_ic(cfg, monkeypatch):
    import sys

    from stockbot.data.loader import synthetic_universe
    from stockbot.feedback.direction import DirectionBoard
    from stockbot.feedback.experience import ExperienceStore
    from stockbot.feedback.review import Review

    frames = synthetic_universe(["AAA", "BBB", "CCC", "DDD", "EEE", "SPY"], n=1300, seed=5)
    train_mod = sys.modules["stockbot.agent.train"]
    monkeypatch.setattr(train_mod, "load_frames", lambda cfg, offline=False, refresh=False, synthetic=False, tickers=None:
                        {t: frames[t] for t in (tickers or frames)})
    store = ExperienceStore(cfg.path("feedback.experience_file"))
    board = DirectionBoard(cfg.path("feedback.direction_file"))
    days = [d for d in frames["AAA"].index if pd.Timestamp("2019-12-16") <= d <= pd.Timestamp("2019-12-20")]
    for k, d in enumerate(days):
        for t in ("AAA", "BBB", "CCC", "DDD", "EEE"):
            px = float(frames[t].loc[d, "close"])
            fills = [{"ticker": t, "side": "buy", "qty": 10, "price": px, "ts": f"{d.date()}T14:31:00+00:00"}] if k == 0 and t == "AAA" else []
            if k == 3 and t == "AAA":
                fills = [{"ticker": t, "side": "sell", "qty": 10, "price": px, "ts": f"{d.date()}T14:31:00+00:00"}]
            store.record(mode="paper", ticker=t, date=d.strftime("%Y-%m-%d"), obs=np.zeros(3), action=1.0, target_exposure=1.0,
                         weight=0.10 if t == "AAA" and k < 3 else 0.0, decision="BUY", price=px, equity=1e5, availability={}, fills=fills, fees=1.0)
            nxt = float(frames[t]["close"].shift(-1).loc[d] / px - 1.0) if k < len(days) - 1 else 0.0
            board.record(ticker=t, date=d.strftime("%Y-%m-%d"), price=px, votes={"seer": {"vote": float(np.sign(nxt)), "score": nxt}}, mode="paper")
    res = Review(cfg).review_period("week", date(2019, 12, 20))
    assert res is not None
    assert "path_oracle" in res["alternatives"] and res["path_oracle"]["switches"] >= 0
    assert res["alternatives"]["path_oracle"] <= res["alternatives"]["oracle"] + 1e-9        # fees make the path oracle no better than daily perfection
    a = res["attribution"]
    assert a["total"] >= 0 and set(a) >= {"missed", "wrong_side", "under_sized", "cost", "biggest"}
    assert res["ic"]["seer"]["ic"] > 0.9                                                 # the perfect input scores a perfect IC
    assert res["regime"]["label"] and "market context" in " ".join(res["lessons"])
    assert any("fee-aware best path" in line for line in res["lessons"])
    assert res["execution"]["fills"] == 2
    if "round_trips" in res:
        assert res["round_trips"]["n"] == 1
