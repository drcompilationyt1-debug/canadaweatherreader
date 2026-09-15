"""Post-trade review (day from session snapshots, week from recorded decisions), the review schedule,
the reliability block, and the session's job deadline."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from stockbot.data.loader import synthetic_universe
from stockbot.feedback.direction import DirectionBoard
from stockbot.feedback.experience import ExperienceStore
from stockbot.feedback.review import Review, period_bounds


def _write_session(cfg, day: str, moves: dict[str, float], held: dict[str, float], votes: dict[str, dict]):
    log_dir = cfg.path("session.log_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    price0 = {t: 100.0 for t in moves}
    decisions = [{"ticker": t, "action": "BUY" if held[t] > 0 else "HOLD", "target_exposure": held[t], "current_exposure": 0.0,
                  "delta_exposure": held[t], "amount_usd": held[t] * 10_000, "shares": held[t] * 100, "price": 100.0, "note": ""} for t in moves]
    recs = [{"type": "decisions", "ts": f"{day}T09:35:00", "decisions": decisions, "votes": votes}]
    for k in (1, 2):
        recs.append({"type": "snapshot", "label": f"t+{k}", "ts": f"{day}T1{k}:00:00", "equity": 100_000.0, "cash": 50_000.0,
                     "n_positions": 2, "mean_move": 0.0, "consensus_hit_rate": None,
                     "tickers": {t: {"price": price0[t] * (1 + moves[t] * k / 2), "move": moves[t] * k / 2, "consensus": 0.0,
                                     "held": held[t] * 100} for t in moves}})
    (log_dir / f"session_{day}.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    (log_dir / f"session_{day}.json").write_text(json.dumps({"date": day, "equity_open": 100_000.0}), encoding="utf-8")


def test_day_review_scores_actual_against_alternatives(cfg):
    votes = {"AAA": {"chronos": {"vote": 1, "score": 0.3}, "trend": {"vote": -1, "score": -0.2}},
             "BBB": {"chronos": {"vote": -1, "score": -0.3}, "trend": {"vote": 1, "score": 0.1}},
             "CCC": {"chronos": {"vote": 1, "score": 0.2}, "trend": {"vote": 1, "score": 0.1}}}
    # AAA +2%, BBB -3%, CCC +1%; we held a full slice of AAA and BBB, nothing in CCC
    _write_session(cfg, "2026-09-14", {"AAA": 0.02, "BBB": -0.03, "CCC": 0.01}, {"AAA": 1.0, "BBB": 1.0, "CCC": 0.0}, votes)
    rev = Review(cfg)
    res = rev.review_day(date(2026, 9, 14))
    assert res is not None and res["period"] == "day"
    # actual: 10% slice each -> +0.2% - 0.3% = -0.1% before fees; oracle holds AAA and CCC: +0.3% before fees
    assert -0.0012 < res["actual_return"] < -0.0009
    assert 0.0027 < res["oracle_return"] < 0.0031 and res["regret"] > 0.0035
    alts = res["alternatives"]
    assert alts["hold"] == 0.0 and "follow:chronos" in alts and "follow:trend" in alts
    assert alts["follow:chronos"] > alts["follow:trend"]                       # chronos called AAA up / BBB down
    assert res["best_model"] == "chronos"
    assert res["per_name"][0]["ticker"] == "BBB" and "trend" not in res["per_name"][0]["voted_down"]
    assert any("mistake: BBB" in line and "chronos had voted down" in line for line in res["lessons"])
    assert any("missed: CCC" in line for line in res["lessons"])
    assert (cfg.path("feedback.review_dir") / "day_2026-09-14.json").exists()


def test_week_review_replays_recorded_weights(cfg, monkeypatch):
    frames = synthetic_universe(["AAA", "BBB", "SPY"], n=1300, seed=5)          # ends 2019-12-25
    import sys

    train_mod = sys.modules["stockbot.agent.train"]          # the package re-exports `train`, so import the module by name
    monkeypatch.setattr(train_mod, "load_frames", lambda cfg, offline=False, refresh=False, synthetic=False, tickers=None:
                        {t: frames[t] for t in (tickers or frames)})
    store = ExperienceStore(cfg.path("feedback.experience_file"))
    board = DirectionBoard(cfg.path("feedback.direction_file"))
    days = [d for d in frames["AAA"].index if pd.Timestamp("2019-12-16") <= d <= pd.Timestamp("2019-12-20")]
    for d in days:
        for t, w in (("AAA", 0.10), ("BBB", 0.05)):
            store.record(mode="paper", ticker=t, date=d.strftime("%Y-%m-%d"), obs=np.zeros(3), action=1.0, target_exposure=1.0,
                         weight=w, decision="HOLD", price=float(frames[t].loc[d, "close"]), equity=1e5, availability={})
            board.record(ticker=t, date=d.strftime("%Y-%m-%d"), price=float(frames[t].loc[d, "close"]),
                         votes={"chronos": {"vote": 1.0 if t == "AAA" else -1.0, "score": 0.1}}, mode="paper")
    rev = Review(cfg)
    res = rev.review_period("week", date(2019, 12, 20))
    assert res is not None and res["days"] == 5 and res["start"] == "2019-12-16"
    r = frames["AAA"]["close"].pct_change().loc[days[1]:days[-1]]
    expected = float((1 + 0.10 * r + 0.05 * frames["BBB"]["close"].pct_change().loc[days[1]:days[-1]]).prod() - 1)
    assert abs(res["actual_return"] - expected) < 0.002                         # replay matches the weights (small fee drag)
    for key in ("hold", "equal", "benchmark", "oracle", "period_oracle", "half", "double", "follow:chronos"):
        assert key in res["alternatives"]
    assert res["oracle_return"] >= res["actual_return"] - 1e-9
    assert "sharpe" in res["stats"] and "benchmark_return" in res["stats"]
    assert (cfg.path("feedback.review_dir") / "week_2019-12-20.json").exists()
    # scheduling: the week review is now done for that Friday; month / year still due
    assert rev.last_review_end("week") == date(2019, 12, 20)
    due = rev.due(date(2019, 12, 21))
    assert "week" not in due and "month" in due and "year" in due
    assert period_bounds("month", date(2026, 9, 14)) == (date(2026, 9, 1), date(2026, 9, 14))
    assert period_bounds("week", date(2026, 9, 17)) == (date(2026, 9, 14), date(2026, 9, 17))


def test_reliability_block_scores_each_input(cfg, frames):
    from stockbot.signals.registry import build_context, build_providers

    cfg.set_path("signals.reliability.window", 40)
    cfg.set_path("signals.reliability.min_obs", 5)
    ctx = build_context(cfg, with_llm=False, with_news=False)
    ctx.extra["frames"] = frames
    rel = next(p for p in build_providers(cfg, ctx) if p.name == "reliability")
    # a perfect 'trend' input (its slope_30 feature = the sign of the next 5-day move) and a random 'chronos' one
    from stockbot.signals.registry import PROVIDER_CLASSES

    trend_names = list(next(k for k in PROVIDER_CLASSES if k.name == "trend").feature_names)
    chronos_names = list(next(k for k in PROVIDER_CLASSES if k.name == "chronos").feature_names)
    per_block = {"trend": {}, "chronos": {}}
    rng = np.random.default_rng(0)
    for t, df in frames.items():
        logc = np.log(df["close"].to_numpy(float))
        fwd = np.zeros(len(df))
        fwd[:-5] = logc[5:] - logc[:-5]
        a = np.zeros((len(df), len(trend_names)), dtype=np.float32)
        a[:, trend_names.index("slope_30")] = np.sign(fwd)
        per_block["trend"][t] = a
        c = rng.normal(size=(len(df), len(chronos_names))).astype(np.float32)
        per_block["chronos"][t] = c
    ctx.extra["per_block"] = per_block
    out = rel.compute_history_all(frames)
    a = out["AAA"]
    assert a.shape == (len(frames["AAA"]), len(rel.feature_names))
    valid = ~np.isnan(a).any(axis=1)
    assert valid.sum() == 0                                                     # only 2 of the inputs have a record: masked (<3)
    # add a third input and the block switches on, with trend near +1 and chronos near 0
    per_block["candles"] = {t: per_block["chronos"][t][:, :1].repeat(25, axis=1) for t in frames}
    out = rel.compute_history_all(frames)
    a = out["AAA"]
    valid = ~np.isnan(a).any(axis=1)
    assert valid[-100:].all()
    i_trend, i_chr = rel.feature_names.index("rel_trend"), rel.feature_names.index("rel_chronos")
    assert a[valid, i_trend].min() > 0.9 and abs(a[valid, i_chr].mean()) < 0.25
    assert rel.cache.path("AAA").exists()
    # live: no per_block in the context -> the cached latest row
    ctx.extra.pop("per_block")
    live = rel.compute_history_all(frames)["AAA"]
    assert np.isnan(live[:-1]).all() and not np.isnan(live[-1]).any() and abs(live[-1, i_trend] - a[-1, i_trend]) < 1e-6


def test_session_respects_job_deadline(cfg, frames):
    from datetime import datetime

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
    clock = Clock(datetime(2026, 9, 14, 9, 31, tzinfo=NY))
    runner = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False, clock=clock)
    # 4-hour watch requested, but the job must end 60 minutes after start: the window is cut to 45 minutes
    session = TradingSession(cfg, mode="paper", hours=4, train=False, clock=clock, sleep=clock.sleep, runner=runner,
                             snapshot_minutes=15, after_open_minutes=0, deadline_minutes=60, end_margin_minutes=15, review_after=True)
    summary = session.run()
    assert "skipped" not in summary
    assert summary["snapshots"] == 4                                            # 45 min / 15 min + the closing snapshot
    ended = datetime.fromisoformat(summary["end"])
    assert ended <= datetime(2026, 9, 14, 10, 32, tzinfo=NY)
    assert "review" in summary and "lessons" in summary["review"]              # the day review ran after the window
    # the same cut from an absolute deadline (the workflow anchors it on the job's start)
    clock2 = Clock(datetime(2026, 9, 14, 9, 31, tzinfo=NY))
    runner2 = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False, clock=clock2)
    session2 = TradingSession(cfg, mode="paper", hours=4, train=False, clock=clock2, sleep=clock2.sleep, runner=runner2, snapshot_minutes=15,
                              after_open_minutes=0, deadline_minutes=0, deadline_at=datetime(2026, 9, 14, 10, 31, tzinfo=NY).isoformat(),
                              end_margin_minutes=15, review_after=False)
    summary2 = session2.run(force=True)
    assert summary2["snapshots"] == 4 and datetime.fromisoformat(summary2["end"]) <= datetime(2026, 9, 14, 10, 32, tzinfo=NY)
