"""Reflection per unit on the paper accounts: the decision ledger (verdict + better move per decision), intraday exits
counted in the day's result, and the rank rule's alternatives compared over the period."""
from __future__ import annotations

import json
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockbot.data.loader import synthetic_universe
from stockbot.feedback.direction import DirectionBoard
from stockbot.feedback.experience import ExperienceStore
from stockbot.feedback.hindsight import better_move, unit_paths, unit_verdicts, verdict_of
from stockbot.feedback.review import Review


def test_verdicts_and_better_moves():
    assert verdict_of(0.0, 1.0) == "missed" and verdict_of(1.0, 0.0) == "wrong side" and verdict_of(0.5, 1.0) == "under-sized"
    assert verdict_of(1.0, 0.5) == "over-sized" and verdict_of(0.5, 0.6) == "right"
    days = ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-12"]
    assert better_move({"path": [0, 0, 1, 1, 1], "k": 0, "days": days}) == "stay out until 2026-09-10"
    assert better_move({"path": [1, 1, 0, 0, 0], "k": 0, "days": days}) == "hold 100% of a slot, sell on 2026-09-10"
    assert better_move({"path": [0, 0, 0, 0, 0], "k": 1, "days": days}) == "stay out for the whole unit"
    rows = [{"ticker": "A", "date": days[0], "action": "BUY", "k": 0, "days": days, "level_best": 0.0, "level_ours": 1.0, "r_day": -0.02,
             "r_to_end": -0.05, "path": [0, 0, 0, 0, 0]},
            {"ticker": "B", "date": days[0], "action": "HOLD", "k": 0, "days": days, "level_best": 1.0, "level_ours": 1.0, "r_day": 0.01,
             "r_to_end": 0.03, "path": [1, 1, 1, 1, 1]}]
    ledger, summ = unit_verdicts(rows, 0.1)
    assert summ["n"] == 2 and summ["counts"]["wrong side"] == 1 and summ["counts"]["right"] == 1 and summ["hit_rate"] == 0.5
    assert ledger[0]["ticker"] == "A" and ledger[0]["regret"] == pytest.approx(0.002) and ledger[0]["better"].startswith("stay out")
    assert summ["regret"] == pytest.approx(0.002)                                  # a full slot held through -2% when the best was out: 0.2% of equity
    rows[0]["r_day"], rows[0]["level_best"], rows[0]["level_ours"] = 0.02, 1.0, 0.0
    _, summ2 = unit_verdicts(rows, 0.1)
    assert summ2["counts"]["missed"] == 1 and summ2["regret"] == pytest.approx(0.002)   # a missed +2% day at a 10% slot: the same 0.2%


def _week_records(cfg, frames, days):
    store = ExperienceStore(cfg.path("feedback.experience_file"))
    board = DirectionBoard(cfg.path("feedback.direction_file"))
    for k, d in enumerate(days):
        for t in ("AAA", "BBB", "CCC"):
            px = float(frames[t].loc[d, "close"])
            store.record(mode="paper", ticker=t, date=d.strftime("%Y-%m-%d"), obs=np.zeros(3), action=1.0, target_exposure=1.0 if t == "AAA" else 0.0,
                         weight=0.10 if t == "AAA" else 0.0, decision="BUY" if t == "AAA" and k == 0 else "HOLD", price=px, equity=1e5,
                         availability={}, chosen=(t == "AAA"))
            board.record(ticker=t, date=d.strftime("%Y-%m-%d"), price=px, votes={"seer": {"vote": 1.0, "score": 0.1}}, mode="paper")
    return store


def test_week_review_lists_every_decision_and_the_rules_alternatives(cfg, frames, monkeypatch):
    from stockbot.env.dataset import MarketDataset
    from stockbot.signals.registry import build_context, build_layout, build_providers

    monkeypatch.setattr(sys.modules["stockbot.agent.train"], "load_frames",
                        lambda cfg, offline=False, refresh=False, synthetic=False, tickers=None: {t: frames[t] for t in (tickers or frames) if t in frames})
    days = [d for d in frames["AAA"].index if pd.Timestamp("2019-12-16") <= d <= pd.Timestamp("2019-12-20")]
    _week_records(cfg, frames, days)
    # the unit's rows: every decision next to the best path
    start, end, rows = unit_paths(cfg, "week", date(2019, 12, 20), ExperienceStore(cfg.path("feedback.experience_file")).load(),
                                  frames=frames, today=date(2019, 12, 31))
    assert (start, end) == (date(2019, 12, 16), date(2019, 12, 20)) and len(rows) == 15
    assert {r["chosen"] for r in rows} == {True, False} and all(len(r["path"]) == 5 for r in rows)
    # a cached dataset covering the unit -> the rule's alternatives are compared over the week
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    ds = MarketDataset.build(frames, providers, build_layout(providers), ctx, fit=True, train_end="2018-12-31")
    monkeypatch.setattr(sys.modules["stockbot.agent.train"], "cached_dataset", lambda cfg, max_age_days=3.0: ds)   # the package re-exports `train`
    cfg.set_path("execution.rank", {"enabled": True, "top_k": 2, "every_bars": 5, "hysteresis": 1, "inputs": {"technical.ret_20": 1.0}, "adaptive": False})
    res = Review(cfg).review_period("week", date(2019, 12, 20))
    assert res is not None
    d = res["decisions"]
    assert d["n"] == 15 and sum(d["counts"].values()) == 15 and len(d["ledger"]) == 15 and "hit_rate" in d
    assert all({"verdict", "better", "regret"} <= set(row) for row in d["ledger"])
    assert any("decisions this week" in line for line in res["lessons"])
    sw = res["strategy_what_if"]
    assert sw["rule"]["name"] == "top2/every5" and "best" in sw and len(sw["variants"]) >= 12
    assert any("strategy what-if this week" in line for line in res["lessons"])


def test_day_review_counts_intraday_exits(cfg):
    from tests.test_review import _write_session

    votes = {t: {"chronos": {"vote": 1, "score": 0.2}} for t in ("AAA", "BBB", "CCC")}
    _write_session(cfg, "2026-09-14", {"AAA": 0.02, "BBB": -0.03, "CCC": 0.01}, {"AAA": 1.0, "BBB": 1.0, "CCC": 0.0}, votes)
    log_dir = cfg.path("session.log_dir")
    f = log_dir / "session_2026-09-14.jsonl"
    recs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    # BBB was sold at 11:00 for 101 (+1%) before it fell 3%: the end snapshot shows it no longer held
    for r in recs:
        if r.get("type") == "snapshot":
            r["tickers"]["BBB"]["held"] = 0.0
    recs.insert(2, {"type": "exit", "ts": "2026-09-14T11:00:00", "ticker": "BBB", "qty": 100.0, "price": 101.0, "fees": 1.99, "prob": 0.7,
                    "ret_open": 0.01, "reason": "take profit"})
    f.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    res = Review(cfg).review_day(date(2026, 9, 14))
    assert res is not None and len(res["exits"]) == 1 and res["exits"][0]["ticker"] == "BBB"
    # actual = AAA full slice +2% and BBB's realised +1% (10% slices): +0.3% before fees
    assert 0.0027 < res["actual_return"] < 0.0031
    assert any("intraday exits: BBB" in line for line in res["lessons"])
