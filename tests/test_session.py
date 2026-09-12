"""Market clock, fee schedule, direction scorecard, the market-open guard and the trading session."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from stockbot.agent.policy import PolicyBundle
from stockbot.data.loader import drop_unfinished_bar
from stockbot.env.portfolio import Portfolio
from stockbot.execution.base import Broker, Fill, Order, Position
from stockbot.execution.fees import FeeSchedule
from stockbot.execution.market_hours import NY, MarketClock, builtin_status, last_completed_session
from stockbot.execution.paper import PaperBroker
from stockbot.execution.runner import TradingRunner
from stockbot.execution.session import TradingSession, session_slot
from stockbot.feedback.direction import DirectionBoard, consensus, votes_from_vector
from stockbot.signals.layout import ObservationLayout
from stockbot.signals.registry import build_context, build_layout, build_providers


def ny(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=NY)


class FakeClock(MarketClock):
    """Built-in calendar driven by a fake wall clock; ``sleep`` advances it."""

    def __init__(self, start: datetime):
        self.t = start
        super().__init__(source="builtin", now_fn=lambda: self.t)

    def sleep(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)


# ---------------------------------------------------------------------- calendar
def test_builtin_calendar():
    assert builtin_status(ny("2026-09-14 10:00")).is_open                       # Monday mid-morning
    st = builtin_status(ny("2026-09-12 10:00"))                                   # Saturday
    assert not st.is_open and st.next_open == ny("2026-09-14 09:30")
    assert builtin_status(ny("2026-09-14 09:00")).minutes_to_open == 30
    assert not builtin_status(ny("2026-11-26 10:00")).is_open                    # Thanksgiving
    half = builtin_status(ny("2026-11-27 12:59"))                                 # half day closes 13:00
    assert half.is_open and half.next_close == ny("2026-11-27 13:00")
    assert not builtin_status(ny("2026-11-27 13:01")).is_open
    assert builtin_status(ny("2026-09-14 16:30")).next_open == ny("2026-09-15 09:30")
    assert last_completed_session(ny("2026-09-14 10:00")) == ny("2026-09-11 00:00").date()
    assert last_completed_session(ny("2026-09-14 16:10")) == ny("2026-09-14 00:00").date()


def test_wait_for_open_and_session_slot():
    clock = FakeClock(ny("2026-09-14 09:20"))
    st = clock.wait_for_open(max_wait_minutes=60, poll_seconds=120, sleep=clock.sleep)
    assert st.is_open and clock.t >= ny("2026-09-14 09:30")
    far = FakeClock(ny("2026-09-12 09:00"))
    assert not far.wait_for_open(max_wait_minutes=60, sleep=far.sleep).is_open   # Saturday: does not wait 2 days
    assert session_slot(FakeClock(ny("2026-09-14 09:15")), 90)[0]
    assert session_slot(FakeClock(ny("2026-09-14 09:50")), 90, 30)[0]
    assert not session_slot(FakeClock(ny("2026-09-14 10:15")), 90, 30)[0]        # the other daylight-saving slot
    assert not session_slot(FakeClock(ny("2026-09-14 07:00")), 90)[0]
    assert not session_slot(FakeClock(ny("2026-11-26 09:15")), 90)[0]            # holiday


def test_drop_unfinished_bar(monkeypatch):
    import stockbot.data.loader as loader

    monkeypatch.setattr(loader, "_last_completed_session", lambda: ny("2026-09-11 00:00").date())
    idx = pd.to_datetime(["2026-09-10", "2026-09-11", "2026-09-14"])
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    out = drop_unfinished_bar(df)
    assert list(out.index.strftime("%Y-%m-%d")) == ["2026-09-10", "2026-09-11"]


# ---------------------------------------------------------------------- fees
def test_moomoo_fee_schedule():
    f = FeeSchedule.from_preset("moomoo")
    # 10 shares at $100: per-share fees are below the minimums -> $0.99 + $1.00
    assert abs(f.cost(10, 100.0, "buy") - 1.99) < 1e-9
    # 1000 shares at $50 -> 4.90 + 5.00 (+ SEC + TAF on the sell side)
    assert abs(f.cost(1000, 50.0, "buy") - 9.90) < 1e-9
    sell = f.cost(1000, 50.0, "sell")
    assert sell > 9.90 and abs(sell - (9.90 + 50_000 * 0.0000206 + 1000 * 0.000195)) < 1e-9
    assert abs(f.cost(0.5, 100.0, "buy") - 0.495) < 1e-9                        # fractional: 0.99% of $50
    assert abs(f.min_trade_usd(25) - 1.99 / 0.0025) < 1e-6                       # $796 for 25 bps
    assert FeeSchedule.from_preset("moomoo_us").cost(10, 100.0, "buy") == 0.0
    assert abs(FeeSchedule.from_preset("moomoo_intl").cost(10, 100.0, "buy") - 0.99) < 1e-9
    assert FeeSchedule.from_preset("none").cost(10, 100.0, "sell") == 0.0
    # the simulator charges a $100k position as if it were a 10% slice: minimums bite 10x harder
    assert abs(f.cost_scaled(100, 100.0, "buy", 0.10) - 1.99 / 0.10) < 1e-9


def test_portfolio_and_paper_broker_use_fees():
    fees = FeeSchedule.from_preset("moomoo")
    p = Portfolio(10_000, commission=0.0, slippage=0.0, fees=fees, fee_scale=1.0)
    p.rebalance(1.0, 100.0)                                                       # buys 100 shares
    assert abs(p.total_costs - 1.99) < 1e-9 and abs(p.cash - (-1.99)) < 1e-9
    legacy = Portfolio(10_000, commission=0.001, slippage=0.0)
    legacy.rebalance(1.0, 100.0)
    assert abs(legacy.total_costs - 10.0) < 1e-9
    b = PaperBroker("unused.json", lambda t: 100.0, initial_cash=10_000, commission=0.0, slippage=0.0, fees=fees)
    b.state_file = b.state_file.with_name("never_saved.json")
    fill = b.submit(Order("AAA", "buy", 10))
    assert abs(fill.cost - 1.99) < 1e-9 and abs(b.cash() - (10_000 - 1000 - 1.99)) < 1e-9


# ---------------------------------------------------------------------- direction board
def test_votes_and_scorecard(tmp_path):
    layout = ObservationLayout([("technical", ["ret_1", "ret_20"]), ("dl_forecast", ["dl_pred", "dl_confidence"]),
                                ("es_agent", ["es_buy", "es_sell", "es_action"])])
    vec = np.zeros(layout.signal_dim, dtype=np.float32)
    tech, dl, es = layout.block("technical"), layout.block("dl_forecast"), layout.block("es_agent")
    vec[tech.offset] = 1.0
    vec[tech.start + 1] = 0.03                     # momentum up
    vec[dl.offset] = 1.0
    vec[dl.start] = -0.5                           # forecaster says down
    vec[es.offset] = 0.0                           # es agent unavailable -> no vote
    votes = votes_from_vector(layout, vec, policy_delta=0.3, deadband=0.1)
    assert votes["momentum"]["vote"] == 1 and votes["dl_forecast"]["vote"] == -1 and votes["policy"]["vote"] == 1
    assert "es_agent" not in votes
    assert votes_from_vector(layout, vec, policy_delta=0.05, deadband=0.1)["policy"]["vote"] == 0   # inside the deadband
    assert consensus(votes) == -1.0                # momentum and policy are excluded from the consensus

    board = DirectionBoard(tmp_path / "d.jsonl")
    board.record(ticker="AAA", date="2026-09-14", price=100.0, votes=votes, mode="paper")
    assert board.settle("AAA", "daily", 101.0, "2026-09-14") is None            # same day: no daily outcome yet
    out = board.settle("AAA", "session", 102.0, "2026-09-14", decision_date="2026-09-14")
    assert out and abs(out["log_return"] - np.log(1.02)) < 1e-9
    assert board.settle("AAA", "session", 103.0, "2026-09-14", decision_date="2026-09-14") is None  # settled once
    assert board.settle("AAA", "daily", 99.0, "2026-09-15")
    sc = board.scorecard("session").set_index("model")
    assert sc.loc["momentum", "hit_rate"] == 1.0 and sc.loc["dl_forecast", "hit_rate"] == 0.0
    daily = board.scorecard("daily").set_index("model")
    assert daily.loc["dl_forecast", "hit_rate"] == 1.0 and daily.loc["momentum", "edge_bps"] < 0
    latest = board.latest()
    assert latest.index.name == "2026-09-14" and latest.loc["AAA", "consensus"] == -1.0
    s = board.summary()
    assert s["votes"] == 1 and s["outcomes"] == 2 and s["best_session"]["model"] == "momentum"


# ---------------------------------------------------------------------- runner guard + session
class ConstantModel:
    def __init__(self, action: float):
        self.action = action
        self.num_timesteps = 0

    def predict(self, obs, deterministic=True):
        return np.array([self.action], dtype=np.float32), None


class RecordingBroker(Broker):
    """A 'real' broker stand-in (not the paper simulator): remembers what was submitted."""

    name = "fake-live"

    def __init__(self, price: float = 100.0):
        self._price = price
        self.orders: list[Order] = []

    def equity(self):
        return 100_000.0

    def cash(self):
        return 100_000.0

    def positions(self):
        return {}

    def price(self, ticker):
        return self._price

    def submit(self, order):
        self.orders.append(order)
        return Fill(order.ticker, order.side, order.qty, self._price, 0.0)


def _bundle(cfg, frames):
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend", "market_regime")]
    layout = build_layout(providers)
    return PolicyBundle(ConstantModel(1.0), layout, {"algo": "ppo"})


def test_runner_refuses_orders_while_closed(cfg, frames):
    bundle = _bundle(cfg, frames)
    broker = RecordingBroker()
    closed = FakeClock(ny("2026-09-12 10:00"))
    runner = TradingRunner(cfg, mode="alpaca", bundle=bundle, frames_loader=lambda refresh: frames, broker=broker, with_llm=False, clock=closed)
    decisions = runner.cycle(dry_run=False, refresh=False)
    assert any(d.action == "BUY" for d in decisions) and broker.orders == []
    assert "market closed" in runner.last_cycle_note
    assert runner.store.summary()["decisions"] == 0                             # nothing recorded either
    # same thing while the exchange is open: orders go out and the moomoo fee floor is applied
    open_clock = FakeClock(ny("2026-09-14 10:00"))
    runner2 = TradingRunner(cfg, mode="alpaca", bundle=bundle, frames_loader=lambda refresh: frames, broker=RecordingBroker(), with_llm=False,
                            clock=open_clock)
    assert runner2.min_trade_usd >= 700
    runner2.cycle(dry_run=False, refresh=False)
    assert len(runner2.broker.orders) == 3 and runner2.last_cycle_note == ""
    # --allow-closed overrides the guard
    runner3 = TradingRunner(cfg, mode="alpaca", bundle=bundle, frames_loader=lambda refresh: frames, broker=RecordingBroker(), with_llm=False,
                            clock=closed, allow_closed=True)
    runner3.cycle(dry_run=False, refresh=False)
    assert len(runner3.broker.orders) == 3


def test_trading_session_end_to_end(cfg, frames, tmp_path):
    bundle = _bundle(cfg, frames)
    clock = FakeClock(ny("2026-09-14 09:20"))                                     # ten minutes before the open
    runner = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False, clock=clock)
    session = TradingSession(cfg, mode="paper", hours=0.5, train=False, clock=clock, sleep=clock.sleep, runner=runner,
                             snapshot_minutes=10, after_open_minutes=5, max_wait_minutes=60)
    summary = session.run()
    assert "skipped" not in summary
    assert summary["start"] >= "2026-09-14T09:30" and summary["orders"] == 3 and summary["snapshots"] == 4
    assert summary["session_votes_settled"] == 3 and summary["equity_end"] > 0
    log_dir = cfg.path("session.log_dir")
    assert (log_dir / "session_2026-09-14.json").exists()
    lines = [json.loads(line) for line in (log_dir / "session_2026-09-14.jsonl").read_text().splitlines()]
    assert lines[0]["type"] == "decisions" and sum(r["type"] == "snapshot" for r in lines) == 4
    board = DirectionBoard(cfg.path("feedback.direction_file"))
    assert board.summary()["outcomes"] == 3
    # a second run on the same day is refused unless forced
    again = TradingSession(cfg, mode="paper", hours=0.5, train=False, clock=clock, sleep=clock.sleep, runner=runner).run()
    assert "already ran" in again["skipped"]
    # the trainer command carries the time budget, the dataset reuse and the rolling split
    cmd = session.trainer_command(42.0)
    assert "--max-minutes" in cmd and "--reuse-dataset-days" in cmd and any(a.startswith("data.train_end=2025-09-01") for a in cmd)


def test_time_budget_stops_training(cfg, frames):
    from stockbot.agent.train import train
    from stockbot.env.dataset import MarketDataset

    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    ds = MarketDataset.build(frames, providers, build_layout(providers), ctx, fit=True, train_end="2018-12-31")
    path = train(cfg, total_timesteps=5_000_000, dataset=ds, n_envs=2, max_minutes=0.05)
    meta = json.loads((path.parent / "meta.json").read_text())
    assert meta["timesteps"] < 5_000_000 and meta["max_minutes"] == 0.05


def test_rolling_train_end():
    from datetime import date

    from stockbot.agent.train import rolling_train_end

    assert rolling_train_end(12, date(2026, 9, 14)) == "2025-09-01"
    assert rolling_train_end(6, date(2026, 3, 1)) == "2025-09-01"
    assert rolling_train_end(0, date(2026, 9, 14)) == "2026-09-01"
