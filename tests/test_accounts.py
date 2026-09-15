"""Two books, one policy: per-account configs, a small account's rules (whole shares, at most N names, weekly cadence),
a second runner that reuses the first one's signals, the session trading both, the budget-aware fee_drag feature,
the predict shim for older policies, hindsight over every account's records, and the moomoo ledger on Alpaca."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stockbot.agent.policy import PolicyBundle
from stockbot.config import account_config, account_names
from stockbot.execution.runner import TradingRunner
from stockbot.signals.registry import build_context, build_layout, build_providers


class BuyEverything:
    num_timesteps = 0

    def predict(self, obs, deterministic=True):
        return np.array([1.0], dtype=np.float32), None


def _tiny(cfg, tmp_path, **execution):
    d = tmp_path / "tiny"
    ex = {"mode": "paper", "state_file": (d / "state.json").as_posix(), "max_names": 2, "max_names_hysteresis": 1, "cadence": "weekly",
          "whole_shares": True, "max_position": 0.3, "max_gross_exposure": 1.0, "min_trade_usd": 50, "cash_reserve": 0.1}
    ex.update(execution)
    cfg.set_path("accounts.tiny", {"enabled": True, "execution": ex, "env": {"initial_cash": 3000},
                                   "feedback": {"experience_file": (d / "trades.jsonl").as_posix(), "direction_file": (d / "direction.jsonl").as_posix(),
                                                "review_dir": (d / "reviews").as_posix()},
                                   "session": {"log_dir": (d / "sessions").as_posix(), "intraday_exit": {"same_day": False}},
                                   "report": {"out": (d / "dashboard.html").as_posix()}})
    return account_config(cfg, "tiny")


def _bundle(cfg):
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend", "market_regime")]
    return PolicyBundle(BuyEverything(), build_layout(providers), {"algo": "ppo"})


def test_account_config_merges_overrides_without_touching_the_base(cfg, tmp_path):
    cfg_t = _tiny(cfg, tmp_path)
    assert cfg_t["account"] == "tiny" and cfg_t.get_path("execution.max_names") == 2 and cfg_t.get_path("execution.cadence") == "weekly"
    assert cfg_t.get_path("env.initial_cash") == 3000 and cfg_t.get_path("env.vol_target") == 0            # the rest is inherited
    assert cfg.get_path("execution.max_names", 0) == 0 and "account" not in cfg
    assert "tiny" in account_names(cfg) and "small" not in account_names(cfg)                              # disabled in the test fixtures
    assert account_config(cfg, None) is cfg and account_config(cfg, "main") is cfg
    with pytest.raises(KeyError):
        account_config(cfg, "nope")


def test_small_account_rules_whole_shares_max_names_weekly(cfg, frames, tmp_path):
    cfg_t = _tiny(cfg, tmp_path)
    bundle = _bundle(cfg_t)
    r = TradingRunner(cfg_t, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False)
    assert r.account == "tiny" and r.whole_shares and r.max_names == 2 and r.cadence == "weekly"
    decisions = r.cycle(dry_run=False, refresh=False)
    bought = [d for d in decisions if d.action == "BUY"]
    assert len(bought) == 2 and all(float(d.shares).is_integer() and d.shares >= 1 for d in bought)         # 2 names, whole shares
    assert len([d for d in decisions if d.action == "HOLD"]) == 1
    assert r.state.get("last_rebalance_week") == r._week_key(str(frames["AAA"].index[-1].date()))
    held = {t: p.shares for t, p in r.broker.positions().items()}
    assert len(held) == 2 and all(float(s).is_integer() for s in held.values())
    # the same week again: nothing is rebalanced, positions are kept
    decisions2 = r.cycle(dry_run=False, refresh=False)
    assert all(d.action == "HOLD" for d in decisions2) and "weekly cadence" in r.last_cycle_note
    assert {t: p.shares for t, p in r.broker.positions().items()} == held
    # the portfolio state carries the budget-aware feature: a $900 slice pays far more per dollar than a $10k one
    from stockbot.signals.layout import PORTFOLIO_FEATURES

    assert PORTFOLIO_FEATURES[-1] == "fee_drag"
    vec, _ = r.portfolio_state("AAA", 100.0, 3000.0)
    assert len(vec) == len(PORTFOLIO_FEATURES) and vec[-1] > 0.2
    big = TradingRunner(cfg, mode="paper", bundle=_bundle(cfg), frames_loader=lambda refresh: frames, with_llm=False)
    assert big.portfolio_state("AAA", 100.0, 100_000.0)[0][-1] < vec[-1]


def test_second_runner_reuses_the_first_ones_signals(cfg, frames, tmp_path, monkeypatch):
    cfg_t = _tiny(cfg, tmp_path, cadence="daily", max_names=0, whole_shares=False)
    r1 = TradingRunner(cfg, mode="paper", bundle=_bundle(cfg), frames_loader=lambda refresh: frames, with_llm=False)
    r1.cycle(dry_run=False, refresh=False)
    assert r1.last_vectors is not None
    r2 = TradingRunner(cfg_t, mode="paper", with_llm=False, shared=r1)
    assert r2.bundle is r1.bundle and r2.providers is r1.providers and r2.broker is not r1.broker and r2.store.path != r1.store.path

    def boom(*a, **k):
        raise AssertionError("the second account must not recompute the signals")

    monkeypatch.setattr(r2, "latest_vectors", boom)
    decisions = r2.cycle(dry_run=False, refresh=False)
    assert len(decisions) == 3 and r2.frames is r1.frames
    assert r2.broker.equity() > 0 and abs(r2.broker.equity() - 3000.0) < 100.0                              # its own $3,000 book
    assert (tmp_path / "tiny" / "trades.jsonl").exists() and (tmp_path / "tiny" / "state.json").exists()


def test_session_trades_both_accounts_from_one_set_of_signals(cfg, frames, tmp_path):
    from stockbot.execution.market_hours import NY, MarketClock
    from stockbot.execution.session import TradingSession

    class Clock(MarketClock):
        def __init__(self, start):
            self.t = start
            super().__init__(source="builtin", now_fn=lambda: self.t)

        def sleep(self, s):
            self.t += timedelta(seconds=s)

    _tiny(cfg, tmp_path, cadence="daily")
    clock = Clock(datetime(2026, 9, 14, 9, 31, tzinfo=NY))
    r1 = TradingRunner(cfg, mode="paper", bundle=_bundle(cfg), frames_loader=lambda refresh: frames, with_llm=False, clock=clock)
    session = TradingSession(cfg, mode="paper", hours=1, train=False, clock=clock, sleep=clock.sleep, runner=r1, snapshot_minutes=30,
                             after_open_minutes=0, review_after=True, learn_before_open=False)
    summary = session.run()
    assert "tiny" in session.extra and session.extra["tiny"].shared is r1
    acc = summary["accounts"]["tiny"]
    assert acc["orders"] == 2 and acc["equity_open"] == pytest.approx(3000.0) and "equity_end" in acc and "session_return" in acc
    assert "review" in acc and "lessons" in acc["review"]
    log_dir = tmp_path / "tiny" / "sessions"
    recs = [json.loads(line) for line in (log_dir / "session_2026-09-14.jsonl").read_text().splitlines() if line.strip()]
    assert recs[0]["type"] == "decisions" and len(recs[0]["decisions"]) == 3
    snaps = [r for r in recs if r["type"] == "snapshot"]
    assert len(snaps) == summary["snapshots"] and all(s["account"] == "tiny" for s in snaps)
    assert set(snaps[-1]["tickers"]) == {"AAA", "BBB", "CCC"} and sum(1 for v in snaps[-1]["tickers"].values() if v["held"] > 0) == 2
    assert (log_dir / "session_2026-09-14.json").exists() and (tmp_path / "tiny" / "dashboard.html").exists()
    assert summary["exits"] == 0


def test_training_episodes_vary_the_budget_and_show_it(cfg, frames):
    from stockbot.config import env_settings
    from stockbot.env.dataset import MarketDataset
    from stockbot.env.trading_env import TradingEnv

    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    layout = build_layout(providers)
    ds = MarketDataset.build(frames, providers, layout, ctx, fit=True, train_end="2018-12-31")
    env_cfg = {**env_settings(cfg), "fees": "moomoo", "fee_scale": 0.1, "cash_range": [5000, 200000]}
    env = TradingEnv(ds, env_cfg, seed=3)
    obs, _ = env.reset()
    assert obs.shape[0] == layout.obs_dim == layout.signal_dim + 7
    cash = {float(env.reset()[0][-1]): env.initial_equity for _ in range(25)}
    equities = list(cash.values())
    assert min(equities) < 30_000 < max(equities)                                                          # different budgets ...
    small = env.reset(options={"cash": 5000})[0][-1]
    big = env.reset(options={"cash": 200000})[0][-1]
    assert small > big > 0 and env.initial_equity == 200000                                                  # ... and the feature says which
    evl = TradingEnv(ds, env_cfg, seed=3, eval_mode=True)
    evl.reset()
    assert evl.initial_equity == float(env_cfg["initial_cash"])                                              # evaluation keeps the fixed budget


def test_policy_trained_before_fee_drag_still_predicts(cfg):
    ctx = build_context(cfg, with_llm=False, with_news=False)
    layout = build_layout([p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")])

    class Old:
        num_timesteps = 0
        observation_space = SimpleNamespace(shape=(layout.obs_dim - 1,))

        def predict(self, obs, deterministic=True):
            assert obs.shape[1] == layout.obs_dim - 1
            return np.array([0.5], dtype=np.float32), None

    b = PolicyBundle(Old(), layout, {"algo": "ppo"})
    assert b.model_obs_dim() == layout.obs_dim - 1
    assert b.predict(np.zeros(layout.obs_dim)) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        b.predict(np.zeros(layout.signal_dim - 1))


def test_hindsight_reads_every_accounts_records(cfg, frames, tmp_path):
    from stockbot.feedback.experience import ExperienceStore
    from stockbot.feedback.hindsight import experience_stores, load_all_records

    cfg_t = _tiny(cfg, tmp_path)
    for c, n in ((cfg, 2), (cfg_t, 3)):
        store = ExperienceStore(c.path("feedback.experience_file"))
        for k in range(n):
            store.record(mode="paper", ticker="AAA", date=f"2019-10-0{k + 1}", obs=np.zeros(4), action=0.0, target_exposure=0.5, weight=0.05,
                         decision="HOLD", price=100.0, equity=1e5, availability={})
    stores = experience_stores(cfg)
    assert len(stores) == 2 and stores[1].path == cfg_t.path("feedback.experience_file")
    recs = load_all_records(cfg)
    assert len(recs) == 5 and recs["_store"].nunique() == 2


def test_alpaca_ledger_and_keys(tmp_path, monkeypatch):
    from stockbot.execution.alpaca import AlpacaBroker, VirtualLedger, alpaca_keys
    from stockbot.execution.fees import FeeSchedule

    monkeypatch.setenv("ALPACA_10K_API_KEY", "k10")
    monkeypatch.setenv("ALPACA_10K_SECRET_KEY", "s10")
    assert alpaca_keys("ALPACA_10K") == ("k10", "s10") and alpaca_keys("ALPACA_10K_") == ("k10", "s10")
    b = AlpacaBroker.__new__(AlpacaBroker)
    b.fees = FeeSchedule.from_preset("moomoo")
    b.fractional = False
    b.ledger = VirtualLedger(tmp_path / "ledger.json")
    b.client = SimpleNamespace(get_account=lambda: SimpleNamespace(equity="100000", cash="60000"))
    b.ledger.add(1.99)
    b.ledger.add(1.99)
    b.save()
    assert b.equity() == pytest.approx(100000 - 3.98) and b.cash() == pytest.approx(60000 - 3.98)
    assert VirtualLedger(tmp_path / "ledger.json").fees == pytest.approx(3.98) and "whole shares" in b.constraints()
    b.fees = None
    assert b.equity() == pytest.approx(100000.0)                                                           # no fee model: Alpaca's own numbers
