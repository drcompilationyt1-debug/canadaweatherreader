"""Multi-market universe: market detection, per-market fees, sleeve routing and the trading cycle."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockbot.agent.policy import PolicyBundle
from stockbot.data.loader import synthetic_universe
from stockbot.env.trading_env import TradingEnv
from stockbot.env.dataset import MarketDataset
from stockbot.execution.fees import FeeBook, FeeSchedule
from stockbot.execution.markets import currency_of, group_by_market, market_of
from stockbot.execution.paper import PaperBroker
from stockbot.execution.routed import RoutedBroker
from stockbot.execution.runner import TradingRunner
from stockbot.signals.registry import build_context, build_layout, build_providers


def test_market_detection_and_fee_book():
    assert market_of("RY.TO") == "ca" and market_of("BABA") == "us" and market_of("XYZ.V") == "ca" and market_of("brk-b") == "us"
    assert currency_of("RY.TO") == "CAD" and currency_of("AAPL") == "USD"
    assert group_by_market(["AAPL", "RY.TO", "JD", "TD.TO"]) == {"us": ["AAPL", "JD"], "ca": ["RY.TO", "TD.TO"]}
    ca = FeeSchedule.from_preset("moomoo_ca")
    assert abs(ca.cost(100, 50.0, "buy") - (0.49 + 1.00)) < 1e-9          # minimums bite: 100 sh x 0.0049 = 0.49
    assert abs(ca.cost(1000, 50.0, "sell") - (4.90 + 10.00)) < 1e-9         # no SEC / TAF on Canadian sells
    book = FeeBook.from_names("moomoo", {"ca": "moomoo_ca"})
    assert book.for_ticker("RY.TO").preset == "moomoo_ca" and book.for_ticker("BABA").preset == "moomoo"
    assert book.names() == {"default": "moomoo", "ca": "moomoo_ca"}


def test_fee_book_from_config(cfg):
    book = FeeBook.from_config(cfg)
    assert book.default.preset == "moomoo" and book.for_market("ca").preset == "moomoo_ca"
    cfg.set_path("fees.by_market", {"ca": "bps"})
    assert FeeBook.from_config(cfg).for_market("ca") is None


def test_env_charges_each_market_its_own_fees(cfg):
    frames = synthetic_universe(["AAA", "BBB.TO"], n=800, seed=3)
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    ds = MarketDataset.build(frames, providers, build_layout(providers), ctx, fit=True, train_end="2018-06-30")
    env_cfg = {"fees": "moomoo", "fees_by_market": {"ca": "moomoo_ca"}, "fee_scale": 0.1, "episode_length": 20, "vol_target": 0}
    env = TradingEnv(ds, env_cfg, tickers=["BBB.TO"], seed=1, eval_mode=True)
    env.reset(seed=1)
    assert env.portfolio.fees.preset == "moomoo_ca"
    env2 = TradingEnv(ds, env_cfg, tickers=["AAA"], seed=1, eval_mode=True)
    env2.reset(seed=1)
    assert env2.portfolio.fees.preset == "moomoo"


class ConstantModel:
    num_timesteps = 0

    def predict(self, obs, deterministic=True):
        return np.array([1.0], dtype=np.float32), None


def test_routed_cycle_uses_sleeve_equity_and_fees(cfg, tmp_path):
    tickers = ["AAA", "BBB.TO", "CCC"]
    frames = synthetic_universe(tickers, n=1300, seed=7)
    cfg.set_path("universe", tickers)
    cfg.set_path("execution.sleeves.ca.state_file", (tmp_path / "paper" / "state_ca.json").as_posix())
    cfg.set_path("execution.sleeves.ca.initial_cash", 50_000)
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend", "market_regime")]
    bundle = PolicyBundle(ConstantModel(), build_layout(providers), {"algo": "ppo"})
    runner = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False)
    assert isinstance(runner.broker, RoutedBroker) and set(runner.broker.sleeves) == {"us", "ca"}
    ca, us = runner.broker.sleeves["ca"], runner.broker.sleeves["us"]
    assert isinstance(ca, PaperBroker) and ca.fees.preset == "moomoo_ca" and us.fees.preset == "moomoo"
    # one book: the simulated Canadian sleeve is funded from the same cash (its 50k seed is not counted), so every name is sized
    # against the whole book
    assert abs(runner.equity_for("BBB.TO") - 100_000) < 1e-6 and abs(runner.equity_for("AAA") - 100_000) < 1e-6
    assert abs(runner.broker.equity() - 100_000) < 1e-6 and runner.broker.summary()["sleeves"]["ca"]["seed"] == 50_000
    assert runner.min_trade_for("BBB.TO") < runner.min_trade_for("AAA")       # C$1.49 vs US$1.99 minimums
    decisions = runner.cycle(dry_run=False, refresh=False)
    assert {d.ticker for d in decisions} == set(tickers) and all(d.action == "BUY" for d in decisions)
    # the Canadian name was filled in the Canadian sleeve, sized against the book, and paid moomoo Canada fees
    pos = ca.position("BBB.TO")
    assert pos.shares > 0 and us.position("BBB.TO").shares == 0
    px = frames["BBB.TO"]["close"].iloc[-1]
    assert abs(pos.shares * px - 0.10 * 100_000) / 10_000 < 0.05
    assert runner.broker.cash() < 100_000 - 0.19 * 100_000                   # both purchases came out of the one book's cash
    assert abs(us.position("AAA").shares * frames["AAA"]["close"].iloc[-1] - 0.10 * 100_000) / 10_000 < 0.05
    assert ca.fills[-1]["cost"] >= 1.49 and us.fills[-1]["cost"] >= 1.99
    summary = runner.broker.summary()
    assert set(summary["sleeves"]) == {"us", "ca"} and summary["sleeves"]["ca"]["currency"] == "CAD"
    assert (tmp_path / "paper" / "state_ca.json").exists()
    # a plain single-market universe still gets a single broker
    cfg.set_path("universe", ["AAA", "CCC"])
    single = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: {t: frames[t] for t in ("AAA", "CCC")}, with_llm=False)
    assert isinstance(single.broker, PaperBroker)
