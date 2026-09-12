import json

import numpy as np
import pandas as pd

from stockbot.agent.decide import decide
from stockbot.agent.evaluate import evaluate, metrics, run_window
from stockbot.agent.policy import PolicyBundle
from stockbot.agent.train import train
from stockbot.env.dataset import MarketDataset
from stockbot.execution.allocator import allocate
from stockbot.execution.base import Order
from stockbot.execution.paper import PaperBroker
from stockbot.execution.runner import TradingRunner
from stockbot.feedback.experience import ExperienceStore
from stockbot.signals.registry import build_context, build_layout, build_providers


def test_decide_mapping():
    d = decide("X", 0.8, 0.0, 10_000, 50.0)
    assert d.action == "BUY" and abs(d.amount_usd - 8000) < 1e-6 and abs(d.shares - 160) < 1e-6
    assert decide("X", 0.2, 0.8, 10_000, 50.0).action == "SELL"
    assert decide("X", -0.5, 0.0, 10_000, 50.0).action == "SHORT"
    assert decide("X", 0.0, -0.5, 10_000, 50.0).action == "COVER"
    assert decide("X", 0.5, -0.5, 10_000, 50.0).action == "BUY"
    assert decide("X", 0.52, 0.5, 10_000, 50.0).action == "HOLD"
    assert decide("X", -0.5, 0.0, 10_000, 50.0, allow_short=False).action == "HOLD"


def test_allocate_caps_gross_exposure():
    w = allocate({"A": 1.0, "B": 1.0, "C": -1.0, "D": 1.0, "E": 1.0, "F": 1.0}, max_position=0.25, max_gross_exposure=1.0)
    assert abs(sum(abs(x) for x in w.values()) - 1.0) < 1e-9
    assert w["C"] < 0
    w2 = allocate({"A": -1.0}, max_position=0.25, allow_short=False)
    assert w2["A"] == 0.0


def test_paper_broker_roundtrip(tmp_path):
    prices = {"AAA": 100.0}
    b = PaperBroker(tmp_path / "s.json", lambda t: prices[t], initial_cash=10_000, commission=0.001, slippage=0.0)
    fill = b.submit(Order("AAA", "buy", 10))
    assert fill.qty == 10 and abs(b.cash() - (10_000 - 1000 - 1.0)) < 1e-9
    prices["AAA"] = 110.0
    assert abs(b.equity() - (8999.0 + 1100.0)) < 1e-9
    b.submit(Order("AAA", "sell", 15))  # flips to short 5
    assert b.position("AAA").shares == -5
    b.save()
    b2 = PaperBroker(tmp_path / "s.json", lambda t: prices[t])
    assert b2.position("AAA").shares == -5 and abs(b2.cash() - b.cash()) < 1e-9
    nb = PaperBroker(tmp_path / "n.json", lambda t: prices[t], allow_short=False)
    assert nb.submit(Order("AAA", "sell", 3)) is None


def test_experience_store(tmp_path):
    st = ExperienceStore(tmp_path / "e.jsonl")
    st.record(mode="paper", ticker="AAA", date="2024-01-02", obs=np.zeros(3), action=0.5, target_exposure=0.5, weight=0.125,
              decision="BUY", price=100.0, equity=1e5, availability={"technical": True})
    assert st.settle("AAA", "2024-01-02", 100.0) is None  # same day -> nothing to settle
    out = st.settle("AAA", "2024-01-03", 102.0)
    assert out and abs(out["pnl_log_return"] - np.log(1.02) * 0.125) < 1e-9
    assert st.settle("AAA", "2024-01-04", 103.0) is None  # already settled
    s = st.summary()
    assert s["decisions"] == 1 and s["outcomes"] == 1 and s["hit_rate"] == 1.0


def test_train_evaluate_and_paper_cycle(cfg, frames, tmp_path):
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "candles", "trend", "market_regime")]
    layout = build_layout(providers)
    ds = MarketDataset.build(frames, providers, layout, ctx, fit=True, train_end="2018-12-31")
    path = train(cfg, total_timesteps=256, dataset=ds, n_envs=2)
    assert path.exists()
    ckpt = cfg.path("train.checkpoint_dir")
    assert (ckpt / "layout.json").exists() and (ckpt / "meta.json").exists()
    bundle = PolicyBundle.load(ckpt, "latest")
    assert bundle.layout.signature() == layout.signature()
    a = bundle.predict(np.zeros(layout.obs_dim))
    assert -1.0 <= a <= 1.0

    _, test = ds.split("2018-12-31")
    summary, curves = evaluate(bundle.model, test, dict(cfg.section("env")), max_bars=120)
    assert len(summary) == 3 and {"sharpe", "max_drawdown", "excess_return"} <= set(summary.columns)
    res = run_window(bundle.model, test, "AAA", dict(cfg.section("env")), length=50)
    assert len(res["equity"]) == 51
    m = metrics(np.array([1, 1.1, 1.05, 1.2]), np.array([1, 1, 1, 1]))
    assert m["total_return"] > 0

    # one paper-trading cycle on the same synthetic frames (no network, no LLM)
    runner = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False)
    decisions = runner.cycle(dry_run=False, refresh=False)
    assert len(decisions) == 3
    summary = runner.broker.summary()
    assert summary["equity"] > 0
    store = ExperienceStore(cfg.path("feedback.experience_file"))
    assert store.summary()["decisions"] == 3
    # second cycle settles the first decisions (prices unchanged -> zero pnl) and keeps working
    decisions2 = runner.cycle(dry_run=True, refresh=False)
    assert len(decisions2) == 3
    state = json.loads(runner.state_file.read_text())
    assert state["cycles"] >= 1
