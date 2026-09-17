"""The live cycle reuses a name's vectors while its last bar is unchanged, and the LLM trader block runs only where it matters."""
import numpy as np

import stockbot.agent  # noqa: F401
from stockbot.agent.policy import PolicyBundle
from stockbot.execution.runner import TradingRunner
from stockbot.signals.base import SignalProvider
from stockbot.signals.registry import build_context, build_layout, build_providers


class BuyEverything:
    num_timesteps = 0

    def predict(self, obs, deterministic=True):
        return np.array([1.0], dtype=np.float32), None


class FakeLLMTrader(SignalProvider):
    name = "llm_trader"
    feature_names = ["lt_dir"]
    live_only = True
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.calls = []

    def availability(self):
        return True, "fake"

    def compute_latest(self, ticker, df):
        self.calls.append(ticker)
        return np.array([0.5], dtype=np.float32)


def _runner(cfg, frames):
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    bundle = PolicyBundle(BuyEverything(), build_layout(providers), {"algo": "ppo"})
    r = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False)
    r.frames = r.frames_loader(False)
    return r


def test_vectors_are_reused_while_the_last_bar_is_unchanged(cfg, frames, monkeypatch):
    r = _runner(cfg, frames)
    tech = next(p for p in r.providers if p.name == "technical")
    calls = {"n": 0}
    orig = tech.safe_history

    def counting(t, df):
        calls["n"] += 1
        return orig(t, df)

    monkeypatch.setattr(tech, "safe_history", counting)
    v1, _ = r.latest_vectors()
    n1 = calls["n"]
    assert n1 == len(frames)
    v2, _ = r.latest_vectors()
    assert calls["n"] == n1                                                            # the same bars: nothing recomputed
    assert all(np.allclose(v1[t]["technical"], v2[t]["technical"]) for t in frames)
    r.frames = {t: df.iloc[:-1] for t, df in frames.items()}                           # a different last bar: computed again
    r.latest_vectors()
    assert calls["n"] == 2 * n1


def test_llm_trader_runs_only_on_held_and_top_consensus_names(cfg, frames):
    r = _runner(cfg, frames)
    fake = FakeLLMTrader(cfg, r.ctx)
    r.providers.append(fake)
    r.llm_trader_top_n, r.llm_trader_budget = 1, 5.0
    r.ctx.extra["positions"] = {"BBB": {"qty": 1}}
    vectors, reasons = r.latest_vectors()
    assert fake.calls[0] == "BBB" and 1 <= len(fake.calls) <= 2 and reasons["llm_trader"] == "fake"   # held first, then the top consensus name
    assert vectors["BBB"]["llm_trader"] is not None and sum(vectors[t]["llm_trader"] is None for t in frames) >= 1
    r.llm_trader_top_n = 0
    fake.calls.clear()
    r.latest_vectors()
    assert sorted(fake.calls) == sorted(frames)                                       # top_n 0: every name, every cycle (never cached)
    fake.calls.clear()
    r.latest_vectors()
    assert sorted(fake.calls) == sorted(frames)


def test_parallel_computation_matches_serial_and_flags_the_live_cycle(cfg, frames):
    r = _runner(cfg, frames)
    serial, _ = r.latest_vectors()
    assert "latest_only" not in r.ctx.extra                                          # set for the cycle, cleared after it
    r2 = _runner(cfg, frames)
    r2.compute_workers = 3
    parallel, _ = r2.latest_vectors()
    for t in frames:
        for block in ("technical", "trend"):
            assert np.allclose(serial[t][block], parallel[t][block])

