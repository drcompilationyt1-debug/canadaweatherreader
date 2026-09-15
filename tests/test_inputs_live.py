"""Keeping the third-party inputs live: qlib carries its last score for a few bars, the Gemini probe picks an answering
model, and the agent loop runs ticker by ticker and drops a framework after repeated failures."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from stockbot.agent.policy import PolicyBundle
from stockbot.execution.runner import TradingRunner
from stockbot.signals.registry import build_context, build_layout, build_providers


def test_qlib_carries_its_last_score_for_a_few_bars(cfg, frames, tmp_path):
    from stockbot.signals.thirdparty.qlib_signal import QlibSignal

    df = frames["AAA"]
    for gap, expect_live in ((3, True), (8, False)):
        pred = pd.DataFrame({"datetime": df.index[: len(df) - gap], "instrument": "AAA", "score": np.linspace(-0.02, 0.02, len(df) - gap)})
        f = tmp_path / f"pred_{gap}.parquet"
        pred.to_parquet(f)
        cfg.set_path("signals.qlib.predictions", f.as_posix())
        cfg.set_path("signals.qlib.max_age_bars", 5)
        ctx = build_context(cfg, with_llm=False, with_news=False)
        q = next(p for p in build_providers(cfg, ctx) if p.name == "qlib")
        assert isinstance(q, QlibSignal)
        a = q.compute_history("AAA", df)
        assert (not np.isnan(a[-1]).any()) is expect_live                     # 3 bars stale: live; 8 bars stale: masked
        assert not np.isnan(a[-gap - 1]).any()


def test_gemini_probe_picks_an_answering_model(monkeypatch):
    from stockbot.llm import probe

    calls = []

    def fake_post(url, params=None, json=None, timeout=None):
        calls.append(url)
        code = 503 if "dead-model" in url else 200
        return SimpleNamespace(status_code=code, text="high demand" if code == 503 else "ok")

    monkeypatch.setattr("requests.post", fake_post)
    probe._CACHE.clear()
    assert probe.first_live_gemini(["dead-model", "live-model"], "key-123456789") == "live-model"
    assert len(calls) == 2
    assert probe.first_live_gemini(["dead-model", "live-model"], "key-123456789") == "live-model" and len(calls) == 2   # cached
    assert probe.first_live_gemini(["dead-model"], "key-123456789") is None
    assert probe.first_live_gemini(["x", "y"], None) == "x"                     # no key: nothing to probe, keep the configured model


class Hold:
    num_timesteps = 0

    def predict(self, obs, deterministic=True):
        return np.array([0.0], dtype=np.float32), None


def test_agent_loop_runs_ticker_by_ticker_and_drops_a_failing_framework(cfg, frames):
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    bundle = PolicyBundle(Hold(), build_layout(providers), {"algo": "ppo"})
    r = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False)

    class Fake:
        live_only, tier, enabled = True, "C", True

        def __init__(self, name, ok):
            self.name, self.ok, self.calls = name, ok, []

        def availability(self):
            return True, "fake"

        def safe_latest(self, t, df):
            self.calls.append(t)
            return np.array([1.0, 1.0, 0.0], dtype=np.float32) if self.ok else None

    good, bad = Fake("good", True), Fake("bad", False)
    r.agent_providers = lambda: [good, bad]
    r.agent_settings = lambda: (3, 5.0)
    cfg.set_path("signals.agents.max_failures", 2)
    r.frames = frames
    vectors, reasons = r.latest_vectors()
    assert good.calls == r.agent_tickers and len(good.calls) == 3                 # the healthy framework ran on every selected ticker
    assert len(bad.calls) == 2                                                   # the failing one was dropped after two failures
    assert all(vectors[t]["good"] is not None for t in r.agent_tickers) and all(vectors[t]["bad"] is None for t in frames)
