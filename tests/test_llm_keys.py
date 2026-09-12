"""Key pools with rotation, model fallback in the OpenAI-compatible backend, and the top-N agent gate."""
from __future__ import annotations

import time
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest

from stockbot.agent.policy import PolicyBundle
from stockbot.execution.runner import TradingRunner
from stockbot.llm.base import LLMQuotaError
from stockbot.llm.keys import KeyPool, load_keys
from stockbot.llm.openai_backend import GroqBackend, OpenAIBackend
from stockbot.signals.base import SignalProvider
from stockbot.signals.layout import ObservationLayout
from stockbot.signals.registry import build_context, build_providers
from stockbot.signals.thirdparty.trading_agents_signal import TradingAgentsSignal


# ---------------------------------------------------------------------- key pools
def test_load_keys_and_rotation(monkeypatch):
    monkeypatch.setenv("ZZZ_API_KEY", "k1")
    monkeypatch.setenv("ZZZ_API_KEY_2", "k2")
    monkeypatch.setenv("ZZZ_API_KEYS", "k3, k1;k4\nk2")
    assert load_keys("ZZZ_API_KEY") == ["k1", "k2", "k3", "k4"]
    pool = KeyPool("ZZZ_API_KEY")
    assert pool.pick() == "k1" and pool.pick() == "k2" and pool.pick() == "k3" and pool.pick() == "k4" and pool.pick() == "k1"
    pool.rest("k2", 600)
    pool.disable("k3", "bad")
    assert set(pool.keys[i] for i in pool.usable()) == {"k1", "k4"}
    assert "resting" in pool.describe() and "rejected" in pool.describe()
    for k in ("k1", "k4"):
        pool.rest(k, 30)
    assert pool.pick() is None and 0 < pool.seconds_until_available() <= 30
    empty = KeyPool("NOPE_API_KEY")
    assert not empty.configured and empty.pick() is None and empty.describe() == "set NOPE_API_KEY"


# ---------------------------------------------------------------------- backend rotation / fallback
def _http_error(cls, status: int, msg: str):
    resp = httpx.Response(status, request=httpx.Request("POST", "https://x/v1/chat/completions"))
    return cls(msg, response=resp, body=None)


class FakeCompletions:
    def __init__(self, behaviour):
        self.behaviour = behaviour   # (key, model) -> exception or text
        self.calls: list[tuple[str, str]] = []

    def create(self, **kw):
        key = kw.pop("_key")
        model = kw["model"]
        self.calls.append((key, model))
        b = self.behaviour.get((key, model)) or self.behaviour.get((key, "*")) or self.behaviour.get(("*", model))
        if isinstance(b, Exception):
            raise b
        text = b or '{"ok": true}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeClient:
    def __init__(self, key, completions):
        self._key = key
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: completions.create(_key=key, **kw)))


def _backend(monkeypatch, behaviour, keys=("A", "B"), model="m1", fallbacks=("m2",)):
    pool = KeyPool("FAKE_API_KEY", keys=list(keys))
    b = OpenAIBackend({"api_key_env": "FAKE_API_KEY", "model": model, "fallback_models": list(fallbacks), "key_cooldown_seconds": 300,
                       "base_url": "https://fake/v1", "require_key": True}, keys=pool)
    fc = FakeCompletions(behaviour)
    monkeypatch.setattr(b, "_client_for", lambda key: FakeClient(key, fc))
    return b, fc, pool


def test_backend_rotates_keys_on_rate_limit(monkeypatch):
    import openai

    b, fc, pool = _backend(monkeypatch, {("A", "*"): _http_error(openai.RateLimitError, 429, "daily cap")})
    out = b.complete_json("sys", "user", {"type": "object"})
    assert out == {"ok": True}
    assert [k for k, _ in fc.calls] == ["A", "B"]                     # A hit the cap, B answered
    assert pool.usable() == [1] or set(pool.keys[i] for i in pool.usable()) == {"B"}
    assert pool.seconds_until_available() == 0.0
    # every key capped -> quota error for the router, with the shortest wait
    b2, fc2, pool2 = _backend(monkeypatch, {("*", "m1"): _http_error(openai.RateLimitError, 429, "cap"), ("*", "m2"): _http_error(openai.RateLimitError, 429, "cap")})
    with pytest.raises(LLMQuotaError) as ei:
        b2.complete_json("sys", "user", {})
    assert ei.value.retry_after is not None and 0 < ei.value.retry_after <= 300


def test_backend_falls_back_to_next_model(monkeypatch):
    import openai

    b, fc, pool = _backend(monkeypatch, {("*", "m1"): _http_error(openai.NotFoundError, 404, "model gone")})
    assert b.complete_json("sys", "user", {}) == {"ok": True}
    assert fc.calls == [("A", "m1"), ("A", "m2")]                     # model fallback keeps the same key
    assert b.models == ["m1", "m2"]
    # bad key is dropped, the other key answers
    b3, fc3, pool3 = _backend(monkeypatch, {("A", "*"): _http_error(openai.AuthenticationError, 401, "invalid key")})
    assert b3.complete_json("sys", "user", {}) == {"ok": True}
    assert 0 in pool3.bad and "invalid" in pool3.bad[0]


def test_backend_presets_and_status(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g1")
    monkeypatch.setenv("GROQ_API_KEYS", "g2,g3")
    g = GroqBackend({})
    ok, why = g.is_configured()
    assert ok and "3 keys" in why and g.model == "openai/gpt-oss-120b" and g.request_kwargs([], 10).get("extra_body") is None
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.delenv("GROQ_API_KEYS")
    ok, why = GroqBackend({}).is_configured()
    assert not ok and "GROQ_API_KEY" in why


# ---------------------------------------------------------------------- top-N agent gate
class ConstantModel:
    num_timesteps = 0

    def predict(self, obs, deterministic=True):
        return np.array([0.9], dtype=np.float32), None


class FakeAgent(SignalProvider):
    name = "trading_agents"
    feature_names = ["ta_decision", "ta_is_buy", "ta_is_sell"]
    live_only = True
    tier = "C"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.calls: list[str] = []

    def availability(self):
        return True, "fake"

    def compute_history(self, ticker, df):
        return None

    def compute_latest(self, ticker, df):
        self.calls.append(ticker)
        return np.array([1.0, 1.0, 0.0], dtype=np.float32)


def test_agents_run_on_top_n_consensus_only(cfg, frames):
    cfg.set_path("signals.agents.top_n", 1)
    cfg.set_path("signals.agents.budget_minutes", 20)
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend", "market_regime")]
    layout = ObservationLayout([(p.name, list(p.feature_names)) for p in providers] + [("trading_agents", FakeAgent.feature_names)])
    bundle = PolicyBundle(ConstantModel(), layout, {"algo": "ppo"})
    runner = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False)
    fake = FakeAgent(cfg.section("signals.trading_agents"), ctx)
    runner.providers = [fake if p.name == "trading_agents" else p for p in runner.providers]
    assert runner.has_agent_frameworks()
    picked = runner.prewarm_agents(refresh=False)
    assert len(picked) == 1 and fake.calls == picked                  # one ticker computed, the others masked
    runner.frames = frames
    vectors, _ = runner.latest_vectors()
    assert sum(v["trading_agents"] is not None for v in vectors.values()) == 1
    assert runner.select_agent_tickers(vectors, 2)[0] == picked[0]
    # an exhausted budget masks the block instead of blocking the cycle
    fake.calls.clear()
    vectors, _ = runner.latest_vectors(budget_minutes=1e-9)
    assert fake.calls == [] and all(v["trading_agents"] is None for v in vectors.values())
    # top_n 0 disables the frameworks entirely
    cfg.set_path("signals.agents.top_n", 0)
    assert not runner.has_agent_frameworks()


def test_agent_cache_key_is_the_bar_date():
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2026-09-10", "2026-09-11"]))
    assert TradingAgentsSignal.cache_day(df) == "2026-09-11"
