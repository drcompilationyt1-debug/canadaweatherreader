import time

import pytest

from stockbot.llm.base import LLMAuthError, LLMBackend, LLMError, LLMQuotaError, LLMTransientError, extract_json
from stockbot.llm.router import LLMRouter


class Fake(LLMBackend):
    def __init__(self, name, behaviour):
        super().__init__({})
        self.name = name
        self.behaviour = behaviour
        self.calls = 0

    def is_configured(self):
        return True, "fake"

    def complete_json(self, system, user, schema, max_tokens=None):
        self.calls += 1
        b = self.behaviour
        if isinstance(b, Exception):
            raise b
        return b


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure:\n```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('prefix {"a": 3} suffix') == {"a": 3}
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_router_falls_back_on_quota_then_recovers(tmp_path):
    a = Fake("a", LLMQuotaError("out", retry_after=0.2))
    b = Fake("b", {"ok": True})
    r = LLMRouter([a, b], cooldown_seconds=100, state_file=tmp_path / "s.json")
    assert r.complete_json("s", "u", {}) == {"ok": True}
    assert a.calls == 1 and b.calls == 1
    assert r.complete_json("s", "u", {}) == {"ok": True}
    assert a.calls == 1  # on cooldown, not retried
    time.sleep(0.25)
    a.behaviour = {"ok": "a"}
    assert r.complete_json("s", "u", {}) == {"ok": "a"}


def test_router_disables_on_auth_and_returns_none_when_all_fail():
    a = Fake("a", LLMAuthError("bad key"))
    b = Fake("b", LLMTransientError("down"))
    r = LLMRouter([a, b], cooldown_seconds=10)
    assert r.complete_json("s", "u", {}) is None
    assert "a" in r.disabled
    assert [x.name for x in r.usable()] == ["b"]  # transient errors do not disable a backend
    b.behaviour = {"x": 1}
    assert r.complete_json("s", "u", {}) == {"x": 1}
    assert a.calls == 1


def test_router_from_config_lists_backends(cfg):
    r = LLMRouter.from_config(cfg)
    names = [b.name for b in r.backends]
    assert names == list(cfg.get_path("llm.order"))
    assert all("configured" in row for row in r.status())
