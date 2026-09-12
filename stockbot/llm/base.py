"""Common LLM backend interface + error taxonomy used by the router for graceful fallback."""
from __future__ import annotations

import json
import re
from typing import Any


class LLMError(Exception):
    """Generic failure - the router tries the next backend."""


class LLMTransientError(LLMError):
    """Network / server error - the next backend is tried, this one is NOT put on cooldown."""


class LLMQuotaError(LLMError):
    """Rate limit or credits exhausted - the backend rests for ``retry_after`` seconds."""

    def __init__(self, msg: str = "quota exhausted", retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class LLMAuthError(LLMError):
    """Bad / missing credentials - the backend is disabled for the rest of the process."""


class LLMBackend:
    name: str = "base"

    def __init__(self, cfg: dict | None = None):
        self.cfg = dict(cfg or {})

    def is_configured(self) -> tuple[bool, str]:
        """(usable, reason).  Must be cheap and must not hit the network."""
        return False, "not implemented"

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int | None = None) -> dict:
        """Return a dict that follows ``schema``.  Raise one of the LLM*Error classes on failure."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        ok, why = self.is_configured()
        return {"backend": self.name, "model": self.cfg.get("model"), "configured": ok, "reason": why}


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict:
    """Parse JSON from a model reply, tolerating code fences and leading prose."""
    if text is None:
        raise LLMError("empty reply")
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = _FENCE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as e:
            raise LLMError(f"reply is not valid JSON: {e}") from e
    raise LLMError("reply contains no JSON object")


def schema_hint(schema: dict) -> str:
    """Plain-text description of a JSON schema for backends without native structured output."""
    return "Respond with ONLY a JSON object (no prose, no markdown) matching this JSON schema:\n" + json.dumps(schema)
