"""Is a Gemini model answering right now?  One tiny request before an agent framework spends ten minutes of retries on
a model that returns 503 "high demand" (what happened on 2026-09-15: every pre-open agent run failed and the other
framework never got its turn).  Results are cached for a few minutes per (model, key)."""
from __future__ import annotations

import time

from ..logging_utils import get_logger

log = get_logger(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_CACHE: dict[tuple[str, str], tuple[float, bool, str]] = {}
TTL = 600.0


def gemini_alive(model: str, key: str, timeout: float = 20.0, cache: bool = True) -> tuple[bool, str]:
    k = (model, key[:10])
    now = time.time()
    if cache and k in _CACHE and now - _CACHE[k][0] < TTL:
        return _CACHE[k][1], _CACHE[k][2]
    try:
        import requests

        r = requests.post(GEMINI_URL.format(model=model), params={"key": key},
                          json={"contents": [{"parts": [{"text": "Reply with OK."}]}], "generationConfig": {"maxOutputTokens": 5}},
                          timeout=timeout)
        ok, why = r.status_code == 200, ("ok" if r.status_code == 200 else f"{r.status_code} {r.text[:80]}")
    except Exception as e:  # noqa: BLE001
        ok, why = False, str(e)[:100]
    _CACHE[k] = (now, ok, why)
    return ok, why


def first_live_gemini(candidates: list[str], key: str | None, timeout: float = 20.0) -> str | None:
    """The first model of ``candidates`` that answers (None when none does or without a key)."""
    if not key:
        return candidates[0] if candidates else None
    for m in [c for c in candidates if c]:
        ok, why = gemini_alive(m, key, timeout=timeout)
        if ok:
            return m
        log.warning("gemini %s not answering (%s) - trying the next model", m, why)
    return None
