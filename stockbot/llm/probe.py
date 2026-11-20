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


RETRIES = 3          # a 503 "high demand" comes and goes within seconds (Nov 20, 2026: the same model answered 4 probes of 6),
RETRY_WAIT = 4.0     # so an overloaded or silent model gets two more tries before the next one; a retired one (404) gets none
RETRY_BUDGET = 45.0  # seconds a single model may cost


def gemini_alive(model: str, key: str, timeout: float = 20.0, cache: bool = True) -> tuple[bool, str]:
    k = (model, key[:10])
    now = time.time()
    if cache and k in _CACHE and now - _CACHE[k][0] < TTL:
        return _CACHE[k][1], _CACHE[k][2]
    ok, why = False, "not tried"
    for attempt in range(max(1, RETRIES)):
        transient = True
        try:
            import requests

            r = requests.post(GEMINI_URL.format(model=model), params={"key": key},
                              json={"contents": [{"parts": [{"text": "Reply with OK."}]}], "generationConfig": {"maxOutputTokens": 5}},
                              timeout=timeout)
            ok, why = r.status_code == 200, ("ok" if r.status_code == 200 else f"{r.status_code} {r.text[:80]}")
            transient = r.status_code in (500, 502, 503, 504)
        except Exception as e:  # noqa: BLE001
            ok, why = False, str(e)[:100]
        if ok or not transient or attempt + 1 >= RETRIES or time.time() - now + RETRY_WAIT > RETRY_BUDGET:
            break
        time.sleep(RETRY_WAIT)
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
