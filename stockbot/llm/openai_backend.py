"""OpenAI-compatible backends.

``OpenAIBackend`` talks to OpenAI or to any server that speaks the OpenAI chat API via
``base_url`` (LM Studio, vLLM, ...).  The presets below point it at providers with a free tier:

* ``openrouter`` - OpenRouter, free models carry a ``:free`` suffix (https://openrouter.ai/models?q=free)
* ``groq``       - Groq free tier (https://console.groq.com)
* ``gemini``     - Google AI Studio free tier, OpenAI-compatible endpoint (https://aistudio.google.com)

Every backend uses a ``KeyPool``: several keys of the same provider (``NAME``, ``NAME_2``, ``NAMES``)
are used round-robin and a key that hits its rate limit rests while the others go on.  Models are
tried in order (``model`` then ``fallback_models``): a model that is gone or over its own quota
is skipped for the next one.  Only when every key x model combination failed does the backend
raise, and the router moves on to the next provider.
"""
from __future__ import annotations

import time

from .base import LLMAuthError, LLMBackend, LLMError, LLMQuotaError, LLMTransientError, extract_json, schema_hint
from .keys import KeyPool, pool

DEFAULT_MODEL = "gpt-4o-mini"


def _retry_after(exc) -> float | None:
    try:
        hdr = exc.response.headers.get("retry-after")
        return float(hdr) if hdr else None
    except Exception:  # noqa: BLE001
        return None


class OpenAIBackend(LLMBackend):
    name = "openai"
    DEFAULTS: dict = {}

    def __init__(self, cfg: dict | None = None, keys: KeyPool | None = None):
        merged = {**self.DEFAULTS, **{k: v for k, v in (cfg or {}).items() if v is not None}}
        super().__init__(merged)
        self.model = self.cfg.get("model") or DEFAULT_MODEL
        self.base_url = self.cfg.get("base_url") or None
        self.api_key_env = self.cfg.get("api_key_env") or "OPENAI_API_KEY"
        self.max_tokens = int(self.cfg.get("max_tokens", 2048))
        self.fallback_models = [m for m in (self.cfg.get("fallback_models") or []) if m]
        # a custom base_url usually means a local server that needs no key
        self.require_key = bool(self.cfg.get("require_key", not self.base_url))
        self.headers = dict(self.cfg.get("headers") or {})
        self.key_cooldown = float(self.cfg.get("key_cooldown_seconds") or self.cfg.get("cooldown_seconds") or 900)
        self.keys = keys or pool(self.api_key_env)
        self._clients: dict[str, object] = {}

    # ------------------------------------------------------------------ status
    def is_configured(self) -> tuple[bool, str]:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "pip install openai"
        if self.require_key and not self.keys.configured:
            return False, f"set {self.api_key_env}"
        where = f" @ {self.base_url}" if self.base_url else ""
        keys = f"; {self.keys.describe()}" if self.require_key else ""
        return True, f"model={self.model}{where}{keys}"

    @property
    def models(self) -> list[str]:
        return [self.model] + [m for m in self.fallback_models if m != self.model]

    def _client_for(self, key: str):
        c = self._clients.get(key)
        if c is None:
            from openai import OpenAI

            c = OpenAI(api_key=key, base_url=self.base_url, max_retries=1, timeout=120.0, default_headers=self.headers or None)
            self._clients[key] = c
        return c

    def request_kwargs(self, messages: list[dict], max_tokens: int | None = None, model: str | None = None) -> dict:
        kw = dict(model=model or self.model, messages=messages, max_completion_tokens=int(max_tokens or self.max_tokens),
                  response_format={"type": "json_object"})
        if self.fallback_models and self.base_url and "openrouter" in self.base_url:
            # OpenRouter also routes server-side to the next model when the first is busy / rate limited
            kw["extra_body"] = {"models": [kw["model"]] + [m for m in self.fallback_models if m != kw["model"]]}
        return kw

    # ------------------------------------------------------------------ one call
    def _call(self, client, kw: dict):
        import openai

        try:
            return client.chat.completions.create(**kw)
        except (TypeError, openai.BadRequestError):
            # servers without json mode / max_completion_tokens support
            kw = dict(kw)
            kw.pop("response_format", None)
            kw.pop("extra_body", None)
            if "max_completion_tokens" in kw:
                kw["max_tokens"] = kw.pop("max_completion_tokens")
            return client.chat.completions.create(**kw)

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int | None = None) -> dict:
        import openai

        messages = [
            {"role": "system", "content": system + "\n\n" + schema_hint(schema)},
            {"role": "user", "content": user},
        ]
        keys = self.keys.usable() if self.require_key else [None]
        if self.require_key and not keys:
            wait = self.keys.seconds_until_available()
            if wait == float("inf"):
                raise LLMAuthError(f"{self.name}: every key was rejected")
            raise LLMQuotaError(f"{self.name}: all {len(self.keys)} keys resting", retry_after=wait)
        last: Exception | None = None
        quota_hits = 0
        for model in self.models:
            model_dead = False
            for idx in list(keys):
                key = self.keys.keys[idx] if idx is not None else "not-needed"
                if idx is not None and (idx in self.keys.bad or self.keys.cooldown_until.get(idx, 0.0) > time.time()):
                    continue
                client = self._client_for(key)
                kw = self.request_kwargs(messages, max_tokens, model)
                try:
                    if idx is not None:
                        self.keys.calls += 1
                    resp = self._call(client, kw)
                except openai.AuthenticationError as e:
                    last = LLMAuthError(f"{self.name} auth: {e}")
                    if idx is not None:
                        self.keys.disable(key, str(e)[:120])
                    continue
                except openai.PermissionDeniedError as e:
                    last = LLMAuthError(f"{self.name} permission: {e}")
                    if idx is not None:
                        self.keys.disable(key, str(e)[:120])
                    continue
                except openai.NotFoundError as e:  # the model is gone / not on this key's plan
                    last = LLMError(f"{self.name}: model {model} unavailable: {e}")
                    model_dead = True
                    break
                except openai.RateLimitError as e:  # per-key daily cap or per-model quota
                    last = LLMQuotaError(f"{self.name} rate limit / quota ({model}): {e}", retry_after=_retry_after(e))
                    quota_hits += 1
                    if idx is not None:
                        self.keys.rest(key, _retry_after(e) or self.key_cooldown)
                    continue
                except openai.APIStatusError as e:
                    if e.status_code in (402, 429):
                        last = LLMQuotaError(f"{self.name} quota ({e.status_code}, {model})", retry_after=_retry_after(e))
                        quota_hits += 1
                        if idx is not None:
                            self.keys.rest(key, _retry_after(e) or self.key_cooldown)
                        continue
                    if e.status_code >= 500:
                        last = LLMTransientError(f"{self.name} server error {e.status_code} ({model})")
                        model_dead = True
                        break
                    last = LLMError(f"{self.name} error {e.status_code} ({model}): {e}")
                    model_dead = True
                    break
                except openai.APIConnectionError as e:
                    last = LLMTransientError(f"{self.name} connection: {e}")
                    model_dead = True
                    break
                if not getattr(resp, "choices", None):
                    last = LLMTransientError(f"{self.name}: empty response from {model} (provider error)")
                    model_dead = True
                    break
                text = resp.choices[0].message.content
                try:
                    return extract_json(text or "")
                except LLMError as e:
                    last = e
                    model_dead = True
                    break
            if model_dead:
                continue
        if isinstance(last, LLMQuotaError) or (quota_hits and self.require_key and not self.keys.usable()):
            wait = self.keys.seconds_until_available() if self.require_key else None
            raise LLMQuotaError(str(last) if last else f"{self.name}: quota", retry_after=None if wait in (None, float("inf")) else wait)
        if isinstance(last, LLMAuthError) and self.require_key and not self.keys.usable():
            raise last
        raise last or LLMError(f"{self.name}: no model answered")


class OpenRouterBackend(OpenAIBackend):
    name = "openrouter"
    DEFAULTS = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "inclusionai/ling-3.0-flash-fin:free",
        "fallback_models": ["nvidia/nemotron-3-super-120b-a12b:free", "nvidia/nemotron-3-ultra-550b-a55b:free", "openrouter/free"],
        "require_key": True,
    }


class GroqBackend(OpenAIBackend):
    name = "groq"
    DEFAULTS = {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
        "fallback_models": ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
        "require_key": True,
    }


class GeminiBackend(OpenAIBackend):
    name = "gemini"
    DEFAULTS = {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-3.8-flash",
        "fallback_models": ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-flash-lite-latest"],
        "require_key": True,
    }
