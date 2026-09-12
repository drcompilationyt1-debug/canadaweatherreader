"""OpenAI-compatible backends.

``OpenAIBackend`` talks to OpenAI or to any server that speaks the OpenAI chat API via
``base_url`` (LM Studio, vLLM, ...).  The presets below point it at providers with a free tier:

* ``openrouter`` - OpenRouter, free models carry a ``:free`` suffix (https://openrouter.ai/models?q=free)
* ``groq``       - Groq free tier (https://console.groq.com)
* ``gemini``     - Google AI Studio free tier, OpenAI-compatible endpoint (https://aistudio.google.com)
"""
from __future__ import annotations

import os

from .base import LLMAuthError, LLMBackend, LLMError, LLMQuotaError, LLMTransientError, extract_json, schema_hint

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIBackend(LLMBackend):
    name = "openai"
    DEFAULTS: dict = {}

    def __init__(self, cfg: dict | None = None):
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
        self._client = None

    def is_configured(self) -> tuple[bool, str]:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "pip install openai"
        if self.require_key and not os.environ.get(self.api_key_env):
            return False, f"set {self.api_key_env}"
        where = f" @ {self.base_url}" if self.base_url else ""
        return True, f"model={self.model}{where}"

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            key = os.environ.get(self.api_key_env) or "not-needed"
            self._client = OpenAI(api_key=key, base_url=self.base_url, max_retries=2, timeout=120.0,
                                  default_headers=self.headers or None)
        return self._client

    def request_kwargs(self, messages: list[dict], max_tokens: int | None = None) -> dict:
        kw = dict(model=self.model, messages=messages, max_completion_tokens=int(max_tokens or self.max_tokens),
                  response_format={"type": "json_object"})
        if self.fallback_models:  # OpenRouter routes to the next model when the first is busy / rate limited
            kw["extra_body"] = {"models": [self.model] + self.fallback_models}
        return kw

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int | None = None) -> dict:
        import openai

        client = self._get_client()
        messages = [
            {"role": "system", "content": system + "\n\n" + schema_hint(schema)},
            {"role": "user", "content": user},
        ]
        kw = self.request_kwargs(messages, max_tokens)
        try:
            try:
                resp = client.chat.completions.create(**kw)
            except (TypeError, openai.BadRequestError):
                # servers without json mode / max_completion_tokens support
                kw.pop("response_format", None)
                kw["max_tokens"] = kw.pop("max_completion_tokens")
                resp = client.chat.completions.create(**kw)
        except openai.AuthenticationError as e:
            raise LLMAuthError(f"{self.name} auth: {e}") from e
        except openai.PermissionDeniedError as e:
            raise LLMAuthError(f"{self.name} permission: {e}") from e
        except openai.RateLimitError as e:  # includes insufficient_quota / free-tier daily caps
            raise LLMQuotaError(f"{self.name} rate limit / quota: {e}") from e
        except openai.APIStatusError as e:
            if e.status_code in (402, 429):
                raise LLMQuotaError(f"{self.name} quota ({e.status_code})") from e
            if e.status_code >= 500:
                raise LLMTransientError(f"{self.name} server error {e.status_code}") from e
            raise LLMError(f"{self.name} error {e.status_code}: {e}") from e
        except openai.APIConnectionError as e:
            raise LLMTransientError(f"{self.name} connection: {e}") from e
        if not getattr(resp, "choices", None):
            raise LLMTransientError(f"{self.name}: empty response (provider error)")
        text = resp.choices[0].message.content
        return extract_json(text or "")


class OpenRouterBackend(OpenAIBackend):
    name = "openrouter"
    DEFAULTS = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "fallback_models": ["deepseek/deepseek-chat-v3-0324:free", "qwen/qwen3-235b-a22b:free"],
        "require_key": True,
    }


class GroqBackend(OpenAIBackend):
    name = "groq"
    DEFAULTS = {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
        "require_key": True,
    }


class GeminiBackend(OpenAIBackend):
    name = "gemini"
    DEFAULTS = {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
        "require_key": True,
    }
