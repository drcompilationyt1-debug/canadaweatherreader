"""Claude backend (official ``anthropic`` SDK)."""
from __future__ import annotations

import os
from pathlib import Path

from .base import LLMAuthError, LLMBackend, LLMError, LLMQuotaError, LLMTransientError, extract_json

DEFAULT_MODEL = "claude-opus-5"


def _has_cli_profile() -> bool:
    # `ant auth login` stores a profile the SDK picks up automatically
    return (Path.home() / ".config" / "anthropic").exists()


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        self.model = self.cfg.get("model") or DEFAULT_MODEL
        self.max_tokens = int(self.cfg.get("max_tokens", 2048))
        self.effort = self.cfg.get("effort", "medium")
        self.server_fallbacks = bool(self.cfg.get("server_fallbacks", True))
        self._client = None

    def is_configured(self) -> tuple[bool, str]:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "pip install anthropic"
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or _has_cli_profile():
            return True, f"model={self.model}"
        return False, "set ANTHROPIC_API_KEY (or run `ant auth login`)"

    def _get_client(self):
        if self._client is None:
            import anthropic

            # zero-arg client: resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / CLI profile
            self._client = anthropic.Anthropic(max_retries=2, timeout=120.0)
        return self._client

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int | None = None) -> dict:
        import anthropic

        client = self._get_client()
        kwargs = dict(
            model=self.model,
            max_tokens=int(max_tokens or self.max_tokens),
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        try:
            if self.server_fallbacks:
                try:
                    # Opus 5 / Fable: let Anthropic re-run a refused request on a fallback model
                    resp = client.beta.messages.create(
                        betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                    )
                except (TypeError, anthropic.BadRequestError):
                    resp = client.messages.create(**kwargs)
            else:
                resp = client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise LLMAuthError(f"anthropic auth: {e.message}") from e
        except anthropic.PermissionDeniedError as e:
            raise LLMAuthError(f"anthropic permission: {e.message}") from e
        except anthropic.RateLimitError as e:
            retry = e.response.headers.get("retry-after") if getattr(e, "response", None) is not None else None
            raise LLMQuotaError(f"anthropic rate limit: {e.message}", retry_after=float(retry) if retry else None) from e
        except anthropic.APIStatusError as e:
            if e.status_code in (402, 429):
                raise LLMQuotaError(f"anthropic quota ({e.status_code}): {e.message}") from e
            if e.status_code >= 500:
                raise LLMTransientError(f"anthropic server error {e.status_code}") from e
            raise LLMError(f"anthropic error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMTransientError(f"anthropic connection: {e}") from e

        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raise LLMError(f"anthropic refused: {getattr(details, 'category', None)}")
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if text is None:
            raise LLMError("anthropic reply had no text block")
        return extract_json(text)
