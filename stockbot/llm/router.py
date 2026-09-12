"""LLM router: try backends in order, rest the ones that ran out of quota, disable the broken ones.

``complete_json`` returns ``None`` when *no* backend can answer - callers treat that as
"signal unavailable" and the trading policy keeps working without the news features.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger
from .anthropic_backend import AnthropicBackend
from .base import LLMAuthError, LLMBackend, LLMError, LLMQuotaError, LLMTransientError
from .ollama_backend import OllamaBackend
from .openai_backend import GeminiBackend, GroqBackend, OpenAIBackend, OpenRouterBackend

log = get_logger(__name__)

BACKENDS: dict[str, type[LLMBackend]] = {
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    "openrouter": OpenRouterBackend,
    "groq": GroqBackend,
    "gemini": GeminiBackend,
    "ollama": OllamaBackend,
}


class LLMRouter:
    def __init__(self, backends: list[LLMBackend], cooldown_seconds: float = 3600.0,
                 state_file: str | Path | None = None):
        self.backends = backends
        self.cooldown = float(cooldown_seconds)
        self.state_file = Path(state_file) if state_file else None
        self.cooldown_until: dict[str, float] = {}
        self.disabled: dict[str, str] = {}
        self.stats: dict[str, dict[str, int]] = {b.name: {"ok": 0, "fail": 0} for b in backends}
        self._load_state()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, cfg) -> "LLMRouter":
        llm = cfg.section("llm")
        order = list(llm.get("order", ["anthropic", "openai", "ollama"]))
        backends: list[LLMBackend] = []
        for name in order:
            klass = BACKENDS.get(name)
            if klass is None:
                log.warning("unknown llm backend %r ignored", name)
                continue
            backends.append(klass(llm.get(name, {})))
        state = cfg.path("llm.state_file", "data/llm_state.json") if hasattr(cfg, "path") else None
        return cls(backends, llm.get("cooldown_seconds", 3600), state)

    # ------------------------------------------------------------------ state persistence
    def _load_state(self) -> None:
        if self.state_file and self.state_file.exists():
            try:
                d = json.loads(self.state_file.read_text(encoding="utf-8"))
                self.cooldown_until = {k: float(v) for k, v in d.get("cooldown_until", {}).items()}
            except Exception:  # noqa: BLE001
                pass

    def _save_state(self) -> None:
        if not self.state_file:
            return
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps({"cooldown_until": self.cooldown_until}), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ queries
    def usable(self) -> list[LLMBackend]:
        now = time.time()
        out = []
        for b in self.backends:
            if b.name in self.disabled or self.cooldown_until.get(b.name, 0) > now:
                continue
            ok, _ = b.is_configured()
            if ok:
                out.append(b)
        return out

    @property
    def available(self) -> bool:
        return bool(self.usable())

    def status(self) -> list[dict[str, Any]]:
        now = time.time()
        rows = []
        for b in self.backends:
            d = b.describe()
            cd = self.cooldown_until.get(b.name, 0) - now
            d["cooldown_s"] = int(cd) if cd > 0 else 0
            d["disabled"] = self.disabled.get(b.name)
            d.update(self.stats.get(b.name, {}))
            rows.append(d)
        return rows

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int | None = None) -> dict | None:
        for b in self.usable():
            try:
                out = b.complete_json(system, user, schema, max_tokens)
                self.stats[b.name]["ok"] += 1
                return out
            except LLMAuthError as e:
                self.disabled[b.name] = str(e)
                log.warning("llm backend %s disabled: %s", b.name, e)
            except LLMQuotaError as e:
                wait = e.retry_after or float(b.cfg.get("cooldown_seconds") or self.cooldown)
                self.cooldown_until[b.name] = time.time() + wait
                self._save_state()
                log.warning("llm backend %s resting %.0fs: %s", b.name, wait, e)
            except LLMTransientError as e:
                log.warning("llm backend %s transient error: %s", b.name, e)
            except LLMError as e:
                log.warning("llm backend %s error: %s", b.name, e)
            except Exception as e:  # noqa: BLE001 - never let an LLM problem crash trading
                log.warning("llm backend %s unexpected error: %s", b.name, e)
            self.stats[b.name]["fail"] += 1
        return None
