"""Local Ollama backend (free, offline) - http://localhost:11434 by default."""
from __future__ import annotations

import requests

from .base import LLMBackend, LLMError, LLMTransientError, extract_json, schema_hint


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        self.model = self.cfg.get("model") or "llama3.1"
        self.host = (self.cfg.get("host") or "http://localhost:11434").rstrip("/")
        self.timeout = float(self.cfg.get("timeout", 120))
        self._reachable: bool | None = None

    def is_configured(self) -> tuple[bool, str]:
        if self._reachable is None:
            try:
                r = requests.get(f"{self.host}/api/tags", timeout=2)
                self._reachable = r.ok
                if r.ok:
                    names = [m.get("name", "") for m in r.json().get("models", [])]
                    if names and not any(n.split(":")[0] == self.model.split(":")[0] for n in names):
                        return False, f"model {self.model} not pulled (ollama pull {self.model})"
            except Exception:  # noqa: BLE001
                self._reachable = False
        return (True, f"model={self.model} @ {self.host}") if self._reachable else (False, f"no ollama server at {self.host}")

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int | None = None) -> dict:
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema,  # ollama >= 0.5 accepts a JSON schema; older servers fall back below
            "options": {"temperature": 0.2, "num_predict": int(max_tokens or 1024)},
            "messages": [
                {"role": "system", "content": system + "\n\n" + schema_hint(schema)},
                {"role": "user", "content": user},
            ],
        }
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            if r.status_code == 400:
                payload["format"] = "json"
                r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            if r.status_code >= 500:
                raise LLMTransientError(f"ollama server error {r.status_code}")
            if not r.ok:
                raise LLMError(f"ollama error {r.status_code}: {r.text[:200]}")
            content = r.json().get("message", {}).get("content", "")
        except requests.RequestException as e:
            self._reachable = None
            raise LLMTransientError(f"ollama connection: {e}") from e
        return extract_json(content)
