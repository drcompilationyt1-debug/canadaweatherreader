"""TradingAgents (TauricResearch) adapter - LLM analyst / researcher / trader / risk-manager debate.

Opt-in (``signals.trading_agents.enabled: true``).  The framework runs in its own virtualenv
(``python scripts/setup_agent_envs.py``) as a subprocess, because its dependency pins clash with
StockBot's.  One run is a dozen or more LLM calls, so results are cached per (ticker, day) and the
block is *live only* (masked during RL training).

Providers: ``llm_provider`` openai / anthropic / google / openrouter / ollama; for free models use
``llm_provider: openai`` with ``backend_url: https://openrouter.ai/api/v1`` - the OpenRouter key is
passed through automatically.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ...logging_utils import get_logger
from ...paths import ROOT
from ..base import SignalProvider
from .agent_runner import resolve_python, run_agent

log = get_logger(__name__)

KEY_FOR_PROVIDER = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY",
                    "xai": "XAI_API_KEY", "openrouter": "OPENROUTER_API_KEY", "ollama": None}
SCRIPT = ROOT / "scripts" / "agents" / "run_trading_agents.py"
CONFIG_KEYS = ("llm_provider", "deep_think_llm", "quick_think_llm", "backend_url", "max_debate_rounds", "max_risk_discuss_rounds")


def decision_code(text: str) -> float:
    t = (text or "").upper()
    if "OVERWEIGHT" in t or "STRONG BUY" in t:
        return 1.0
    if "UNDERWEIGHT" in t or "STRONG SELL" in t:
        return -1.0
    if "BUY" in t:
        return 0.7
    if "SELL" in t:
        return -0.7
    return 0.0


class TradingAgentsSignal(SignalProvider):
    name = "trading_agents"
    feature_names = ["ta_decision", "ta_is_buy", "ta_is_sell"]
    live_only = True
    tier = "C"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.cache_dir = ctx.models_dir / "cache" / "trading_agents"
        self.timeout = float(self.cfg.get("timeout", 1800))

    def _key_env(self) -> tuple[str | None, dict]:
        """(missing key name or None, extra env for the subprocess)."""
        provider = str(self.cfg.get("llm_provider", "openai")).lower()
        backend = str(self.cfg.get("backend_url") or "")
        extra: dict = {}
        if provider == "openai" and "openrouter" in backend and not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENROUTER_API_KEY"):
            extra["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
            return None, extra
        key = KEY_FOR_PROVIDER.get(provider, f"{provider.upper()}_API_KEY")
        if key and not os.environ.get(key):
            return key, extra
        return None, extra

    def availability(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "disabled (signals.trading_agents.enabled)"
        py, why = resolve_python(self.cfg.get("python"), ".venv-tradingagents", "tradingagents")
        if py is None:
            return False, why
        missing, _ = self._key_env()
        if missing:
            return False, f"set {missing}"
        return True, f"{why}; provider={self.cfg.get('llm_provider', 'openai')}"

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        return None

    def compute_latest(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        f = self.cache_dir / f"{ticker}_{day}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
        else:
            py, _ = resolve_python(self.cfg.get("python"), ".venv-tradingagents", "tradingagents")
            _, extra = self._key_env()
            overrides = {k: self.cfg.get(k) for k in CONFIG_KEYS if self.cfg.get(k) is not None}
            d = run_agent(py, SCRIPT, [ticker, day, "--config-json", json.dumps(overrides)], self.timeout, extra)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(d), encoding="utf-8")
        code = decision_code(d.get("decision", ""))
        return np.array([code, float(code > 0.3), float(code < -0.3)], dtype=np.float32)
