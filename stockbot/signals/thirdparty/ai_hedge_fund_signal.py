"""ai-hedge-fund (virattt) adapter - LLM investor personas (Buffett, Munger, Burry, ...) + quant models.

The repo is organised as a *fund* (YAML mandate) staffed with alpha models; every model returns a
``Signal(value in [-1, 1])`` for a ticker as of a date.  This adapter runs the staffed models of a
mandate (default: the repo's ``hedge_fund/fund/example.yaml``) in the framework's own virtualenv
(``python scripts/setup_agent_envs.py``) and reports the average conviction plus the share of
bullish / bearish models.

Live only, cached per (ticker, bar date).  Needs an LLM key (the GEMINI_API_KEY pool is mapped to
GOOGLE_API_KEY, or OPENAI_API_KEY / ANTHROPIC_API_KEY ...) and FINANCIAL_DATASETS_API_KEY: the
framework's data source requires a key for every ticker (free sign-up at financialdatasets.ai).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ...logging_utils import get_logger
from ...paths import ROOT, THIRD_PARTY
from ..base import SignalProvider
from .agent_runner import resolve_python, run_agent

log = get_logger(__name__)

LLM_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY")
SCRIPT = ROOT / "scripts" / "agents" / "run_ai_hedge_fund.py"
# financialdatasets.ai (the framework's data source) requires an API key for every ticker since 2026;
# a free key comes from https://financialdatasets.ai - without it every model in the mandate fails
DATA_KEY = "FINANCIAL_DATASETS_API_KEY"


class AIHedgeFundSignal(SignalProvider):
    name = "ai_hedge_fund"
    feature_names = ["ahf_conviction", "ahf_bullish_share", "ahf_bearish_share"]
    live_only = True
    tier = "C"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.cache_dir = ctx.models_dir / "cache" / "ai_hedge_fund"
        self.mandate = self.cfg.get("mandate") or str(THIRD_PARTY / "ai-hedge-fund" / "hedge_fund" / "fund" / "example.yaml")
        self.timeout = float(self.cfg.get("timeout", 1800))

    def _llm_env(self) -> dict:
        """Env for the subprocess: the configured model and, for Google models, the next Gemini key of the pool."""
        from ...llm.keys import pool

        extra: dict = {}
        model = self.cfg.get("llm_model")
        if model:
            extra["HEDGE_FUND_LLM_MODEL"] = str(model)
        if not os.environ.get("GOOGLE_API_KEY"):
            key = pool("GEMINI_API_KEY").pick()
            if key:
                extra["GOOGLE_API_KEY"] = key
        data_key = pool(DATA_KEY).pick()          # several financialdatasets.ai keys share the daily quota, round-robin
        if data_key:
            extra[DATA_KEY] = data_key
        return extra

    def availability(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "disabled (signals.ai_hedge_fund.enabled)"
        py, why = resolve_python(self.cfg.get("python"), ".venv-aihf", "hedge_fund")
        if py is None:
            return False, why
        if not any(os.environ.get(k) for k in LLM_KEYS) and not self._llm_env().get("GOOGLE_API_KEY"):
            return False, "set an LLM key (GEMINI_API_KEY pool / OPENAI_API_KEY / ANTHROPIC_API_KEY ...)"
        if not Path(self.mandate).exists():
            return False, f"mandate file not found: {self.mandate}"
        from ...llm.keys import pool

        if not pool(DATA_KEY).configured:
            return False, f"set {DATA_KEY} (free key at financialdatasets.ai - its data API needs one for every ticker)"
        return True, f"{why}; mandate={Path(self.mandate).name}; financialdatasets.ai {pool(DATA_KEY).describe()}"

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        return None

    @staticmethod
    def cache_day(df: pd.DataFrame) -> str:
        """Cache key: the last complete bar's date, so a pre-open run and the open-time cycle share it."""
        try:
            return pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001
            return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def compute_latest(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        from ...llm.keys import pool

        if not pool(DATA_KEY).configured:
            return None
        day = self.cache_day(df)
        f = self.cache_dir / f"{ticker}_{day}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
        else:
            py, _ = resolve_python(self.cfg.get("python"), ".venv-aihf", "hedge_fund")
            d = run_agent(py, SCRIPT, [ticker, day, "--mandate", str(self.mandate)], self.timeout, self._llm_env())
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(d), encoding="utf-8")
        vals = np.array([s["value"] for s in d.get("signals", [])], dtype=float)
        if len(vals) == 0:
            return None
        return np.array([float(np.clip(vals.mean(), -1, 1)), float((vals > 0.2).mean()), float((vals < -0.2).mean())], dtype=np.float32)
