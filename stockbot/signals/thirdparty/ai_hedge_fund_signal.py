"""ai-hedge-fund (virattt) adapter - LLM investor personas (Buffett, Munger, Burry, ...) + quant models.

The repo is organised as a *fund* (YAML mandate) staffed with alpha models; every model returns a
``Signal(value in [-1, 1])`` for a ticker as of a date.  This adapter runs the staffed models of a
mandate (default: the repo's ``hedge_fund/fund/example.yaml``) in the framework's own virtualenv
(``python scripts/setup_agent_envs.py``) and reports the average conviction plus the share of
bullish / bearish models.

Opt-in (``signals.ai_hedge_fund.enabled: true``), live only, cached per (ticker, day).  Needs an
LLM key (OPENAI_API_KEY / ANTHROPIC_API_KEY ...) and FINANCIAL_DATASETS_API_KEY for tickers outside
the free tier (AAPL, GOOGL, MSFT, NVDA, TSLA).
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

    def availability(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "disabled (signals.ai_hedge_fund.enabled)"
        py, why = resolve_python(self.cfg.get("python"), ".venv-aihf", "hedge_fund")
        if py is None:
            return False, why
        if not any(os.environ.get(k) for k in LLM_KEYS):
            return False, "set an LLM key (OPENAI_API_KEY / ANTHROPIC_API_KEY ...)"
        if not Path(self.mandate).exists():
            return False, f"mandate file not found: {self.mandate}"
        fd = "FINANCIAL_DATASETS_API_KEY set" if os.environ.get("FINANCIAL_DATASETS_API_KEY") else "no FINANCIAL_DATASETS_API_KEY (free tickers only)"
        return True, f"{why}; mandate={Path(self.mandate).name}; {fd}"

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        return None

    def compute_latest(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        f = self.cache_dir / f"{ticker}_{day}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
        else:
            py, _ = resolve_python(self.cfg.get("python"), ".venv-aihf", "hedge_fund")
            d = run_agent(py, SCRIPT, [ticker, day, "--mandate", str(self.mandate)], self.timeout)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(d), encoding="utf-8")
        vals = np.array([s["value"] for s in d.get("signals", [])], dtype=float)
        if len(vals) == 0:
            return None
        return np.array([float(np.clip(vals.mean(), -1, 1)), float((vals > 0.2).mean()), float((vals < -0.2).mean())], dtype=np.float32)
