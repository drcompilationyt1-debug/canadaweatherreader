"""LLM trader decision - the NoFxAiOS/nofx mechanic: an LLM reads a compact market briefing
(price action, indicators, what the textbook strategies say, the current position and its PnL),
applies nofx's risk rules and answers with their decision JSON (action, confidence, reasoning),
adapted from leveraged crypto perpetuals to cash equities.

Live only (masked in training), one cached call per ticker per day through the LLM router, so it
runs on the free tiers and disappears cleanly when no backend answers.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.strategies import compute_strategies
from ..features.technical import atr, rsi
from ..logging_utils import get_logger
from .base import SignalProvider

log = get_logger(__name__)

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["HOLD", "PARTIAL_CLOSE", "FULL_CLOSE", "ADD_POSITION", "OPEN_NEW", "WAIT"]},
        "direction": {"type": "string", "enum": ["long", "short", "none"], "description": "side for OPEN_NEW / ADD_POSITION"},
        "position_size_pct": {"type": "number", "description": "0..100 percent of the capital slice for OPEN_NEW / ADD_POSITION"},
        "confidence": {"type": "number", "description": "0..100"},
        "reasoning": {"type": "string"},
    },
    "required": ["action", "direction", "position_size_pct", "confidence", "reasoning"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a professional discretionary trader (the nofx AI-trader role) managing one stock position for a long-term investor.

## Risk Management Rules
- A single position losing -8% from entry must be closed (stocks, no leverage)
- Protect capital first, then consider profit
- When position PnL retraces 30% from its peak, consider partial or full take-profit
- Enter only when the trend on several horizons agrees; never chase a move that already ran
- Scale in: a first entry should not exceed 50% of the capital slice; only add to profitable positions

## Output
Answer ONLY with the JSON decision object (fields: action, direction, position_size_pct, confidence 0-100, reasoning).
- HOLD: keep the current position   - PARTIAL_CLOSE / FULL_CLOSE: reduce or exit
- ADD_POSITION: add to the current position   - OPEN_NEW: open a position (give direction)   - WAIT: stay flat
"""

ACTION_DIRECTION = {"OPEN_NEW": 1.0, "ADD_POSITION": 0.5, "HOLD": 0.0, "WAIT": 0.0, "PARTIAL_CLOSE": -0.5, "FULL_CLOSE": -1.0}


def market_briefing(ticker: str, df: pd.DataFrame, position: dict | None) -> str:
    c = df["close"].astype(float)
    last = float(c.iloc[-1])
    rets = {k: float(c.iloc[-1] / c.iloc[-1 - k] - 1.0) * 100 for k in (1, 5, 20, 60, 120) if len(c) > k}
    r = float(rsi(c, 14).iloc[-1])
    a = float(atr(df, 14).iloc[-1] / last * 100)
    hi52 = float(c.tail(252).max())
    lo52 = float(c.tail(252).min())
    strat = compute_strategies(df).iloc[-1]
    votes = ", ".join(f"{k}={int(v):+d}" for k, v in strat.items() if k != "vote" and not np.isnan(v))
    daily = ", ".join(f"{x:+.1f}%" for x in (c.pct_change().tail(10) * 100).tolist())
    lines = [
        f"Symbol: {ticker}   Date: {df.index[-1].date()}   Last close: {last:.2f}",
        f"Returns: " + ", ".join(f"{k}d {v:+.1f}%" for k, v in rets.items()),
        f"Last 10 daily moves: {daily}",
        f"RSI14 {r:.0f}   ATR {a:.1f}% of price   52w high {hi52:.2f} ({(last / hi52 - 1) * 100:+.1f}%)   52w low {lo52:.2f} ({(last / lo52 - 1) * 100:+.1f}%)",
        f"Textbook strategies (+1 long / -1 short / 0 flat): {votes}; vote {float(strat['vote']):+.2f}",
    ]
    if position:
        lines.append(f"Current position: exposure {position.get('exposure', 0):+.2f} of the capital slice, "
                     f"unrealised PnL {position.get('pnl_pct', 0):+.2f}%, peak PnL {position.get('peak_pnl_pct', 0):+.2f}%, "
                     f"held {position.get('days', 0)} days")
    else:
        lines.append("Current position: flat")
    lines.append("Make your decision.")
    return "\n".join(lines)


def decision_to_vector(d: dict) -> np.ndarray:
    action = str(d.get("action", "WAIT")).upper()
    direction = str(d.get("direction", "none")).lower()
    base = ACTION_DIRECTION.get(action, 0.0)
    if action in ("OPEN_NEW", "ADD_POSITION"):
        base = base if direction != "short" else -base
    try:
        conf = float(np.clip(float(d.get("confidence", 50)) / 100.0, 0, 1))
        size = float(np.clip(float(d.get("position_size_pct", 0)) / 100.0, 0, 1))
    except (TypeError, ValueError):
        conf, size = 0.5, 0.0
    return np.array([base, conf, size, float(action in ("OPEN_NEW", "ADD_POSITION")), float(action in ("PARTIAL_CLOSE", "FULL_CLOSE"))],
                    dtype=np.float32)


class LLMTraderSignal(SignalProvider):
    name = "llm_trader"
    feature_names = ["nofx_direction", "nofx_confidence", "nofx_size", "nofx_open", "nofx_close"]
    live_only = True
    tier = "C"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.cache_dir: Path = ctx.models_dir / "cache" / "llm_trader"

    def availability(self) -> tuple[bool, str]:
        if self.ctx.llm is None:
            return False, "no LLM router in context"
        usable = self.ctx.llm.usable()
        if not usable:
            return False, "no LLM backend configured (set OPENROUTER_API_KEY / GROQ_API_KEY / ... or run ollama)"
        return True, "nofx-style decision via " + ", ".join(b.name for b in usable)

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        return None

    def compute_latest(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        f = self.cache_dir / f"{ticker}_{day}.json"
        if f.exists():
            return decision_to_vector(json.loads(f.read_text(encoding="utf-8")))
        position = (self.ctx.extra.get("positions") or {}).get(ticker)
        res = self.ctx.llm.complete_json(SYSTEM_PROMPT, market_briefing(ticker, df.tail(400), position), DECISION_SCHEMA, max_tokens=800)
        if res is None:
            return None
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(res), encoding="utf-8")
        log.info("llm_trader %s: %s (%s, conf %s)", ticker, res.get("action"), res.get("direction"), res.get("confidence"))
        return decision_to_vector(res)
