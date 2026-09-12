"""LLM news reader: an external model (Claude / OpenAI-compatible / Ollama) reads the recent
headlines for a ticker and returns a calibrated JSON assessment that becomes a signal block.

* Results are cached per (ticker, day, headline-hash) so a day is never scored twice.
* When no backend is configured / all are out of quota, ``compute_latest`` returns ``None`` and
  the block is simply masked out for the policy - nothing breaks.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..news.fetcher import NewsItem, headlines_hash
from .base import SignalProvider

log = get_logger(__name__)

NEWS_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "number", "description": "-1 (very negative) .. 1 (very positive) for the stock"},
        "impact": {"type": "number", "description": "0 (noise) .. 1 (thesis-changing) expected effect on the price"},
        "confidence": {"type": "number", "description": "0 .. 1 how sure you are"},
        "horizon": {"type": "string", "enum": ["days", "weeks", "months"]},
        "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "key_points": {"type": "array", "items": {"type": "string"}, "description": "up to 5 short bullet points"},
    },
    "required": ["sentiment", "impact", "confidence", "horizon", "direction", "key_points"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a buy-side equity analyst supporting a long-term, risk-aware investor. "
    "Read the recent headlines for one stock and judge how they should change the investor's view "
    "over the next days to months. Be calibrated: most headlines are noise (impact near 0); reserve "
    "high impact for earnings surprises, guidance changes, regulatory / legal events, M&A, major "
    "product or management news. Ignore clickbait and generic market commentary. Output only JSON."
)

GRADE_SCHEMA = {
    "type": "object",
    "properties": {"relevant": {"type": "array", "items": {"type": "integer"}, "description": "indices of headlines relevant to the stock's outlook"}},
    "required": ["relevant"],
    "additionalProperties": False,
}
GRADE_PROMPT = ("You are a grader assessing the relevance of retrieved news headlines to the question "
                "'What should a long-term investor expect for this stock?'. Return the indices of the headlines that are "
                "relevant (company- or sector-specific, material); drop generic market chatter, listicles and duplicates.")

HORIZON_CODE = {"days": -1.0, "weeks": 0.0, "months": 1.0}
DIRECTION_CODE = {"bearish": -1.0, "neutral": 0.0, "bullish": 1.0}


def build_prompt(ticker: str, items: list[NewsItem], as_of: str) -> str:
    lines = [f"Stock: {ticker}", f"Date: {as_of}", "", "Recent headlines (newest first):"]
    for it in items:
        src = f" [{it.source}]" if it.source else ""
        lines.append(f"- ({it.published[:10]}){src} {it.text()}")
    lines.append("")
    lines.append("Assess the combined effect on the stock and return the JSON object.")
    return "\n".join(lines)


def result_to_vector(res: dict, n_items: int) -> np.ndarray:
    def num(k, lo, hi):
        try:
            return float(np.clip(float(res.get(k, 0.0)), lo, hi))
        except (TypeError, ValueError):
            return 0.0

    return np.array([
        num("sentiment", -1, 1),
        num("impact", 0, 1),
        num("confidence", 0, 1),
        HORIZON_CODE.get(str(res.get("horizon", "weeks")).lower(), 0.0),
        DIRECTION_CODE.get(str(res.get("direction", "neutral")).lower(), 0.0),
        min(n_items / 10.0, 2.0),
    ], dtype=np.float32)


class LLMNewsSignal(SignalProvider):
    name = "news_llm"
    feature_names = ["llm_sentiment", "llm_impact", "llm_confidence", "llm_horizon", "llm_direction", "llm_n_items"]
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.lookback_days = int(self.cfg.get("lookback_days", 3))
        self.max_items = int(self.cfg.get("max_items", 25))
        self.score_history = bool(self.cfg.get("score_history", False))
        self.max_history_calls = int(self.cfg.get("max_history_calls", 200))
        self.history_granularity = str(self.cfg.get("history_granularity", "week")).lower()  # week | day
        # retrieve -> grade -> generate, the stocks-insights-ai-agent news graph (one extra call per scoring)
        self.rag_grading = bool(self.cfg.get("rag_grading", False))
        self.cache_dir: Path = (ctx.news.cache_dir if ctx.news is not None else ctx.models_dir) / "llm"

    # ------------------------------------------------------------------ availability
    def availability(self) -> tuple[bool, str]:
        if self.ctx.news is None:
            return False, "no news fetcher in context"
        if self.ctx.llm is None:
            return False, "no LLM router in context"
        usable = self.ctx.llm.usable()
        cached = self.cache_dir.exists() and any(self.cache_dir.rglob("*.json"))
        if not usable and not cached:
            return False, "no LLM backend configured (set ANTHROPIC_API_KEY / OPENAI_API_KEY or run ollama)"
        if not usable:
            return True, "no live LLM backend - using cached scores only"
        return True, "backends: " + ", ".join(b.name for b in usable)

    # ------------------------------------------------------------------ cache
    def _cache_file(self, ticker: str, day: str) -> Path:
        return self.cache_dir / ticker / f"{day}.json"

    def _read_cache(self, ticker: str, day: str, h: str | None = None) -> dict | None:
        f = self._cache_file(ticker, day)
        if not f.exists():
            return None
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        if h is not None and d.get("hash") != h:
            return None
        return d

    def _write_cache(self, ticker: str, day: str, h: str, result: dict, n_items: int) -> None:
        f = self._cache_file(ticker, day)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"hash": h, "n_items": n_items, "result": result,
                                 "scored_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")

    # ------------------------------------------------------------------ scoring
    def score_items(self, ticker: str, items: list[NewsItem], as_of: str) -> dict | None:
        items = items[: self.max_items]
        if not items:
            return {"sentiment": 0.0, "impact": 0.0, "confidence": 0.5, "horizon": "weeks", "direction": "neutral",
                    "key_points": ["no recent news"]}
        h = headlines_hash(items)
        cached = self._read_cache(ticker, as_of, h)
        if cached:
            return cached["result"]
        if self.ctx.llm is None:
            return None
        if self.rag_grading and len(items) > 3:
            listing = "\n".join(f"[{i}] ({it.published[:10]}) {it.text()}" for i, it in enumerate(items))
            graded = self.ctx.llm.complete_json(GRADE_PROMPT, f"Stock: {ticker}\n\nHeadlines:\n{listing}", GRADE_SCHEMA, max_tokens=400)
            if graded and isinstance(graded.get("relevant"), list):
                keep = [items[i] for i in graded["relevant"] if isinstance(i, int) and 0 <= i < len(items)]
                if keep:
                    items = keep
        res = self.ctx.llm.complete_json(SYSTEM_PROMPT, build_prompt(ticker, items, as_of), NEWS_SCHEMA)
        if res is None:
            return None
        self._write_cache(ticker, as_of, h, res, len(items))
        return res

    def compute_latest(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        items = self.ctx.news.fetch(ticker, self.lookback_days)
        as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        res = self.score_items(ticker, items, as_of)
        if res is None:
            return None
        return result_to_vector(res, len(items))

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        """Historical values come from the on-disk cache (and optionally from scoring local CSV news)."""
        out = np.full((len(df), self.size), np.nan, dtype=np.float32)
        if self.score_history and self.ctx.llm is not None and self.ctx.llm.available:
            self._score_local_history(ticker)
        folder = self.cache_dir / ticker
        if not folder.exists():
            return out
        rows: dict[pd.Timestamp, np.ndarray] = {}
        for f in folder.glob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                rows[pd.Timestamp(f.stem)] = result_to_vector(d["result"], int(d.get("n_items", 0)))
            except Exception:  # noqa: BLE001
                continue
        if not rows:
            return out
        ser = pd.DataFrame.from_dict(rows, orient="index").sort_index()
        limit = 5 if self.history_granularity == "week" else self.lookback_days
        ser = ser.reindex(df.index.union(ser.index)).ffill(limit=limit).reindex(df.index)
        return ser.to_numpy(np.float32)

    def _score_local_history(self, ticker: str) -> None:
        hist = self.ctx.news.history(ticker)
        if hist is None or len(hist) == 0:
            return
        # one LLM call per week (default) or per day; scored periods are cached so this is resumable
        key = hist["date"].dt.to_period("W-FRI").dt.start_time if self.history_granularity == "week" else hist["date"]
        calls = 0
        for period, grp in hist.groupby(key):
            day_s = pd.Timestamp(period).strftime("%Y-%m-%d")
            if self._cache_file(ticker, day_s).exists():
                continue
            if calls >= self.max_history_calls:
                log.info("news_llm: history scoring budget reached for %s (%d calls)", ticker, calls)
                break
            grp = grp.sort_values("date").tail(self.max_items)
            items = [NewsItem(ticker, str(r.title), pd.Timestamp(r.date).strftime("%Y-%m-%d"), "local",
                              str(r.summary) if isinstance(r.summary, str) else "") for r in grp.itertuples()]
            if self.score_items(ticker, items, day_s) is None:
                log.info("news_llm: no backend available - history scoring for %s stops at %s", ticker, day_s)
                break
            calls += 1
