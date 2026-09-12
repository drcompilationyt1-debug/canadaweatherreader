"""Headline sentiment with VADER - the stocksight approach (nltk VADER + TextBlob) without an LLM."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .base import SignalProvider

log = get_logger(__name__)


def _analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()
    except ImportError:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer as NltkSIA

            return NltkSIA()
        except Exception:  # noqa: BLE001
            return None


def score_texts(texts: list[str], sia) -> np.ndarray:
    if not texts:
        return np.zeros(4, dtype=np.float32)
    comp = np.array([sia.polarity_scores(t)["compound"] for t in texts], dtype=float)
    return np.array([
        comp.mean(),
        (comp > 0.05).mean() - (comp < -0.05).mean(),
        min(np.log1p(len(texts)) / 3.0, 2.0),
        comp[np.argmax(np.abs(comp))],
    ], dtype=np.float32)


class SentimentSignal(SignalProvider):
    name = "sentiment"
    feature_names = ["vader_mean", "vader_posneg", "vader_n", "vader_extreme"]
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.lookback_days = int(self.cfg.get("lookback_days", 3))
        self._sia = None

    def availability(self) -> tuple[bool, str]:
        if self._sia is None:
            self._sia = _analyzer()
        if self._sia is None:
            return False, "pip install vaderSentiment"
        if self.ctx.news is None:
            return False, "no news fetcher in context"
        return True, "VADER on headlines (history only where data/news/*.csv exists)"

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        hist = self.ctx.news.history(ticker)
        out = np.full((len(df), self.size), np.nan, dtype=np.float32)
        if hist is None or len(hist) == 0:
            return out  # all-unavailable rows (block mask = 0)
        comp = np.array([self._sia.polarity_scores(t)["compound"] for t in hist["title"].astype(str)])
        daily = pd.DataFrame({"c": comp, "date": hist["date"].to_numpy()})
        start, end = daily["date"].min(), daily["date"].max()
        rows = []
        idx = df.index
        by_day = daily.groupby("date")["c"].apply(list)
        for d in idx:
            if d < start or d > end:
                rows.append([np.nan] * 4)
                continue
            texts = []
            for k in range(self.lookback_days):
                day = d - pd.Timedelta(days=k)
                texts.extend(by_day.get(day, []))
            if not texts:
                rows.append([0.0, 0.0, 0.0, 0.0])
            else:
                c = np.array(texts)
                rows.append([c.mean(), (c > 0.05).mean() - (c < -0.05).mean(), min(np.log1p(len(c)) / 3.0, 2.0), c[np.argmax(np.abs(c))]])
        return np.array(rows, dtype=np.float32)

    def compute_latest(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        items = self.ctx.news.fetch(ticker, self.lookback_days)
        return score_texts([i.text() for i in items], self._sia)
