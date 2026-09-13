"""FinBERT (ProsusAI/finBERT) - a BERT fine-tuned on financial text, run locally on the CPU.

Unlike the LLM news blocks (live only, quota bound) FinBERT scores every headline in the local
news history (``data/news/*.csv`` from ``stockbot news-history``), so the block has years of
training history.  Per bar: mean (positive - negative) probability over the last ``lookback_days``
of headlines, mean positive, mean negative, and the (log) number of headlines.  Scores are
cached per headline under ``models/signals/finbert`` so each headline is scored once; a fresh
run scores at most ``max_texts`` new headlines per ticker (newest first) to bound the time.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ...logging_utils import get_logger
from ..base import SignalProvider

log = get_logger(__name__)


class FinBERTSignal(SignalProvider):
    name = "finbert"
    feature_names = ["fb_net", "fb_pos", "fb_neg", "fb_n"]
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.model_id = str(self.cfg.get("model", "ProsusAI/finbert"))
        self.lookback_days = int(self.cfg.get("lookback_days", 3))
        self.max_texts = int(self.cfg.get("max_texts", 4000))
        self.batch_size = int(self.cfg.get("batch_size", 32))
        self.csv_dir = ctx.cfg.path("news.local_csv_dir", "data/news")
        self.folder = ctx.state_dir() / "finbert"
        self._pipe = None

    def availability(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False, "pip install transformers (FinBERT)"
        return True, f"{self.model_id} headline sentiment, {self.lookback_days}-day window"

    # ------------------------------------------------------------------ scoring
    def _pipeline(self):
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline("text-classification", model=self.model_id, top_k=None, device=-1, truncation=True, max_length=64)
        return self._pipe

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.strip().lower().encode("utf-8", "ignore")).hexdigest()[:16]

    def _score_path(self, ticker: str):
        return self.folder / f"{ticker.replace('/', '_')}_scores.parquet"

    def _load_scores(self, ticker: str) -> pd.DataFrame:
        p = self._score_path(ticker)
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:  # noqa: BLE001
                pass
        return pd.DataFrame(columns=["key", "pos", "neg", "neu"]).set_index("key")

    def score_texts(self, ticker: str, texts: list[str]) -> pd.DataFrame:
        """Scores (pos, neg, neu) for every text, from the cache or freshly computed (bounded by max_texts)."""
        scores = self._load_scores(ticker)
        keys = [self._key(t) for t in texts]
        todo = [(k, t) for k, t in zip(keys, texts) if k not in scores.index and t.strip()]
        seen: set[str] = set()
        todo = [(k, t) for k, t in todo if not (k in seen or seen.add(k))][-self.max_texts:]
        if todo:
            pipe = self._pipeline()
            rows = []
            for i in range(0, len(todo), self.batch_size):
                chunk = todo[i:i + self.batch_size]
                res = pipe([t for _, t in chunk])
                for (k, _), labels in zip(chunk, res):
                    d = {x["label"].lower(): float(x["score"]) for x in labels}
                    rows.append({"key": k, "pos": d.get("positive", 0.0), "neg": d.get("negative", 0.0), "neu": d.get("neutral", 0.0)})
            new = pd.DataFrame(rows).set_index("key")
            scores = pd.concat([scores, new]) if len(scores) else new
            scores = scores[~scores.index.duplicated(keep="last")]
            self.folder.mkdir(parents=True, exist_ok=True)
            scores.to_parquet(self._score_path(ticker))
            log.info("finbert %s: scored %d new headlines", ticker, len(rows))
        return scores.reindex(keys)

    # ------------------------------------------------------------------ features
    def _daily(self, dates: pd.Series, scored: pd.DataFrame) -> pd.DataFrame:
        d = pd.DataFrame({"date": pd.to_datetime(dates).dt.normalize().to_numpy(), "pos": scored["pos"].to_numpy(),
                          "neg": scored["neg"].to_numpy()}).dropna()
        if len(d) == 0:
            return pd.DataFrame(columns=["pos", "neg", "n"])
        g = d.groupby("date").agg(pos=("pos", "sum"), neg=("neg", "sum"), n=("pos", "size"))
        return g

    def _features(self, daily: pd.DataFrame, index: pd.DatetimeIndex) -> np.ndarray:
        out = np.full((len(index), 4), np.nan, dtype=np.float32)
        if len(daily) == 0:
            return out
        cal = pd.date_range(min(daily.index.min(), index.min()), index.max(), freq="D")
        s = daily.reindex(cal).fillna(0.0).rolling(self.lookback_days, min_periods=1).sum()
        s = s.reindex(index)
        n = s["n"].to_numpy(float)
        has = n > 0
        pos = np.where(has, s["pos"].to_numpy(float) / np.maximum(n, 1), np.nan)
        neg = np.where(has, s["neg"].to_numpy(float) / np.maximum(n, 1), np.nan)
        out[:, 0] = (pos - neg) * 2.0
        out[:, 1] = pos * 2.0 - 1.0
        out[:, 2] = neg * 2.0 - 1.0
        out[:, 3] = np.where(has, np.log1p(n) / 3.0, np.nan)
        return out

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        f = self.csv_dir / f"google_{ticker}.csv"
        if not f.exists():
            return None
        news = pd.read_csv(f)
        if "title" not in news.columns or "date" not in news.columns or len(news) == 0:
            return None
        news = news.dropna(subset=["title", "date"]).sort_values("date")
        scored = self.score_texts(ticker, news["title"].astype(str).tolist())
        return self._features(self._daily(news["date"], scored), df.index)

    def compute_latest(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        items = []
        if self.ctx.news is not None:
            try:
                items = self.ctx.news.fetch(ticker, self.lookback_days)
            except Exception as e:  # noqa: BLE001
                log.debug("finbert news fetch %s: %s", ticker, e)
        if not items:
            hist = self.compute_history(ticker, df)
            if hist is None or len(hist) == 0 or np.isnan(hist[-1]).any():
                return None
            return hist[-1]
        texts = [it.title for it in items if it.title]
        scored = self.score_texts(ticker, texts)
        dates = pd.Series([it.ts.tz_localize(None) if it.ts.tzinfo else it.ts for it in items if it.title])
        daily = self._daily(dates, scored)
        end = pd.Timestamp(df.index[-1]) if len(df) else pd.Timestamp.utcnow().normalize()
        end = max(end, daily.index.max()) if len(daily) else end
        feats = self._features(daily, pd.DatetimeIndex([end]))
        row = feats[-1]
        return None if np.isnan(row).any() else row
