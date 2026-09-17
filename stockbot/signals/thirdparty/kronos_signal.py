"""Kronos (shiyu-coder/Kronos, AAAI 2026) - a foundation model pre-trained on candlesticks.

Kronos tokenises OHLCV bars from 45 exchanges and autoregressively predicts the next bars.  We
feed it the last ``context`` daily bars and read the predicted 1-bar and 5-bar log return of the
close and the average predicted range.  It is sampled (autoregressive), so history is computed on
every ``stride``-th bar of the last ``history_years`` and forward-filled; live trading predicts
only the newest bar.  Results are cached under ``models/signals/kronos``.  The repo is not a
package: it is imported from the ``third_party/Kronos`` submodule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...logging_utils import get_logger
from ...paths import add_submodule_to_syspath, third_party_dir
from ..base import SignalProvider
from ._cache import IncrementalCache

log = get_logger(__name__)

TOKENIZER_FOR = {"NeoQuasar/Kronos-mini": "NeoQuasar/Kronos-Tokenizer-2k",
                 "NeoQuasar/Kronos-small": "NeoQuasar/Kronos-Tokenizer-base",
                 "NeoQuasar/Kronos-base": "NeoQuasar/Kronos-Tokenizer-base"}


class KronosSignal(SignalProvider):
    name = "kronos"
    parallel_ok = False            # runs its own threads (torch / TensorFlow): one ticker at a time
    feature_names = ["kr_ret_1", "kr_ret_5", "kr_range_5"]
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.model_id = str(self.cfg.get("model", "NeoQuasar/Kronos-small"))
        self.tokenizer_id = str(self.cfg.get("tokenizer") or TOKENIZER_FOR.get(self.model_id, "NeoQuasar/Kronos-Tokenizer-base"))
        self.context = int(self.cfg.get("context", 400))
        self.pred_len = int(self.cfg.get("pred_len", 5))
        self.temperature = float(self.cfg.get("temperature", 0.8))   # <1 = less sampling noise in the predicted path
        self.batch_size = int(self.cfg.get("batch_size", 32))
        self.history_years = float(self.cfg.get("history_years", 6))
        self.stride = max(1, int(self.cfg.get("stride", 5)))
        self._predictor = None
        self.cache = IncrementalCache(ctx.state_dir() / "kronos", self.feature_names, version=f"{self.model_id.split('/')[-1]}-{self.context}")

    def availability(self) -> tuple[bool, str]:
        if third_party_dir("Kronos") is None:
            return False, "submodule Kronos not cloned (python scripts/setup_submodules.py --only Kronos)"
        try:
            import einops  # noqa: F401
            import huggingface_hub  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False, "pip install einops huggingface_hub safetensors (Kronos requirements)"
        return True, f"{self.model_id.split('/')[-1]} K-line foundation model, {self.pred_len}-bar paths (history every {self.stride} bars)"

    def _get_predictor(self):
        if self._predictor is None:
            add_submodule_to_syspath("Kronos")
            from model import Kronos, KronosPredictor, KronosTokenizer

            tok = KronosTokenizer.from_pretrained(self.tokenizer_id)
            mdl = Kronos.from_pretrained(self.model_id)
            self._predictor = KronosPredictor(mdl, tok, device="cpu", max_context=min(512, self.context))
        return self._predictor

    @staticmethod
    def _bars(df: pd.DataFrame) -> pd.DataFrame:
        x = df[["open", "high", "low", "close", "volume"]].astype(float).copy()
        x["amount"] = x["close"] * x["volume"]
        return x

    def _forecast(self, df: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
        """Predict ``pred_len`` bars after each row index in ``rows`` -> (n, 3) features."""
        pred = self._get_predictor()
        bars = self._bars(df)
        out = np.full((len(rows), 3), np.nan, dtype=np.float32)
        for i in range(0, len(rows), self.batch_size):
            chunk = rows[i:i + self.batch_size]
            xs, xts, yts = [], [], []
            for r in chunk:
                x = bars.iloc[r + 1 - self.context:r + 1]
                xs.append(x.reset_index(drop=True))
                xts.append(pd.Series(x.index))
                yts.append(pd.Series(pd.bdate_range(x.index[-1] + pd.Timedelta(days=1), periods=self.pred_len)))
            preds = pred.predict_batch(df_list=xs, x_timestamp_list=xts, y_timestamp_list=yts, pred_len=self.pred_len,
                                       T=self.temperature, top_p=0.9, sample_count=1, verbose=False)
            for k, (r, p) in enumerate(zip(chunk, preds)):
                last = float(bars["close"].iloc[r])
                c = p["close"].to_numpy(float)
                rng = ((p["high"] - p["low"]) / p["close"].clip(lower=1e-9)).to_numpy(float)
                if last <= 0 or not np.all(np.isfinite(c)) or c.min() <= 0:
                    continue
                out[i + k, 0] = np.clip(np.log(c[0] / last) * 20.0, -5, 5)
                out[i + k, 1] = np.clip(np.log(c[min(4, len(c) - 1)] / last) * 20.0, -5, 5)
                out[i + k, 2] = np.clip(float(np.nanmean(rng)) * 30.0, 0, 5)
        return out

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        idx = df.index
        eligible = np.arange(self.context, len(df))
        if self.history_years > 0:
            cutoff = idx[-1] - pd.Timedelta(days=365.25 * self.history_years)
            eligible = eligible[idx[eligible] >= cutoff]
        if self.stride > 1:
            eligible = np.array(sorted(set(eligible[::-self.stride].tolist())), dtype=int)
        missing = self.cache.missing_dates(ticker, idx[eligible])
        if len(missing) > 1 and self.ctx.extra.get("latest_only"):    # a live cycle: today's window now, the history at the weekly build
            missing = missing[-1:]
        if len(missing):
            rows = np.array([idx.get_loc(d) for d in missing], dtype=int)
            feats = self._forecast(df, rows)
            self.cache.merge(ticker, pd.DataFrame(feats, index=idx[rows], columns=self.feature_names))
            log.info("kronos %s: %d new forecasts", ticker, len(rows))
        return self.cache.aligned(ticker, idx, ffill_limit=self.stride - 1)

    def compute_latest(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        if len(df) <= self.context:
            return None
        cached = self.cache.load(ticker)
        if cached is not None and df.index[-1] in cached.index:
            row = cached.loc[df.index[-1]].to_numpy(np.float32)
        else:
            row = self._forecast(df, np.array([len(df) - 1]))[0]
            self.cache.merge(ticker, pd.DataFrame(row[None, :], index=df.index[-1:], columns=self.feature_names))
        return None if np.isnan(row).any() else row
