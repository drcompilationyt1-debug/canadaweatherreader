"""Chronos-Bolt (amazon-science/chronos-forecasting) - a zero-shot time-series foundation model.

The model was pre-trained on ~100B time points from many domains and forecasts a series it has
never seen, with quantiles, in a single forward pass.  For every bar we hand it the last
``context`` log closes and read the median forecast 5 and 20 bars out, the probability of an up
move (share of quantiles above today's close) and the width of the 10-90 % band.  Chronos-Bolt
small runs at a few hundred forecasts per second on a CPU, so the whole history is computed and
cached (``models/signals/chronos``); live trading only forecasts the newest bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...logging_utils import get_logger
from ..base import SignalProvider
from ._cache import IncrementalCache

log = get_logger(__name__)

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class ChronosSignal(SignalProvider):
    name = "chronos"
    feature_names = ["chr_ret_5", "chr_ret_20", "chr_up_5", "chr_spread_5"]
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.model_id = str(self.cfg.get("model", "amazon/chronos-bolt-small"))
        self.context = int(self.cfg.get("context", 256))
        self.batch_size = int(self.cfg.get("batch_size", 64))
        self.history_years = float(self.cfg.get("history_years", 0) or 0)   # 0 = every bar
        self.stride = max(1, int(self.cfg.get("stride", 1)))
        self._pipe = None
        self.cache = IncrementalCache(ctx.state_dir() / "chronos", self.feature_names, version=f"{self.model_id.split('/')[-1]}-{self.context}")

    # ------------------------------------------------------------------ model
    def availability(self) -> tuple[bool, str]:
        try:
            import chronos  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False, "pip install chronos-forecasting (third_party/chronos-forecasting)"
        return True, f"zero-shot {self.model_id.split('/')[-1]}, context {self.context}"

    def _pipeline(self):
        if self._pipe is None:
            import torch
            from chronos import BaseChronosPipeline

            self._pipe = BaseChronosPipeline.from_pretrained(self.model_id, device_map="cpu", torch_dtype=torch.float32)
        return self._pipe

    def _forecast(self, windows: np.ndarray) -> np.ndarray:
        """(n, context) log-price windows -> (n, 4) features."""
        import torch

        pipe = self._pipeline()
        out = np.full((len(windows), 4), np.nan, dtype=np.float32)
        for i in range(0, len(windows), self.batch_size):
            batch = torch.tensor(windows[i:i + self.batch_size], dtype=torch.float32)
            q, _mean = pipe.predict_quantiles(batch, prediction_length=20, quantile_levels=QUANTILES)
            q = q.detach().cpu().numpy()                      # (b, 20, 9)
            last = windows[i:i + self.batch_size, -1][:, None]
            med5, med20 = q[:, 4, 4], q[:, 19, 4]
            up5 = (q[:, 4, :] > last).mean(axis=1)
            spread5 = q[:, 4, 8] - q[:, 4, 0]
            out[i:i + len(q), 0] = np.clip((med5 - last[:, 0]) * 20.0, -5, 5)
            out[i:i + len(q), 1] = np.clip((med20 - last[:, 0]) * 10.0, -5, 5)
            out[i:i + len(q), 2] = up5 * 2.0 - 1.0
            out[i:i + len(q), 3] = np.clip(spread5 * 20.0, 0, 5)
        return out

    # ------------------------------------------------------------------ features
    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        logc = np.log(df["close"].astype(float).clip(lower=1e-9)).to_numpy()
        idx = df.index
        eligible = np.arange(self.context, len(df))
        if self.history_years > 0:
            cutoff = idx[-1] - pd.Timedelta(days=365.25 * self.history_years)
            eligible = eligible[idx[eligible] >= cutoff]
        if self.stride > 1:
            keep = set(eligible[::-self.stride].tolist())          # always includes the newest bar
            eligible = np.array(sorted(keep), dtype=int)
        missing = self.cache.missing_dates(ticker, idx[eligible])
        if len(missing):
            rows = np.array([idx.get_loc(d) for d in missing], dtype=int)
            windows = np.stack([logc[r - self.context:r + 1][-self.context:] for r in rows])
            feats = self._forecast(windows)
            self.cache.merge(ticker, pd.DataFrame(feats, index=idx[rows], columns=self.feature_names))
            log.info("chronos %s: %d new forecasts", ticker, len(rows))
        return self.cache.aligned(ticker, idx, ffill_limit=self.stride - 1)

    def compute_latest(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        if len(df) <= self.context:
            return None
        cached = self.cache.load(ticker)
        if cached is not None and df.index[-1] in cached.index:
            row = cached.loc[df.index[-1]].to_numpy(np.float32)
        else:
            logc = np.log(df["close"].astype(float).clip(lower=1e-9)).to_numpy()
            row = self._forecast(logc[-self.context:][None, :])[0]
            self.cache.merge(ticker, pd.DataFrame(row[None, :], index=df.index[-1:], columns=self.feature_names))
        return None if np.isnan(row).any() else row
