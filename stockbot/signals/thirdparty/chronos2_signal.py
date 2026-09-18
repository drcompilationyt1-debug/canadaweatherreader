"""Chronos-2 (Amazon, 2025) with covariates: the zero-shot forecaster conditioned on the name's own volume and the index.

The first-generation Chronos block forecasts the price path from the price path alone.  Chronos-2 is a single model
that takes past covariates - here log volume and the log index level - and shares information across them, which is
where its reported gains over Chronos-Bolt come from.  It is a heavier model, so this block forecasts every ``stride``-th
bar of the last ``history_years`` (like TimesFM) and keeps its own incremental cache; the original block stays as it is
and the pruner decides which of the two earns its place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...logging_utils import get_logger
from ._cache import IncrementalCache
from .chronos_signal import QUANTILES, ChronosSignal

log = get_logger(__name__)


class Chronos2Signal(ChronosSignal):
    name = "chronos2"
    feature_names = ["c2_ret_5", "c2_ret_20", "c2_up5", "c2_spread5"]
    parallel_ok = False

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.model_id = str(self.cfg.get("model", "amazon/chronos-2"))
        self.history_years = float(self.cfg.get("history_years", 6))
        self.stride = max(1, int(self.cfg.get("stride", 5)))
        self.batch_size = int(self.cfg.get("batch_size", 16))
        self.benchmark = str(self.cfg.get("benchmark", "SPY"))
        self.cache = IncrementalCache(ctx.state_dir() / "chronos2", self.feature_names, version=f"{self.model_id.split('/')[-1]}-{self.context}")
        self._pipe = None

    def availability(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            from chronos import Chronos2Pipeline  # noqa: F401
        except ImportError:
            return False, "pip install chronos-forecasting>=2 (third_party/chronos-forecasting)"
        return True, f"zero-shot {self.model_id.split('/')[-1]} with volume + {self.benchmark} covariates, context {self.context} (history every {self.stride} bars)"

    def _pipeline(self):
        if self._pipe is None:
            import torch
            from chronos import Chronos2Pipeline

            self._pipe = Chronos2Pipeline.from_pretrained(self.model_id, device_map="cpu", torch_dtype=torch.float32)
        return self._pipe

    # ------------------------------------------------------------------ covariates
    def _covariate_series(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        vol = np.log1p(df["volume"].astype(float).clip(lower=0.0)).to_numpy()
        cov = {"log_volume": vol}
        frames = self.ctx.extra.get("frames") or {}
        bench = frames.get(self.benchmark)
        if bench is not None and self.benchmark != getattr(df, "ticker", None):
            b = np.log(bench["close"].astype(float).clip(lower=1e-9)).reindex(df.index).ffill().bfill().to_numpy()
            cov["log_index"] = b
        return cov

    def _forecast_inputs(self, logc: np.ndarray, cov: dict[str, np.ndarray], rows: np.ndarray) -> list[dict]:
        inputs = []
        for r in rows:
            lo = max(0, r + 1 - self.context)
            item = {"target": logc[lo:r + 1].astype(np.float32), "past_covariates": {k: v[lo:r + 1].astype(np.float32) for k, v in cov.items()}}
            inputs.append(item)
        return inputs

    def _forecast_dicts(self, inputs: list[dict]) -> np.ndarray:
        """Chronos-2 quantile forecasts -> the block's 4 features (the same arithmetic as the first-generation block)."""
        pipe = self._pipeline()
        out = np.full((len(inputs), 4), np.nan, dtype=np.float32)
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            qs = pipe.predict_quantiles(batch, prediction_length=20, quantile_levels=QUANTILES)
            if isinstance(qs, tuple):
                qs = qs[0]
            for j, q in enumerate(qs):
                q = np.asarray(q.detach().cpu().numpy() if hasattr(q, "detach") else q, dtype=np.float32)
                if q.ndim == 3:
                    q = q[0]                                            # (n_variates, horizon, quantiles) -> the target
                last = float(batch[j]["target"][-1])
                med5, med20 = q[4, 4], q[19, 4]
                up5 = float((q[4, :] > last).mean())
                spread5 = q[4, 8] - q[4, 0]
                out[i + j] = [np.clip((med5 - last) * 20.0, -5, 5), np.clip((med20 - last) * 10.0, -5, 5), up5 * 2.0 - 1.0,
                              np.clip(spread5 * 20.0, 0, 5)]
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
            eligible = np.array(sorted(set(eligible[::-self.stride].tolist())), dtype=int)
        missing = self.cache.missing_dates(ticker, idx[eligible])
        if len(missing) > 1 and self.ctx.extra.get("latest_only"):
            missing = missing[-1:]
        if len(missing):
            rows = np.array([idx.get_loc(d) for d in missing], dtype=int)
            feats = self._forecast_dicts(self._forecast_inputs(logc, self._covariate_series(df), rows))
            self.cache.merge(ticker, pd.DataFrame(feats, index=idx[rows], columns=self.feature_names))
            log.info("chronos2 %s: %d new forecasts", ticker, len(rows))
        return self.cache.aligned(ticker, idx, ffill_limit=self.stride - 1)

    def compute_latest(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        if len(df) <= self.context:
            return None
        logc = np.log(df["close"].astype(float).clip(lower=1e-9)).to_numpy()
        return self._forecast_dicts(self._forecast_inputs(logc, self._covariate_series(df), np.array([len(df) - 1])))[0]
