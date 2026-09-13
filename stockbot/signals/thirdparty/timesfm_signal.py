"""TimesFM (google-research/timesfm) - Google's decoder-only time-series foundation model.

Zero-shot like Chronos, but a different architecture and training corpus, so its errors are not
the same errors: the policy gets a second opinion.  The 200M checkpoint is heavier on a CPU
than Chronos-Bolt, so history is computed every ``stride``-th bar of the last ``history_years``
and forward-filled; live trading forecasts only the newest bar.  Cached under
``models/signals/timesfm``.  Works with the ``timesfm`` package's 2.5 torch API and falls back
to the 3.0 API when present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...logging_utils import get_logger
from ..base import SignalProvider
from ._cache import IncrementalCache

log = get_logger(__name__)


class TimesFMSignal(SignalProvider):
    name = "timesfm"
    feature_names = ["tfm_ret_5", "tfm_ret_20", "tfm_spread_5"]
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.model_id = str(self.cfg.get("model", "google/timesfm-2.5-200m-pytorch"))
        self.context = int(self.cfg.get("context", 512))
        self.batch_size = int(self.cfg.get("batch_size", 32))
        self.history_years = float(self.cfg.get("history_years", 6))
        self.stride = max(1, int(self.cfg.get("stride", 5)))
        self._model = None
        self._api = None
        self.cache = IncrementalCache(ctx.state_dir() / "timesfm", self.feature_names, version=f"{self.model_id.split('/')[-1]}-{self.context}")

    def availability(self) -> tuple[bool, str]:
        try:
            import timesfm  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False, "pip install 'timesfm[torch]' (third_party/timesfm)"
        return True, f"zero-shot {self.model_id.split('/')[-1]}, context {self.context} (history every {self.stride} bars)"

    def _load(self):
        if self._model is not None:
            return self._model
        import timesfm

        if hasattr(timesfm, "TimesFM_2p5_200M_torch") and "2.5" in self.model_id:
            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_id)
            m.compile(timesfm.ForecastConfig(max_context=self.context, max_horizon=20, normalize_inputs=True,
                                             use_continuous_quantile_head=True, fix_quantile_crossing=True,
                                             per_core_batch_size=self.batch_size))
            self._api = "2.5"
        elif hasattr(timesfm, "TimesFM3Forecaster"):
            from timesfm import ModelConfig, TimesFM3Forecaster

            m = TimesFM3Forecaster(ModelConfig(checkpoint_path=self.model_id, per_core_batch_size=self.batch_size, device="cpu"))
            self._api = "3.0"
        else:
            raise RuntimeError("unsupported timesfm package version")
        self._model = m
        return m

    def _forecast(self, windows: np.ndarray) -> np.ndarray:
        """(n, context) log-price windows -> (n, 3) features."""
        m = self._load()
        out = np.full((len(windows), 3), np.nan, dtype=np.float32)
        for i in range(0, len(windows), self.batch_size):
            batch = [w.astype(np.float32) for w in windows[i:i + self.batch_size]]
            if self._api == "2.5":
                point, quant = m.forecast(horizon=20, inputs=batch)          # (b, 20), (b, 20, 10) [mean + 9 quantiles]
                point, quant = np.asarray(point), np.asarray(quant)
                q10 = quant[:, 4, 1] if quant.shape[-1] >= 10 else point[:, 4]
                q90 = quant[:, 4, 9] if quant.shape[-1] >= 10 else point[:, 4]
            else:
                res = list(m.predict_batch(contexts=batch, horizon=20, return_quantiles=True))
                point = np.stack([np.asarray(r.forecast).reshape(-1)[:20] for r in res])
                qs = np.stack([np.asarray(r.quantiles).reshape(20, -1) for r in res])
                q10, q90 = qs[:, 4, 0], qs[:, 4, -1]
            last = windows[i:i + len(point), -1]
            out[i:i + len(point), 0] = np.clip((point[:, 4] - last) * 20.0, -5, 5)
            out[i:i + len(point), 1] = np.clip((point[:, 19] - last) * 10.0, -5, 5)
            out[i:i + len(point), 2] = np.clip(np.abs(q90 - q10) * 20.0, 0, 5)
        return out

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
        if len(missing):
            rows = np.array([idx.get_loc(d) for d in missing], dtype=int)
            windows = np.stack([logc[r + 1 - self.context:r + 1] for r in rows])
            feats = self._forecast(windows)
            self.cache.merge(ticker, pd.DataFrame(feats, index=idx[rows], columns=self.feature_names))
            log.info("timesfm %s: %d new forecasts", ticker, len(rows))
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
