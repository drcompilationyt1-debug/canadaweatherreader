"""Reliability: how right each input has been lately, as an input in its own right.

The post-trade review keeps finding which models called the moves; this block hands that lesson
to the policy every day so it can learn to lean on inputs that are currently working and ignore
the ones that are not (signal-decay monitoring, in quant terms).  For every direction voter
(the same list ``stockbot forecast`` scores) it computes, per ticker, the trailing hit rate of
the voter's sign against the realised ``horizon``-day return over the last ``window`` settled
bars - using only outcomes that were known at the time - scaled to -1..1, plus how many voters
have a track record.  Runs last, reads the other blocks from ``ctx.extra["per_block"]`` during
the dataset build and serves the cached latest row live (refreshed by the weekend retrain).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..feedback.voters import VOTERS
from ..logging_utils import get_logger
from .base import SignalProvider
from .thirdparty._cache import IncrementalCache

log = get_logger(__name__)

VOTER_NAMES = [v for v in VOTERS if v != "policy"]


class ReliabilitySignal(SignalProvider):
    name = "reliability"
    feature_names = [f"rel_{v}" for v in VOTER_NAMES] + ["rel_mean", "rel_n"]
    needs_universe = True
    tier = "A"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.horizon = int(self.cfg.get("horizon", 5))
        self.window = int(self.cfg.get("window", 60))
        self.min_obs = int(self.cfg.get("min_obs", 10))
        self.cache = IncrementalCache(ctx.state_dir() / "reliability", self.feature_names, version=f"h{self.horizon}w{self.window}")

    def availability(self) -> tuple[bool, str]:
        return True, f"trailing {self.window}-bar hit rate of {len(VOTER_NAMES)} inputs at {self.horizon} days"

    # ------------------------------------------------------------------ history (build)
    def _series(self, per_block: dict, ticker: str, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        logc = np.log(df["close"].astype(float).clip(lower=1e-9)).to_numpy()
        fwd = np.full(n, np.nan)
        fwd[:-self.horizon] = logc[self.horizon:] - logc[:-self.horizon]
        out = np.zeros((n, len(self.feature_names)), dtype=np.float32)
        known = np.zeros(n, dtype=np.float32)
        for k, voter in enumerate(VOTER_NAMES):
            block, feature = VOTERS[voter]
            arrs = per_block.get(block)
            a = arrs.get(ticker) if arrs else None
            if a is None or a.ndim != 2:
                continue
            spec = self._feature_index(block, feature, a.shape[1])
            if spec is None:
                continue
            v = a[:, spec]
            valid = ~np.isnan(a).any(axis=1) & (np.abs(v) > 1e-6) & ~np.isnan(fwd)
            hit = np.where(valid, (np.sign(v) == np.sign(fwd)).astype(float), np.nan)
            # the outcome of bar s is known at s + horizon: shift, then trailing mean over the window
            s = pd.Series(hit).shift(self.horizon)
            rate = s.rolling(self.window, min_periods=self.min_obs).mean().to_numpy()
            ok = ~np.isnan(rate)
            out[ok, k] = (rate[ok] * 2.0 - 1.0).astype(np.float32)
            known += ok
        rel = out[:, : len(VOTER_NAMES)]
        with np.errstate(invalid="ignore"):
            out[:, -2] = np.where(known > 0, rel.sum(axis=1) / np.maximum(known, 1.0), 0.0)
        out[:, -1] = known / max(len(VOTER_NAMES), 1) * 2.0 - 1.0
        out[known < 3] = np.nan            # not enough track record yet -> masked
        return out

    def _feature_index(self, block: str, feature: str, size: int) -> int | None:
        from .registry import PROVIDER_CLASSES

        for klass in PROVIDER_CLASSES:
            if klass.name == block:
                names = list(klass.feature_names)
                if feature in names and len(names) == size:
                    return names.index(feature)
                return None
        return None

    def compute_history_all(self, frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray | None]:
        per_block = self.ctx.extra.get("per_block")
        out: dict[str, np.ndarray | None] = {}
        if per_block:
            for t, df in frames.items():
                arr = self._series(per_block, t, df)
                out[t] = arr
                keep = ~np.isnan(arr).any(axis=1)
                if keep.any():
                    self.cache.save(t, pd.DataFrame(arr[keep], index=df.index[keep], columns=self.feature_names))
            return out
        for t, df in frames.items():        # live: the last cached row (refreshed by the weekend build)
            cached = self.cache.load(t)
            arr = np.full((len(df), len(self.feature_names)), np.nan, dtype=np.float32)
            if cached is not None and len(cached):
                arr[-1] = cached.iloc[-1].to_numpy(np.float32)
            out[t] = arr
        return out

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        frames = self.ctx.extra.get("frames") or {}
        return self.compute_history_all(frames).get(ticker) if ticker in frames else None
