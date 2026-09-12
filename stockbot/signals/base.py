"""Signal providers: every "input model" exposes the same tiny interface.

A provider emits a fixed-size block of features for each bar.  When it cannot run (missing
dependency, missing API key, quota exhausted, no data...) it returns ``None`` and the
registry writes zeros plus an availability flag of 0 into the observation, so the policy
always sees the same layout and simply learns to ignore blocks that are switched off.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from ..llm.router import LLMRouter
    from ..news.fetcher import NewsFetcher

log = get_logger(__name__)


@dataclass
class SignalContext:
    """Shared services handed to every provider."""

    cfg: Config
    models_dir: Path
    llm: "LLMRouter | None" = None
    news: "NewsFetcher | None" = None
    extra: dict[str, Any] = field(default_factory=dict)

    def state_dir(self) -> Path:
        d = self.models_dir / "signals"
        d.mkdir(parents=True, exist_ok=True)
        return d


class SignalProvider:
    """Base class.  Subclasses set ``name`` and ``feature_names`` and implement ``compute_history``."""

    name: str = "base"
    feature_names: list[str] = []
    live_only: bool = False        # provider has no historical values (e.g. LLM agent frameworks)
    needs_universe: bool = False   # provider is cross-sectional: prefers ``compute_history_all``
    tier: str = "A"

    def __init__(self, cfg: Config, ctx: SignalContext):
        self.cfg = cfg.section(f"signals.{self.name}")
        self.root_cfg = cfg
        self.ctx = ctx

    # ------------------------------------------------------------------ meta
    @property
    def size(self) -> int:
        return len(self.feature_names)

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", True))

    def availability(self) -> tuple[bool, str]:
        """(available, human readable reason)."""
        return True, "ok"

    def state_path(self, filename: str) -> Path:
        return self.ctx.state_dir() / filename

    # ------------------------------------------------------------------ fitting / persistence
    def fit(self, frames: dict[str, pd.DataFrame], train_end: str | None = None) -> None:
        """Fit provider-internal state on the *training* part of the data (optional)."""

    def save_state(self) -> None:
        """Persist fitted state under ``models/signals`` (optional)."""

    def load_state(self) -> bool:
        """Load fitted state; return True on success (optional)."""
        return True

    # ------------------------------------------------------------------ computation
    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        """Return a (T, size) array aligned to ``df.index`` (NaN rows -> unavailable) or None."""
        raise NotImplementedError

    def compute_history_all(self, frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray | None]:
        out: dict[str, np.ndarray | None] = {}
        for t, df in frames.items():
            out[t] = self.safe_history(t, df)
        return out

    def compute_latest(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        """Features for the most recent bar (live trading).  Default: last row of history."""
        hist = self.compute_history(ticker, df)
        if hist is None or len(hist) == 0:
            return None
        row = hist[-1]
        return None if np.isnan(row).any() else row

    # ------------------------------------------------------------------ guarded wrappers
    def safe_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        ok, why = self.availability()
        if not ok:
            return None
        try:
            arr = self.compute_history(ticker, df)
        except Exception as e:  # noqa: BLE001
            log.warning("signal %s failed for %s: %s", self.name, ticker, e)
            return None
        return self._check(arr, len(df))

    def safe_latest(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        ok, why = self.availability()
        if not ok:
            return None
        try:
            arr = self.compute_latest(ticker, df)
        except Exception as e:  # noqa: BLE001
            log.warning("signal %s (latest) failed for %s: %s", self.name, ticker, e)
            return None
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.size or np.isnan(arr).any():
            return None
        return arr

    def _check(self, arr: np.ndarray | None, n: int) -> np.ndarray | None:
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape != (n, self.size):
            log.warning("signal %s returned shape %s, expected %s - ignoring", self.name, arr.shape, (n, self.size))
            return None
        return arr

    def describe(self) -> dict[str, Any]:
        ok, why = self.availability()
        return {"name": self.name, "tier": self.tier, "enabled": self.enabled, "available": ok,
                "reason": why, "size": self.size, "live_only": self.live_only}
