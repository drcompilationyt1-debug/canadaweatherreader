from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.candles import CANDLE_FEATURES, PatternStats, detect_patterns, pattern_features
from ..logging_utils import get_logger
from .base import SignalProvider

log = get_logger(__name__)


class CandleSignal(SignalProvider):
    """Tier A - candlestick patterns + empirical forward-return statistics per pattern."""

    name = "candles"
    feature_names = list(CANDLE_FEATURES)
    tier = "A"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.horizons = [int(h) for h in self.cfg.get("horizons", [1, 5, 20])]
        self.decay = [float(d) for d in self.cfg.get("decay", [1.0, 0.6, 0.3])]
        self.stats = PatternStats(horizons=self.horizons)

    def fit(self, frames: dict[str, pd.DataFrame], train_end: str | None = None) -> None:
        self.stats = PatternStats(horizons=self.horizons).fit(frames, end=train_end)
        log.info("candle pattern stats fitted on %d tickers (%d bars)", len(frames), self.stats.n_bars)

    def save_state(self) -> None:
        if self.stats.fitted:
            self.stats.save(self.state_path("pattern_stats.json"))

    def load_state(self) -> bool:
        p = self.state_path("pattern_stats.json")
        if p.exists():
            try:
                self.stats = PatternStats.load(p)
                return True
            except Exception as e:  # noqa: BLE001
                log.warning("could not load pattern stats: %s", e)
        return False

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        if not self.stats.fitted:
            self.load_state()
        pat = detect_patterns(df)
        return pattern_features(pat, self.stats if self.stats.fitted else None, self.decay).to_numpy(np.float32)
