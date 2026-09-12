from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.trend import TREND_FEATURES, compute_trend
from .base import SignalProvider


class TrendSignal(SignalProvider):
    """Tier A - regression slopes / R^2 / ADX / moving-average regime / trend age."""

    name = "trend"
    feature_names = list(TREND_FEATURES)
    tier = "A"

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        return compute_trend(df).to_numpy(np.float32)
