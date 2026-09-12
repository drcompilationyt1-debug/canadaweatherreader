from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.technical import TECH_FEATURES, compute_technical
from .base import SignalProvider


class TechnicalSignal(SignalProvider):
    """Tier A - classic indicators (RSI, MACD, Bollinger, ATR, moving averages, volume, ...)."""

    name = "technical"
    feature_names = list(TECH_FEATURES)
    tier = "A"

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        return compute_technical(df).to_numpy(np.float32)
