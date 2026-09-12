from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.strategies import STRATEGY_FEATURES, compute_strategies
from .base import SignalProvider


class StrategyZooSignal(SignalProvider):
    """Tier A - positions of classic strategies ported from Lean, huseinzol05's agents and
    akurgat's indications ("what would the textbook systems do today")."""

    name = "strategy_zoo"
    feature_names = list(STRATEGY_FEATURES)
    tier = "A"

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        return compute_strategies(df).to_numpy(np.float32)
