"""freqtrade adapter: indicators from freqtrade's ``qtpylib`` vendor package.

Recent freqtrade versions re-export ``technical.qtpylib`` (``pip install technical``), the
older ones ship the implementation inline; both paths are tried.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...logging_utils import get_logger
from ...paths import THIRD_PARTY, import_optional, load_module_from_file
from ..base import SignalProvider

log = get_logger(__name__)


def _qtpylib():
    mod = import_optional("technical.qtpylib")
    if mod is not None and hasattr(mod, "heikinashi"):
        return mod
    inline = THIRD_PARTY / "freqtrade" / "freqtrade" / "vendor" / "qtpylib" / "indicators.py"
    mod = load_module_from_file("freqtrade_qtpylib_indicators", inline)
    if mod is not None and hasattr(mod, "heikinashi"):
        return mod
    return None


class FreqtradeSignal(SignalProvider):
    name = "freqtrade"
    feature_names = ["ha_streak", "hma_ratio", "vwap_ratio"]
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self._mod = None
        self._checked = False

    def availability(self) -> tuple[bool, str]:
        if not self._checked:
            self._mod = _qtpylib()
            self._checked = True
        if self._mod is None:
            return False, "pip install technical (freqtrade's qtpylib indicators)"
        return True, "qtpylib heikin-ashi / hull MA / vwap"

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        q = self._mod
        # qtpylib indexes bars positionally -> give it a RangeIndex and restore df.index on output
        bars = df[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
        ha = q.heikinashi(bars)
        up = (ha["close"] > ha["open"]).to_numpy()
        streak = np.zeros(len(df))
        for i in range(1, len(df)):
            if up[i] == up[i - 1]:
                streak[i] = streak[i - 1] + (1 if up[i] else -1)
            else:
                streak[i] = 1 if up[i] else -1
        hma = q.hull_moving_average(bars["close"], window=20)
        vwap = q.rolling_vwap(bars, window=20)
        out = pd.DataFrame(index=df.index)
        out["ha_streak"] = np.clip(streak / 5.0, -3, 3)
        out["hma_ratio"] = ((bars["close"] / pd.Series(np.asarray(hma, float)) - 1.0) * 20.0).to_numpy()
        out["vwap_ratio"] = ((bars["close"] / pd.Series(np.asarray(vwap, float)) - 1.0) * 20.0).to_numpy()
        return out.replace([np.inf, -np.inf], np.nan).clip(-5, 5).to_numpy(np.float32)
