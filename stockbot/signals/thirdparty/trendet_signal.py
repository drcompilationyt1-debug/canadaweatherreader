"""trendet adapter (alvarobartt/trendet) - running-mean trend runs.

trendet labels a trend only once it has ended, which would leak the future into a feature.
This adapter therefore ports trendet's rule in *causal* form: at every bar it reports the length
of the up / down run that is alive right now (the run counter trendet keeps internally).  The
original (non-causal) labelling is exposed through ``label_trends`` for analysis / plots and uses
the real trendet package when it is importable.
"""
from __future__ import annotations

from statistics import mean

import numpy as np
import pandas as pd

from ...logging_utils import get_logger
from ...paths import add_submodule_to_syspath, import_optional
from ..base import SignalProvider

log = get_logger(__name__)


def causal_runs(element: np.ndarray) -> np.ndarray:
    """Length of the trendet run alive at each bar (trendet's inner loop, evaluated online)."""
    out = np.zeros(len(element), dtype=float)
    limit = None
    values: list[float] = []
    for i, value in enumerate(element):
        if np.isnan(value):
            limit, values = None, []
            continue
        if limit is not None and limit > value:
            values.append(value)
            limit = mean(values)
        elif limit is not None and limit < value:
            limit, values = None, []
        else:
            values = [value]
            limit = value
        out[i] = len(values)
    return out


def trendet_labels_native(values: np.ndarray, window_size: int = 5) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Non-causal port of ``trendet.identify_df_trends`` returning (up_trends, down_trends) index spans."""

    def find(element):
        limit, vals, trends, from_trend = None, [], [], 0
        for index, value in enumerate(element):
            if limit is not None and limit > value:
                vals.append(value)
                limit = mean(vals)
            elif limit is not None and limit < value:
                if len(vals) > window_size:
                    counter = vals.index(min(vals))
                    trends.append((from_trend, from_trend + counter))
                limit, vals = None, []
            else:
                from_trend = index
                vals.append(value)
                limit = mean(vals)
        return trends

    up = find(-np.asarray(values, float))
    down = find(np.asarray(values, float))

    def overlaps(a, b):
        return b[0] < a[0] < b[1] or b[0] < a[1] < b[1]

    up_keep = [u for u in up if all(not overlaps(u, d) or (u[1] - u[0]) > (d[1] - d[0]) for d in down)]
    down_keep = [d for d in down if all(not overlaps(d, u) or (d[1] - d[0]) >= (u[1] - u[0]) for u in up)]
    return up_keep, down_keep


def label_trends(df: pd.DataFrame, column: str = "close", window_size: int = 5) -> pd.DataFrame:
    """Return df with 'Up Trend' / 'Down Trend' label columns (real trendet if importable)."""
    add_submodule_to_syspath("trendet")
    mod = import_optional("trendet")
    if mod is not None:
        try:
            return mod.identify_df_trends(df=df.copy(), column=column, window_size=window_size, identify="both")
        except Exception as e:  # noqa: BLE001
            log.debug("trendet failed, using native port: %s", e)
    up, down = trendet_labels_native(df[column].to_numpy(float), window_size)
    out = df.copy()
    out["Up Trend"] = np.nan
    out["Down Trend"] = np.nan
    for k, (a, b) in enumerate(up):
        out.iloc[a:b + 1, out.columns.get_loc("Up Trend")] = chr(65 + k % 26)
    for k, (a, b) in enumerate(down):
        out.iloc[a:b + 1, out.columns.get_loc("Down Trend")] = chr(65 + k % 26)
    return out


class TrendetSignal(SignalProvider):
    name = "trendet"
    feature_names = ["tt_up", "tt_down", "tt_up_len", "tt_down_len"]
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.window_size = int(self.cfg.get("window_size", 5))

    def availability(self) -> tuple[bool, str]:
        add_submodule_to_syspath("trendet")
        real = import_optional("trendet") is not None
        return True, "causal port of trendet runs" + (" (trendet package importable)" if real else " (trendet needs investpy+unidecode; native port used)")

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        close = df["close"].to_numpy(float)
        up = causal_runs(-close)
        down = causal_runs(close)
        out = np.column_stack([
            (up > self.window_size).astype(float),
            (down > self.window_size).astype(float),
            np.minimum(up / 20.0, 2.0),
            np.minimum(down / 20.0, 2.0),
        ]).astype(np.float32)
        out[: self.window_size] = np.nan
        return out
