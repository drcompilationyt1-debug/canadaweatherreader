"""Per-ticker incremental feature cache for the expensive third-party models.

Foundation-model forecasts and transformer sentiment scores are slow to compute over 18 years of
bars, so every adapter stores what it computed in ``models/signals/<name>/<ticker>.parquet``
(date index, one column per feature) and only computes the dates it has not seen.  The folder
lives under ``models/signals`` so the GitHub ``state`` branch carries it between runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class IncrementalCache:
    def __init__(self, folder: str | Path, columns: list[str], version: str = "1"):
        self.folder = Path(folder)
        self.columns = list(columns)
        self.version = str(version)

    def path(self, ticker: str) -> Path:
        safe = ticker.replace("/", "_").replace("^", "idx_")
        return self.folder / f"{safe}_v{self.version}.parquet"

    def load(self, ticker: str) -> pd.DataFrame | None:
        p = self.path(ticker)
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            if list(df.columns) != self.columns:
                return None
            return df[~df.index.duplicated(keep="last")].sort_index()
        except Exception:  # noqa: BLE001
            return None

    def save(self, ticker: str, df: pd.DataFrame) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        out = df[self.columns].copy()
        out.index = pd.to_datetime(out.index)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out.to_parquet(self.path(ticker))

    def merge(self, ticker: str, new: pd.DataFrame) -> pd.DataFrame:
        """Union of the cached rows and ``new`` (new wins), saved and returned."""
        old = self.load(ticker)
        df = new if old is None or len(old) == 0 else pd.concat([old, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        self.save(ticker, df)
        return df

    def aligned(self, ticker: str, index: pd.DatetimeIndex, ffill_limit: int = 0) -> np.ndarray:
        """Cached rows re-indexed to ``index`` (NaN where unknown), optionally forward-filled a few bars."""
        df = self.load(ticker)
        if df is None:
            return np.full((len(index), len(self.columns)), np.nan, dtype=np.float32)
        out = df.reindex(index)
        if ffill_limit > 0:
            out = out.ffill(limit=ffill_limit)
        return out.to_numpy(np.float32)

    def missing_dates(self, ticker: str, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        df = self.load(ticker)
        if df is None:
            return index
        return index.difference(df.index)
