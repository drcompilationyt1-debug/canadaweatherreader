"""Microsoft Qlib adapter.

Qlib models are trained with ``scripts/qlib_train.py`` (Alpha158 -> LightGBM, the standard
qlib workflow) which writes a prediction table ``models/qlib/pred.parquet`` with columns
``datetime, instrument, score``.  This provider turns that table into a signal block:

    qlib_score  - the raw model score (expected short-horizon excess return), scaled
    qlib_rank   - cross-sectional rank of the score among the tickers present that day

If qlib is not installed or no predictions exist, the block is masked out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...logging_utils import get_logger
from ...paths import import_optional, resolve
from ..base import SignalProvider

log = get_logger(__name__)


class QlibSignal(SignalProvider):
    name = "qlib"
    feature_names = ["qlib_score", "qlib_rank"]
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.pred_path = resolve(self.cfg.get("predictions", "models/qlib/pred.parquet"))
        self._table: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame | None:
        if self._table is not None:
            return self._table
        if not self.pred_path.exists():
            return None
        try:
            if self.pred_path.suffix in (".pkl", ".pickle"):
                obj = pd.read_pickle(self.pred_path)
                df = obj.to_frame("score") if isinstance(obj, pd.Series) else obj
                df = df.reset_index()
            else:
                df = pd.read_parquet(self.pred_path)
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index()
            df.columns = [str(c).lower() for c in df.columns]
            score_col = "score" if "score" in df.columns else [c for c in df.columns if c not in ("datetime", "instrument")][0]
            df = df.rename(columns={score_col: "score"})
            df["instrument"] = df["instrument"].astype(str).str.upper().str.replace(".", "-", regex=False)
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
            df["rank"] = df.groupby("datetime")["score"].rank(pct=True) * 2.0 - 1.0
            self._table = df[["datetime", "instrument", "score", "rank"]]
            return self._table
        except Exception as e:  # noqa: BLE001
            log.warning("could not read qlib predictions %s: %s", self.pred_path, e)
            return None

    def availability(self) -> tuple[bool, str]:
        table = self._load()
        if table is None:
            has_qlib = import_optional("qlib", "qlib") is not None
            hint = "run scripts/qlib_train.py" if has_qlib else "pip install pyqlib, then run scripts/qlib_train.py"
            return False, f"no predictions at {self.pred_path} ({hint})"
        return True, f"{len(table)} predictions, {table['instrument'].nunique()} instruments, last {table['datetime'].max().date()}"

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        table = self._load()
        if table is None:
            return None
        sub = table[table["instrument"] == ticker.upper()].set_index("datetime")
        out = np.full((len(df), 2), np.nan, dtype=np.float32)
        if len(sub) == 0:
            return out
        sub = sub[~sub.index.duplicated(keep="last")].reindex(df.index)
        out[:, 0] = np.clip(sub["score"].to_numpy(float) * 50.0, -5, 5)
        out[:, 1] = sub["rank"].to_numpy(float)
        return out
