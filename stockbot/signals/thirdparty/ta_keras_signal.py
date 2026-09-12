"""akurgat/automating-technical-analysis adapter - its two trained Keras models, run for real.

The repo ships ``action_prediction_model.h5`` (Buy / Hold / Sell from a 60-day window of 28
indicator features) and ``price_prediction_model.h5`` (next close).  They need TensorFlow, which
does not fit next to PyTorch here, so they run in ``.venv-tf`` through
``scripts/agents/run_ta_keras.py``; results are cached per (ticker, last bar).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...logging_utils import get_logger
from ...paths import ROOT, THIRD_PARTY
from ..base import SignalProvider
from .agent_runner import env_python, run_agent

log = get_logger(__name__)

SCRIPT = ROOT / "scripts" / "agents" / "run_ta_keras.py"
REPO = THIRD_PARTY / "automating-technical-analysis"
ACTION_CODE = {"Buy": 1.0, "Hold": 0.0, "Sell": -1.0}


class TAKerasSignal(SignalProvider):
    name = "ta_keras"
    feature_names = ["akm_action", "akm_p_buy", "akm_p_sell", "akm_price_chg"]
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.timeout = float(self.cfg.get("timeout", 900))
        self.latest_bars = int(self.cfg.get("latest_bars", 400))
        self.cache_dir: Path = ctx.models_dir / "cache" / "ta_keras"

    def _python(self) -> Path | None:
        configured = self.cfg.get("python")
        if configured:
            p = Path(configured)
            p = p if p.is_absolute() else ROOT / p
            return p if p.exists() else None
        return env_python(".venv-tf")

    def availability(self) -> tuple[bool, str]:
        if not (REPO / "models" / "action_prediction_model.h5").exists():
            return False, "submodule automating-technical-analysis not cloned"
        if self._python() is None:
            return False, "run: python scripts/setup_agent_envs.py --only tf   (TensorFlow env for the Keras models)"
        return True, "Keras action + price models in .venv-tf"

    # ------------------------------------------------------------------ running
    def _run(self, ticker: str, df: pd.DataFrame) -> dict:
        key = hashlib.sha1(f"{ticker}|{df.index[0].date()}|{df.index[-1].date()}|{len(df)}".encode()).hexdigest()[:12]
        cache = self.cache_dir / f"{ticker}_{key}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        parquet = self.cache_dir / f"{ticker}_input.parquet"
        df[["open", "high", "low", "close", "volume"]].to_parquet(parquet)
        out_file = self.cache_dir / f"{ticker}_{key}.tmp.json"
        run_agent(self._python(), SCRIPT, ["--parquet", str(parquet), "--out", str(out_file)], timeout=self.timeout)
        res = json.loads(out_file.read_text(encoding="utf-8"))
        out_file.unlink(missing_ok=True)
        if "error" in res:
            raise RuntimeError(res["error"])
        cache.write_text(json.dumps(res), encoding="utf-8")
        for old in self.cache_dir.glob(f"{ticker}_*.json"):  # keep only the newest result per ticker
            if old != cache:
                old.unlink(missing_ok=True)
        return res

    @staticmethod
    def _to_frame(res: dict) -> pd.DataFrame:
        dates = pd.to_datetime(res["dates"])
        close = np.asarray(res["close"], float)
        pred = np.asarray(res["price_pred"], float)
        chg = np.clip((pred / np.maximum(close, 1e-9) - 1.0) * 100.0, -20, 20)
        return pd.DataFrame({
            "akm_action": [ACTION_CODE.get(a, 0.0) for a in res["action"]],
            "akm_p_buy": np.asarray(res["p_buy"], float),
            "akm_p_sell": np.asarray(res["p_sell"], float),
            "akm_price_chg": chg,
        }, index=dates)

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        res = self._run(ticker, df)
        frame = self._to_frame(res)
        frame = frame[~frame.index.duplicated(keep="last")].reindex(df.index)
        return frame.to_numpy(np.float32)

    def compute_latest(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        tail = df.tail(self.latest_bars)
        frame = self._to_frame(self._run(ticker, tail))
        if len(frame) == 0:
            return None
        row = frame.iloc[-1].to_numpy(np.float32)
        return None if np.isnan(row).any() else row
