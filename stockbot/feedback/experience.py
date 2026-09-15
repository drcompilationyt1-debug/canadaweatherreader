"""Experience store: every paper / live decision is logged with its observation, and settled with
the realised outcome on the next cycle.  ``stockbot retrain`` uses this to (a) report how the
policy is doing for real and (b) continue training on data that now includes those days.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)


class ExperienceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _append(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")

    def record(self, *, mode: str, ticker: str, date: str, obs: np.ndarray, action: float, target_exposure: float,
               weight: float, decision: str, price: float, equity: float, availability: dict[str, bool],
               fills: list[dict] | None = None, fees: float = 0.0, conviction: float | None = None,
               chosen: bool | None = None, rank: int | None = None) -> None:
        self._append({
            "type": "decision", "ts": datetime.now(timezone.utc).isoformat(), "mode": mode, "ticker": ticker,
            "date": date, "obs": np.asarray(obs, dtype=float).round(5).tolist(), "action": float(action),
            "target_exposure": float(target_exposure), "weight": float(weight), "decision": decision,
            "price": float(price), "equity": float(equity), "availability": availability, "fills": fills or [],
            "fees": float(fees), "conviction": None if conviction is None else float(conviction),
            "chosen": chosen, "rank": rank,           # the rank layer's verdict: only chosen names are sized by the policy
        })

    def settle(self, ticker: str, date: str, price: float) -> dict | None:
        """Attach the realised log return to the most recent unsettled decision of ``ticker``."""
        recs = self.load()
        if recs is None or len(recs) == 0:
            return None
        dec = recs[(recs["type"] == "decision") & (recs["ticker"] == ticker)]
        if len(dec) == 0:
            return None
        last = dec.iloc[-1]
        settled = recs[(recs["type"] == "outcome") & (recs["ticker"] == ticker) & (recs["decision_date"] == last["date"])]
        if len(settled) or last["date"] == date or last["price"] <= 0:
            return None
        ret = float(np.log(price / last["price"]))
        out = {"type": "outcome", "ts": datetime.now(timezone.utc).isoformat(), "ticker": ticker, "decision_date": last["date"],
               "date": date, "price_then": float(last["price"]), "price_now": float(price), "asset_log_return": ret,
               "weight": float(last["weight"]), "pnl_log_return": ret * float(last["weight"])}
        self._append(out)
        return out

    def load(self) -> pd.DataFrame | None:
        if not self.path.exists():
            return None
        rows = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if not rows:
            return None
        df = pd.DataFrame(rows)
        for col in ("decision_date", "weight", "price", "pnl_log_return", "asset_log_return"):
            if col not in df.columns:
                df[col] = np.nan
        return df

    def summary(self) -> dict:
        df = self.load()
        if df is None:
            return {"decisions": 0, "outcomes": 0}
        dec = df[df["type"] == "decision"]
        out = df[df["type"] == "outcome"]
        res = {"decisions": int(len(dec)), "outcomes": int(len(out)), "tickers": sorted(dec["ticker"].dropna().unique().tolist()) if len(dec) else []}
        if len(out):
            pnl = out["pnl_log_return"].astype(float)
            res.update({
                "realized_pnl_log_return": float(pnl.sum()), "mean_daily_pnl_bps": float(pnl.mean() * 1e4),
                "hit_rate": float((pnl > 0).mean()), "avg_abs_weight": float(out["weight"].abs().mean()),
                "first": str(out["decision_date"].min()), "last": str(out["date"].max()),
            })
        if len(dec):
            res["actions"] = dec["decision"].value_counts().to_dict()
            if "fees" in dec.columns:
                res["fees_paid"] = float(dec["fees"].fillna(0.0).astype(float).sum())
        return res


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
