"""WorldQuant's "101 Formulaic Alphas" (Kakushadze 2015) via yli188/WorldQuant_alpha101_code.

Twelve of the classic price/volume alphas, computed cross-sectionally over the whole universe
(the paper's semantics: ``rank`` compares stocks on the same day).  The submodule's ``rank``
helper ranks over time instead - which would leak the future into every bar - so it is replaced
with the cross-sectional version before use.  Each alpha is reported as its cross-sectional
percentile rank scaled to -1..1, plus their mean as a composite.  Alphas that need market cap or
industry classification are not available and are left out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...logging_utils import get_logger
from ...paths import THIRD_PARTY, load_module_from_file
from ..base import SignalProvider

log = get_logger(__name__)

ALPHAS = ["alpha001", "alpha003", "alpha006", "alpha012", "alpha014", "alpha018", "alpha026", "alpha033", "alpha038",
          "alpha041", "alpha044", "alpha101"]   # the ones that run cleanly in this implementation (005/021/023/027/031/039 do not)
SOURCE = THIRD_PARTY / "WorldQuant_alpha101_code" / "101Alpha_code_1.py"


def _module():
    mod = load_module_from_file("wq_alpha101", SOURCE)
    if mod is not None:
        mod.rank = lambda df: df.rank(axis=1, pct=True)          # cross-sectional, as in the paper
    return mod


class Alpha101Signal(SignalProvider):
    name = "alpha101"
    feature_names = [f"a{a[-3:]}" for a in ALPHAS] + ["a101_mean"]
    needs_universe = True
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.alphas = [a for a in (self.cfg.get("alphas") or ALPHAS) if a in ALPHAS] or ALPHAS
        self._mod = None

    def availability(self) -> tuple[bool, str]:
        if not SOURCE.exists():
            return False, "submodule WorldQuant_alpha101_code not cloned"
        if self._mod is None:
            self._mod = _module()
        if self._mod is None or not hasattr(self._mod, "Alphas"):
            return False, "could not import 101Alpha_code_1.py"
        return True, f"{len(self.alphas)} WorldQuant alphas, cross-sectional ranks"

    @staticmethod
    def _wide(frames: dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
        return pd.DataFrame({t: df[col].astype(float) for t, df in frames.items()}).sort_index()

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        frames = self.ctx.extra.get("frames") or {}
        if ticker not in frames:
            return None
        return self.compute_history_all(frames).get(ticker)

    def compute_history_all(self, frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray | None]:
        ok, _ = self.availability()
        if not ok or not frames:
            return {t: None for t in frames}
        o, h, l, c, v = (self._wide(frames, k) for k in ("open", "high", "low", "close", "volume"))
        data = {"S_DQ_OPEN": o, "S_DQ_HIGH": h, "S_DQ_LOW": l, "S_DQ_CLOSE": c, "S_DQ_VOLUME": v / 100.0,
                "S_DQ_AMOUNT": (c * v) / 1000.0, "S_DQ_PCTCHANGE": c.pct_change()}
        ranks: dict[str, pd.DataFrame] = {}
        for name in self.alphas:
            try:
                stock = self._mod.Alphas({k: d.copy() for k, d in data.items()})   # fresh copy: some alphas mutate inputs
                val = getattr(stock, name)()
                val = pd.DataFrame(val).reindex(index=c.index, columns=c.columns).replace([np.inf, -np.inf], np.nan)
                r = val.rank(axis=1)                                        # 1..n per date
                n = val.notna().sum(axis=1)
                ranks[name] = r.sub(1.0, axis=0).div((n - 1).clip(lower=1), axis=0) * 2.0 - 1.0   # lowest -1, highest +1
            except Exception as e:  # noqa: BLE001 - one broken formula must not mask the whole block
                log.warning("alpha101 %s failed (%s) - reported as neutral 0", name, e)
                ranks[name] = pd.DataFrame(0.0, index=c.index, columns=c.columns)
        if not ranks:
            return {t: None for t in frames}
        composite = pd.concat(ranks.values(), keys=ranks.keys()).groupby(level=1).mean().reindex(c.index)
        out: dict[str, np.ndarray | None] = {}
        for t, df in frames.items():
            cols = [ranks[n][t].reindex(df.index) if n in ranks else pd.Series(np.nan, index=df.index) for n in self.alphas]
            cols.append(composite[t].reindex(df.index))
            arr = pd.concat(cols, axis=1).to_numpy(np.float32)
            out[t] = arr
        return out

    def compute_latest(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        frames = self.ctx.extra.get("frames") or {}
        if ticker not in frames:
            return None
        arr = self.compute_history_all(frames).get(ticker)
        if arr is None or len(arr) == 0:
            return None
        row = arr[-1]
        return None if np.isnan(row).any() else row
