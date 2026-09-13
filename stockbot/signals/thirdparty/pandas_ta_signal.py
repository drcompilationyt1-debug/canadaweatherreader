"""pandas-ta (community fork ``pandas-ta-classic``) - 150+ indicators; we take the ones the native
technical block does not already have: SuperTrend direction, TTM squeeze, Vortex, KST, CCI, MFI,
Chaikin money flow, awesome oscillator, TSI, ultimate oscillator, efficiency ratio and the
parabolic SAR side.  Column names differ between pandas-ta versions, so columns are found by
prefix.  Everything is scaled to roughly -1..1 for the policy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...logging_utils import get_logger
from ...paths import import_optional
from ..base import SignalProvider

log = get_logger(__name__)

FEATURES = ["pta_supertrend", "pta_squeeze", "pta_vortex", "pta_kst", "pta_cci", "pta_mfi", "pta_cmf", "pta_ao",
            "pta_tsi", "pta_uo", "pta_er", "pta_psar"]


def _ta():
    return import_optional("pandas_ta_classic") or import_optional("pandas_ta")


def _col(df: pd.DataFrame | pd.Series | None, prefix: str) -> pd.Series | None:
    if df is None:
        return None
    if isinstance(df, pd.Series):
        return df
    for c in df.columns:
        if str(c).startswith(prefix):
            return df[c]
    return None


class PandasTASignal(SignalProvider):
    name = "pandas_ta"
    feature_names = list(FEATURES)
    tier = "B"

    def availability(self) -> tuple[bool, str]:
        ta = _ta()
        if ta is None:
            return False, "pip install pandas-ta-classic (third_party/pandas-ta-classic)"
        return True, f"pandas-ta {getattr(ta, '__version__', '?')}: 12 extra indicators"

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        ta = _ta()
        if ta is None:
            return None
        o, h, l, c, v = (df[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
        n = len(df)
        out = pd.DataFrame(np.nan, index=df.index, columns=FEATURES, dtype=float)

        def safe(fn, *a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as e:  # noqa: BLE001
                log.debug("pandas_ta %s failed: %s", getattr(fn, "__name__", fn), e)
                return None

        st = _col(safe(ta.supertrend, h, l, c, length=10, multiplier=3.0), "SUPERTd")
        if st is not None:
            out["pta_supertrend"] = st.to_numpy(float)
        sq = _col(safe(ta.squeeze, h, l, c), "SQZ_ON")
        if sq is not None:
            out["pta_squeeze"] = sq.to_numpy(float) * 2.0 - 1.0
        vx = safe(ta.vortex, h, l, c, length=14)
        vp, vm = _col(vx, "VTXP"), _col(vx, "VTXM")
        if vp is not None and vm is not None:
            out["pta_vortex"] = np.clip((vp - vm).to_numpy(float) * 3.0, -3, 3)
        ks = safe(ta.kst, c)
        k, ksig = _col(ks, "KST_"), _col(ks, "KSTs")
        if k is not None and ksig is not None:
            out["pta_kst"] = np.clip((k - ksig).to_numpy(float) / 20.0, -3, 3)
        cci = _col(safe(ta.cci, h, l, c, length=20), "CCI")
        if cci is not None:
            out["pta_cci"] = np.clip(cci.to_numpy(float) / 100.0, -3, 3)
        mfi = _col(safe(ta.mfi, h, l, c, v, length=14), "MFI")
        if mfi is not None:
            out["pta_mfi"] = mfi.to_numpy(float) / 50.0 - 1.0
        cmf = _col(safe(ta.cmf, h, l, c, v, length=20), "CMF")
        if cmf is not None:
            out["pta_cmf"] = np.clip(cmf.to_numpy(float) * 3.0, -3, 3)
        ao = _col(safe(ta.ao, h, l), "AO")
        if ao is not None:
            out["pta_ao"] = np.clip((ao / c).to_numpy(float) * 20.0, -3, 3)
        tsi = _col(safe(ta.tsi, c), "TSI_")
        if tsi is not None:
            out["pta_tsi"] = np.clip(tsi.to_numpy(float) / 30.0, -3, 3)
        uo = _col(safe(ta.uo, h, l, c), "UO")
        if uo is not None:
            out["pta_uo"] = uo.to_numpy(float) / 50.0 - 1.0
        er = _col(safe(ta.er, c, length=10), "ER")
        if er is not None:
            out["pta_er"] = er.to_numpy(float)
        ps = safe(ta.psar, h, l, c)
        psl = _col(ps, "PSARl")
        if psl is not None:
            out["pta_psar"] = np.where(psl.notna().to_numpy(), 1.0, -1.0)
            out.loc[out.index[:5], "pta_psar"] = np.nan
        arr = out.to_numpy(np.float32)
        if n and np.isnan(arr).all():
            return None
        return arr
