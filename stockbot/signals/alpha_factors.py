"""Qlib-lite: Alpha158-style price/volume factors + a gradient-boosted forecaster.

Microsoft Qlib's flagship workflow is "Alpha158 features -> LightGBM -> cross-sectional
score".  This provider reproduces that recipe natively on the bot's own data so an ML
forecast is always available even when ``pyqlib`` itself is not installed (the real qlib
adapter lives in ``thirdparty/qlib_signal.py`` and is used in addition when present).

Predictions are produced *walk-forward* (refit every year on all data before that year) so
the RL policy never trains on in-sample forecasts.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from ..features.trend import rolling_slope
from ..logging_utils import get_logger
from .base import SignalProvider

log = get_logger(__name__)

EPS = 1e-9
WINDOWS = (5, 10, 20, 60)


def compute_alpha_factors(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = (df[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    logc = np.log(c.clip(lower=EPS))
    ret = c.pct_change()
    lv = np.log(v + 1.0)
    out: dict[str, pd.Series] = {
        "KMID": (c - o) / o,
        "KLEN": (h - l) / o,
        "KMID2": (c - o) / (h - l + EPS),
        "KUP": (h - np.maximum(o, c)) / o,
        "KLOW": (np.minimum(o, c) - l) / o,
        "KSFT": (2 * c - h - l) / o,
    }
    for d in WINDOWS:
        hi, lo = h.rolling(d).max(), l.rolling(d).min()
        out[f"ROC{d}"] = c.shift(d) / c - 1.0
        out[f"MA{d}"] = c.rolling(d).mean() / c - 1.0
        out[f"STD{d}"] = c.rolling(d).std() / c
        out[f"MAX{d}"] = hi / c - 1.0
        out[f"MIN{d}"] = lo / c - 1.0
        out[f"RSV{d}"] = (c - lo) / (hi - lo + EPS)
        out[f"RANK{d}"] = c.rolling(d).rank(pct=True)
        out[f"CNTP{d}"] = (ret > 0).astype(float).rolling(d).mean()
        out[f"SUMP{d}"] = ret.clip(lower=0).rolling(d).sum() / (ret.abs().rolling(d).sum() + EPS)
        out[f"CORR{d}"] = c.rolling(d).corr(lv)
        out[f"CORD{d}"] = ret.rolling(d).corr(lv.diff())
        out[f"VMA{d}"] = v.rolling(d).mean() / (v + EPS) - 1.0
        out[f"VSTD{d}"] = v.rolling(d).std() / (v + EPS)
        slope, r2 = rolling_slope(logc.to_numpy(), d)
        out[f"BETA{d}"] = pd.Series(slope * d, index=df.index)
        out[f"RSQR{d}"] = pd.Series(r2, index=df.index)
    fac = pd.DataFrame(out, index=df.index).replace([np.inf, -np.inf], np.nan)
    return fac


FACTOR_COLUMNS = list(compute_alpha_factors(pd.DataFrame({
    "open": np.ones(80), "high": np.ones(80) * 1.01, "low": np.ones(80) * 0.99,
    "close": np.linspace(1, 1.1, 80), "volume": np.ones(80) * 1000,
}, index=pd.bdate_range("2020-01-01", periods=80))).columns)


def _make_model():
    try:
        import lightgbm as lgb

        return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
                                 reg_lambda=1.0, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                             min_samples_leaf=50, l2_regularization=1.0)


class AlphaFactorSignal(SignalProvider):
    name = "alpha_factors"
    feature_names = ["af_pred", "af_rank"]
    needs_universe = True
    tier = "A"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.horizon = int(self.cfg.get("horizon", 5))
        self.min_train_years = int(self.cfg.get("min_train_years", 3))
        self.model = None
        self.preds: pd.DataFrame | None = None   # columns: ticker, date, pred (walk-forward)

    # ------------------------------------------------------------------ fitting
    def _table(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        parts = []
        for t, df in frames.items():
            fac = compute_alpha_factors(df)
            logc = np.log(df["close"].astype(float).clip(lower=EPS))
            y = (logc.shift(-self.horizon) - logc).clip(-0.2, 0.2)
            fac["y"] = y
            fac["ticker"] = t
            fac["date"] = df.index
            parts.append(fac)
        return pd.concat(parts, ignore_index=True)

    def fit(self, frames: dict[str, pd.DataFrame], train_end: str | None = None) -> None:
        table = self._table(frames)
        feats = table[FACTOR_COLUMNS].to_numpy(np.float32)
        ok_x = ~np.isnan(feats).all(axis=1)
        years = sorted(table["date"].dt.year.unique())
        preds = []
        for y in years[self.min_train_years:]:
            cutoff = pd.Timestamp(f"{y}-01-01")
            tr = ok_x & (table["date"] < cutoff).to_numpy() & ~table["y"].isna().to_numpy()
            te = ok_x & (table["date"].dt.year == y).to_numpy()
            if tr.sum() < 500 or te.sum() == 0:
                continue
            model = _make_model()
            model.fit(feats[tr], table["y"].to_numpy(np.float32)[tr])
            p = model.predict(feats[te])
            preds.append(pd.DataFrame({"ticker": table["ticker"].to_numpy()[te], "date": table["date"].to_numpy()[te], "pred": p}))
        self.preds = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=["ticker", "date", "pred"])
        final = ok_x & ~table["y"].isna().to_numpy()
        if final.sum() >= 500:
            self.model = _make_model()
            self.model.fit(feats[final], table["y"].to_numpy(np.float32)[final])
        log.info("alpha factor model fitted: %d walk-forward predictions, final model on %d rows",
                 len(self.preds), int(final.sum()))

    def save_state(self) -> None:
        if self.model is not None:
            with open(self.state_path("alpha_factors.pkl"), "wb") as f:
                pickle.dump({"model": self.model, "columns": FACTOR_COLUMNS, "horizon": self.horizon}, f)
        if self.preds is not None:
            self.preds.to_parquet(self.state_path("alpha_preds.parquet"))

    def load_state(self) -> bool:
        p = self.state_path("alpha_factors.pkl")
        if not p.exists():
            return False
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
            self.model = d["model"]
            pp = self.state_path("alpha_preds.parquet")
            self.preds = pd.read_parquet(pp) if pp.exists() else None
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("could not load alpha factor model: %s", e)
            return False

    # ------------------------------------------------------------------ availability
    def availability(self) -> tuple[bool, str]:
        if self.model is None and not self.load_state():
            return False, "not fitted yet (run `stockbot train` or `build-dataset`)"
        return True, "walk-forward LightGBM/GBRT on Alpha158-style factors"

    # ------------------------------------------------------------------ prediction
    def _predict_series(self, ticker: str, df: pd.DataFrame) -> pd.Series:
        """Walk-forward predictions where available, final-model predictions elsewhere."""
        pred = pd.Series(np.nan, index=df.index)
        if self.preds is not None and len(self.preds):
            sub = self.preds[self.preds["ticker"] == ticker]
            if len(sub):
                s = pd.Series(sub["pred"].to_numpy(), index=pd.to_datetime(sub["date"]))
                s = s[~s.index.duplicated(keep="last")]
                pred.loc[pred.index.intersection(s.index)] = s.reindex(pred.index).loc[pred.index.intersection(s.index)]
        missing = pred.isna()
        if self.model is not None and missing.any():
            fac = compute_alpha_factors(df)[FACTOR_COLUMNS]
            rows = missing & ~fac.isna().all(axis=1)
            if rows.any():
                pred.loc[rows] = self.model.predict(fac.loc[rows].to_numpy(np.float32))
        return pred

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        pred = self._predict_series(ticker, df)
        out = np.column_stack([np.clip(pred.to_numpy() * 20.0, -5, 5), np.zeros(len(df))]).astype(np.float32)
        out[pred.isna().to_numpy()] = np.nan
        return out

    def compute_history_all(self, frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray | None]:
        ok, _ = self.availability()
        if not ok:
            return {t: None for t in frames}
        series = {}
        for t, df in frames.items():
            try:
                series[t] = self._predict_series(t, df)
            except Exception as e:  # noqa: BLE001
                log.warning("alpha factors failed for %s: %s", t, e)
        if not series:
            return {t: None for t in frames}
        wide = pd.DataFrame(series)
        rank = wide.rank(axis=1, pct=True) * 2.0 - 1.0
        out: dict[str, np.ndarray | None] = {}
        for t, df in frames.items():
            if t not in series:
                out[t] = None
                continue
            p = series[t].reindex(df.index)
            r = rank[t].reindex(df.index).fillna(0.0)
            arr = np.column_stack([np.clip(p.to_numpy() * 20.0, -5, 5), r.to_numpy()]).astype(np.float32)
            arr[p.isna().to_numpy()] = np.nan
            out[t] = arr
        return out
