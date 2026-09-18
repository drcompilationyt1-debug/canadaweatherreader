"""Cross-sectional ranking head: which names will beat the others over the next month?

Every other block scores a ticker on its own.  Quant funds instead rank the whole universe every
day and hold the top of the list.  This block trains a LightGBM regressor on the outputs of all
the other blocks (their features plus availability flags) to predict each ticker's percentile
rank of forward ``horizon``-day return within the universe on the same date - a relative, not
absolute, target.  Fitting is walk-forward (yearly refits, only past data), so the policy sees
out-of-sample ranks; a final model on all data serves the live session.  Features per bar: the
predicted score, its cross-sectional rank (-1..1), and top / bottom-decile flags.

It runs last in the layout and reads the other blocks from ``ctx.extra["per_block"]`` (dataset
build) or ``ctx.extra["latest_vectors"]`` (live).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .base import SignalProvider

log = get_logger(__name__)

EXCLUDE = {"xs_rank", "xs_nn", "trading_agents", "ai_hedge_fund", "llm_trader"}   # the ranking heads themselves and the live-only blocks


class XSRankSignal(SignalProvider):
    name = "xs_rank"
    feature_names = ["xs_score", "xs_rank", "xs_top", "xs_bottom"]
    STATE_FILE = "xs_rank.joblib"
    PREDS_FILE = "xs_rank_preds.parquet"
    needs_universe = True
    tier = "A"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.horizon = int(self.cfg.get("horizon", 20))
        self.min_train_years = int(self.cfg.get("min_train_years", 3))
        self.refit_every = int(self.cfg.get("refit_every", 1))
        self.n_estimators = int(self.cfg.get("n_estimators", 300))
        self.max_train_rows = int(self.cfg.get("max_train_rows", 400_000))
        self.spec: list[tuple[str, int]] = []      # (block, size) in feature order
        self.model = None
        self.preds: pd.DataFrame | None = None

    def availability(self) -> tuple[bool, str]:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            return False, "pip install lightgbm"
        return True, f"LightGBM cross-sectional ranker on the other blocks, {self.horizon}-day relative return, walk-forward"

    # ------------------------------------------------------------------ feature matrix
    def _spec_from(self, per_block: dict[str, dict[str, np.ndarray | None]]) -> list[tuple[str, int]]:
        spec = []
        for block, arrs in per_block.items():
            if block in EXCLUDE:
                continue
            size = next((a.shape[1] for a in arrs.values() if a is not None and a.ndim == 2), None)
            if size:
                spec.append((block, int(size)))
        return spec

    def _matrix(self, per_ticker: dict[str, np.ndarray | None], n: int) -> np.ndarray:
        """(n, F) features: per block [flag, values...] with NaN -> 0 and the flag = row complete."""
        cols = []
        for block, size in self.spec:
            a = per_ticker.get(block)
            if a is None or a.shape != (n, size):
                cols.append(np.zeros((n, size + 1), dtype=np.float32))
                continue
            ok = ~np.isnan(a).any(axis=1)
            vals = np.where(ok[:, None], np.nan_to_num(a, nan=0.0), 0.0).astype(np.float32)
            cols.append(np.column_stack([ok.astype(np.float32), np.clip(vals, -10, 10)]))
        return np.concatenate(cols, axis=1) if cols else np.zeros((n, 0), dtype=np.float32)

    def _lgbm(self):
        import lightgbm as lgb

        return lgb.LGBMRegressor(n_estimators=self.n_estimators, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                                 subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, n_jobs=4, verbose=-1)

    # ------------------------------------------------------------------ history (build) mode
    def _history(self, frames: dict[str, pd.DataFrame], per_block: dict) -> dict[str, np.ndarray | None]:
        self.spec = self._spec_from(per_block)
        if not self.spec:
            return {t: None for t in frames}
        X, tick, dates, fwd = [], [], [], []
        for t, df in frames.items():
            n = len(df)
            X.append(self._matrix({b: per_block[b].get(t) for b, _ in self.spec}, n))
            tick.append(np.full(n, t))
            dates.append(df.index.to_numpy(dtype="datetime64[ns]"))
            logc = np.log(df["close"].astype(float).clip(lower=1e-9)).to_numpy()
            f = np.full(n, np.nan)
            f[:-self.horizon] = logc[self.horizon:] - logc[:-self.horizon]
            fwd.append(f)
        X = np.concatenate(X)
        tick = np.concatenate(tick)
        dates = np.concatenate(dates)
        fwd = np.concatenate(fwd)
        panel = pd.DataFrame({"ticker": tick, "date": dates, "fwd": fwd})
        panel["y"] = panel.groupby("date")["fwd"].rank(pct=True) - 0.5          # relative target
        panel.loc[panel["fwd"].isna(), "y"] = np.nan
        years = pd.DatetimeIndex(dates).year.to_numpy()
        uniq = sorted(set(years.tolist()))
        pred = np.full(len(panel), np.nan)
        fit_years = [yr for i, yr in enumerate(uniq[self.min_train_years:]) if i % self.refit_every == 0]
        y = panel["y"].to_numpy(float)
        rng = np.random.default_rng(0)
        for k, yr in enumerate(fit_years):
            until = fit_years[k + 1] if k + 1 < len(fit_years) else uniq[-1] + 1
            tr = np.flatnonzero((years < yr) & ~np.isnan(y) & (X[:, ::1].any(axis=1)))
            te = np.flatnonzero((years >= yr) & (years < until))
            if len(tr) < 5000 or len(te) == 0:
                continue
            if len(tr) > self.max_train_rows:
                tr = rng.choice(tr, self.max_train_rows, replace=False)
            m = self._lgbm().fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        final = np.flatnonzero(~np.isnan(y) & X.any(axis=1))
        if len(final) >= 5000:
            if len(final) > self.max_train_rows:
                final = rng.choice(final, self.max_train_rows, replace=False)
            self.model = self._lgbm().fit(X[final], y[final])
        panel["pred"] = pred
        self.preds = panel[["ticker", "date", "pred"]].dropna().reset_index(drop=True)
        log.info("xs_rank: %d out-of-sample scores over %d refits, %d features from %d blocks", len(self.preds), len(fit_years),
                 X.shape[1], len(self.spec))
        self.save_state()
        return self._features_from(panel, frames)

    @staticmethod
    def _rank_features(panel: pd.DataFrame) -> pd.DataFrame:
        p = panel.dropna(subset=["pred"]).copy()
        p["rank"] = p.groupby("date")["pred"].rank(pct=True)
        n = p.groupby("date")["pred"].transform("size")
        p["top"] = ((p["rank"] >= 0.9) & (n >= 5)).astype(float)
        p["bottom"] = ((p["rank"] <= 0.1) & (n >= 5)).astype(float)
        p["xs_score"] = np.clip(p["pred"] * 6.0, -3, 3)
        p["xs_rank"] = p["rank"] * 2.0 - 1.0
        return p

    def _features_from(self, panel: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray | None]:
        p = self._rank_features(panel).set_index(["ticker", "date"])
        out = {}
        for t, df in frames.items():
            arr = np.full((len(df), 4), np.nan, dtype=np.float32)
            if t in p.index.get_level_values(0):
                sub = p.loc[t].reindex(pd.DatetimeIndex(df.index))
                arr[:, 0] = sub["xs_score"].to_numpy(np.float32)
                arr[:, 1] = sub["xs_rank"].to_numpy(np.float32)
                arr[:, 2] = sub["top"].to_numpy(np.float32)
                arr[:, 3] = sub["bottom"].to_numpy(np.float32)
            out[t] = arr
        return out

    # ------------------------------------------------------------------ live mode
    def _latest(self, frames: dict[str, pd.DataFrame], vectors: dict[str, dict[str, np.ndarray | None]]) -> dict[str, np.ndarray | None]:
        if self.model is None and not self.load_state():
            return {t: None for t in frames}
        rows, names = [], []
        for t in frames:
            vecs = vectors.get(t) or {}
            per = {b: (None if vecs.get(b) is None else np.asarray(vecs[b], dtype=np.float32).reshape(1, -1)) for b, _ in self.spec}
            rows.append(self._matrix(per, 1)[0])
            names.append(t)
        X = np.stack(rows)
        pred = self.model.predict(X)
        panel = pd.DataFrame({"ticker": names, "date": pd.Timestamp("2000-01-01"), "pred": pred})
        p = self._rank_features(panel).set_index("ticker")
        out = {}
        for t, df in frames.items():
            arr = np.full((len(df), 4), np.nan, dtype=np.float32)
            arr[-1] = [p.loc[t, "xs_score"], p.loc[t, "xs_rank"], p.loc[t, "top"], p.loc[t, "bottom"]]
            out[t] = arr
        return out

    def compute_history_all(self, frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray | None]:
        ok, _ = self.availability()
        if not ok:
            return {t: None for t in frames}
        per_block = self.ctx.extra.get("per_block")
        if per_block:
            return self._history(frames, per_block)
        vectors = self.ctx.extra.get("latest_vectors")
        if vectors:
            return self._latest(frames, vectors)
        log.warning("xs_rank: no other blocks available in the context - masked")
        return {t: None for t in frames}

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        frames = self.ctx.extra.get("frames") or {}
        return self.compute_history_all(frames).get(ticker) if ticker in frames else None

    # ------------------------------------------------------------------ persistence
    def save_state(self) -> None:
        if self.model is None:
            return
        import joblib

        joblib.dump({"model": self.model, "spec": self.spec, "horizon": self.horizon,
                     "saved_at": datetime.now(timezone.utc).isoformat()}, self.state_path(self.STATE_FILE))
        if self.preds is not None:
            self.preds.to_parquet(self.state_path(self.PREDS_FILE))

    def load_state(self) -> bool:
        p = self.state_path(self.STATE_FILE)
        if not p.exists():
            return False
        try:
            import joblib

            d = joblib.load(p)
            self.model, self.spec = d["model"], [tuple(x) for x in d["spec"]]
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("could not load xs_rank state: %s", e)
            return False
