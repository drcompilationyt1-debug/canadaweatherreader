"""PyPortfolioOpt (robertmartin8/PyPortfolioOpt) - risk-based allocation as an input.

Big allocators do not size positions by conviction alone; they ask what a risk model would hold.
Every ``refit_every`` bars the block re-runs hierarchical risk parity (HRP, Lopez de Prado) and
inverse-volatility weighting on the trailing ``window`` days of returns of the whole universe and
tells the policy, per ticker: the HRP weight and the inverse-vol weight (both scaled so 1.0 = an
equal-weight slice), the ticker's correlation with the universe average, and its share of the
portfolio's risk under HRP.  Cross-sectional, so it runs once for all tickers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...logging_utils import get_logger
from ...paths import import_optional
from ..base import SignalProvider

log = get_logger(__name__)


class PortfolioOptSignal(SignalProvider):
    name = "pypfopt"
    feature_names = ["po_hrp_w", "po_invvol_w", "po_corr_mkt", "po_risk_share"]
    needs_universe = True
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.window = int(self.cfg.get("window", 252))
        self.refit_every = int(self.cfg.get("refit_every", 21))
        self.min_assets = int(self.cfg.get("min_assets", 5))

    def availability(self) -> tuple[bool, str]:
        if import_optional("pypfopt") is None:
            return False, "pip install PyPortfolioOpt (third_party/PyPortfolioOpt)"
        return True, f"HRP + inverse-vol weights on a {self.window}-day window, refit every {self.refit_every} bars"

    @staticmethod
    def _hrp(returns: pd.DataFrame) -> pd.Series:
        from pypfopt.hierarchical_portfolio import HRPOpt

        hrp = HRPOpt(returns)
        w = hrp.optimize()
        return pd.Series(w, dtype=float).reindex(returns.columns).fillna(0.0)

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        frames = self.ctx.extra.get("frames") or {}
        if ticker not in frames:
            return None
        return self.compute_history_all(frames).get(ticker)

    def compute_history_all(self, frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray | None]:
        ok, _ = self.availability()
        if not ok or not frames:
            return {t: None for t in frames}
        closes = pd.DataFrame({t: df["close"].astype(float) for t, df in frames.items()}).sort_index()
        rets = np.log(closes.clip(lower=1e-9)).diff()
        n_assets = closes.shape[1]
        feats = pd.DataFrame(np.nan, index=closes.index, columns=pd.MultiIndex.from_product([closes.columns, self.feature_names]))
        dates = closes.index
        for end in range(self.window, len(dates), self.refit_every):
            win = rets.iloc[end - self.window:end]
            valid = win.columns[win.notna().sum() >= int(self.window * 0.9)]
            if len(valid) < self.min_assets:
                continue
            w = win[valid].dropna()
            if len(w) < self.window // 2:
                continue
            try:
                hrp = self._hrp(w)
            except Exception as e:  # noqa: BLE001
                log.debug("hrp failed at %s: %s", dates[end - 1], e)
                continue
            vol = w.std().replace(0.0, np.nan)
            inv = (1.0 / vol) / (1.0 / vol).sum()
            mkt = w.mean(axis=1)
            corr = w.corrwith(mkt)
            cov = w.cov()
            port_var = float(hrp.to_numpy() @ cov.to_numpy() @ hrp.to_numpy())
            risk_share = (hrp * (cov @ hrp)) / port_var if port_var > 0 else hrp * 0.0
            stop = min(end + self.refit_every, len(dates))
            span = dates[end:stop]
            for t in valid:
                feats.loc[span, (t, "po_hrp_w")] = float(np.clip(hrp[t] * len(valid), 0, 5))
                feats.loc[span, (t, "po_invvol_w")] = float(np.clip(inv[t] * len(valid), 0, 5))
                feats.loc[span, (t, "po_corr_mkt")] = float(np.clip(corr[t], -1, 1))
                feats.loc[span, (t, "po_risk_share")] = float(np.clip(risk_share[t] * len(valid), 0, 5))
        out: dict[str, np.ndarray | None] = {}
        for t, df in frames.items():
            sub = feats[t].reindex(df.index)
            out[t] = sub.to_numpy(np.float32)
        return out

    def compute_latest(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        frames = self.ctx.extra.get("frames") or {}
        if ticker not in frames:
            return None
        arr = self.compute_history_all(frames).get(ticker)
        if arr is None or len(arr) == 0:
            return None
        row = arr[-1]
        if np.isnan(row).any():  # the last refit block may not cover the newest bar yet: use the latest known row
            known = np.flatnonzero(~np.isnan(arr).any(axis=1))
            if len(known) == 0 or len(arr) - 1 - known[-1] > self.refit_every:
                return None
            row = arr[known[-1]]
        return row
