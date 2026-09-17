"""Deep-learning price forecaster - the recurrent-network forecasters of huseinzol05/
Stock-Prediction-Models (``deep-learning/``), crypto-code/Stock-Market-AI-GUI and the many LSTM
stock predictors, rebuilt in PyTorch with an honest protocol.

A small GRU reads the last ``window`` days of (log return, volume change, range) and predicts the
next ``horizon``-day log return.  It is trained pooled across tickers and refit walk-forward (every
``refit_every`` years on all data before that year), so the RL policy never sees in-sample
forecasts.  Expect it to be noisy: it is one more opinion for the policy, not an oracle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .base import SignalProvider

log = get_logger(__name__)


def sequence_inputs(df: pd.DataFrame) -> np.ndarray:
    """(T, 3) per-bar inputs: log return, log volume change, high-low range (all scaled)."""
    c = df["close"].astype(float).clip(lower=1e-9)
    v = df["volume"].astype(float)
    r = np.log(c).diff().fillna(0.0) * 20.0
    dv = np.log(v + 1.0).diff().fillna(0.0)
    rng = ((df["high"] - df["low"]) / c * 30.0 - 1.0).fillna(0.0)
    return np.column_stack([r.to_numpy(), dv.to_numpy(), rng.to_numpy()]).astype(np.float32).clip(-5, 5)


def _windows(x: np.ndarray, window: int) -> np.ndarray:
    """(T, window, F) sliding windows ending at each bar (zero-padded at the start)."""
    T, F = x.shape
    pad = np.zeros((window - 1, F), dtype=np.float32)
    xp = np.concatenate([pad, x])
    idx = np.arange(window)[None, :] + np.arange(T)[:, None]
    return xp[idx]


class DLForecastSignal(SignalProvider):
    name = "dl_forecast"
    parallel_ok = False            # runs its own threads (torch / TensorFlow): one ticker at a time
    feature_names = ["dl_pred", "dl_confidence"]
    needs_universe = True
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.window = int(self.cfg.get("window", 30))
        self.horizon = int(self.cfg.get("horizon", 5))
        self.hidden = int(self.cfg.get("hidden", 32))
        self.epochs = int(self.cfg.get("epochs", 3))
        self.min_train_years = int(self.cfg.get("min_train_years", 3))
        self.refit_every = int(self.cfg.get("refit_every", 2))
        self.max_train_rows = int(self.cfg.get("max_train_rows", 150_000))
        self.state = None            # final model state dict (for live prediction)
        self.preds: pd.DataFrame | None = None

    # ------------------------------------------------------------------ model
    def _build(self):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self, hidden: int):
                super().__init__()
                self.gru = nn.GRU(3, hidden, batch_first=True)
                self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

            def forward(self, x):
                _, h = self.gru(x)
                return self.head(h[-1]).squeeze(-1)

        torch.manual_seed(0)
        return Net(self.hidden)

    def _train(self, X: np.ndarray, y: np.ndarray):
        import torch

        net = self._build()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
        n = len(Xt)
        bs = 512
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        net.eval()
        return net

    def _predict(self, net, X: np.ndarray) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                out.append(net(torch.from_numpy(X[i:i + 4096])).numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------------ fitting (walk-forward)
    def fit(self, frames: dict[str, pd.DataFrame], train_end: str | None = None) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            log.warning("dl_forecast needs torch")
            return
        parts = []
        for t, df in frames.items():
            x = _windows(sequence_inputs(df), self.window)
            logc = np.log(df["close"].astype(float).clip(lower=1e-9)).to_numpy()
            y = np.full(len(df), np.nan)
            y[:-self.horizon] = (logc[self.horizon:] - logc[:-self.horizon]) * 20.0
            parts.append((t, df.index, x, np.clip(y, -5, 5)))
        dates = np.concatenate([p[1].to_numpy() for p in parts])
        X = np.concatenate([p[2] for p in parts])
        y = np.concatenate([p[3] for p in parts])
        tick = np.concatenate([np.full(len(p[1]), p[0]) for p in parts])
        years = np.array(pd.DatetimeIndex(dates).year)
        uniq = sorted(set(years))
        preds = []
        fit_years = [yr for i, yr in enumerate(uniq[self.min_train_years:]) if i % self.refit_every == 0]
        for k, yr in enumerate(fit_years):
            until = fit_years[k + 1] if k + 1 < len(fit_years) else uniq[-1] + 1
            tr = (years < yr) & ~np.isnan(y)
            te = (years >= yr) & (years < until)
            if tr.sum() < 2000 or te.sum() == 0:
                continue
            tr_idx = np.flatnonzero(tr)
            if len(tr_idx) > self.max_train_rows:
                tr_idx = np.random.default_rng(0).choice(tr_idx, self.max_train_rows, replace=False)
            net = self._train(X[tr_idx], y[tr_idx])
            p = self._predict(net, X[te])
            preds.append(pd.DataFrame({"ticker": tick[te], "date": dates[te], "pred": p}))
        self.preds = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=["ticker", "date", "pred"])
        final = ~np.isnan(y)
        if final.sum() >= 2000:
            idx = np.flatnonzero(final)
            if len(idx) > self.max_train_rows:
                idx = np.random.default_rng(1).choice(idx, self.max_train_rows, replace=False)
            net = self._train(X[idx], y[idx])
            self.state = {k: v.clone() for k, v in net.state_dict().items()}
        log.info("dl_forecast fitted: %d walk-forward predictions (%d refits)", len(self.preds), len(fit_years))

    def save_state(self) -> None:
        if self.state is not None:
            import torch

            torch.save({"state": self.state, "window": self.window, "hidden": self.hidden, "horizon": self.horizon},
                       self.state_path("dl_forecast.pt"))
        if self.preds is not None:
            self.preds.to_parquet(self.state_path("dl_preds.parquet"))

    def load_state(self) -> bool:
        p = self.state_path("dl_forecast.pt")
        if not p.exists():
            return False
        try:
            import torch

            d = torch.load(p, map_location="cpu")
            if d.get("window") != self.window or d.get("hidden") != self.hidden:
                return False
            self.state = d["state"]
            pp = self.state_path("dl_preds.parquet")
            self.preds = pd.read_parquet(pp) if pp.exists() else None
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("could not load dl_forecast state: %s", e)
            return False

    def availability(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "needs torch"
        if self.state is None and not self.load_state():
            return False, "not fitted yet (run `stockbot build-dataset`)"
        return True, "walk-forward GRU forecaster (huseinzol05 / LSTM-predictor port)"

    # ------------------------------------------------------------------ prediction
    def _series(self, ticker: str, df: pd.DataFrame) -> pd.Series:
        pred = pd.Series(np.nan, index=df.index)
        if self.preds is not None and len(self.preds):
            sub = self.preds[self.preds["ticker"] == ticker]
            if len(sub):
                s = pd.Series(sub["pred"].to_numpy(), index=pd.to_datetime(sub["date"]))
                s = s[~s.index.duplicated(keep="last")]
                common = pred.index.intersection(s.index)
                pred.loc[common] = s.loc[common]
        missing = pred.isna().to_numpy()
        if self.state is not None and missing.any():
            net = self._build()
            net.load_state_dict(self.state)
            net.eval()
            X = _windows(sequence_inputs(df), self.window)
            rows = np.flatnonzero(missing)
            rows = rows[rows >= self.window]
            if len(rows):
                pred.iloc[rows] = self._predict(net, X[rows])
        return pred

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        pred = self._series(ticker, df)
        p = pred.to_numpy(float)
        out = np.column_stack([np.clip(p, -5, 5), np.clip(np.abs(p), 0, 5)]).astype(np.float32)
        out[np.isnan(p)] = np.nan
        return out

    def compute_history_all(self, frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray | None]:
        ok, _ = self.availability()
        if not ok:
            return {t: None for t in frames}
        return {t: self.safe_history(t, df) for t, df in frames.items()}
