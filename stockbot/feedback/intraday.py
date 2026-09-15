"""Intraday exit: learn from the broker's bars when a position should have been sold during the day.

"We were up 1% at 11:00, held, and gave it all back by 14:00" is a pattern the daily policy cannot
act on - it decides once, at the open.  This layer sits on top of it: at every snapshot of the watch
window it looks at the path of each held name since the open (move so far, high-water mark, pull-back
from it, time of day, intraday volatility, the overnight gap) and asks a small model whether the rest
of the day is more likely to take the gain away than to add to it.  The model is fitted on Alpaca's
15-minute bars for the whole universe (thousands of name-days, not just ours), labelled in hindsight:
did the close end below where we could have sold now, net of the round-trip fee?  It is re-fitted
every morning and scored on the last few days held out, and the session only sells when the
probability is high and the move is worth the fee (``session.intraday_exit``).  The day review uses
the same bars to show, for every name we held, the best exit that was available and what the exit
model would have done.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)

STEPS_PER_DAY = 26            # 15-minute bars from 09:30 to 15:45
FEATURES = ["ret_open", "ret_max", "ret_min", "dd_from_max", "gain_vs_max", "time_frac", "vol", "last_ret", "last4", "gap"]


# ---------------------------------------------------------------------- features / labels
def features_at(closes: np.ndarray, k: int, open_px: float, prev_close: float | None) -> dict:
    """The state of one held name ``k`` bars after the open (``closes[0]`` = the first bar's close)."""
    c = np.asarray(closes[: k + 1], dtype=float)
    rets = c / open_px - 1.0
    ret_open = float(rets[-1])
    ret_max = float(rets.max())
    ret_min = float(rets.min())
    lr = np.diff(np.log(np.maximum(c, 1e-9))) if k >= 1 else np.zeros(0)
    return {"ret_open": ret_open, "ret_max": ret_max, "ret_min": ret_min, "dd_from_max": float(c[-1] / c.max() - 1.0),
            "gain_vs_max": ret_max - ret_open, "time_frac": float((k + 1) / STEPS_PER_DAY),
            "vol": float(lr.std(ddof=1)) if len(lr) >= 2 else 0.0, "last_ret": float(c[-1] / c[-2] - 1.0) if k >= 1 else 0.0,
            "last4": float(c[-1] / c[max(0, k - 4)] - 1.0), "gap": float(open_px / prev_close - 1.0) if prev_close else 0.0}


def label_at(closes: np.ndarray, k: int, fee_rt: float) -> int:
    """1 when selling at bar ``k`` beats holding to the close, net of the round-trip fee."""
    return int(float(closes[-1]) < float(closes[k]) * (1.0 - fee_rt))


def what_if_exit(closes: np.ndarray, open_px: float, fee_rt: float) -> dict:
    """The best exit that was available on the day versus holding to the close."""
    c = np.asarray(closes, dtype=float)
    if len(c) == 0 or open_px <= 0:
        return {}
    hold = float(c[-1] / open_px - 1.0)
    exits = c * (1.0 - fee_rt) / open_px - 1.0
    k = int(exits.argmax())
    return {"hold_return": hold, "best_exit_k": k, "best_exit_return": float(exits[k]), "gain_vs_hold": float(exits[k] - hold),
            "high_water": float(c.max() / open_px - 1.0)}


def paths_from_bars(bars: dict[str, pd.DataFrame]) -> dict[tuple[str, date], dict]:
    """Regular-hours close paths per (ticker, day) from intraday bars, with the day's open and the previous close."""
    out: dict[tuple[str, date], dict] = {}
    for t, df in bars.items():
        if df is None or len(df) == 0:
            continue
        mins = df.index.hour * 60 + df.index.minute
        d = df[(mins >= 9 * 60 + 30) & (mins < 16 * 60)]
        prev_close = None
        for day, g in d.groupby(d.index.date):
            if len(g) >= 3:
                out[(t, day)] = {"closes": g["close"].to_numpy(float), "open": float(g["open"].iloc[0]), "prev_close": prev_close,
                                 "times": [ts.strftime("%H:%M") for ts in g.index]}
            prev_close = float(g["close"].iloc[-1]) if len(g) else prev_close
    return out


def build_dataset(paths: dict[tuple[str, date], dict], fee_rt: float) -> pd.DataFrame:
    rows = []
    for (t, day), p in paths.items():
        c = np.asarray(p["closes"], dtype=float)
        if len(c) < 4 or p["open"] <= 0:
            continue
        for k in range(1, len(c) - 1):
            f = features_at(c, k, p["open"], p.get("prev_close"))
            f.update({"ticker": t, "day": day, "k": k, "label": label_at(c, k, fee_rt),
                      "gain_if_exit": float(c[k] * (1.0 - fee_rt) / c[-1] - 1.0)})
            rows.append(f)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- the model
def _auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    pos, neg = p[y > 0.5], p[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(p).rank().to_numpy()
    return float((ranks[y > 0.5].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


class ExitModel:
    def __init__(self, model=None, meta: dict | None = None):
        self.model = model
        self.meta = meta or {}

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, holdout_days: int = 5, min_prob: float = 0.6) -> dict:
        days = sorted(df["day"].unique())
        hold = set(days[-holdout_days:]) if holdout_days > 0 and len(days) > holdout_days else set()
        tr = df[~df["day"].isin(hold)]
        te = df[df["day"].isin(hold)]
        X, y = tr[FEATURES].to_numpy(np.float32), tr["label"].to_numpy(int)
        try:
            import lightgbm as lgb

            self.model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50, subsample=0.8,
                                            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
            self.model.fit(X, y)
            kind = "lightgbm"
        except Exception as e:  # noqa: BLE001 - fall back to a logistic fit
            log.info("lightgbm unavailable for the exit model (%s) - logistic fallback", e)
            self.model = _Logistic().fit(X, y)
            kind = "logistic"
        metrics = {"kind": kind, "n_train": int(len(tr)), "n_holdout": int(len(te)), "days": int(len(days)), "holdout_days": sorted(str(d) for d in hold),
                   "base_rate": float(y.mean()) if len(y) else float("nan")}
        if len(te):
            p = self.predict(te)
            metrics.update({"auc": _auc(te["label"].to_numpy(), p), "hit_rate": float(((p >= 0.5) == (te["label"].to_numpy() > 0.5)).mean()),
                            "exit_rate": float((p >= min_prob).mean()),
                            # value of following the model: average gain vs holding, over the moments it says "exit"
                            "gain_when_exit": float(te.loc[p >= min_prob, "gain_if_exit"].mean()) if (p >= min_prob).any() else 0.0,
                            "gain_all": float(te["gain_if_exit"].mean())})
        self.meta = {"features": FEATURES, "fitted_at": datetime.now(timezone.utc).isoformat(), "metrics": metrics}
        return metrics

    def predict(self, rows: pd.DataFrame | dict | list[dict]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("exit model not fitted")
        df = pd.DataFrame([rows]) if isinstance(rows, dict) else pd.DataFrame(rows) if isinstance(rows, list) else rows
        X = df[FEATURES].to_numpy(np.float32)
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X))[:, 1]
        return np.asarray(self.model.predict(X), dtype=float)

    # ------------------------------------------------------------------ live use
    def advice(self, closes: np.ndarray, open_px: float, prev_close: float | None, fee_rt: float, min_prob: float = 0.6,
               min_gain: float = 0.005, stop_loss: float = 0.02) -> dict:
        """Should a held name be sold now?  Only when the model is confident the close will be lower (net of the
        fee) and the move is worth acting on: a gain of at least ``min_gain`` to take, or a loss past ``stop_loss``."""
        c = np.asarray(closes, dtype=float)
        if len(c) < 2 or open_px <= 0:
            return {"exit": False, "prob": None, "reason": "too early"}
        f = features_at(c, len(c) - 1, open_px, prev_close)
        p = float(self.predict(f)[0])
        gain = f["ret_open"]
        worth = gain >= min_gain or gain <= -abs(stop_loss)
        exit_now = p >= min_prob and worth and gain > -1.0 and abs(gain) > fee_rt
        reason = ("take profit" if gain > 0 else "cut loss") if exit_now else ("not worth the fee" if p >= min_prob else "hold")
        return {"exit": bool(exit_now), "prob": p, "ret_open": gain, "ret_max": f["ret_max"], "dd_from_max": f["dd_from_max"], "reason": reason}

    def replay(self, closes: np.ndarray, open_px: float, prev_close: float | None, fee_rt: float, **kw) -> dict:
        """Walk one day's path and report where the model would have sold and what that was worth vs holding."""
        c = np.asarray(closes, dtype=float)
        for k in range(1, len(c)):
            a = self.advice(c[: k + 1], open_px, prev_close, fee_rt, **kw)
            if a["exit"]:
                return {"exit_k": k, "exit_return": float(c[k] * (1.0 - fee_rt) / open_px - 1.0), "hold_return": float(c[-1] / open_px - 1.0), "prob": a["prob"]}
        return {"exit_k": None, "exit_return": float(c[-1] / open_px - 1.0), "hold_return": float(c[-1] / open_px - 1.0), "prob": None}

    # ------------------------------------------------------------------ persistence
    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "booster_"):
            self.model.booster_.save_model(str(folder / "model.txt"))
            self.meta["file"] = "model.txt"
        else:
            (folder / "logistic.json").write_text(json.dumps(self.model.to_dict()), encoding="utf-8")
            self.meta["file"] = "logistic.json"
        (folder / "meta.json").write_text(json.dumps(self.meta, indent=1, default=str), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> "ExitModel | None":
        folder = Path(folder)
        meta_file = folder / "meta.json"
        if not meta_file.exists():
            return None
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("file") == "model.txt" and (folder / "model.txt").exists():
                import lightgbm as lgb

                return cls(_BoosterWrap(lgb.Booster(model_file=str(folder / "model.txt"))), meta)
            if (folder / "logistic.json").exists():
                return cls(_Logistic.from_dict(json.loads((folder / "logistic.json").read_text(encoding="utf-8"))), meta)
        except Exception as e:  # noqa: BLE001
            log.warning("exit model in %s could not be loaded: %s", folder, e)
        return None


class _BoosterWrap:
    def __init__(self, booster):
        self.booster = booster

    def predict_proba(self, X):
        p = np.asarray(self.booster.predict(X)).reshape(-1)
        return np.stack([1.0 - p, p], axis=1)


class _Logistic:
    """Standardised logistic regression by gradient descent - no sklearn needed."""

    def __init__(self, w=None, b=0.0, mu=None, sd=None):
        self.w, self.b, self.mu, self.sd = w, b, mu, sd

    def fit(self, X, y, iters: int = 400, lr: float = 0.1, l2: float = 1e-3):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = (X - self.mu) / self.sd
        w = np.zeros(Z.shape[1])
        b = 0.0
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
            g = Z.T @ (p - y) / len(y) + l2 * w
            w -= lr * g
            b -= lr * float((p - y).mean())
        self.w, self.b = w, b
        return self

    def predict_proba(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        p = 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))
        return np.stack([1.0 - p, p], axis=1)

    def to_dict(self):
        return {"w": self.w.tolist(), "b": float(self.b), "mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["w"]), float(d["b"]), np.asarray(d["mu"]), np.asarray(d["sd"]))


# ---------------------------------------------------------------------- fit from the broker's bars
def round_trip_fee(cfg, ticker: str, notional: float = 10_000.0, price: float = 100.0) -> float:
    from ..execution.fees import FeeBook

    sched = FeeBook.from_config(cfg).for_ticker(ticker)
    if sched is None or notional <= 0:
        return 0.0
    return 2.0 * float(sched.cost(notional / price, price, "buy")) / notional


def fit_from_alpaca(cfg, days: int = 40, timeframe: str = "15Min", holdout_days: int = 5, end: date | None = None) -> dict:
    """Fit the exit model on Alpaca's intraday bars for the US names of the universe and save it."""
    from ..execution.alpaca_history import AlpacaHistory
    from ..execution.markets import market_of

    ie = dict(cfg.get_path("session.intraday_exit", {}) or {})
    folder = cfg.path("session.intraday_exit.model_dir", "models/intraday_exit")
    if not AlpacaHistory.available():
        return {"skipped": "no Alpaca keys"}
    tickers = [t for t in cfg.get("universe", []) if market_of(t) == "us"]
    end = end or (datetime.now(timezone.utc).date() - timedelta(days=1))
    start = end - timedelta(days=int(days * 1.5) + 3)
    hist = AlpacaHistory(cfg)
    bars = hist.bars_cached(tickers, start, end, timeframe)
    paths = paths_from_bars(bars)
    if not paths:
        return {"skipped": "no intraday bars"}
    fee_rt = round_trip_fee(cfg, "SPY", notional=float(cfg.get_path("env.initial_cash", 100_000)) * float(cfg.get_path("execution.max_position", 0.1)))
    df = build_dataset(paths, fee_rt)
    if len(df) < 500:
        return {"skipped": f"only {len(df)} samples"}
    model = ExitModel()
    metrics = model.fit(df, holdout_days=holdout_days, min_prob=float(ie.get("min_prob", 0.6)))
    model.meta.update({"tickers": len(bars), "timeframe": timeframe, "fee_rt": fee_rt, "start": str(start), "end": str(end)})
    model.save(folder)
    report = {"model_dir": str(folder), "samples": int(len(df)), "name_days": int(len(paths)), "tickers": int(len(bars)), "fee_rt": fee_rt, **metrics}
    log.info("intraday exit model: %d samples over %d name-days, holdout auc %.3f, exits %.0f%% worth %+.2f%% each vs hold",
             report["samples"], report["name_days"], metrics.get("auc", float("nan")), 100 * metrics.get("exit_rate", 0.0),
             100 * metrics.get("gain_when_exit", 0.0))
    return report
