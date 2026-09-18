"""Market timing by complexity (Kelly, Malamud and Zhou, Journal of Finance 2024, "The Virtue of Complexity in Return
Prediction"): a very wide random-feature ridge regression of the next month's market return on a handful of standard
predictors beats sparse models out of sample, and the gain grows with the number of features even past the point where
the model interpolates the training data, as long as the ridge shrinkage is heavy.

Here the predictors are the portfolio agent's market features (index momentum, volatility, trend, breadth, dispersion),
the target the index's next 20-day return, the features ``n_features`` random Fourier features of the standardised
predictors, and the fit a ridge in its dual (kernel) form so a few thousand features cost nothing.  Walk-forward: refit
every ``refit_every`` bars on all rows whose targets were known by then, predictions for the bars in between.  The output
is one more market feature for the portfolio agent - which is what decides whether it is worth anything.
"""
from __future__ import annotations

import numpy as np

HORIZON = 20


class ComplexityTimer:
    def __init__(self, n_features: int = 3000, gamma: float = 0.5, ridge: float = 10.0, seed: int = 0):
        self.n_features, self.gamma, self.ridge, self.seed = int(n_features), float(gamma), float(ridge), int(seed)
        self.W = self.b = self.mu = self.sd = None
        self.alpha = None                 # dual coefficients
        self.Z_train = None

    # ------------------------------------------------------------------ features
    def _init(self, dim: int) -> None:
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0.0, self.gamma, size=(dim, self.n_features)).astype(np.float32)
        self.b = rng.uniform(0.0, 2 * np.pi, size=self.n_features).astype(np.float32)

    def _rff(self, F: np.ndarray) -> np.ndarray:
        Z = (np.asarray(F, dtype=np.float32) - self.mu) / self.sd
        return np.sqrt(2.0 / self.n_features) * np.cos(Z @ self.W + self.b)

    # ------------------------------------------------------------------ fit / predict
    def fit(self, F: np.ndarray, y: np.ndarray) -> "ComplexityTimer":
        F, y = np.asarray(F, dtype=np.float32), np.asarray(y, dtype=np.float64)
        if self.W is None or self.W.shape[0] != F.shape[1]:
            self._init(F.shape[1])
        self.mu, self.sd = F.mean(axis=0), F.std(axis=0) + 1e-6
        Z = self._rff(F)
        K = Z @ Z.T                                                        # (T, T): dual ridge, cheap for a few thousand rows
        self.alpha = np.linalg.solve(K + self.ridge * np.eye(len(Z)), y - y.mean())
        self.y_mean = float(y.mean())
        self.Z_train = Z
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        if self.alpha is None:
            return np.zeros(len(F), dtype=np.float32)
        return (self._rff(F) @ self.Z_train.T @ self.alpha + self.y_mean).astype(np.float32)

    # ------------------------------------------------------------------ walk-forward
    def walk_forward(self, F: np.ndarray, y: np.ndarray, min_train: int = 500, refit_every: int = 21, horizon: int = HORIZON,
                     max_train: int = 3000) -> np.ndarray:
        """Predictions for every row from ``min_train`` on, each from a fit on rows whose targets were known by then
        (rows s with s + horizon <= t), refit every ``refit_every`` rows on the last ``max_train`` usable rows."""
        F, y = np.asarray(F, dtype=np.float32), np.asarray(y, dtype=np.float64)
        n = len(F)
        out = np.full(n, np.nan, dtype=np.float32)
        t = min_train
        while t < n:
            usable = np.flatnonzero(np.isfinite(y[: max(0, t - horizon)]))
            if len(usable) >= 50:
                usable = usable[-max_train:]
                self.fit(F[usable], y[usable])
                stop = min(n, t + refit_every)
                out[t:stop] = self.predict(F[t:stop])
            t += refit_every
        return out

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"n_features": self.n_features, "gamma": self.gamma, "ridge": self.ridge, "seed": self.seed,
                "mu": None if self.mu is None else self.mu.tolist(), "sd": None if self.sd is None else self.sd.tolist(),
                "alpha": None if self.alpha is None else self.alpha.tolist(), "y_mean": getattr(self, "y_mean", 0.0),
                "Z_train": None if self.Z_train is None else self.Z_train.tolist(), "dim": None if self.W is None else int(self.W.shape[0])}

    @classmethod
    def from_dict(cls, d: dict) -> "ComplexityTimer":
        t = cls(int(d["n_features"]), float(d["gamma"]), float(d["ridge"]), int(d["seed"]))
        if d.get("dim"):
            t._init(int(d["dim"]))
        if d.get("mu") is not None:
            t.mu, t.sd = np.asarray(d["mu"], dtype=np.float32), np.asarray(d["sd"], dtype=np.float32)
            t.alpha, t.y_mean = np.asarray(d["alpha"], dtype=np.float64), float(d.get("y_mean", 0.0))
            t.Z_train = np.asarray(d["Z_train"], dtype=np.float32)
        return t
