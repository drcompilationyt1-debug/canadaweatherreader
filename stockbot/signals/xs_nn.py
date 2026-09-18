"""A second cross-sectional ranking head: a small neural network on the same panel the LightGBM head ranks.

Gu, Kelly and Xiu (2020, "Empirical Asset Pricing via Machine Learning") found shallow neural networks the strongest
out-of-sample return predictors on a 30,000-stock panel, ahead of boosted trees, with the two making different errors -
which is why an average of the two rankers beats either.  This head is that second opinion: the same features and the
same walk-forward yearly refits as ``xs_rank`` (a 20-day relative-return target), scored by a 2-layer network with early
stopping, and its ``nn_score`` sits in the rank blend's candidate inputs so the weekend tuner decides its weight.
"""
from __future__ import annotations

import numpy as np

from ..logging_utils import get_logger
from .xs_rank import XSRankSignal

log = get_logger(__name__)


class _MLP:
    """Standardise, then a small MLP with early stopping (scikit-learn); the fit/predict shape the ranking head expects."""

    def __init__(self, hidden=(64, 32), seed: int = 0, max_iter: int = 60):
        self.hidden, self.seed, self.max_iter = tuple(int(h) for h in hidden), int(seed), int(max_iter)
        self.mu = self.sd = None
        self.net = None

    def fit(self, X, y):
        from sklearn.neural_network import MLPRegressor

        X = np.asarray(X, dtype=np.float32)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.net = MLPRegressor(hidden_layer_sizes=self.hidden, activation="relu", solver="adam", alpha=1e-4, batch_size=2048,
                                learning_rate_init=1e-3, max_iter=self.max_iter, early_stopping=True, validation_fraction=0.1,
                                n_iter_no_change=6, random_state=self.seed)
        self.net.fit(Z, np.asarray(y, dtype=np.float32))
        return self

    def predict(self, X):
        Z = (np.asarray(X, dtype=np.float32) - self.mu) / self.sd
        return self.net.predict(Z)


class XSNNSignal(XSRankSignal):
    name = "xs_nn"
    feature_names = ["nn_score", "nn_rank", "nn_top", "nn_bottom"]
    STATE_FILE = "xs_nn.joblib"
    PREDS_FILE = "xs_nn_preds.parquet"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.hidden = tuple(int(h) for h in (self.cfg.get("hidden", [64, 32]) or [64, 32]))
        self.max_iter = int(self.cfg.get("max_iter", 60))
        self.max_train_rows = int(self.cfg.get("max_train_rows", 150_000))

    def availability(self) -> tuple[bool, str]:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            return False, "pip install scikit-learn"
        return True, f"neural cross-sectional ranker {self.hidden} on the other blocks, {self.horizon}-day relative return, walk-forward"

    def _lgbm(self):                       # the ranking head's model factory: a network instead of the trees
        return _MLP(self.hidden, seed=0, max_iter=self.max_iter)
