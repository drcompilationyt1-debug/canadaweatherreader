"""A third cross-sectional ranking head: TabPFN, the tabular foundation model, on the same panel.

TabPFN (Hollmann et al., Nature 2025; v2.5 in late 2025) is a transformer pre-trained on synthetic tables that predicts
by in-context learning - no training run, no tuning - and beats tuned boosted trees on small and medium tables.  Our
panel is far larger than its context, so each yearly walk-forward refit sees a random sample of ``max_train_rows`` of
the past and only the last ``years_back`` years are refit (CPU inference is slow); earlier years stay unscored, which
the blend tuner treats as an absent input.  The weights need a one-time licence acceptance at priorlabs.ai and the
``TABPFN_TOKEN`` secret; without it the block is masked and says so.
"""
from __future__ import annotations

import os

import numpy as np

from ..logging_utils import get_logger
from .xs_rank import XSRankSignal

log = get_logger(__name__)


class _TabPFN:
    """fit/predict around TabPFNRegressor: a random sample of the training rows as context, predictions in chunks."""

    def __init__(self, max_rows: int = 8000, chunk: int = 4000, seed: int = 0, model_cls=None):
        self.max_rows, self.chunk, self.seed, self.model_cls = int(max_rows), int(chunk), int(seed), model_cls
        self.model = None

    def fit(self, X, y):
        cls = self.model_cls
        if cls is None:
            from tabpfn import TabPFNRegressor

            cls = TabPFNRegressor
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        if len(X) > self.max_rows:
            keep = np.random.default_rng(self.seed).choice(len(X), self.max_rows, replace=False)
            X, y = X[keep], y[keep]
        self.model = cls(device="cpu") if cls is not None and "device" in getattr(cls.__init__, "__code__", type("c", (), {"co_varnames": ()})).co_varnames else cls()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float32)
        for i in range(0, len(X), self.chunk):
            out[i:i + self.chunk] = np.asarray(self.model.predict(X[i:i + self.chunk]), dtype=np.float32)
        return out


class XSTabPFNSignal(XSRankSignal):
    name = "xs_tabpfn"
    feature_names = ["pfn_score", "pfn_rank", "pfn_top", "pfn_bottom"]
    STATE_FILE = "xs_tabpfn.joblib"
    PREDS_FILE = "xs_tabpfn_preds.parquet"
    model_cls = None                      # tests inject a stand-in

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.max_train_rows = int(self.cfg.get("max_train_rows", 8000))
        self.chunk = int(self.cfg.get("chunk", 4000))
        self.years_back = int(self.cfg.get("years_back", 4))

    def availability(self) -> tuple[bool, str]:
        if self.model_cls is not None:
            return True, "tabular foundation model ranker (test stand-in)"
        try:
            import tabpfn  # noqa: F401
        except ImportError:
            return False, "pip install tabpfn"
        if not os.environ.get("TABPFN_TOKEN"):
            return False, "set TABPFN_TOKEN (accept the licence at ux.priorlabs.ai, then the token from your account)"
        return True, f"TabPFN ranker on the other blocks, {self.horizon}-day relative return, walk-forward over the last {self.years_back} years"

    def _lgbm(self):
        return _TabPFN(self.max_train_rows, self.chunk, seed=0, model_cls=self.model_cls)
