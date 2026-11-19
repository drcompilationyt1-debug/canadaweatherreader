"""A third cross-sectional ranking head: TabPFN, the tabular foundation model, on the same panel.

TabPFN (Hollmann et al., Nature 2025; v2.5 in late 2025) is a transformer pre-trained on synthetic tables that predicts
by in-context learning - no training run, no tuning - and beats tuned boosted trees on small and medium tables.  Our
panel is far larger than its context, so each yearly walk-forward refit sees a random sample of ``max_train_rows`` of
the past, only the last ``years_back`` years are refit and only every ``stride``-th row is scored and carried forward
(CPU inference runs at about 50 ms a row); earlier years stay unscored, which
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

    # TabPFN refuses a context over 5000 rows on a CPU; the GPU-trained state keeps 10000, so the daily CPU session failed on
    # every load from Oct 2026 on and the live ranking ran without this head - on a CPU the context is cut to the CPU recipe's size
    CPU_ROWS = 4000

    def __init__(self, max_rows: int = 4000, chunk: int = 2000, seed: int = 0, model_cls=None, stride: int = 5):
        self.max_rows, self.chunk, self.seed, self.model_cls = int(max_rows), int(chunk), int(seed), model_cls
        self.stride = max(1, int(stride))     # CPU inference is slow: score every stride-th row and carry it (rows are per name, in date order)
        self.model = None
        self.X_ctx = self.y_ctx = None        # the sampled training rows: for TabPFN the fit *is* this context

    @staticmethod
    def _device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _make(self):
        cls = self.model_cls
        if cls is None:
            from tabpfn import TabPFNRegressor

            cls = TabPFNRegressor
        code = getattr(cls.__init__, "__code__", None)
        return cls(device=self._device()) if code is not None and "device" in code.co_varnames else cls()

    def _sample(self, X, y, n: int):
        if len(X) > n:
            keep = np.random.default_rng(self.seed).choice(len(X), n, replace=False)
            X, y = X[keep], y[keep]
        return X, y

    def _fit_model(self, X, y):
        if self._device() == "cpu":
            X, y = self._sample(X, y, self.CPU_ROWS)
        m = self._make()
        m.fit(X, y)
        self.model = m                        # only a finished fit is kept (a failed one left an unfitted model for the next call)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        X, y = self._sample(X, y, self.max_rows)
        self.X_ctx, self.y_ctx = X, y
        self.model = None
        self._fit_model(X, y)
        return self

    def _ensure(self):
        if self.model is None and self.X_ctx is not None:        # after a load: rebuild the fit from the saved context
            self._fit_model(self.X_ctx, self.y_ctx)

    def __getstate__(self):
        d = dict(self.__dict__)
        d["model"] = None                     # the foundation model's weights are 900 MB and downloadable: never in the state
        return d

    def predict(self, X):
        self._ensure()
        X = np.asarray(X, dtype=np.float32)
        if len(X) == 0:
            return np.empty(0, dtype=np.float32)
        sub = X[:: self.stride] if self.stride > 1 else X
        out = np.empty(len(sub), dtype=np.float32)
        for i in range(0, len(sub), self.chunk):
            out[i:i + self.chunk] = np.asarray(self.model.predict(sub[i:i + self.chunk]), dtype=np.float32)
        return np.repeat(out, self.stride)[: len(X)] if self.stride > 1 else out


class XSTabPFNSignal(XSRankSignal):
    name = "xs_tabpfn"
    feature_names = ["pfn_score", "pfn_rank", "pfn_top", "pfn_bottom"]
    STATE_FILE = "xs_tabpfn.joblib"
    PREDS_FILE = "xs_tabpfn_preds.parquet"
    model_cls = None                      # tests inject a stand-in

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        gpu = False
        try:
            import torch

            gpu = torch.cuda.is_available()
        except ImportError:
            pass
        g = dict(self.cfg.get("gpu", {}) or {}) if gpu else {}             # a GPU affords the full recipe
        self.max_train_rows = int(g.get("max_train_rows", self.cfg.get("max_train_rows", 4000)))
        self.chunk = int(g.get("chunk", self.cfg.get("chunk", 2000)))
        self.years_back = int(g.get("years_back", self.cfg.get("years_back", 2)))
        self.stride = int(g.get("stride", self.cfg.get("stride", 5)))
        self.trained_on = "gpu" if gpu else "cpu"

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
        return _TabPFN(self.max_train_rows, self.chunk, seed=0, model_cls=self.model_cls, stride=self.stride)
