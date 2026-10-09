"""A second cross-sectional ranking head: an ensemble of small neural networks on the same panel the LightGBM head ranks.

Gu, Kelly and Xiu (2020, "Empirical Asset Pricing via Machine Learning") found shallow neural networks the strongest
out-of-sample return predictors on a 30,000-stock panel - their best, "NN3" (32-16-8 with batch norm, an L1 penalty, early
stopping and an ensemble of ten seeds) - ahead of boosted trees, with the two making different errors.  This head is that
second opinion: the same features and walk-forward yearly refits as ``xs_rank`` (a 20-day relative-return target).  It trains
on whatever device it finds: a GPU (Kaggle / Colab / a PC, ``stockbot gpu-train``) gets the full recipe, a CPU runner a
lighter one - and with ``pretrained`` the runner reuses a GPU-trained state and only scores new days.
"""
from __future__ import annotations

import numpy as np

from ..logging_utils import get_logger
from .xs_rank import XSRankSignal

log = get_logger(__name__)


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class _TorchEnsemble:
    """Standardise, then ``seeds`` MLPs with batch norm, dropout, an L1 penalty and early stopping on a held-out 15%; the
    prediction is their average.  Pickles to CPU weights only (loads on any machine)."""

    def __init__(self, hidden=(32, 16, 8), seeds: int = 10, epochs: int = 60, patience: int = 6, lr: float = 1e-3, l1: float = 1e-5,
                 batch: int = 4096, device: str | None = None):
        self.hidden, self.seeds, self.epochs, self.patience = tuple(int(h) for h in hidden), int(seeds), int(epochs), int(patience)
        self.lr, self.l1, self.batch = float(lr), float(l1), int(batch)
        self.device = device or _device()
        self.mu = self.sd = None
        self.states: list[dict] = []
        self._nets = None

    def _net(self, d_in: int):
        from torch import nn

        layers, prev = [], d_in
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        return nn.Sequential(*layers, nn.Linear(prev, 1))

    def fit(self, X, y):
        import torch

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-6
        Z = (X - self.mu) / self.sd
        dev = torch.device(self.device)
        Xt, yt = torch.tensor(Z, device=dev), torch.tensor(y, device=dev)
        self.states, self._nets = [], None
        for seed in range(self.seeds):
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(Z), generator=g)
            n_val = max(1, int(0.15 * len(Z)))
            va, tr = perm[:n_val].to(dev), perm[n_val:].to(dev)
            torch.manual_seed(seed)
            net = self._net(Z.shape[1]).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            best, best_state, bad = np.inf, None, 0
            bs = int(min(self.batch, max(128, len(tr) // 32)))           # at least ~32 steps an epoch, whatever the data size
            for _ in range(self.epochs):
                net.train()
                order = tr[torch.randperm(len(tr), device=dev)]
                for b in range(0, len(order), bs):
                    j = order[b:b + bs]
                    if len(j) < 2:
                        continue
                    opt.zero_grad()
                    loss = ((net(Xt[j]).squeeze(-1) - yt[j]) ** 2).mean()
                    loss = loss + self.l1 * sum(p.abs().sum() for n_, p in net.named_parameters() if n_.endswith("weight"))
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = float(((net(Xt[va]).squeeze(-1) - yt[va]) ** 2).mean())
                if v < best - 1e-6:
                    best, bad = v, 0
                    best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            self.states.append(best_state)
        return self

    def _ensure(self):
        if self._nets is None:
            nets = []
            for st in self.states:
                net = self._net(len(self.mu))
                net.load_state_dict(st)
                net.eval()
                nets.append(net)
            self._nets = nets

    def predict(self, X):
        import torch

        self._ensure()
        Z = torch.tensor((np.asarray(X, dtype=np.float32) - self.mu) / self.sd)
        out = np.zeros(len(Z), dtype=np.float64)
        with torch.no_grad():
            for net in self._nets:
                for b in range(0, len(Z), 65536):
                    out[b:b + 65536] += net(Z[b:b + 65536]).squeeze(-1).numpy()
        return (out / max(1, len(self._nets))).astype(np.float32)

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_nets"] = None
        d["device"] = "cpu"
        return d


class XSNNSignal(XSRankSignal):
    name = "xs_nn"
    feature_names = ["nn_score", "nn_rank", "nn_top", "nn_bottom"]
    STATE_FILE = "xs_nn.joblib"
    PREDS_FILE = "xs_nn_preds.parquet"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.device = _device()
        gpu = self.device == "cuda"
        self.hidden = tuple(int(h) for h in (self.cfg.get("hidden", [32, 16, 8]) or [32, 16, 8]))
        self.seeds = int(self.cfg.get("seeds_gpu" if gpu else "seeds_cpu", 10 if gpu else 2))
        self.epochs = int(self.cfg.get("epochs_gpu" if gpu else "epochs_cpu", 60 if gpu else 15))
        self.max_train_rows = int(self.cfg.get("max_train_rows_gpu" if gpu else "max_train_rows", 2_000_000 if gpu else 150_000))
        self.trained_on = "gpu" if gpu else "cpu"

    def availability(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "pip install torch"
        return True, (f"neural-network ensemble ({self.seeds} x {self.hidden}) on the other blocks, {self.horizon}-day relative return, "
                      f"walk-forward on {self.device}" + (", reusing a pretrained state when fresh" if self.pretrained else ""))

    def _lgbm(self):                       # the ranking head's model factory: a network ensemble instead of the trees
        return _TorchEnsemble(self.hidden, seeds=self.seeds, epochs=self.epochs, device=self.device)
