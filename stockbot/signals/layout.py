"""Observation layout: the fixed ordering of signal blocks (each = 1 availability flag + features)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PORTFOLIO_FEATURES = ["exposure", "equity_ret", "drawdown", "position_age", "is_long", "is_short"]


@dataclass(frozen=True)
class Block:
    name: str
    size: int                 # number of features (excluding the availability flag)
    feature_names: tuple[str, ...]
    offset: int               # index of the availability flag inside the signal vector

    @property
    def start(self) -> int:   # first feature column
        return self.offset + 1

    @property
    def end(self) -> int:     # one past the last feature column
        return self.offset + 1 + self.size


class ObservationLayout:
    def __init__(self, blocks: list[tuple[str, list[str]]]):
        self.blocks: list[Block] = []
        off = 0
        for name, names in blocks:
            self.blocks.append(Block(name, len(names), tuple(names), off))
            off += 1 + len(names)
        self.signal_dim = off
        self.portfolio_dim = len(PORTFOLIO_FEATURES)
        self.obs_dim = self.signal_dim + self.portfolio_dim

    # ------------------------------------------------------------------ helpers
    def block(self, name: str) -> Block:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(name)

    @property
    def names(self) -> list[str]:
        return [b.name for b in self.blocks]

    def column_names(self) -> list[str]:
        cols: list[str] = []
        for b in self.blocks:
            cols.append(f"{b.name}.available")
            cols.extend(f"{b.name}.{f}" for f in b.feature_names)
        cols.extend(f"portfolio.{f}" for f in PORTFOLIO_FEATURES)
        return cols

    def signature(self) -> str:
        payload = json.dumps([(b.name, list(b.feature_names)) for b in self.blocks])
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {"blocks": [{"name": b.name, "features": list(b.feature_names)} for b in self.blocks],
                "signature": self.signature(), "obs_dim": self.obs_dim}

    @classmethod
    def from_dict(cls, d: dict) -> "ObservationLayout":
        return cls([(b["name"], list(b["features"])) for b in d["blocks"]])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ObservationLayout":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ assembly
    def assemble(self, n_rows: int, arrays: dict[str, np.ndarray | None]) -> np.ndarray:
        """Stack per-block arrays (T, size) into a (T, signal_dim) float32 matrix with flags."""
        out = np.zeros((n_rows, self.signal_dim), dtype=np.float32)
        for b in self.blocks:
            arr = arrays.get(b.name)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.float32)
            if arr.shape != (n_rows, b.size):
                continue
            ok = ~np.isnan(arr).any(axis=1)
            out[:, b.offset] = ok.astype(np.float32)
            feats = np.where(ok[:, None], np.nan_to_num(arr, nan=0.0), 0.0)
            out[:, b.start:b.end] = np.clip(feats, -10.0, 10.0)
        return out

    def assemble_latest(self, vectors: dict[str, np.ndarray | None]) -> np.ndarray:
        """Same as ``assemble`` for a single bar."""
        arrays = {k: (None if v is None else np.asarray(v, dtype=np.float32).reshape(1, -1)) for k, v in vectors.items()}
        return self.assemble(1, arrays)[0]

    def availability_of(self, signal_vec: np.ndarray) -> dict[str, bool]:
        return {b.name: bool(signal_vec[b.offset] > 0.5) for b in self.blocks}
