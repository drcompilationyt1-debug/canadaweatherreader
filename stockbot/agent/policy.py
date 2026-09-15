"""Policy construction / persistence (stable-baselines3 PPO or SAC)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_utils import get_logger
from ..signals.layout import ObservationLayout

log = get_logger(__name__)

ALGOS = ("ppo", "sac")


def build_model(algo: str, env, train_cfg: dict, seed: int = 0):
    algo = (algo or "ppo").lower()
    net_arch = [int(x) for x in train_cfg.get("net_arch", [256, 256])]
    device = train_cfg.get("device", "cpu")
    lr = float(train_cfg.get("learning_rate", 3e-4))
    if algo == "ppo":
        from stable_baselines3 import PPO

        return PPO(
            "MlpPolicy", env, learning_rate=lr, n_steps=int(train_cfg.get("n_steps", 512)),
            batch_size=int(train_cfg.get("batch_size", 256)), n_epochs=int(train_cfg.get("n_epochs", 10)),
            gamma=float(train_cfg.get("gamma", 0.99)), gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
            ent_coef=float(train_cfg.get("ent_coef", 0.001)), clip_range=float(train_cfg.get("clip_range", 0.2)),
            policy_kwargs={"net_arch": dict(pi=net_arch, vf=net_arch)}, seed=seed, device=device, verbose=0,
        )
    if algo == "sac":
        from stable_baselines3 import SAC

        return SAC(
            "MlpPolicy", env, learning_rate=lr, buffer_size=int(train_cfg.get("buffer_size", 200_000)),
            batch_size=int(train_cfg.get("batch_size", 256)), gamma=float(train_cfg.get("gamma", 0.99)),
            train_freq=1, gradient_steps=1, learning_starts=int(train_cfg.get("learning_starts", 5000)),
            policy_kwargs={"net_arch": net_arch}, seed=seed, device=device, verbose=0,
        )
    raise ValueError(f"unknown algo {algo!r}; choose from {ALGOS}")


def load_model(path: str | Path, algo: str = "ppo", env=None, device: str = "cpu"):
    algo = (algo or "ppo").lower()
    if algo == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(str(path), env=env, device=device)
    from stable_baselines3 import SAC

    return SAC.load(str(path), env=env, device=device)


class EnsemblePolicy:
    """Several independently trained policies whose deterministic actions are averaged.

    Averaging over seeds removes most of the run-to-run variance of PPO and makes the live
    behaviour far more stable than any single run.
    """

    def __init__(self, members: list[Any], paths: list[str] | None = None):
        self.members = members
        self.paths = paths or []

    def predict(self, obs, deterministic: bool = True):
        acts = [np.asarray(m.predict(obs, deterministic=deterministic)[0], dtype=np.float64) for m in self.members]
        return np.mean(acts, axis=0), None

    @property
    def num_timesteps(self) -> int:
        return int(sum(getattr(m, "num_timesteps", 0) for m in self.members))

    def save(self, path: str) -> None:  # members are saved by their own training runs
        return None


@dataclass
class PolicyBundle:
    """A trained model + the observation layout it expects + training metadata."""

    model: Any
    layout: ObservationLayout
    meta: dict
    folder: Path | None = None

    def model_obs_dim(self) -> int | None:
        """The observation size the trained network actually takes (None for models without an observation space)."""
        m = self.model.members[0] if isinstance(self.model, EnsemblePolicy) and self.model.members else self.model
        space = getattr(m, "observation_space", None)
        shape = getattr(space, "shape", None)
        return int(shape[0]) if shape else None

    def fit_obs(self, obs: np.ndarray) -> np.ndarray:
        """Trim portfolio features added after the model was trained (they are always appended at the end), so a
        policy from the last retrain keeps trading until the next one learns the new feature."""
        obs = np.asarray(obs, dtype=np.float32)
        expected = self.model_obs_dim() or self.layout.obs_dim
        if obs.shape[-1] == expected:
            return obs
        if obs.shape[-1] > expected >= self.layout.signal_dim:
            return obs[..., :expected]
        raise ValueError(f"observation has {obs.shape[-1]} dims, model expects {expected}")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> float:
        obs = self.fit_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1))
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))

    @property
    def algo(self) -> str:
        return self.meta.get("algo", "ppo")

    def save(self, folder: str | Path, name: str = "latest") -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if isinstance(self.model, EnsemblePolicy):
            (folder / "ensemble.json").write_text(json.dumps({"members": self.model.paths, "signature": self.layout.signature()}, indent=1), encoding="utf-8")
        else:
            self.model.save(str(folder / f"{name}.zip"))
        self.layout.save(folder / "layout.json")
        meta = dict(self.meta)
        meta.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
        meta["signature"] = self.layout.signature()
        (folder / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        self.folder = folder
        return folder / f"{name}.zip"

    @classmethod
    def load(cls, folder: str | Path, name: str = "latest", device: str = "cpu") -> "PolicyBundle":
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8")) if (folder / "meta.json").exists() else {}
        layout = ObservationLayout.load(folder / "layout.json")
        ens = folder / "ensemble.json"
        if ens.exists():
            info = json.loads(ens.read_text(encoding="utf-8"))
            members, paths = [], []
            for p in info.get("members", []):
                mp = Path(p)
                mp = mp if mp.is_absolute() else folder / mp
                if mp.exists():
                    members.append(load_model(mp, meta.get("algo", "ppo"), device=device))
                    paths.append(str(mp))
            if members:
                log.info("loaded ensemble of %d policies from %s (layout %s)", len(members), folder, layout.signature())
                return cls(EnsemblePolicy(members, paths), layout, meta, folder)
        path = folder / f"{name}.zip"
        if not path.exists():
            path = folder / "latest.zip"
        model = load_model(path, meta.get("algo", "ppo"), device=device)
        log.info("loaded policy %s (%s, layout %s)", path, meta.get("algo", "ppo"), layout.signature())
        return cls(model, layout, meta, folder)

    @classmethod
    def exists(cls, folder: str | Path, name: str = "latest") -> bool:
        folder = Path(folder)
        if not (folder / "layout.json").exists():
            return False
        return (folder / "ensemble.json").exists() or (folder / f"{name}.zip").exists() or (folder / "latest.zip").exists()
