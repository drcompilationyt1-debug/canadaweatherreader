"""Parallel simulators (vectorised environments) for training."""
from __future__ import annotations

import sys

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from .dataset import MarketDataset
from .trading_env import TradingEnv


def make_vec_env(dataset: MarketDataset, env_cfg: dict, n_envs: int = 8, seed: int = 0, vec: str = "auto",
                 tickers: list[str] | None = None, recent_bias: float = 0.0, recent_days: int = 400,
                 eval_mode: bool = False) -> VecEnv:
    def factory(rank: int):
        def _init():
            env = TradingEnv(dataset, env_cfg, tickers=tickers, seed=seed + rank, eval_mode=eval_mode,
                             recent_bias=recent_bias, recent_days=recent_days)
            return Monitor(env)
        return _init

    fns = [factory(i) for i in range(max(1, n_envs))]
    if vec == "auto":
        vec = "dummy" if (sys.platform == "win32" or n_envs == 1) else "subproc"
    if vec == "subproc":
        return SubprocVecEnv(fns, start_method="fork" if sys.platform != "win32" else "spawn")
    return DummyVecEnv(fns)
