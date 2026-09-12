"""DQN trading agent - a PyTorch port of henryboisdequin/AI-Stock-Trader (state = shares owned,
price, cash; actions sell / hold / buy; MLP with one hidden layer of 32; gamma 0.95; epsilon 1 ->
0.01 decaying by 0.995; replay buffer 500; batch 32; 20 episodes) with the price-window input of
romaingrx/stockBot's conv1d agent (look-back 15) added to the state.

Trained per ticker on the training period only.  Features: how much the agent prefers buying
when flat and selling when long (Q-value gaps), plus its greedy action.  Like the ES agent it is
an overfitted little trader whose opinion the policy learns to weigh.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .base import SignalProvider

log = get_logger(__name__)


def price_windows(close: np.ndarray, look_back: int) -> np.ndarray:
    pct = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-9) * 100.0
    T = len(close)
    out = np.zeros((T, look_back), dtype=np.float32)
    for k in range(look_back):
        out[k:, look_back - 1 - k] = pct[: T - k]
    return np.clip(out, -20, 20)


class SingleStockEnv:
    """AI-Stock-Trader's StockTradingEnv for one stock: 0 sell all, 1 hold, 2 buy as much as cash allows."""

    def __init__(self, close: np.ndarray, windows: np.ndarray, initial_investment: float = 20_000.0):
        self.close, self.windows, self.initial = close, windows, float(initial_investment)
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.shares = 0.0
        self.cash = self.initial
        return self.obs()

    def value(self) -> float:
        return self.cash + self.shares * self.close[self.t]

    def obs(self) -> np.ndarray:
        return np.concatenate([[self.shares, self.close[self.t], self.cash], self.windows[self.t]]).astype(np.float32)

    def step(self, action: int):
        before = self.value()
        price = self.close[self.t]
        if action == 0 and self.shares > 0:
            self.cash += self.shares * price
            self.shares = 0.0
        elif action == 2 and self.cash >= price:
            n = np.floor(self.cash / price)
            self.shares += n
            self.cash -= n * price
        self.t += 1
        done = self.t >= len(self.close) - 1
        return self.obs(), self.value() - before, done


def _scaler(env: SingleStockEnv, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Their get_scaler: StandardScaler fitted on states from a random-play episode."""
    states = []
    s = env.reset()
    done = False
    while not done:
        states.append(s)
        s, _, done = env.step(int(rng.integers(3)))
    arr = np.array(states)
    return arr.mean(axis=0), arr.std(axis=0) + 1e-6


class DQNAgentSignal(SignalProvider):
    name = "dqn_agent"
    feature_names = ["dqn_buy_pref", "dqn_sell_pref", "dqn_action"]
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.look_back = int(self.cfg.get("look_back", 15))
        self.episodes = int(self.cfg.get("episodes", 12))
        self.train_bars = int(self.cfg.get("train_bars", 1000))
        self.hidden = int(self.cfg.get("hidden", 32))
        # fitted on bars ending holdout_years before train_end; outputs over the fit window are masked
        self.holdout_years = float(self.cfg.get("holdout_years", 4))
        self.models: dict[str, dict] = {}

    # ------------------------------------------------------------------ network
    def _net(self):
        import torch
        from torch import nn

        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(3 + self.look_back, self.hidden), nn.ReLU(), nn.Linear(self.hidden, 3))

    def _train_one(self, close: np.ndarray, seed: int) -> dict | None:
        import torch

        rng = np.random.default_rng(seed)
        windows = price_windows(close, self.look_back)
        env = SingleStockEnv(close, windows)
        mean, std = _scaler(env, rng)
        net = self._net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        gamma, eps, eps_min, eps_decay = 0.95, 1.0, 0.01, 0.995
        buf_s, buf_a, buf_r, buf_s2, buf_d = [], [], [], [], []
        for _ in range(self.episodes):
            s = (env.reset() - mean) / std
            done = False
            while not done:
                if rng.random() <= eps:
                    a = int(rng.integers(3))
                else:
                    with torch.no_grad():
                        a = int(torch.argmax(net(torch.from_numpy(s[None]))).item())
                s2, r, done = env.step(a)
                s2 = (s2 - mean) / std
                buf_s.append(s); buf_a.append(a); buf_r.append(r / env.initial * 100.0); buf_s2.append(s2); buf_d.append(float(done))
                if len(buf_s) > 500:
                    buf_s.pop(0); buf_a.pop(0); buf_r.pop(0); buf_s2.pop(0); buf_d.pop(0)
                s = s2
                if len(buf_s) >= 32:
                    idx = rng.integers(len(buf_s), size=32)
                    S = torch.from_numpy(np.array([buf_s[i] for i in idx], dtype=np.float32))
                    S2 = torch.from_numpy(np.array([buf_s2[i] for i in idx], dtype=np.float32))
                    A = torch.tensor([buf_a[i] for i in idx])
                    R = torch.tensor([buf_r[i] for i in idx], dtype=torch.float32)
                    D = torch.tensor([buf_d[i] for i in idx], dtype=torch.float32)
                    with torch.no_grad():
                        target = R + (1 - D) * gamma * net(S2).max(dim=1).values
                    q = net(S).gather(1, A[:, None]).squeeze(1)
                    loss = torch.nn.functional.mse_loss(q, target)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    if eps > eps_min:
                        eps *= eps_decay
        return {"state": {k: v.detach().clone() for k, v in net.state_dict().items()}, "mean": mean, "std": std}

    # ------------------------------------------------------------------ fit / persistence
    def fit(self, frames: dict[str, pd.DataFrame], train_end: str | None = None) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            return
        cutoff = (pd.Timestamp(train_end) - pd.DateOffset(years=self.holdout_years)) if train_end else None
        self.models = {}
        for i, (t, df) in enumerate(frames.items()):
            sub = df[df.index <= cutoff] if cutoff is not None else df
            if len(sub) < self.look_back + 100:  # short history: fit on the first 60% of the training data instead
                end_ts = pd.Timestamp(train_end) if train_end else df.index[-1]
                train_part = df[df.index <= end_ts]
                sub = train_part.iloc[: int(len(train_part) * 0.6)]
            close = sub["close"].to_numpy(float)[-self.train_bars:]
            if len(close) < self.look_back + 100:
                continue
            m = self._train_one(close, seed=i)
            if m is not None:
                m["fit_end"] = str(sub.index[-1].date())
                self.models[t] = m
        log.info("DQN agents fitted for %d tickers (fit window ends %s)", len(self.models),
                 cutoff.date() if cutoff is not None else "end of data")

    def save_state(self) -> None:
        if self.models:
            with open(self.state_path("dqn_agent.pkl"), "wb") as f:
                pickle.dump({"look_back": self.look_back, "hidden": self.hidden, "models": self.models}, f)

    def load_state(self) -> bool:
        p = self.state_path("dqn_agent.pkl")
        if not p.exists():
            return False
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
            if d.get("look_back") == self.look_back and d.get("hidden") == self.hidden:
                self.models = d["models"]
                return True
        except Exception as e:  # noqa: BLE001
            log.warning("could not load dqn_agent state: %s", e)
        return False

    def availability(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "needs torch"
        if not self.models and not self.load_state():
            return False, "not fitted yet (run `stockbot build-dataset`)"
        return True, f"DQN agents for {len(self.models)} tickers (AI-Stock-Trader / stockBot port)"

    # ------------------------------------------------------------------ features
    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        import torch

        m = self.models.get(ticker)
        if m is None:
            return None
        net = self._net()
        net.load_state_dict(m["state"])
        net.eval()
        close = df["close"].to_numpy(float)
        windows = price_windows(close, self.look_back)
        cash = 20_000.0
        flat = np.column_stack([np.zeros(len(close)), close, np.full(len(close), cash), windows]).astype(np.float32)
        shares = np.floor(cash / np.maximum(close, 1e-9))
        long_ = np.column_stack([shares, close, np.zeros(len(close)), windows]).astype(np.float32)
        with torch.no_grad():
            q_flat = net(torch.from_numpy((flat - m["mean"]) / m["std"])).numpy()
            q_long = net(torch.from_numpy((long_ - m["mean"]) / m["std"])).numpy()
        buy_pref = np.tanh(q_flat[:, 2] - q_flat[:, 1])
        sell_pref = np.tanh(q_long[:, 0] - q_long[:, 1])
        action = np.where(buy_pref > 0, 1.0, np.where(sell_pref > 0, -1.0, 0.0))
        out = np.column_stack([buy_pref, sell_pref, action]).astype(np.float32)
        out[: self.look_back] = np.nan
        if m.get("fit_end"):  # never expose the agent's own training window
            out[df.index <= pd.Timestamp(m["fit_end"])] = np.nan
        return out
