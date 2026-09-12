"""Evolution-strategy trading agent - a port of huseinzol05/Stock-Prediction-Models
``realtime-agent`` (also the agent inside crypto-code/Stock-Market-AI-GUI).

The original: a one-hidden-layer network maps the last ``window_size`` price changes to
{hold, buy, sell}; its weights are optimised with a Deep Evolution Strategy (population 15,
sigma 0.1, lr 0.03) to maximise the profit of trading one unit per signal.  Here the agent is
trained per ticker on the training period only, and its live buy / sell probabilities become
two features for the policy.  It overfits happily - that is fine, the policy learns how much to
trust it - and it is cheap: a couple of seconds per ticker.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .base import SignalProvider

log = get_logger(__name__)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def make_states(close: np.ndarray, window: int) -> np.ndarray:
    """(T, window) matrix of the last ``window`` percentage changes (x100), zero-padded at the start."""
    pct = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-9) * 100.0
    T = len(close)
    out = np.zeros((T, window), dtype=np.float64)
    for k in range(window):
        out[k:, window - 1 - k] = pct[: T - k]
    return np.clip(out, -20, 20)


class ESModel:
    def __init__(self, input_size: int, layer_size: int, output_size: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.standard_normal((input_size, layer_size)) * np.sqrt(1.0 / (input_size + layer_size)),
            rng.standard_normal((layer_size, output_size)) * np.sqrt(1.0 / (layer_size + output_size)),
            np.zeros((1, layer_size)),
        ]

    def predict(self, x: np.ndarray, weights: list[np.ndarray] | None = None) -> np.ndarray:
        w = weights or self.weights
        h = np.maximum(x @ w[0] + w[2], 0.0)  # the original uses a linear hidden layer; ReLU keeps it stable
        return h @ w[1]


def simulate_profit(logits: np.ndarray, close: np.ndarray, capital: float) -> float:
    """Original reward: buy one unit on 'buy' when affordable, sell the oldest unit on 'sell'."""
    actions = np.argmax(logits, axis=1)
    cash = capital
    inventory: list[float] = []
    for t in range(len(close) - 1):
        a, price = actions[t], close[t]
        if a == 1 and cash >= price:
            inventory.append(price)
            cash -= price
        elif a == 2 and inventory:
            cash += price
            inventory.pop(0)
    final = cash + len(inventory) * close[-1]
    return (final - capital) / capital * 100.0


class DeepEvolutionStrategy:
    def __init__(self, model: ESModel, states: np.ndarray, close: np.ndarray, population: int = 15,
                 sigma: float = 0.1, lr: float = 0.03, seed: int = 0):
        self.model, self.states, self.close = model, states, close
        self.population, self.sigma, self.lr = population, sigma, lr
        self.rng = np.random.default_rng(seed)
        self.capital = float(close.mean() * 10)  # enough for a handful of units

    def train(self, iterations: int = 100) -> float:
        best = -np.inf
        for _ in range(iterations):
            noise = [[self.rng.standard_normal(w.shape) for w in self.model.weights] for _ in range(self.population)]
            rewards = np.zeros(self.population)
            for k in range(self.population):
                trial = [w + self.sigma * n for w, n in zip(self.model.weights, noise[k])]
                rewards[k] = simulate_profit(self.model.predict(self.states, trial), self.close, self.capital)
            best = max(best, float(rewards.max()))
            std = rewards.std()
            if std < 1e-9:
                continue
            adv = (rewards - rewards.mean()) / std
            for i, w in enumerate(self.model.weights):
                grad = sum(adv[k] * noise[k][i] for k in range(self.population)) / (self.population * self.sigma)
                self.model.weights[i] = w + self.lr * grad
        return best


class ESAgentSignal(SignalProvider):
    name = "es_agent"
    feature_names = ["es_buy", "es_sell", "es_action"]
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.window = int(self.cfg.get("window_size", 20))
        self.layer_size = int(self.cfg.get("layer_size", 64))
        self.iterations = int(self.cfg.get("iterations", 100))
        self.train_bars = int(self.cfg.get("train_bars", 1500))
        # the agent is fitted on bars ending this many years before train_end and its outputs over
        # its own fit window are masked, so the policy only ever sees it out of sample
        self.holdout_years = float(self.cfg.get("holdout_years", 4))
        self.models: dict[str, list[np.ndarray]] = {}
        self.fit_end: dict[str, str] = {}

    def _fit_cutoff(self, train_end: str | None) -> pd.Timestamp | None:
        if not train_end:
            return None
        return pd.Timestamp(train_end) - pd.DateOffset(years=self.holdout_years)

    def fit(self, frames: dict[str, pd.DataFrame], train_end: str | None = None) -> None:
        cutoff = self._fit_cutoff(train_end)
        self.models, self.fit_end = {}, {}
        for i, (t, df) in enumerate(frames.items()):
            sub = df[df.index <= cutoff] if cutoff is not None else df
            if len(sub) < self.window + 50:  # short history: fit on the first 60% of the training data instead
                end_ts = pd.Timestamp(train_end) if train_end else df.index[-1]
                train_part = df[df.index <= end_ts]
                sub = train_part.iloc[: int(len(train_part) * 0.6)]
            close = sub["close"].to_numpy(float)[-self.train_bars:]
            if len(close) < self.window + 50:
                continue
            states = make_states(close, self.window)
            model = ESModel(self.window, self.layer_size, 3, seed=i)
            best = DeepEvolutionStrategy(model, states, close, seed=i).train(self.iterations)
            self.models[t] = [w.copy() for w in model.weights]
            self.fit_end[t] = str(sub.index[-1].date())
            log.debug("es_agent %s: best training profit %.1f%%", t, best)
        log.info("evolution-strategy agents fitted for %d tickers (fit window ends %s)", len(self.models),
                 cutoff.date() if cutoff is not None else "end of data")

    def save_state(self) -> None:
        if self.models:
            with open(self.state_path("es_agent.pkl"), "wb") as f:
                pickle.dump({"window": self.window, "layer_size": self.layer_size, "models": self.models,
                             "fit_end": self.fit_end}, f)

    def load_state(self) -> bool:
        p = self.state_path("es_agent.pkl")
        if not p.exists():
            return False
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
            if d.get("window") == self.window and d.get("layer_size") == self.layer_size:
                self.models = d["models"]
                self.fit_end = d.get("fit_end", {})
                return True
        except Exception as e:  # noqa: BLE001
            log.warning("could not load es_agent state: %s", e)
        return False

    def availability(self) -> tuple[bool, str]:
        if not self.models and not self.load_state():
            return False, "not fitted yet (run `stockbot build-dataset`)"
        return True, f"ES agents for {len(self.models)} tickers (huseinzol05 port)"

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        weights = self.models.get(ticker)
        if weights is None:
            return None
        close = df["close"].to_numpy(float)
        model = ESModel(self.window, self.layer_size, 3)
        probs = softmax(model.predict(make_states(close, self.window), weights))
        action = np.argmax(probs, axis=1).astype(float)  # 0 hold, 1 buy, 2 sell
        out = np.column_stack([probs[:, 1], probs[:, 2], np.where(action == 1, 1.0, np.where(action == 2, -1.0, 0.0))]).astype(np.float32)
        out[: self.window] = np.nan
        fit_end = self.fit_end.get(ticker)
        if fit_end:  # never expose the agent's own training window
            out[df.index <= pd.Timestamp(fit_end)] = np.nan
        return out
