"""Gymnasium trading environment.

One episode = one ticker, one random window of ``episode_length`` bars.  Many of these run in
parallel (``make_vec_env``) and the *same* policy is trained on all of them, so it learns
behaviour that transfers across stocks and market regimes.

Action  : target exposure in [-1, 1] (fraction of equity; negative = short).  The mapping to
          BUY / SELL / SHORT / COVER / HOLD + amount is in ``agent.decide``.
Reward  : daily log return of equity (transaction costs, borrow fees included), optionally
          benchmark-relative, with penalties for shorting / turnover / variance / drawdown.
Robustness: at reset each signal block is hidden for the whole episode with probability
          ``signal_dropout`` so the policy never depends on any single input model.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..agent.sizing import conviction_to_exposure, realized_vol_series, size_exposure
from ..signals.layout import PORTFOLIO_FEATURES
from .dataset import MarketDataset, TickerData
from .portfolio import Portfolio

DEFAULT_ENV_CFG: dict[str, Any] = dict(
    episode_length=252, initial_cash=100_000.0, commission=0.0005, slippage=0.0005, allow_short=True,
    max_leverage=1.0, short_borrow_rate_annual=0.03, short_penalty=0.02, turnover_penalty=0.0,
    reward="log_return", benchmark_mix=0.0, variance_penalty=0.0, drawdown_penalty=0.0, reward_scale=100.0,
    signal_dropout=0.15, deadband=0.05, blowup_equity_frac=0.3,
    vol_target=0.0, vol_window=20, vol_max_scale=1.5,
)


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset: MarketDataset, env_cfg: dict | None = None, tickers: list[str] | None = None,
                 seed: int | None = None, eval_mode: bool = False, recent_bias: float = 0.0, recent_days: int = 400):
        super().__init__()
        self.ds = dataset
        self.layout = dataset.layout
        self.tickers = [t for t in (tickers or dataset.tickers) if t in dataset.data]
        if not self.tickers:
            raise ValueError("no tickers available in dataset")
        c = {**DEFAULT_ENV_CFG, **(env_cfg or {})}
        self.c = c
        self.episode_length = int(c["episode_length"])
        self.eval_mode = eval_mode
        self.signal_dropout = 0.0 if eval_mode else float(c["signal_dropout"])
        self.recent_bias = float(recent_bias)
        self.recent_days = int(recent_days)
        self.reward_scale = float(c["reward_scale"])
        fees = c.get("fees")
        self.fee_book = None
        if isinstance(fees, str):  # preset name from the config (e.g. "moomoo"); "bps" = proportional commission
            from ..execution.fees import FeeBook

            # one schedule per market: a Canadian episode pays moomoo Canada's fees, a US one moomoo's US fees
            self.fee_book = FeeBook.from_names(fees.lower(), {m: str(p).lower() for m, p in (c.get("fees_by_market") or {}).items()})
            fees = self.fee_book.default
        self.portfolio = Portfolio(c["initial_cash"], c["commission"], c["slippage"], c["allow_short"],
                                   c["short_borrow_rate_annual"], c["max_leverage"], fees=fees,
                                   fee_scale=float(c.get("fee_scale", 1.0) or 1.0))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.layout.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.block_mask = np.ones(self.layout.signal_dim, dtype=np.float32)
        self.td: TickerData | None = None
        self.t = 0
        self.end = 0
        self.vol_target = float(c.get("vol_target", 0.0) or 0.0)
        self._vol_cache: dict[str, np.ndarray] = {}
        self._vol: np.ndarray | None = None

    def _vol_for(self, td: TickerData) -> np.ndarray:
        if td.ticker not in self._vol_cache:
            self._vol_cache[td.ticker] = realized_vol_series(td.close, int(self.c.get("vol_window", 20)))
        return self._vol_cache[td.ticker]

    # ------------------------------------------------------------------ episode setup
    def _pick_window(self, options: dict | None) -> tuple[TickerData, int, int]:
        options = options or {}
        ticker = options.get("ticker") or self.tickers[int(self.rng.integers(len(self.tickers)))]
        td = self.ds.data[ticker]
        length = int(options.get("length") or self.episode_length)
        T = len(td)
        min_start = td.min_start
        max_start = T - 2 - length
        if max_start < min_start:
            length = T - 2 - min_start
            max_start = min_start
            if length < 5:
                raise ValueError(f"{ticker}: not enough bars ({T}) for an episode")
        if "start" in options and options["start"] is not None:
            start = int(min(max(options["start"], min_start), max_start))
        elif self.recent_bias > 0 and self.rng.random() < self.recent_bias:
            lo = max(min_start, T - 2 - self.recent_days - length)
            start = int(self.rng.integers(lo, max_start + 1)) if max_start >= lo else max_start
        else:
            start = int(self.rng.integers(min_start, max_start + 1))
        return td, start, start + length

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.td, self.t, self.end = self._pick_window(options)
        self._vol = self._vol_for(self.td) if self.vol_target > 0 else None
        if self.fee_book is not None:
            self.portfolio.fees = self.fee_book.for_ticker(self.td.ticker)
        self.portfolio.reset()
        self.initial_equity = self.portfolio.initial_cash
        self.peak = self.initial_equity
        self.pos_age = 0
        self.prev_exposure = 0.0
        self.equity_hist = [self.initial_equity]
        self.block_mask[:] = 1.0
        if self.signal_dropout > 0:
            for b in self.layout.blocks:
                if self.rng.random() < self.signal_dropout:
                    self.block_mask[b.offset:b.end] = 0.0
        return self._obs(), self._info(reward_raw=0.0, cost=0.0, bench=0.0)

    # ------------------------------------------------------------------ observation
    def _portfolio_state(self, price: float) -> np.ndarray:
        eq = self.portfolio.equity(price)
        exp = self.portfolio.exposure(price)
        return np.array([
            exp,
            np.clip(eq / self.initial_equity - 1.0, -1.0, 3.0) * 2.0,
            (1.0 - eq / self.peak) * 5.0,
            min(self.pos_age / 252.0, 2.0),
            float(exp > 0.01),
            float(exp < -0.01),
        ], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        sig = self.td.signals[self.t] * self.block_mask
        return np.concatenate([sig, self._portfolio_state(float(self.td.close[self.t]))]).astype(np.float32)

    def _info(self, **kw) -> dict:
        price = float(self.td.close[self.t])
        info = {"ticker": self.td.ticker, "t": self.t, "date": str(pd.Timestamp(self.td.dates[self.t]).date()),
                "price": price, "equity": self.portfolio.equity(price), "exposure": self.portfolio.exposure(price)}
        info.update(kw)
        return info

    # ------------------------------------------------------------------ step
    def step(self, action):
        a = conviction_to_exposure(float(np.asarray(action, dtype=np.float64).reshape(-1)[0]), bool(self.c["allow_short"]))
        if self._vol is not None:  # volatility-targeted sizing: the policy decides conviction, vol decides size
            target = size_exposure(a, float(self._vol[self.t]), self.vol_target, float(self.c["max_leverage"]),
                                   float(self.c.get("vol_max_scale", 1.5)))
        else:
            target = a * float(self.c["max_leverage"])
        target = self.portfolio.clip_target(target)
        price = float(self.td.close[self.t])
        eq_prev = self.portfolio.equity(price)
        cur = self.portfolio.exposure(price)
        if abs(target - cur) < float(self.c["deadband"]):
            target = cur
        trade = self.portfolio.rebalance(target, price)
        exp_after = self.portfolio.exposure(price)

        self.t += 1
        price_next = float(self.td.close[self.t])
        self.portfolio.accrue(price_next)
        eq_now = self.portfolio.equity(price_next)
        eq_now = max(eq_now, 1e-6)
        r = float(np.log(eq_now / max(eq_prev, 1e-6)))
        bench = float(np.log(price_next / price)) if price > 0 else 0.0

        reward = (r - float(self.c["benchmark_mix"]) * bench) * self.reward_scale
        if self.c["reward"] == "excess_log_return":
            reward = (r - bench) * self.reward_scale
        reward -= float(self.c["short_penalty"]) * max(0.0, -exp_after)
        reward -= float(self.c["turnover_penalty"]) * abs(exp_after - cur)
        reward -= float(self.c["variance_penalty"]) * (r * self.reward_scale) ** 2
        old_peak = self.peak
        self.peak = max(self.peak, eq_now)
        dd_increase = max(0.0, (1.0 - eq_now / self.peak) - (1.0 - eq_prev / old_peak))
        reward -= float(self.c["drawdown_penalty"]) * dd_increase * self.reward_scale

        if abs(exp_after) > 0.01 and np.sign(exp_after) == np.sign(self.prev_exposure):
            self.pos_age += 1
        else:
            self.pos_age = 0 if abs(exp_after) <= 0.01 else 1
        self.prev_exposure = exp_after
        self.equity_hist.append(eq_now)

        terminated = eq_now < float(self.c["blowup_equity_frac"]) * self.initial_equity
        truncated = self.t >= self.end
        info = self._info(reward_raw=r, bench=bench, cost=trade.cost, traded=trade.traded_value, target=target)
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------ helpers
    @property
    def portfolio_feature_names(self) -> list[str]:
        return list(PORTFOLIO_FEATURES)
