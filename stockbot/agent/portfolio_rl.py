"""Portfolio-level reinforcement learning on top of the ranker: when to rebalance and how much to expose.

The per-name policy and the ranking head say *what* to hold; this agent decides *when* to act and *how much* to risk.
State: the market (index momentum at three horizons, realised volatility, the 200-day trend, breadth, the cross-
sectional dispersion of returns and of the rank scores, how many names carry a score, the book's own recent return) and
the book (exposure, bars since the last rebalance relative to the cadence, the book's realised volatility, its drawdown,
the turnover of the last rebalance, the slot count).  Action: an exposure level {50%, 75%, 100%} of the satellite share
and whether to rebalance today (a rebalance re-picks the top-K with the usual hysteresis and sector cap and sets every
slot to the chosen exposure; holding means no trades at all).  Reward: the book's daily log return net of fees, less a
penalty on squared returns (a risk-averse investor, not a return maximiser).  Episodes are one year from a random start
at a random budget (the fee book follows the budget).

Evaluated walk-forward: for each of the last N one-year windows the agent is trained on everything before the window
and run through it, phase-averaged over start dates like the rule; it is adopted for live use only when it beats the
fixed-cadence rule on Sharpe in most windows and on average - the same discipline as the weekend tuners.  Until then
the report is just evidence and the runner keeps the rule.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..execution.ranking import select_top
from ..logging_utils import get_logger

log = get_logger(__name__)

EXPOSURES = (0.5, 0.75, 1.0)
MARKET_FEATURES = ["spy_ret_20", "spy_ret_60", "spy_ret_120", "spy_vol_20", "spy_trend_200", "breadth_50", "xs_disp_20", "score_disp",
                   "scored_frac", "book_ret_60"]
BOOK_FEATURES = ["exposure", "since_rebalance", "book_vol_20", "drawdown", "last_turnover", "slots"]
FEATURES = MARKET_FEATURES + BOOK_FEATURES
RISK_PENALTY = 2.0                 # reward = log(1 + r) - RISK_PENALTY * r^2: log utility plus extra curvature
EPISODE_BARS = 252
POLICY_DIR = "portfolio_policy"


# ---------------------------------------------------------------------------------------------------------------- features
def market_features(px: pd.DataFrame, score: pd.DataFrame | None, eligible: pd.DataFrame | None, benchmark: str = "SPY",
                    book_daily: pd.Series | None = None) -> pd.DataFrame:
    """(dates x MARKET_FEATURES) from closes, rank scores and eligibility; every value is a trailing-window statistic of the
    bar's own past, scaled to about [-1, 1] and NaN-free (0 where there is no history yet)."""
    bench = px[benchmark].ffill() if benchmark in px.columns else px.ffill().mean(axis=1)
    r = bench.pct_change(fill_method=None)
    el = eligible.reindex(index=px.index, columns=px.columns).fillna(True).astype(bool) if eligible is not None else pd.DataFrame(True, index=px.index, columns=px.columns)
    out = pd.DataFrame(index=px.index)
    out["spy_ret_20"] = (bench / bench.shift(20) - 1.0) * 2.0
    out["spy_ret_60"] = (bench / bench.shift(60) - 1.0) * 2.0
    out["spy_ret_120"] = (bench / bench.shift(120) - 1.0) * 2.0
    out["spy_vol_20"] = r.rolling(20, min_periods=10).std() * np.sqrt(252) / 0.4 - 1.0
    out["spy_trend_200"] = (bench / bench.rolling(200, min_periods=100).mean() - 1.0) * 3.0
    above = ((px > px.rolling(50, min_periods=30).mean()) & el)
    n_el = el.sum(axis=1).clip(lower=1)
    out["breadth_50"] = (above.sum(axis=1) / n_el) * 2.0 - 1.0
    ret20 = (px / px.shift(20) - 1.0).where(el)
    out["xs_disp_20"] = ret20.std(axis=1) * 5.0 - 1.0
    if score is not None and len(score.columns):
        sc = score.reindex(index=px.index)
        out["score_disp"] = sc.std(axis=1).fillna(0.0) - 1.0
        out["scored_frac"] = (sc.notna().sum(axis=1) / max(len(px.columns), 1)) * 2.0 - 1.0
    else:
        out["score_disp"], out["scored_frac"] = -1.0, -1.0
    bd = book_daily.reindex(px.index).fillna(0.0) if book_daily is not None else pd.Series(0.0, index=px.index)
    out["book_ret_60"] = bd.rolling(60, min_periods=1).sum() * 2.0
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0).astype(np.float32)


def book_features(exposure: float, since: int | None, every: int, book: list[float] | np.ndarray, drawdown: float, last_turnover: float,
                  k: int) -> np.ndarray:
    b = np.asarray([x for x in (book or []) if x is not None and np.isfinite(x)], dtype=float)
    vol = float(np.std(b[-20:], ddof=1) * np.sqrt(252)) if len(b) >= 10 else 0.2
    since_rel = min(float(since) / max(every, 1), 3.0) if since is not None else 3.0
    return np.clip(np.asarray([exposure * 2.0 - 1.0, since_rel - 1.0, vol / 0.4 - 1.0, float(drawdown) / 0.3 * 2.0 + 1.0,
                               min(float(last_turnover), 2.0) - 1.0, k / 25.0 - 1.0], dtype=np.float32), -3.0, 3.0)


# ---------------------------------------------------------------------------------------------------------------- panel
@dataclass
class Panel:
    dates: pd.DatetimeIndex
    names: list[str]
    rets: np.ndarray                 # (T, N) simple returns, 0 where a name has no bar
    score: np.ndarray                # (T, N) rank scores, NaN where none
    eligible: np.ndarray             # (T, N) bool
    market: np.ndarray               # (T, len(MARKET_FEATURES))
    k: int
    every: int
    hysteresis: int
    sectors: dict[str, str] | None
    max_per_sector: int
    reserve: float
    cash_range: tuple[float, float]
    fee_of: object                   # budget -> round-trip bps
    first_valid: int                 # first row with a score

    @property
    def satellite(self) -> float:
        return max(0.0, 1.0 - float(self.reserve))


def build_panel(cfg, ds, account: str = "main") -> Panel:
    """Everything the environment needs, from the dataset and the account's rank settings (the tuned structure if any)."""
    from ..config import account_config
    from .backtest import blended_scores, closes, eligibility, round_trip_bps, simulate
    from ..execution.ranking import every_bars_of, load_tuned_inputs, load_tuned_profile

    c = account_config(cfg, None if account == "main" else account)
    rk = dict(c.get_path("execution.rank", {}) or {})
    inputs = dict(rk.get("inputs") or {})
    if rk.get("adaptive", True):
        inputs = load_tuned_inputs(c.path("models_dir", "models"), inputs or None)
    k, every, hyst = int(rk.get("top_k", 20)), every_bars_of(rk.get("every_bars", 10)), int(rk.get("hysteresis", 3))
    prof = load_tuned_profile(c.path("models_dir", "models"), account, {"top_k": k, "every_bars": every, "hysteresis": hyst, "core_share": 0.0})
    if prof:
        k, every, hyst = int(prof.get("top_k", k)), every_bars_of(prof.get("every_bars", every)), int(prof.get("hysteresis", hyst))
    px = closes(ds)
    score = blended_scores(ds, inputs or None)
    elig = eligibility(c, px)
    reserve = float(c.get_path("execution.cash_reserve", 0.1) or 0.0)
    sectors = None
    max_sec = int(rk.get("max_per_sector", 0) or 0)
    if max_sec > 0:
        try:
            from ..data.sectors import load_sectors

            sectors = load_sectors(c, list(px.columns), refresh=False) or None
        except Exception as e:  # noqa: BLE001
            log.debug("sectors unavailable for the portfolio agent: %s", e)
    rets = px.pct_change(fill_method=None).fillna(0.0)
    first_valid = int(np.argmax(score.notna().sum(axis=1).to_numpy() >= 5)) if score.notna().any().any() else len(px) - 1
    rule = simulate(px, score, px.index[first_valid], k=k, every=every, hysteresis=hyst, fee_bps=round_trip_bps(c, 90_000, k),
                    reserve=reserve, sectors=sectors, max_per_sector=max_sec, eligible=elig)
    market = market_features(px, score, elig, book_daily=rule["daily"])
    cr = c.get_path("env.cash_range") or [c.get_path("env.initial_cash", 100_000)] * 2
    el_np = (elig.reindex(index=px.index, columns=px.columns).fillna(True).astype(bool).to_numpy() if elig is not None
             else np.ones(px.shape, dtype=bool))
    return Panel(px.index, list(px.columns), rets.to_numpy(np.float64), score.reindex(index=px.index, columns=px.columns).to_numpy(np.float64),
                 el_np, market.to_numpy(np.float32), k, every, hyst, sectors, max_sec, reserve, (float(cr[0]), float(cr[-1])),
                 lambda budget: round_trip_bps(c, budget, k), first_valid)


# ---------------------------------------------------------------------------------------------------------------- environment
class PortfolioEnv:
    """gymnasium environment over a Panel; ``lo``/``hi`` bound the episode starts (rows), so training never sees the test window."""

    metadata = {"render_modes": []}

    def __init__(self, panel: Panel, lo: int, hi: int, episode_bars: int = EPISODE_BARS, seed: int = 0, budget: float | None = None):
        import gymnasium as gym
        from gymnasium import spaces

        self.panel, self.lo, self.hi, self.L = panel, int(lo), int(hi), int(episode_bars)
        self.fixed_budget = budget
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(-3.0, 3.0, shape=(len(FEATURES),), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([len(EXPOSURES), 2])
        self.spec = None
        self._gym = gym
        self.reset()

    # gymnasium API ------------------------------------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        p = self.panel
        start = int(options["start"]) if options and "start" in options else int(self.rng.integers(self.lo, max(self.lo + 1, self.hi - self.L)))
        self.t0 = self.t = max(start, p.first_valid)
        self.end = min(int(options["end"]) if options and options.get("end") else self.t0 + self.L, len(p.dates) - 1)
        lo, hi = p.cash_range
        self.budget = float(self.fixed_budget) if self.fixed_budget else float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))
        self.fee_bps = float(p.fee_of(self.budget * p.satellite))
        self.w = np.zeros(len(p.names))
        self.held: list[str] = []
        self.exposure, self.since, self.last_turnover = 1.0, None, 0.0
        self.equity, self.peak = 1.0, 1.0
        self.book: list[float] = []
        self.daily: list[float] = []
        self.rebalances = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        p = self.panel
        dd = self.equity / self.peak - 1.0
        return np.concatenate([p.market[self.t], book_features(self.exposure, self.since, p.every, self.book, dd, self.last_turnover, p.k)]).astype(np.float32)

    def _rebalance(self, exposure: float) -> float:
        p = self.panel
        row = p.score[self.t]
        ok = np.isfinite(row) & p.eligible[self.t]
        scores = {p.names[i]: float(row[i]) for i in np.flatnonzero(ok)}
        self.held = select_top(scores, self.held, p.k, p.hysteresis, sectors=p.sectors, max_per_sector=p.max_per_sector) if scores else self.held
        new_w = np.zeros(len(p.names))
        if self.held:
            idx = [p.names.index(t) for t in self.held]
            new_w[idx] = exposure * p.satellite / max(p.k, 1)
        turnover = float(np.abs(new_w - self.w).sum())
        self.w, self.exposure, self.since, self.last_turnover = new_w, exposure, 0, turnover
        self.rebalances += 1
        return turnover * self.fee_bps / 1e4

    def step(self, action):
        p = self.panel
        exp_level, do_reb = int(action[0]), int(action[1])
        fee = self._rebalance(EXPOSURES[exp_level]) if (do_reb or not self.held) else 0.0
        if not (do_reb or fee):
            self.since = (self.since + 1) if self.since is not None else 1
        ret = float((self.w * p.rets[self.t + 1]).sum()) - fee
        self.equity *= (1.0 + ret)
        self.peak = max(self.peak, self.equity)
        self.book.append(ret)
        self.daily.append(ret)
        reward = float(np.log1p(max(ret, -0.99)) - RISK_PENALTY * ret * ret)
        self.t += 1
        done = self.t >= self.end
        return self._obs(), reward, bool(done), False, {"ret": ret, "fee": fee}

    def render(self):
        return None

    def close(self):
        return None


def _wrap(panel: Panel, lo: int, hi: int, seed: int, budget: float | None = None):
    import gymnasium as gym

    class _Env(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.inner = PortfolioEnv(panel, lo, hi, seed=seed, budget=budget)
            self.observation_space, self.action_space = self.inner.observation_space, self.inner.action_space

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.inner.reset(seed=seed, options=options)

        def step(self, action):
            return self.inner.step(action)

    return _Env


def train_agent(panel: Panel, lo: int, hi: int, timesteps: int = 200_000, seed: int = 0, n_envs: int = 4, max_minutes: float | None = None):
    """PPO on episodes starting in [lo, hi); returns the SB3 model."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv([lambda i=i: _wrap(panel, lo, hi, seed + i)() for i in range(n_envs)])
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=256, learning_rate=3e-4, gamma=0.97, gae_lambda=0.95, ent_coef=0.01,
                policy_kwargs={"net_arch": [64, 64]}, seed=seed, verbose=0, device="cpu")
    t0 = time.time()
    if max_minutes:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Deadline(BaseCallback):
            def _on_step(self) -> bool:
                return (time.time() - t0) < max_minutes * 60.0

        model.learn(total_timesteps=int(timesteps), callback=_Deadline())
    else:
        model.learn(total_timesteps=int(timesteps))
    log.info("portfolio agent: %d steps in %.0fs (episodes from rows %d..%d)", timesteps, time.time() - t0, lo, hi)
    return model


def run_policy(model, panel: Panel, start: int, end: int, budget: float) -> dict:
    """The trained agent through rows [start, end) deterministically; the rule's stats dict (total, sharpe, maxdd, turnover)."""
    env = PortfolioEnv(panel, start, end, episode_bars=end - start, seed=0, budget=budget)
    obs, _ = env.reset(options={"start": start, "end": end})
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
    daily = pd.Series(env.daily, index=panel.dates[start + 1:start + 1 + len(env.daily)])
    eq = (1.0 + daily).cumprod()
    return {"daily": daily, "total": float(eq.iloc[-1] - 1.0) if len(eq) else 0.0,
            "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 1 and daily.std() > 0 else 0.0,
            "max_drawdown": float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0,
            "rebalances_per_year": float(env.rebalances / max(len(daily), 1) * 252), "days": int(len(daily))}


# ---------------------------------------------------------------------------------------------------------------- walk-forward
def walk_forward(cfg, ds, account: str = "main", windows: int = 3, timesteps: int = 200_000, offsets=(0, 3, 6, 9), out_dir=None, seed: int = 0,
                 min_gain: float = 0.05, max_minutes: float | None = None) -> dict:
    """Train before each of the last ``windows`` one-year windows, run through it, compare with the fixed-cadence rule
    (both phase-averaged over ``offsets``); write the report and, when adopted, the final agent trained on everything."""
    from ..config import account_config
    from .backtest import closes, eligibility, round_trip_bps, simulate

    c = account_config(cfg, None if account == "main" else account)
    panel = build_panel(cfg, ds, account)
    px = closes(ds)
    score = pd.DataFrame(panel.score, index=panel.dates, columns=panel.names)      # the same blend the agent saw
    budget = float(c.get_path("env.initial_cash", 100_000))
    fee = round_trip_bps(c, budget * panel.satellite, panel.k)
    elig = eligibility(c, px)
    last = panel.dates[-1]
    rows = []
    t_start = time.time()
    for w in range(windows, 0, -1):
        start, end = last - pd.DateOffset(years=w), last - pd.DateOffset(years=w - 1)
        i0, i1 = int(panel.dates.searchsorted(start)), int(panel.dates.searchsorted(end, side="right"))
        if i0 - panel.first_valid < 500 or i1 - i0 < 120:
            log.info("portfolio agent: window %s..%s skipped (not enough history before it)", start.date(), end.date())
            continue
        budget_left = None if max_minutes is None else max(1.0, (max_minutes * 60 - (time.time() - t_start)) / 60 / (w + 1))
        model = train_agent(panel, panel.first_valid, i0 - EPISODE_BARS, timesteps, seed, max_minutes=budget_left)
        rl, rule = [], []
        for off in offsets:
            if i0 + off + 60 >= i1:
                continue
            rl.append(run_policy(model, panel, i0 + off, i1, budget))
            rule.append(simulate(px, score, panel.dates[i0 + off], k=panel.k, every=panel.every, hysteresis=panel.hysteresis, fee_bps=fee,
                                 end=panel.dates[i1 - 1], reserve=panel.reserve, sectors=panel.sectors, max_per_sector=panel.max_per_sector, eligible=elig))
        if not rl:
            continue
        stat = lambda rs, key: float(np.mean([r[key] for r in rs]))  # noqa: E731
        rows.append({"window": f"{panel.dates[i0].date()}..{panel.dates[i1 - 1].date()}",
                     "agent": {k: stat(rl, k) for k in ("total", "sharpe", "max_drawdown", "rebalances_per_year")},
                     "rule": {k: stat(rule, k) for k in ("total", "sharpe", "max_drawdown")}, "starts": len(rl)})
        log.info("portfolio agent %s: agent %+.1f%% sharpe %.2f dd %.0f%% | rule %+.1f%% sharpe %.2f dd %.0f%%", rows[-1]["window"],
                 100 * rows[-1]["agent"]["total"], rows[-1]["agent"]["sharpe"], 100 * rows[-1]["agent"]["max_drawdown"],
                 100 * rows[-1]["rule"]["total"], rows[-1]["rule"]["sharpe"], 100 * rows[-1]["rule"]["max_drawdown"])
    wins = sum(r["agent"]["sharpe"] >= r["rule"]["sharpe"] for r in rows)
    mean_a = {k: float(np.mean([r["agent"][k] for r in rows])) for k in ("total", "sharpe", "max_drawdown")} if rows else {}
    mean_r = {k: float(np.mean([r["rule"][k] for r in rows])) for k in ("total", "sharpe", "max_drawdown")} if rows else {}
    accepted = bool(rows) and wins * 3 >= len(rows) * 2 and mean_a["sharpe"] >= mean_r["sharpe"] + min_gain and mean_a["total"] >= mean_r["total"] - 0.01
    rep = {"account": account, "tuned_at": str(date.today()), "windows": rows, "wins": int(wins), "n": len(rows), "agent_mean": mean_a,
           "rule_mean": mean_r, "accepted": accepted, "timesteps": int(timesteps), "features": FEATURES, "exposures": list(EXPOSURES),
           "structure": {"top_k": panel.k, "every_bars": panel.every, "hysteresis": panel.hysteresis, "max_per_sector": panel.max_per_sector,
                         "reserve": panel.reserve},
           "reason": ("the agent beat the rule on Sharpe in most windows and on average - in force" if accepted else
                      "the fixed-cadence rule is as good - kept" if rows else "not enough history for a walk-forward")}
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if accepted:
            final = train_agent(panel, panel.first_valid, len(panel.dates) - EPISODE_BARS, timesteps, seed,
                                max_minutes=None if max_minutes is None else max(1.0, (max_minutes * 60 - (time.time() - t_start)) / 60))
            final.save(out / "policy.zip")
            rep["policy"] = str(out / "policy.zip")
        (out / "report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    log.info("portfolio agent (%s): %s (%d/%d windows)", account, rep["reason"], wins, len(rows))
    return rep


def format_report(rep: dict) -> str:
    lines = [f"portfolio agent ({rep['account']}, {rep['timesteps']} steps per window): {rep['reason']}",
             f"{'window':24s} {'agent':>8s} {'sharpe':>7s} {'maxdd':>6s} {'reb/yr':>7s} | {'rule':>8s} {'sharpe':>7s} {'maxdd':>6s}"]
    for r in rep["windows"]:
        a, b = r["agent"], r["rule"]
        lines.append(f"{r['window']:24s} {100 * a['total']:+7.1f}% {a['sharpe']:7.2f} {100 * a['max_drawdown']:5.0f}% {a['rebalances_per_year']:7.1f} | "
                     f"{100 * b['total']:+7.1f}% {b['sharpe']:7.2f} {100 * b['max_drawdown']:5.0f}%")
    if rep.get("agent_mean"):
        a, b = rep["agent_mean"], rep["rule_mean"]
        lines.append(f"{'mean':24s} {100 * a['total']:+7.1f}% {a['sharpe']:7.2f} {100 * a['max_drawdown']:5.0f}%         | "
                     f"{100 * b['total']:+7.1f}% {b['sharpe']:7.2f} {100 * b['max_drawdown']:5.0f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------- live use
class PortfolioAgent:
    """The adopted agent for the runner: ``decide`` -> (exposure, rebalance_now) from the live market frames and the book."""

    def __init__(self, model, meta: dict):
        self.model, self.meta = model, meta

    @classmethod
    def load(cls, models_dir, account: str = "main") -> "PortfolioAgent | None":
        d = Path(models_dir) / f"{POLICY_DIR}_{account}"
        rep_file, pol = d / "report.json", d / "policy.zip"
        if not rep_file.exists() or not pol.exists():
            return None
        try:
            rep = json.loads(rep_file.read_text(encoding="utf-8"))
            if not rep.get("accepted"):
                return None
            from stable_baselines3 import PPO

            return cls(PPO.load(pol, device="cpu"), rep)
        except Exception as e:  # noqa: BLE001
            log.warning("portfolio agent unavailable: %s", e)
            return None

    def decide(self, frames: dict[str, pd.DataFrame], scores: dict[str, float], eligible: list[str] | None, exposure: float, since: int | None,
               every: int, book: list[float], drawdown: float, last_turnover: float, k: int, benchmark: str = "SPY") -> tuple[float, bool]:
        px = pd.DataFrame({t: df["close"].astype(float).tail(300) for t, df in frames.items()})
        sc = pd.DataFrame([{t: v for t, v in scores.items()}], index=[px.index[-1]]).reindex(columns=px.columns)
        el = pd.DataFrame(True, index=px.index, columns=px.columns)
        if eligible is not None:
            for t in px.columns:
                if t not in eligible:
                    el[t] = False
        bd = pd.Series(list(book)[-60:], index=px.index[-len(list(book)[-60:]):]) if book else None
        m = market_features(px, sc, el, benchmark=benchmark, book_daily=bd).iloc[-1].to_numpy(np.float32)
        obs = np.concatenate([m, book_features(exposure, since, every, book, drawdown, last_turnover, k)]).astype(np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return float(EXPOSURES[int(action[0])]), bool(int(action[1]))
