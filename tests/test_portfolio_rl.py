"""The portfolio agent: causal features that match live, an environment that holds without trades, a trained agent that runs
and loads only once the walk-forward adopted it."""
import numpy as np
import pandas as pd

import stockbot.agent  # noqa: F401
from stockbot.agent.portfolio_rl import (EXPOSURES, FEATURES, Panel, PortfolioAgent, PortfolioEnv, book_features, market_features,
                                         run_policy, train_agent)


def _panel(n=900, names=("A", "B", "C", "D", "E", "SPY"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    px = pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, n)) for t in names}, index=idx)
    score = pd.DataFrame(rng.normal(size=(n, len(names))), index=idx, columns=names)
    score.iloc[:260] = np.nan
    elig = pd.DataFrame(True, index=idx, columns=names)
    rets = px.pct_change(fill_method=None).fillna(0.0)
    market = market_features(px, score, elig)
    panel = Panel(idx, list(names), rets.to_numpy(), score.to_numpy(), elig.to_numpy(), market.to_numpy(np.float32), 2, 5, 1, None, 0, 0.1,
                  (10_000.0, 100_000.0), lambda b: 10.0, 260)
    return panel, px, score, elig


def test_market_features_are_causal_bounded_and_the_same_live():
    panel, px, score, elig = _panel()
    full = market_features(px, score, elig)
    assert list(full.columns) == FEATURES[:11] and np.isfinite(full.to_numpy()).all() and (full.abs() <= 3).all().all()
    t = 700
    trunc = market_features(px.iloc[t - 299:t + 1], score.iloc[t - 299:t + 1], elig.iloc[t - 299:t + 1])
    assert np.allclose(full.iloc[t].to_numpy(), trunc.iloc[-1].to_numpy(), atol=1e-5)      # a 300-bar live window reproduces the row
    px2 = px.copy()
    px2.iloc[t + 1:] *= 1.5                                                                # the future does not leak into row t
    assert np.allclose(full.iloc[t].to_numpy(), market_features(px2, score, elig).iloc[t].to_numpy())
    assert book_features(1.0, None, 10, [], 0.0, 0.0, 20).shape == (6,)


def test_env_holds_without_trades_and_rebalances_on_demand():
    panel, *_ = _panel()
    env = PortfolioEnv(panel, 300, 600, episode_bars=50, seed=1)
    obs, _ = env.reset(options={"start": 300})
    assert obs.shape == (len(FEATURES),) and np.isfinite(obs).all()
    obs, r, done, _, info = env.step(np.array([2, 0]))                 # the first step always fills the book
    assert env.held and env.rebalances == 1 and info["fee"] > 0 and abs(env.w.sum() - 0.9) < 1e-9   # exposure 1.0 x satellite 0.9
    w = env.w.copy()
    obs, r, done, _, info = env.step(np.array([0, 0]))                 # hold: no trades, no fee
    assert info["fee"] == 0 and np.array_equal(env.w, w) and env.since == 1
    env.step(np.array([0, 1]))                                          # rebalance at 50% exposure
    assert abs(env.w.sum() - 0.45) < 1e-9 and env.since == 0 and env.rebalances == 2 and env.t == 303
    for _ in range(60):
        obs, r, done, _, _ = env.step(env.action_space.sample())
        if done:
            break
    assert done and len(env.daily) == 50 and np.isfinite(obs).all() and np.isfinite(r)


def test_agent_trains_runs_and_loads_only_when_adopted(tmp_path):
    panel, px, score, elig = _panel()
    model = train_agent(panel, 260, 500, timesteps=1024, seed=0, n_envs=2)
    r = run_policy(model, panel, 600, 800, 50_000)
    assert r["days"] == 200 and np.isfinite(r["total"]) and r["rebalances_per_year"] > 0 and len(r["daily"]) == 200
    r_end = run_policy(model, panel, 700, len(panel.dates), 50_000)                     # up to the last bar, no overrun
    assert r_end["days"] == len(panel.dates) - 701
    d = tmp_path / "portfolio_policy_main"
    d.mkdir()
    model.save(d / "policy.zip")
    (d / "report.json").write_text('{"accepted": false}', encoding="utf-8")
    assert PortfolioAgent.load(tmp_path, "main") is None                                # not adopted: the runner keeps the rule
    (d / "report.json").write_text('{"accepted": true}', encoding="utf-8")
    agent = PortfolioAgent.load(tmp_path, "main")
    assert agent is not None
    frames = {t: pd.DataFrame({"close": px[t]}) for t in px.columns}
    exposure, reb = agent.decide(frames, {t: 0.1 * i for i, t in enumerate(px.columns) if t != "SPY"}, None, 0.9, 3, 5, [0.001] * 30, -0.02, 0.5, 2)
    assert exposure in EXPOSURES and isinstance(reb, bool)
