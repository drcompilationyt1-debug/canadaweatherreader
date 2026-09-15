import numpy as np

from stockbot.env.dataset import MarketDataset
from stockbot.env.portfolio import Portfolio
from stockbot.env.trading_env import TradingEnv
from stockbot.env.vec import make_vec_env
from stockbot.signals.registry import build_context, build_layout, build_providers


def _dataset(cfg, frames):
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "candles", "trend")]
    layout = build_layout(providers)
    return MarketDataset.build(frames, providers, layout, ctx, fit=True, train_end="2018-12-31")


def test_portfolio_accounting():
    p = Portfolio(1000.0, commission=0.001, slippage=0.0, allow_short=True)
    r = p.rebalance(0.5, 10.0)
    assert abs(r.delta_shares - 50) < 1e-9 and abs(r.cost - 0.5) < 1e-9
    assert abs(p.equity(10.0) - 999.5) < 1e-9
    assert abs(p.exposure(20.0) - (1000 / (1499.5))) < 1e-6
    p.rebalance(-0.5, 20.0)
    assert p.shares < 0
    fee = p.accrue(20.0)
    assert fee > 0
    q = Portfolio(1000.0, allow_short=False)
    assert q.clip_target(-0.7) == 0.0 and q.clip_target(3.0) == 1.0


def test_env_step_matches_manual_accounting(cfg, frames):
    ds = _dataset(cfg, frames)
    env_cfg = dict(cfg.section("env"))
    env_cfg.update(signal_dropout=0.0, deadband=0.0, short_penalty=0.0, commission=0.001, slippage=0.0,
                   benchmark_mix=0.0, vol_target=0.0, turnover_penalty=0.0, allow_short=True)
    env = TradingEnv(ds, env_cfg, tickers=["AAA"], seed=1)
    obs, info = env.reset(options={"ticker": "AAA", "start": 300, "length": 50})
    assert obs.shape == (ds.layout.obs_dim,)
    td = ds.data["AAA"]
    p0, p1 = td.close[300], td.close[301]
    obs, reward, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
    cash0 = env_cfg["initial_cash"]
    shares = cash0 / p0
    cost = shares * p0 * 0.001
    eq = cash0 - shares * p0 - cost + shares * p1
    assert abs(info["equity"] - eq) < 1e-6
    assert abs(reward - np.log(eq / cash0) * 100) < 1e-6
    assert abs(info["exposure"] - shares * p1 / eq) < 1e-9
    # going short is possible and flagged in the portfolio state
    obs, reward, term, trunc, info = env.step(np.array([-1.0], dtype=np.float32))
    assert info["exposure"] < 0 and obs[-2] == 1.0 and obs[-3] == 0.0 and obs[-1] >= 0.0   # is_short, is_long, fee_drag
    # run the episode to the end
    done = term or trunc
    while not done:
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        done = term or trunc
    assert info["t"] == 350


def test_env_no_short_and_dropout(cfg, frames):
    ds = _dataset(cfg, frames)
    env_cfg = dict(cfg.section("env"))
    env_cfg.update(allow_short=False, signal_dropout=1.0)
    env = TradingEnv(ds, env_cfg, seed=3)
    obs, _ = env.reset()
    assert (obs[: ds.layout.signal_dim] == 0).all()  # every block dropped
    _, _, _, _, info = env.step(np.array([-1.0], dtype=np.float32))
    assert info["exposure"] == 0.0                      # long-only: -1 means flat ...
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert 0.3 < info["exposure"] < 0.7                 # ... 0 means half invested
    ev = TradingEnv(ds, env_cfg, seed=3, eval_mode=True)
    obs, _ = ev.reset()
    assert (obs[: ds.layout.signal_dim] != 0).any()  # no dropout in eval mode


def test_vec_env_runs(cfg, frames):
    ds = _dataset(cfg, frames)
    vec = make_vec_env(ds, dict(cfg.section("env")), n_envs=3, seed=0, vec="dummy")
    obs = vec.reset()
    assert obs.shape == (3, ds.layout.obs_dim)
    for _ in range(5):
        obs, rewards, dones, infos = vec.step(np.random.uniform(-1, 1, size=(3, 1)).astype(np.float32))
    assert obs.shape == (3, ds.layout.obs_dim) and rewards.shape == (3,)
    vec.close()
