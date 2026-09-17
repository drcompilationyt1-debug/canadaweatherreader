from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stockbot.config import load_config  # noqa: E402
from stockbot.data.loader import synthetic_universe  # noqa: E402

TICKERS = ["AAA", "BBB", "CCC"]


@pytest.fixture()
def cfg(tmp_path):
    return load_config(overrides=[
        f"universe=[{','.join(TICKERS)}]",
        "data.train_end=2018-12-31",
        f"data.cache_dir={(tmp_path / 'cache').as_posix()}",
        f"models_dir={(tmp_path / 'models').as_posix()}",
        f"train.checkpoint_dir={(tmp_path / 'models' / 'policy').as_posix()}",
        f"news.cache_dir={(tmp_path / 'news_cache').as_posix()}",
        f"news.local_csv_dir={(tmp_path / 'news').as_posix()}",
        f"llm.state_file={(tmp_path / 'llm_state.json').as_posix()}",
        f"execution.state_file={(tmp_path / 'paper' / 'state.json').as_posix()}",
        f"feedback.experience_file={(tmp_path / 'exp' / 'trades.jsonl').as_posix()}",
        f"feedback.direction_file={(tmp_path / 'exp' / 'direction.jsonl').as_posix()}",
        f"session.log_dir={(tmp_path / 'sessions').as_posix()}",
        f"feedback.review_dir={(tmp_path / 'reviews').as_posix()}",
        "signals.alpha_factors.min_train_years=1",
        "signals.news_llm.enabled=true",
        "signals.trading_agents.enabled=false",
        "signals.ai_hedge_fund.enabled=false",
        "train.n_envs=2",
        "signals.chronos.enabled=false", "signals.kronos.enabled=false", "signals.timesfm.enabled=false", "signals.finbert.enabled=false",
        "train.seeds=1",
        "env.vol_target=0",
        "accounts.small.enabled=false",
        "env.cash_range=null",
        "env.cash_choices=null",
        "train.eval_cash=null",
        "execution.rank.enabled=false",
        "execution.rank.max_per_sector=0",
        "signals.compute_workers=1",
        "execution.rank.vol_target.enabled=false",
        "env.fee_choices=null",
        f"session.intraday_exit.model_dir={(tmp_path / 'intraday_exit').as_posix()}",
        "train.n_steps=64",
        "train.batch_size=64",
        "train.n_epochs=2",
        "train.eval_freq=128",
        "train.eval_tickers=2",
        "env.episode_length=60",
    ])


@pytest.fixture()
def frames():
    return synthetic_universe(TICKERS, n=1300, seed=7)


@pytest.fixture()
def rng():
    return np.random.default_rng(0)
