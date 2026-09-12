"""Out-of-sample evaluation: run the deterministic policy through full windows and score it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..env.dataset import MarketDataset
from ..env.trading_env import TradingEnv
from ..logging_utils import get_logger

log = get_logger(__name__)


def metrics(equity: np.ndarray, bench: np.ndarray, exposures: np.ndarray | None = None) -> dict:
    equity = np.asarray(equity, float)
    bench = np.asarray(bench, float)
    n = len(equity) - 1
    if n < 2:
        return {}
    r = np.diff(np.log(equity))
    rb = np.diff(np.log(bench))
    years = n / 252.0
    total = equity[-1] / equity[0] - 1.0
    bh = bench[-1] / bench[0] - 1.0
    cagr = (equity[-1] / equity[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    sharpe = float(r.mean() / (r.std() + 1e-12) * np.sqrt(252))
    sharpe_bh = float(rb.mean() / (rb.std() + 1e-12) * np.sqrt(252))
    peak = np.maximum.accumulate(equity)
    mdd = float((1.0 - equity / peak).max())
    peak_b = np.maximum.accumulate(bench)
    mdd_bh = float((1.0 - bench / peak_b).max())
    out = {
        "bars": n, "total_return": float(total), "cagr": float(cagr), "sharpe": sharpe, "max_drawdown": mdd,
        "bh_return": float(bh), "bh_sharpe": sharpe_bh, "bh_max_drawdown": mdd_bh, "excess_return": float(total - bh),
        "vol": float(r.std() * np.sqrt(252)),
    }
    if exposures is not None and len(exposures):
        e = np.asarray(exposures, float)
        out["avg_exposure"] = float(np.mean(e))
        out["avg_abs_exposure"] = float(np.mean(np.abs(e)))
        out["short_share"] = float(np.mean(e < -0.01))
        out["turnover"] = float(np.sum(np.abs(np.diff(e, prepend=0.0))) / max(len(e) / 252.0, 1e-9))
    return out


def run_window(model, dataset: MarketDataset, ticker: str, env_cfg: dict, start: int | None = None,
               length: int | None = None, deterministic: bool = True, seed: int = 0) -> dict:
    """Run the policy over one contiguous window; returns curves + metrics."""
    env = TradingEnv(dataset, env_cfg, tickers=[ticker], seed=seed, eval_mode=True)
    td = dataset.data[ticker]
    if start is None:
        start = td.min_start
    if length is None:
        length = len(td) - 2 - start
    obs, info = env.reset(options={"ticker": ticker, "start": start, "length": length})
    equity = [info["equity"]]
    bench = [info["price"]]
    dates = [info["date"]]
    exposures, actions, rewards = [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs.reshape(1, -1), deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(np.asarray(action).reshape(-1))
        equity.append(info["equity"])
        bench.append(info["price"])
        dates.append(info["date"])
        exposures.append(info["exposure"])
        actions.append(float(np.asarray(action).reshape(-1)[0]))
        rewards.append(reward)
        done = terminated or truncated
    eq = np.array(equity)
    bh = np.array(bench) / bench[0] * equity[0]
    m = metrics(eq, bh, np.array(exposures))
    m.update({"ticker": ticker, "start": dates[0], "end": dates[-1], "total_reward": float(np.sum(rewards))})
    return {"metrics": m, "equity": eq, "bench": bh, "dates": dates, "exposures": np.array(exposures), "actions": np.array(actions)}


def evaluate(model, dataset: MarketDataset, env_cfg: dict, tickers: list[str] | None = None,
             max_bars: int | None = None, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    tickers = tickers or dataset.tickers
    rows, curves = [], {}
    for t in tickers:
        if t not in dataset.data:
            continue
        td = dataset.data[t]
        length = len(td) - 2 - td.min_start
        if max_bars:
            length = min(length, max_bars)
        if length < 10:
            continue
        try:
            res = run_window(model, dataset, t, env_cfg, start=td.min_start, length=length, seed=seed)
        except Exception as e:  # noqa: BLE001
            log.warning("evaluation failed for %s: %s", t, e)
            continue
        rows.append(res["metrics"])
        curves[t] = res
    summary = pd.DataFrame(rows)
    return summary, curves


def aggregate(summary: pd.DataFrame) -> dict:
    if summary is None or len(summary) == 0:
        return {}
    cols = ["total_return", "cagr", "sharpe", "max_drawdown", "bh_return", "bh_sharpe", "excess_return",
            "avg_exposure", "short_share", "turnover"]
    cols = [c for c in cols if c in summary.columns]
    out = {f"mean_{c}": float(summary[c].mean()) for c in cols}
    out["median_excess_return"] = float(summary["excess_return"].median()) if "excess_return" in summary else 0.0
    out["win_rate_vs_bh"] = float((summary["excess_return"] > 0).mean()) if "excess_return" in summary else 0.0
    out["n_tickers"] = int(len(summary))
    return out


def format_summary(summary: pd.DataFrame) -> str:
    if summary is None or len(summary) == 0:
        return "(no evaluation rows)"
    cols = ["ticker", "start", "end", "total_return", "bh_return", "excess_return", "sharpe", "bh_sharpe",
            "max_drawdown", "avg_exposure", "short_share", "turnover"]
    cols = [c for c in cols if c in summary.columns]
    df = summary[cols].copy()
    for c in ("total_return", "bh_return", "excess_return", "max_drawdown", "avg_exposure", "short_share"):
        if c in df:
            df[c] = (df[c] * 100).round(1)
    for c in ("sharpe", "bh_sharpe", "turnover"):
        if c in df:
            df[c] = df[c].round(2)
    return df.to_string(index=False)
