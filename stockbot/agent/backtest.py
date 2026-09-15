"""Portfolio backtest of the rank-core decision layer on a dataset: the yardstick that matters (beat SPY).

The policy evaluation scores one name at a time; this replays what the runner actually does - hold the
top-K names by the blended rank, rebalance every N bars with hysteresis - as a portfolio, net of the fees
a given budget pays per round trip, and compares it with SPY and the equal-weight universe.  It is run
after every retrain (``stockbot backtest``) for the out-of-sample year and the last three years, at the
$10k and $100k budgets.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..execution.ranking import DEFAULT_INPUTS, every_bars_of, select_top
from ..logging_utils import get_logger

log = get_logger(__name__)


def blended_scores(ds, inputs: dict[str, float] | None = None) -> pd.DataFrame:
    """Date x ticker blended percentile-rank score from the dataset's signal arrays (masked blocks skipped)."""
    inputs = inputs or DEFAULT_INPUTS
    lay = ds.layout
    parts: list[tuple[pd.DataFrame, float]] = []
    for key, w in inputs.items():
        block, feat = key.split(".", 1)
        try:
            b = lay.block(block)
            j = b.start + list(b.feature_names).index(feat)
        except (KeyError, ValueError):
            continue
        cols = {}
        for t, td in ds.data.items():
            v = td.signals[:, j].astype(float)
            v[td.signals[:, b.offset] < 0.5] = np.nan
            cols[t] = pd.Series(v, index=pd.DatetimeIndex(td.dates))
        parts.append((pd.DataFrame(cols).sort_index().rank(axis=1, pct=True), float(w)))
    if not parts:
        raise ValueError("none of the rank inputs is in the dataset layout")
    num = sum(df * w for df, w in parts)
    den = sum(df.notna() * w for df, w in parts)
    return num / den.replace(0.0, np.nan)


def closes(ds) -> pd.DataFrame:
    return pd.DataFrame({t: pd.Series(td.close, index=pd.DatetimeIndex(td.dates)) for t, td in ds.data.items()}).sort_index()


def simulate(px: pd.DataFrame, score: pd.DataFrame | None, start, k: int = 20, every: int = 10, hysteresis: int = 3,
             fee_bps: float = 8.0, end=None) -> dict:
    """Daily portfolio returns of the rank-core rule (``score`` None = equal-weight everything), fees on turnover."""
    idx = px.index[(px.index >= pd.Timestamp(start)) & ((px.index <= pd.Timestamp(end)) if end is not None else True)]
    rets = px.pct_change().reindex(idx).fillna(0.0)
    wts = pd.DataFrame(0.0, index=idx, columns=px.columns)
    held: list[str] = []
    for i, d in enumerate(idx):
        if score is None:
            wts.loc[d] = 1.0 / px.shape[1]
            continue
        if i % every == 0:
            s = score.loc[d].dropna() if d in score.index else pd.Series(dtype=float)
            held = select_top(s.to_dict(), held, k, hysteresis) if len(s) else held
        if held:
            wts.loc[d, held] = 1.0 / len(held)
    prev = wts.shift(1).fillna(0.0)
    turnover = (wts - prev).abs().sum(axis=1)
    daily = (prev * rets).sum(axis=1) - turnover * fee_bps / 1e4
    eq = (1.0 + daily).cumprod()
    return {"daily": daily, "total": float(eq.iloc[-1] - 1.0) if len(eq) else 0.0,
            "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 1 and daily.std() > 0 else 0.0,
            "max_drawdown": float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0,
            "turnover_per_year": float(turnover.sum() / max(len(idx), 1) * 252), "days": int(len(idx))}


def round_trip_bps(cfg, budget: float, k: int, price: float = 100.0, ticker: str = "SPY") -> float:
    """Fee per unit of turnover (buy + sell of one slot) in bps for a budget split into ``k`` slots."""
    from ..execution.fees import FeeBook

    sched = FeeBook.from_config(cfg).for_ticker(ticker)
    slot = max(float(budget) / max(k, 1), 1.0)
    if sched is None:
        return 0.0
    return 1e4 * (float(sched.cost(slot / price, price, "buy")) + float(sched.cost(slot / price, price, "sell"))) / slot


def run_backtests(cfg, ds, budgets: dict[str, dict] | None = None, oos_start=None, years: int = 3, benchmark: str = "SPY") -> dict:
    """The rank rule at each budget's settings vs SPY and equal-weight, for the out-of-sample year and the last ``years``."""
    rk = dict(cfg.get_path("execution.rank", {}) or {})
    inputs = dict(rk.get("inputs") or DEFAULT_INPUTS)
    px = closes(ds)
    score = blended_scores(ds, inputs)
    last = px.index[-1]
    oos_start = pd.Timestamp(oos_start or cfg.get_path("data.train_end") or (last - pd.DateOffset(years=1)))
    windows = {"oos": oos_start, f"{years}y": last - pd.DateOffset(years=years)}
    if budgets is None:
        budgets = {"100k": {"budget": 100_000, "k": int(rk.get("top_k", 20)), "every": every_bars_of(rk.get("every_bars", 10))}}
        try:
            from ..config import account_config, account_names

            for name in account_names(cfg):
                ca = account_config(cfg, name)
                r2 = dict(ca.get_path("execution.rank", {}) or {})
                budgets[name] = {"budget": float(ca.get_path("env.initial_cash", 10_000)), "k": int(r2.get("top_k", 10)),
                                 "every": every_bars_of(r2.get("every_bars", 21))}
        except Exception as e:  # noqa: BLE001
            log.debug("account budgets: %s", e)
    hyst = int(rk.get("hysteresis", 3))
    out: dict = {"inputs": inputs, "windows": {k: str(v.date()) for k, v in windows.items()}, "results": {}}
    for wname, start in windows.items():
        res = {}
        if benchmark in px.columns:
            b = px[benchmark].pct_change().reindex(px.index[px.index >= start]).fillna(0.0)
            eq = (1 + b).cumprod()
            res[benchmark] = {"total": float(eq.iloc[-1] - 1), "sharpe": float(b.mean() / b.std() * np.sqrt(252)) if b.std() > 0 else 0.0,
                              "max_drawdown": float((eq / eq.cummax() - 1).min())}
        ew = simulate(px, None, start)
        res["equal_weight"] = {k: v for k, v in ew.items() if k != "daily"}
        for bname, bset in budgets.items():
            fee = round_trip_bps(cfg, bset["budget"], bset["k"])
            r = simulate(px, score, start, k=bset["k"], every=bset["every"], hysteresis=hyst, fee_bps=fee)
            res[f"rank_{bname}"] = {**{k: v for k, v in r.items() if k != "daily"}, "k": bset["k"], "every_bars": bset["every"], "fee_bps": fee,
                                    "excess_vs_benchmark": r["total"] - res.get(benchmark, {}).get("total", 0.0)}
        out["results"][wname] = res
    return out


def format_report(rep: dict) -> str:
    lines = [f"rank inputs: {rep['inputs']}"]
    for wname, res in rep["results"].items():
        lines.append(f"\n== {wname} (from {rep['windows'][wname]}) ==")
        lines.append(f"{'strategy':16s} {'total':>8s} {'sharpe':>7s} {'maxdd':>6s} {'turn/yr':>8s} {'fee bps':>8s}")
        for name, r in res.items():
            lines.append(f"{name:16s} {100 * r['total']:+7.1f}% {r['sharpe']:7.2f} {100 * r['max_drawdown']:5.0f}% "
                         f"{r.get('turnover_per_year', float('nan')):8.1f} {r.get('fee_bps', 0.0):8.1f}")
    return "\n".join(lines)
