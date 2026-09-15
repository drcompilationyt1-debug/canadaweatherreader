"""The quant-desk half of the post-trade review: where the regret came from and what the best play was.

* :func:`attribute_regret` - splits ``oracle - actual`` exactly into *missed* (not held, went up),
  *wrong side* (held, went down), *under-sized* (held less than a full slice, went up) and *cost*
  (fees), the way a desk's P&L attribution does.
* :func:`path_oracle` - the fee-aware best exposure *path* of one name over a period (dynamic
  programme over exposure levels, paying the fee on every switch).  The best week is not five best
  days strung together: with fees the path oracle holds through noise instead of flipping daily.
* :func:`what_if_day` - a parallel search over the moves that were possible in one session: every
  exposure level x entry snapshot x exit snapshot on the recorded price path, fees included, then
  the regret of what we did split into *sizing* (right side, wrong size) and *timing*.
* :func:`round_trips` - trade-level round-trip statistics via pyfolio (win rate, profit factor,
  holding time), when pyfolio-reloaded is installed.
* :func:`ic_by_voter` - each input's information coefficient (cross-sectional rank correlation of
  its score with the next day's return), the standard signal-quality measure.
* :func:`regime` - tags a period with the benchmark's move and volatility so lessons are read in context.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)

LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


# ---------------------------------------------------------------------- attribution
def attribute_regret(exposure: dict[str, float], moves: dict[str, float], slice_frac: float, cost: float = 0.0) -> dict:
    """Split the gap between perfect one-period foresight and what we did.

    ``exposure`` = fraction of a slice held in each name (0..1), ``moves`` = the name's return over
    the period, ``slice_frac`` = slice / equity, ``cost`` = fees paid as a fraction of equity.
    The parts add up exactly to ``oracle_gross - actual_gross + cost``."""
    missed = wrong = under = 0.0
    for t, r in moves.items():
        e = float(np.clip(exposure.get(t, 0.0), 0.0, 1.0))
        if r > 0:
            if e <= 1e-9:
                missed += r * slice_frac
            else:
                under += r * (1.0 - e) * slice_frac
        elif r < 0 and e > 1e-9:
            wrong += -r * e * slice_frac
    total = missed + wrong + under + cost
    parts = {"missed": missed, "wrong_side": wrong, "under_sized": under, "cost": cost}
    biggest = max(parts, key=parts.get) if total > 0 else None
    return {**parts, "total": total, "biggest": biggest}


def attribute_regret_daily(W: pd.DataFrame, R: pd.DataFrame, max_position: float, cost: float = 0.0) -> dict:
    """The same split over every (day, name) of a period: ``W`` = weights held into each day (fraction of
    equity), ``R`` = that day's returns."""
    prev = W.shift(1).fillna(0.0)
    Rr = R.reindex(columns=W.columns).fillna(0.0)
    missed = wrong = under = 0.0
    for t in W.columns:
        e = (prev[t] / max_position).clip(0.0, 1.0) if max_position > 0 else prev[t] * 0.0
        r = Rr[t]
        up, down = r > 0, r < 0
        missed += float((r[up & (e <= 1e-9)] * max_position).sum())
        under += float((r[up & (e > 1e-9)] * (1.0 - e[up & (e > 1e-9)]) * max_position).sum())
        wrong += float((-r[down & (e > 1e-9)] * e[down & (e > 1e-9)] * max_position).sum())
    total = missed + wrong + under + cost
    parts = {"missed": missed, "wrong_side": wrong, "under_sized": under, "cost": cost}
    return {**parts, "total": total, "biggest": max(parts, key=parts.get) if total > 0 else None}


# ---------------------------------------------------------------------- fee-aware path oracle
def path_oracle(returns: np.ndarray, fee_frac: float, levels: tuple[float, ...] = LEVELS, start_level: float = 0.0,
                exit_at_end: bool = False) -> tuple[float, np.ndarray]:
    """Best exposure path for one name: ``returns`` are the name's per-bar simple returns, exposure is a
    fraction of one slice chosen from ``levels`` before each bar, and every change of level costs
    ``fee_frac * |change|`` (fee as a fraction of a full-slice trade).  Returns the total P&L per slice
    and the level held during each bar.  Dynamic programme (Viterbi), exact for the additive P&L used here."""
    r = np.asarray(returns, dtype=float)
    r = np.where(np.isfinite(r), r, 0.0)
    L = np.asarray(levels, dtype=float)
    n = len(r)
    if n == 0:
        return 0.0, np.zeros(0)
    switch = float(fee_frac) * np.abs(L[:, None] - L[None, :])          # cost of going from level j to level k
    start = np.abs(L - float(start_level)) * float(fee_frac)
    dp = -start + L * r[0]
    back = np.zeros((n, len(L)), dtype=int)
    for i in range(1, n):
        cand = dp[:, None] - switch                                      # from j (rows) to k (columns)
        back[i] = cand.argmax(axis=0)
        dp = cand.max(axis=0) + L * r[i]
    final = dp - (switch[:, 0] if exit_at_end else 0.0)
    k = int(final.argmax())
    best = float(final[k])
    path = np.zeros(n)
    for i in range(n - 1, -1, -1):
        path[i] = L[k]
        if i > 0:
            k = int(back[i][k])
    return best, path


def portfolio_path_oracle(R: pd.DataFrame, fee_fracs: dict[str, float], max_position: float, gross_cap: float = 1.0,
                          levels: tuple[float, ...] = LEVELS, start_weights: dict[str, float] | None = None) -> dict:
    """Fee-aware best path for every name, then the portfolio it implies: names are ranked by their
    best P&L and taken until the gross cap is full (a joint cap in the dynamic programme is not
    tractable, this greedy cut is the usual approximation).  Returns the portfolio return and per-name paths."""
    per_name = {}
    for t in R.columns:
        r = R[t].fillna(0.0).to_numpy(float)
        start = float((start_weights or {}).get(t, 0.0)) / max_position if max_position > 0 else 0.0
        best, path = path_oracle(r, fee_fracs.get(t, 0.0), levels, start_level=min(1.0, max(0.0, start)))
        per_name[t] = {"pnl_per_slice": best, "path": path, "switches": int((np.diff(np.concatenate([[start], path])) != 0).sum()),
                       "avg_level": float(path.mean()) if len(path) else 0.0}
    n_slots = max(1, int(np.floor(gross_cap / max_position + 1e-9))) if max_position > 0 else len(per_name)
    ranked = sorted(per_name, key=lambda t: -per_name[t]["pnl_per_slice"])
    chosen = [t for t in ranked if per_name[t]["pnl_per_slice"] > 0][:n_slots]
    total = float(sum(per_name[t]["pnl_per_slice"] for t in chosen) * max_position)
    return {"return": total, "names": chosen, "per_name": per_name,
            "switches": int(sum(per_name[t]["switches"] for t in chosen))}


# ---------------------------------------------------------------------- one session: search over moves
def _best_move_for_name(prices: np.ndarray, level_us: float, slice_cap: float, fee_fn, levels: tuple[float, ...]) -> dict:
    """All (level, entry snapshot, exit snapshot | hold) moves on one name's snapshot price path."""
    p = np.asarray(prices, dtype=float)
    S = len(p) - 1                                                        # snapshots after the decision price
    if S < 1 or not np.all(p > 0):
        return {}
    best = {"pnl": -np.inf}
    same_timing_best = -np.inf
    for lvl in levels:
        notional = lvl * slice_cap
        if lvl <= 0:
            pnl_flat = 0.0
            if pnl_flat > best["pnl"]:
                best = {"pnl": 0.0, "level": 0.0, "entry": 0, "exit": None}
            same_timing_best = max(same_timing_best, 0.0)
            continue
        for i in range(0, S):                                             # enter at snapshot i (0 = the open decision)
            fee_in = fee_fn(notional, p[i], "buy")
            # hold to the end
            pnl_hold = notional * (p[S] / p[i] - 1.0) - fee_in
            if i == 0:
                same_timing_best = max(same_timing_best, pnl_hold)
            if pnl_hold > best["pnl"]:
                best = {"pnl": pnl_hold, "level": lvl, "entry": i, "exit": None}
            for j in range(i + 1, S):                                     # or sell at an earlier snapshot j
                pnl = notional * (p[j] / p[i] - 1.0) - fee_in - fee_fn(notional, p[j], "sell")
                if pnl > best["pnl"]:
                    best = {"pnl": pnl, "level": lvl, "entry": i, "exit": j}
    lvl_us = float(np.clip(level_us, 0.0, 1.0))
    pnl_us = lvl_us * slice_cap * (p[S] / p[0] - 1.0) - (fee_fn(lvl_us * slice_cap, p[0], "buy") if lvl_us > 0 else 0.0)
    sizing = max(0.0, same_timing_best - pnl_us)
    timing = max(0.0, best["pnl"] - same_timing_best)
    return {**best, "pnl_us": pnl_us, "level_us": lvl_us, "regret": max(0.0, best["pnl"] - pnl_us),
            "sizing_regret": sizing, "timing_regret": timing}


def what_if_day(prices: dict[str, list[float]], held: dict[str, float], slice_cap: float, equity: float, fee_fn,
                labels: list[str] | None = None, levels: tuple[float, ...] = LEVELS, gross_cap: float = 1.0,
                workers: int = 4) -> dict:
    """Search, in parallel across names, every move that was available during the session and score what
    we did against the best one.  ``prices[t]`` = [decision price, snapshot prices...], ``held[t]`` = the
    fraction of a slice we ended the session with (entered at the open), ``fee_fn(notional, price, side)``
    -> fee in currency.  Returns per-name best moves, the joint best portfolio (best names up to the gross
    cap), and the regret split into sizing and timing, all as fractions of ``equity``."""
    names = [t for t in prices if len(prices[t]) >= 2]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        found = dict(zip(names, ex.map(lambda t: _best_move_for_name(prices[t], held.get(t, 0.0), slice_cap, fee_fn, levels), names)))
    found = {t: v for t, v in found.items() if v}
    n_slots = max(1, int(np.floor(gross_cap / (slice_cap / equity) + 1e-9))) if equity > 0 and slice_cap > 0 else len(found)
    ranked = sorted(found, key=lambda t: -found[t]["pnl"])
    chosen = [t for t in ranked if found[t]["pnl"] > 0][:n_slots]
    best_total = sum(found[t]["pnl"] for t in chosen) / equity if equity > 0 else 0.0
    actual_total = sum(v["pnl_us"] for v in found.values()) / equity if equity > 0 else 0.0
    per_name = []
    for t, v in found.items():
        row = {"ticker": t, "level_us": v["level_us"], "pnl_us": v["pnl_us"] / equity, "best_level": v["level"],
               "best_entry": labels[v["entry"]] if labels and v["entry"] < len(labels) else v["entry"],
               "best_exit": (labels[v["exit"]] if labels and v["exit"] is not None and v["exit"] < len(labels) else v["exit"]) if v["exit"] is not None else "hold",
               "best_pnl": v["pnl"] / equity, "regret": v["regret"] / equity, "sizing_regret": v["sizing_regret"] / equity,
               "timing_regret": v["timing_regret"] / equity}
        per_name.append(row)
    per_name.sort(key=lambda r: -r["regret"])
    return {"best_return": best_total, "actual_return": actual_total, "best_names": chosen, "per_name": per_name,
            "sizing_regret": sum(r["sizing_regret"] for r in per_name), "timing_regret": sum(r["timing_regret"] for r in per_name),
            "moves_searched": int(sum(1 for _ in found) * len(levels) * max(1, (len(next(iter(prices.values()))) - 1) ** 2 // 2))}


# ---------------------------------------------------------------------- trade round trips (pyfolio)
def round_trips(fills: list[dict], equity: pd.Series | float | None = None) -> dict | None:
    """Round-trip statistics of the fills in a window (a round trip = a position opened and closed).
    Uses pyfolio-reloaded's ``extract_round_trips`` when available; returns None without fills or pyfolio."""
    rows = [f for f in fills if f.get("qty") and f.get("price")]
    if not rows:
        return None
    try:
        from pyfolio.round_trips import extract_round_trips
    except ImportError:
        return None
    txn = pd.DataFrame({"amount": [float(f["qty"]) * (1.0 if f.get("side", "buy") == "buy" else -1.0) for f in rows],
                        "price": [float(f["price"]) for f in rows], "symbol": [f.get("ticker", "?") for f in rows]},
                       index=pd.to_datetime([f.get("ts") for f in rows], utc=True).tz_convert(None))
    txn = txn.sort_index()
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rts = extract_round_trips(txn)
    except Exception as e:  # noqa: BLE001
        log.debug("round trips failed: %s", e)
        return None
    if rts is None or len(rts) == 0:
        return {"n": 0, "open_positions": int((txn.groupby("symbol")["amount"].sum().abs() > 1e-9).sum())}
    pnl = rts["pnl"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    eq = float(equity.iloc[-1]) if isinstance(equity, pd.Series) and len(equity) else (float(equity) if equity else None)
    out = {"n": int(len(rts)), "win_rate": float((pnl > 0).mean()), "avg_pnl": float(pnl.mean()), "total_pnl": float(pnl.sum()),
           "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else (float("inf") if len(wins) else 0.0),
           "avg_win": float(wins.mean()) if len(wins) else 0.0, "avg_loss": float(losses.mean()) if len(losses) else 0.0,
           "avg_holding_days": float(rts["duration"].dt.total_seconds().mean() / 86400.0),
           "best": {"symbol": str(rts.loc[pnl.idxmax(), "symbol"]), "pnl": float(pnl.max())},
           "worst": {"symbol": str(rts.loc[pnl.idxmin(), "symbol"]), "pnl": float(pnl.min())},
           "open_positions": int((txn.groupby("symbol")["amount"].sum().abs() > 1e-9).sum())}
    if eq:
        out["total_return"] = out["total_pnl"] / eq
    return out


# ---------------------------------------------------------------------- input quality and context
def ic_by_voter(votes: dict[tuple, dict], R: pd.DataFrame, min_names: int = 5) -> dict[str, dict]:
    """Information coefficient per input: the daily cross-sectional Spearman correlation between the
    input's score and the next day's return, averaged over the days with at least ``min_names`` names."""
    if not votes or R is None or len(R) < 2:
        return {}
    nxt = R.shift(-1)
    by_day: dict[pd.Timestamp, dict[str, dict[str, float]]] = {}
    for (d, t), vv in votes.items():
        d = pd.Timestamp(d).normalize()
        for voter, v in (vv or {}).items():
            if voter in ("policy", "momentum"):
                continue
            by_day.setdefault(d, {}).setdefault(voter, {})[t] = float(v.get("score", v.get("vote", 0.0)))
    acc: dict[str, list[float]] = {}
    for d, per_voter in by_day.items():
        if d not in nxt.index:
            continue
        fwd = nxt.loc[d]
        for voter, scores in per_voter.items():
            s = pd.Series(scores)
            f = fwd.reindex(s.index)
            ok = f.notna() & s.notna()
            if ok.sum() < min_names or s[ok].nunique() < 2:
                continue
            ic = s[ok].corr(f[ok], method="spearman")
            if np.isfinite(ic):
                acc.setdefault(voter, []).append(float(ic))
    out = {}
    for voter, ics in acc.items():
        a = np.asarray(ics)
        out[voter] = {"ic": float(a.mean()), "ir": float(a.mean() / a.std()) if len(a) > 1 and a.std() > 0 else 0.0, "days": int(len(a))}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["ic"]))


def regime(bench_daily: pd.Series | None) -> dict:
    """Context for the lessons: how the benchmark moved and how rough it was."""
    if bench_daily is None or len(bench_daily.dropna()) == 0:
        return {}
    b = bench_daily.fillna(0.0)
    ret = float((1.0 + b).prod() - 1.0)
    vol = float(b.std() * np.sqrt(252)) if len(b) > 1 else 0.0
    trend = "up" if ret > 0.005 else "down" if ret < -0.005 else "flat"
    rough = "volatile" if vol > 0.25 else "calm" if vol < 0.12 else "normal"
    return {"benchmark_return": ret, "benchmark_vol": vol, "label": f"{trend}-{rough}"}
