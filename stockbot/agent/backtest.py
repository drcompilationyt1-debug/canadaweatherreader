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

from ..execution.ranking import ANCHOR, CANDIDATE_INPUTS, DEFAULT_INPUTS, every_bars_of, load_tuned_profile, select_top, vol_scale
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
    num = sum((df * w).fillna(0.0) for df, w in parts)                 # a missing input drops out of the blend (it must not NaN the score)
    den = sum(df.notna() * w for df, w in parts)
    return num / den.replace(0.0, np.nan)


def take_profit_step(held: list[str], price, entry: dict, vol, pct, tp: dict, i: int, sold: dict, k: int, ranked=None,
                     stats: dict | None = None, proj=None, mult: dict | None = None, peak: dict | None = None) -> tuple[list[str], list[tuple]]:
    """One day of the take-profit re-evaluation (the backtest and the live runner share it).

    ``entry`` {name: (reference price, bar, projection)}, ``sold`` {name: (sale price, bar, amount sold in slot units)},
    ``mult`` {name: slot multiplier, 1 = a full slot} and ``peak`` {name: highest price since its peak} are updated in place;
    ``price``, ``vol`` (daily volatility), ``pct`` (today's score percentile) and ``proj`` (the models' 20-day log-return
    projection) are mappings by name; ``ranked`` lists the choosable names best first (for ``replace``).

    What counts as a peak is tuned: ``pct`` (a fixed gain), ``z`` (a move of z times the name's own volatility, scaled by the
    square root of the bars held) or ``fz`` (the price beat the projection made at the reference by fz volatilities).  At a
    peak the name is re-evaluated: kept while it ranks at least ``keep_pct`` (reference and projection reset), otherwise sold -
    all of it, or the ``scale`` fraction (scale-out: the rest runs on).  Practitioners' exits, each optional:
    ``trail`` - after a peak, sell what is left when the price falls ``trail`` monthly volatilities below its high since the peak
    (a trailing stop protects the gain); ``stop`` - sell when the price falls ``stop`` below the reference (Han, Zhou and Zhu
    2016: a 10% stop more than doubled momentum's Sharpe ratio); ``rebuy_dip`` - a name sold on a peak is bought back when it
    falls that far below the sale price and still ranks in the top half, with the whole sale amount (the same money buys more
    shares); ``replace`` - a slot emptied by a sale goes to the best-ranked name not held.  Returns (held list, events)."""
    held = list(held)
    mult = mult if mult is not None else {}
    peak = peak if peak is not None else {}
    events: list[tuple] = []
    has_peak = any(key in tp for key in ("pct", "z", "fz"))

    def sell(t: str, p: float, kind: str, gain: float, frac: float = 1.0) -> None:
        amount = mult.get(t, 1.0) * frac
        prev = sold.get(t)
        sold[t] = (float(p), int(i), (prev[2] if prev and len(prev) > 2 else 0.0) + amount)
        if frac >= 1.0 - 1e-9:
            held.remove(t)
            entry.pop(t, None)
            peak.pop(t, None)
            mult.pop(t, None)
        else:
            mult[t] = mult.get(t, 1.0) * (1.0 - frac)
        events.append((kind, t, gain, frac))
        if stats is not None:
            stats["sold"] += 1

    for t in list(held):
        ref = entry.get(t)
        p = price.get(t) if hasattr(price, "get") else None
        if ref is None or p is None or not np.isfinite(p) or ref[0] <= 0:
            continue
        p = float(p)
        gain, days = p / float(ref[0]) - 1.0, max(1, int(i) - int(ref[1]))
        v = vol.get(t, np.nan) if hasattr(vol, "get") else np.nan
        ok_v = v is not None and np.isfinite(v) and v > 0
        if "stop" in tp and gain <= -float(tp["stop"]):                   # the stop-loss
            sell(t, p, "stopped", gain)
            continue
        if t in peak and "trail" in tp and ok_v:                          # the trailing stop after a peak
            peak[t] = max(peak[t], p)
            if p <= peak[t] * (1.0 - float(tp["trail"]) * float(v) * np.sqrt(20.0)):
                sell(t, p, "trailed", gain)
                continue
        if not has_peak:
            continue
        if "fz" in tp:
            mu = float(ref[2]) if len(ref) > 2 and ref[2] is not None and np.isfinite(ref[2]) else 0.0
            hit = ok_v and np.log1p(gain) > mu * min(days, 20) / 20.0 + float(tp["fz"]) * float(v) * np.sqrt(days)
        elif "z" in tp:
            hit = ok_v and gain > float(tp["z"]) * float(v) * np.sqrt(days)
        else:
            hit = gain > float(tp.get("pct", 0.15))
        if not hit:
            continue
        if stats is not None:
            stats["triggers"] += 1
        mu_now = proj.get(t, np.nan) if (proj is not None and hasattr(proj, "get")) else np.nan
        new_ref = (p, int(i), float(mu_now) if mu_now is not None and np.isfinite(mu_now) else 0.0)
        if float(pct.get(t, 0.0) or 0.0) >= float(tp.get("keep_pct", 1.01)):
            entry[t] = new_ref
            peak[t] = max(peak.get(t, p), p)
            events.append(("kept", t, gain, 0.0))
            if stats is not None:
                stats["kept"] += 1
            continue
        frac = float(tp.get("scale", 1.0) or 1.0)
        sell(t, p, "sold", gain, frac)
        if frac < 1.0 - 1e-9:                                             # scaled out: the rest runs on from here
            entry[t] = new_ref
            peak[t] = max(peak.get(t, p), p)
        elif tp.get("replace") and ranked:
            for c in ranked:
                if c not in held and c not in sold:
                    held.append(c)
                    mult[c] = 1.0
                    events.append(("in", c, 0.0, 1.0))
                    break
    dip = float(tp.get("rebuy_dip", 0.0) or 0.0)
    if dip > 0:                                                       # the flip: the whole sale buys back in on the dip
        for t, rec in list(sold.items()):
            sp, amount = rec[0], (rec[2] if len(rec) > 2 else 1.0)
            p = price.get(t) if hasattr(price, "get") else None
            if p is None or not np.isfinite(p) or p > sp * (1.0 - dip) or float(pct.get(t, 0.0) or 0.0) < 0.5:
                continue
            if t in held:
                mult[t] = mult.get(t, 1.0) + amount
            elif len(held) < k:
                held.append(t)
                mult[t] = amount
                mu_now = proj.get(t, np.nan) if (proj is not None and hasattr(proj, "get")) else np.nan
                entry[t] = (float(p), int(i), float(mu_now) if mu_now is not None and np.isfinite(mu_now) else 0.0)
            else:
                continue
            sold.pop(t)
            events.append(("rebought", t, float(p) / sp - 1.0, amount))
            if stats is not None:
                stats["rebought"] += 1
    return held, events


def apply_deals(scores: dict[str, float], held: list[str], k: int, deals: dict, sectors: dict[str, str] | None = None,
                max_per_sector: int = 0) -> list[str]:
    """The opportunistic layer for one day between rebalances (see ``simulate``): returns the new held list."""
    import numpy as np

    enter, exit_, max_swaps = float(deals.get("enter_pct", 0.95)), float(deals.get("exit_pct", 0.0)), int(deals.get("max_swaps", 1))
    gap = float(deals.get("min_gap", 0.0))
    names = [t for t, v in scores.items() if np.isfinite(v)]
    if len(names) < 10 or max_swaps <= 0:
        return held
    order = sorted(names, key=lambda t: scores[t])
    pct = {t: (i + 0.5) / len(order) for i, t in enumerate(order)}          # percentile of each name's score today
    held = list(held)
    changes = 0
    if exit_ > 0:                                                            # a holding that collapsed leaves
        for t in sorted([t for t in held if t in pct], key=lambda t: pct[t]):
            if pct[t] < exit_ and changes < max_swaps:
                held.remove(t)
                changes += 1
    counts: dict[str, int] = {}
    for t in held:
        sec = (sectors or {}).get(t)
        if sec and sec not in ("Unknown", "Index"):
            counts[sec] = counts.get(sec, 0) + 1
    for t in sorted([t for t in names if t not in held and pct[t] >= enter], key=lambda t: -pct[t]):
        if changes >= max_swaps:
            break
        sec = (sectors or {}).get(t)
        if max_per_sector and sec and sec not in ("Unknown", "Index") and counts.get(sec, 0) >= max_per_sector:
            continue
        if len(held) < k:
            held.append(t)
        else:
            weakest = min([h for h in held if h in pct], key=lambda h: pct[h], default=None)
            if weakest is None or pct[t] - pct[weakest] < gap:
                continue
            held.remove(weakest)
            wsec = (sectors or {}).get(weakest)
            if wsec and wsec in counts:
                counts[wsec] -= 1
            held.append(t)
        if sec and sec not in ("Unknown", "Index"):
            counts[sec] = counts.get(sec, 0) + 1
        changes += 1
    return held


def eligibility(cfg, px: pd.DataFrame) -> pd.DataFrame | None:
    """(dates x tickers) True where a name may be chosen: from its first S&P 500 membership date (``config/universe_membership.json``),
    so a backtest never picks a name because it later grew into the index; names without a record are always eligible."""
    import json
    from pathlib import Path

    from ..paths import resolve

    f = Path(resolve(str(cfg.get("universe_membership", "config/universe_membership.json"))))
    if not f.exists():
        return None
    try:
        member_from = dict(json.loads(f.read_text(encoding="utf-8")).get("member_from", {}))
    except Exception as e:  # noqa: BLE001
        log.warning("membership file %s unreadable: %s", f, e)
        return None
    el = pd.DataFrame(True, index=px.index, columns=px.columns)
    for t, day in member_from.items():
        if t in el.columns:
            el.loc[el.index < pd.Timestamp(day), t] = False
    return el


PROJECTION_INPUTS = (("timesfm.tfm_ret_20", 10.0), ("chronos.chr_ret_20", 10.0), ("chronos2.c2_ret_20", 10.0))


def feature_frame(ds, key: str, scale: float = 1.0) -> pd.DataFrame | None:
    """Date x ticker values of one block feature (NaN where the block is absent), divided by ``scale``; None if not in the layout."""
    block, feat = key.split(".", 1)
    try:
        b = ds.layout.block(block)
        j = b.start + list(b.feature_names).index(feat)
    except (KeyError, ValueError):
        return None
    cols = {}
    for t, td in ds.data.items():
        v = td.signals[:, j].astype(float) / scale
        v[td.signals[:, b.offset] < 0.5] = np.nan
        cols[t] = pd.Series(v, index=pd.DatetimeIndex(td.dates))
    return pd.DataFrame(cols).sort_index()


def projection_frame(ds) -> pd.DataFrame | None:
    """The models' 20-day log-return projection per date and name: TimesFM, then Chronos, then Chronos-2 where it is missing."""
    out = None
    for key, scale in PROJECTION_INPUTS:
        f = feature_frame(ds, key, scale)
        if f is not None:
            out = f if out is None else out.combine_first(f)
    return out


def closes(ds) -> pd.DataFrame:
    return pd.DataFrame({t: pd.Series(td.close, index=pd.DatetimeIndex(td.dates)) for t, td in ds.data.items()}).sort_index()


def trend_state(px: pd.DataFrame, benchmark: str = "SPY", sma: int = 200, band: float = 0.02) -> pd.Series:
    """Faber-style trend switch: on while the benchmark closes above its ``sma``-day average (a ``band`` either side to
    avoid whipsaw), off below it.  Checked daily; it flips a couple of times a year."""
    c = px[benchmark].astype(float).dropna()           # the benchmark's own bars: the union index has gaps (other markets' holidays)
    m = c.rolling(int(sma)).mean()
    states = {}
    on = True
    for d in c.index:
        cm, mm = c.get(d), m.get(d)
        if pd.notna(cm) and pd.notna(mm):
            if on and cm < mm * (1.0 - band):
                on = False
            elif not on and cm > mm * (1.0 + band):
                on = True
        states[d] = float(on)
    return pd.Series(states).reindex(px.index).ffill().fillna(1.0)


def simulate(px: pd.DataFrame, score: pd.DataFrame | None, start, k: int = 20, every: int = 10, hysteresis: int = 3,
             fee_bps: float = 8.0, end=None, core: dict | None = None, trend: dict | None = None, reserve: float = 0.0,
             sectors: dict[str, str] | None = None, max_per_sector: int = 0, vol_target: float = 0.0, vol_window: int = 20,
             vol_floor: float = 0.4, eligible: pd.DataFrame | None = None, deals: dict | None = None,
             take_profit: dict | None = None, projection: pd.DataFrame | None = None) -> dict:
    """Daily portfolio returns of the rank-core rule (``score`` None = equal-weight everything), fees on turnover.
    ``core`` = {ticker, share}: a buy-and-hold slice bought once and never sold; ``trend`` = {benchmark, sma, band}: the
    rank slots go to cash while the benchmark is below its moving average; ``reserve`` = cash never invested;
    ``deals`` = {enter_pct, exit_pct, max_swaps, min_gap}: between rebalance days a name whose score percentile reaches
    ``enter_pct`` may enter (a free slot, else replacing the weakest holding when it beats it by ``min_gap`` percentiles) and a
    holding whose percentile falls below ``exit_pct`` leaves, at most ``max_swaps`` changes a day;
    ``take_profit`` = {z | pct, keep_pct, replace, rebuy_dip, vol_days}: between rebalance days a held name whose gain since
    entry exceeds ``z`` standard deviations of its own daily moves (scaled by the square root of the days held) or ``pct`` is
    re-evaluated: kept while its score percentile is at least ``keep_pct`` (its reference price resets, so the next trigger
    needs another extreme move), sold otherwise.  ``replace`` refills the slot with the best-ranked name not held;
    ``rebuy_dip`` buys a name sold on a spike back when it falls that far below the sale price and still ranks in the top half."""
    idx = px.index[(px.index >= pd.Timestamp(start)) & ((px.index <= pd.Timestamp(end)) if end is not None else True)]
    rets = px.pct_change(fill_method=None).reindex(idx).fillna(0.0)
    wts = pd.DataFrame(0.0, index=idx, columns=px.columns)
    core_t, core_share = (str(core.get("ticker", "SPY")), float(core.get("share", 0.0))) if core else (None, 0.0)
    core_series = core.get("series") if core else None                    # the core's weight per day (the model's timing)
    if core_t is not None and core_t not in px.columns:
        core_t, core_share, core_series = None, 0.0, None
    on = trend_state(px, str(trend.get("benchmark", "SPY")), int(trend.get("sma", 200)), float(trend.get("band", 0.02))) if trend else None
    el = eligible.reindex(index=idx, columns=px.columns).fillna(True).astype(bool) if eligible is not None else None
    held: list[str] = []
    book: list[float] = []                                            # the book's own daily returns so far (vol targeting)
    prev_row = pd.Series(0.0, index=px.columns)
    tp = dict(take_profit or {})
    entry: dict[str, tuple[float, int]] = {}                          # name -> (reference price, bar index) for the take-profit test
    sold_tp: dict[str, tuple] = {}                                    # name -> (sale price, bar index, amount) of a take-profit sale
    tp_mult: dict[str, float] = {}                                    # name -> slot multiplier after a scale-out or a buy-back
    tp_peak: dict[str, float] = {}                                    # name -> highest price since its peak (trailing stop)
    tp_vol = (px.pct_change(fill_method=None).rolling(int(tp.get("vol_days", 60)), min_periods=20).std().reindex(idx)
              if tp else None)
    tp_stats = {"triggers": 0, "sold": 0, "kept": 0, "rebought": 0}
    proj = projection.reindex(index=idx, columns=px.columns).ffill(limit=5) if (tp and projection is not None) else None
    for i, d in enumerate(idx):
        if i > 0:
            book.append(float((prev_row * rets.loc[d]).sum()))
        share_d = float(core_series.get(d, core_share)) if core_series is not None else core_share
        satellite = max(0.0, 1.0 - share_d - float(reserve))
        if core_t is not None:
            wts.loc[d, core_t] = share_d
        if score is None:
            others = [t for t in px.columns if t != core_t and (el is None or el.at[d, t])]
            if others:
                wts.loc[d, others] = satellite / len(others)
            prev_row = wts.loc[d]
            continue
        risk_on = True if on is None else bool(on.get(d, 1.0) >= 0.5)
        if not risk_on:
            held = []                                                     # the trend filter is off: satellite in cash
        elif i % every == 0 or not held:
            s = score.loc[d].dropna() if d in score.index else pd.Series(dtype=float)
            if core_t is not None:
                s = s.drop(core_t, errors="ignore")
            if el is not None and len(s):
                s = s[[t for t in s.index if el.at[d, t]]]                # not yet in the index that day: not choosable
            held = select_top(s.to_dict(), held, k, hysteresis, sectors=sectors, max_per_sector=max_per_sector) if len(s) else held
            sold_tp.clear()
            tp_mult.clear()
            tp_peak.clear()
        elif tp and held:
            s = score.loc[d].dropna() if d in score.index else pd.Series(dtype=float)
            if core_t is not None:
                s = s.drop(core_t, errors="ignore")
            if el is not None and len(s):
                s = s[[t for t in s.index if el.at[d, t]]]
            if len(s) >= 10:
                held, _ = take_profit_step(held, px.loc[d], entry, tp_vol.loc[d], s.rank(pct=True), tp, i, sold_tp, k,
                                           ranked=list(s.sort_values(ascending=False).index), stats=tp_stats,
                                           proj=proj.loc[d] if proj is not None and d in proj.index else None, mult=tp_mult, peak=tp_peak)
        elif deals:
            s = score.loc[d].dropna() if d in score.index else pd.Series(dtype=float)
            if core_t is not None:
                s = s.drop(core_t, errors="ignore")
            if el is not None and len(s):
                s = s[[t for t in s.index if el.at[d, t]]]
            if len(s) >= 10:
                held = apply_deals(s.to_dict(), held, k, deals, sectors=sectors, max_per_sector=max_per_sector)
        scale = vol_scale(book, vol_target, vol_window, vol_floor, 1.0) if vol_target > 0 else 1.0
        if held:
            wts.loc[d, held] = satellite * scale / max(k, 1)              # a slot is full or empty: no trims
            if tp_mult:                                                   # ... except a scale-out or a buy-back of the whole sale
                for t in held:
                    if t in tp_mult:
                        wts.at[d, t] = satellite * scale / max(k, 1) * tp_mult[t]
        if tp:
            for t in held:
                if t not in entry:
                    mu = proj.at[d, t] if proj is not None and t in proj.columns else np.nan
                    entry[t] = (float(px.at[d, t]), i, float(mu) if np.isfinite(mu) else 0.0)
            for t in [t for t in entry if t not in held]:
                entry.pop(t)
        prev_row = wts.loc[d]
    prev = wts.shift(1).fillna(0.0)
    turnover = (wts - prev).abs().sum(axis=1)
    daily = (prev * rets).sum(axis=1) - turnover * fee_bps / 1e4
    eq = (1.0 + daily).cumprod()
    return {"daily": daily, "total": float(eq.iloc[-1] - 1.0) if len(eq) else 0.0,
            "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 1 and daily.std() > 0 else 0.0,
            "max_drawdown": float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0,
            "turnover_per_year": float(turnover.sum() / max(len(idx), 1) * 252), "days": int(len(idx)),
            **({"take_profit": tp_stats} if tp else {})}


def core_series_from_policy(cfg, ds, ticker: str, start, min_share: float, max_share: float, every: int = 21, band: float = 0.05,
                            baseline: float | None = None) -> pd.Series | None:
    """The core's weight per day as the current policy would time it: its conviction on the core name (from a run of the
    policy along the window) placed between min_share and max_share every ``every`` bars, held between decisions, moved
    only when the change clears ``band`` - the runner's rule.  None without a policy."""
    try:
        from ..config import env_settings
        from .evaluate import run_window
        from .policy import PolicyBundle

        ckpt = cfg.path("train.checkpoint_dir", "models/policy")
        if ticker not in ds.data or not PolicyBundle.exists(ckpt):
            return None
        bundle = PolicyBundle.load(ckpt)
        td = ds.data[ticker]
        dates = pd.DatetimeIndex(td.dates)
        i0 = max(int(dates.searchsorted(pd.Timestamp(start))) - 1, td.min_start)
        length = len(td) - 2 - i0
        if length < 5:
            return None

        class _Fit:
            def predict(self, obs, deterministic=True):
                return bundle.model.predict(bundle.fit_obs(np.asarray(obs, np.float32)), deterministic=deterministic)

        res = run_window(_Fit(), ds, ticker, env_settings(cfg), start=i0, length=length)
        conv = pd.Series(np.asarray(res["exposures"], float), index=pd.to_datetime(res["dates"][1:]))
        out, cur, last_i = {}, float(baseline if baseline is not None else (min_share + max_share) / 2), None
        for i, d in enumerate(conv.index):
            if last_i is None or i - last_i >= every:
                target = min_share + (max_share - min_share) * float(np.clip(conv.iloc[i], 0.0, 1.0))
                if abs(target - cur) >= band:
                    cur = target
                last_i = i
            out[d] = cur
        return pd.Series(out)
    except Exception as e:  # noqa: BLE001
        log.debug("core timing from the policy unavailable: %s", e)
        return None


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
    def _profile(c, rk_, budget):
        core = dict(c.get_path("execution.core", {}) or {})
        base = {"top_k": int(rk_.get("top_k", 20)), "every_bars": every_bars_of(rk_.get("every_bars", 10)), "hysteresis": int(rk_.get("hysteresis", 3)),
                "core_share": float(core.get("share", 0.0) or 0.0)}
        base["deals"] = deals_of(rk_)
        tuned = load_tuned_profile(c.path("models_dir", "models"), str(c.get("account") or "main"), base) if rk_.get("adaptive", True) else None
        deals = tuned.get("deals", base["deals"]) if tuned else base["deals"]
        if tuned:                                                          # what the runner actually uses
            rk_ = {**rk_, "top_k": tuned.get("top_k", rk_.get("top_k", 20)), "every_bars": tuned.get("every_bars", rk_.get("every_bars", 10)),
                   "hysteresis": tuned.get("hysteresis", rk_.get("hysteresis", 3))}
            cs = float(tuned.get("core_share", core.get("share", 0.0) or 0.0))
            core = {**core, "share": cs, "min_share": cs / 2.0, "max_share": cs, "decide": "model"} if cs > 0 else {**core, "share": 0.0}
        tf = dict(rk_.get("trend_filter", {}) or {})
        share = float(core.get("share", 0.0) or 0.0)
        core_spec = None
        if share > 0:
            core_spec = {"ticker": str(core.get("ticker", "SPY")), "share": share, "decide": str(core.get("decide", "model")),
                         "min_share": float(core.get("min_share", share * 0.5)), "max_share": float(core.get("max_share", share * 1.3)),
                         "every": every_bars_of(core.get("every_bars", 21)), "band": float(core.get("band", 0.05))}
        vt = dict(rk_.get("vol_target", {}) or {})
        return {"budget": float(budget), "k": int(rk_.get("top_k", 20)), "every": every_bars_of(rk_.get("every_bars", 10)),
                "hysteresis": int(rk_.get("hysteresis", 3)), "reserve": float(c.get_path("execution.cash_reserve", 0.1) or 0.0),
                "core": core_spec, "max_per_sector": int(rk_.get("max_per_sector", 0) or 0), "deals": deals,
                "vol_target": float(vt.get("target", 0.0) or 0.0) if vt.get("enabled") else 0.0, "vol_window": int(vt.get("window", 20)),
                "vol_floor": float(vt.get("floor", 0.4)),
                "trend": {"benchmark": str(tf.get("benchmark", "SPY")), "sma": int(tf.get("sma", 200)), "band": float(tf.get("band", 0.02))} if tf.get("enabled") else None}

    if budgets is None:
        budgets = {"100k": _profile(cfg, rk, cfg.get_path("env.initial_cash", 100_000))}
        try:
            from ..config import account_config, account_names

            for name in account_names(cfg):
                ca = account_config(cfg, name)
                r2 = dict(ca.get_path("execution.rank", {}) or {})
                budgets[name] = _profile(ca, r2, ca.get_path("env.initial_cash", 10_000))
        except Exception as e:  # noqa: BLE001
            log.debug("account budgets: %s", e)
    hyst = int(rk.get("hysteresis", 3))
    elig = eligibility(cfg, px)
    sectors = None
    if any(b.get("max_per_sector") for b in budgets.values()):
        try:
            from ..data.sectors import load_sectors

            sectors = load_sectors(cfg, list(px.columns), refresh=False) or None
        except Exception as e:  # noqa: BLE001
            log.debug("sectors unavailable for the backtest: %s", e)
    out: dict = {"inputs": inputs, "windows": {k: str(v.date()) for k, v in windows.items()}, "results": {},
                 "point_in_time": elig is not None}
    for wname, start in windows.items():
        res = {}
        if benchmark in px.columns:
            b = px[benchmark].pct_change(fill_method=None).reindex(px.index[px.index >= start]).fillna(0.0)
            eq = (1 + b).cumprod()
            res[benchmark] = {"total": float(eq.iloc[-1] - 1), "sharpe": float(b.mean() / b.std() * np.sqrt(252)) if b.std() > 0 else 0.0,
                              "max_drawdown": float((eq / eq.cummax() - 1).min())}
        ew = simulate(px, None, start, eligible=elig)
        res["equal_weight"] = {k: v for k, v in ew.items() if k != "daily"}
        for bname, bset in budgets.items():
            core = dict(bset["core"]) if bset.get("core") else None
            if core and core.get("decide", "model") == "model":                  # the policy's own timing of the core
                series = core_series_from_policy(cfg, ds, core["ticker"], start, core["min_share"], core["max_share"], core["every"], core["band"],
                                                 baseline=core["share"])
                if series is not None:
                    core["series"] = series
            bset = {**bset, "core": core}
            core_share = (bset.get("core") or {}).get("share", 0.0) if bset.get("core") else 0.0
            satellite_budget = bset["budget"] * max(0.0, 1.0 - core_share - bset.get("reserve", 0.0))
            fee = round_trip_bps(cfg, satellite_budget, bset["k"])
            extra_kw = {"sectors": sectors, "max_per_sector": bset.get("max_per_sector", 0), "vol_target": bset.get("vol_target", 0.0),
                        "vol_window": bset.get("vol_window", 20), "vol_floor": bset.get("vol_floor", 0.4), "eligible": elig,
                        "deals": bset.get("deals")}
            r = simulate(px, score, start, k=bset["k"], every=bset["every"], hysteresis=bset.get("hysteresis", hyst), fee_bps=fee,
                         core=bset.get("core"), trend=bset.get("trend"), reserve=bset.get("reserve", 0.0), **extra_kw)
            # the same rule started 2..20 bars later: a concentrated book's result depends on the rebalance phase, so the
            # phase-averaged figure is the one to believe (mean / worst / best over the offsets)
            phase = []
            idx_w = px.index[px.index >= pd.Timestamp(start)]
            for off in range(2, 21, 3):
                if off < len(idx_w) - 30:
                    rp = simulate(px, score, idx_w[off], k=bset["k"], every=bset["every"], hysteresis=bset.get("hysteresis", hyst), fee_bps=fee,
                                  core=bset.get("core"), trend=bset.get("trend"), reserve=bset.get("reserve", 0.0), **extra_kw)
                    phase.append((rp["total"], rp["sharpe"], rp["max_drawdown"]))
            phase_stats = ({"mean_total": float(np.mean([p[0] for p in phase])), "min_total": float(min(p[0] for p in phase)),
                            "max_total": float(max(p[0] for p in phase)), "mean_sharpe": float(np.mean([p[1] for p in phase])),
                            "n": len(phase) + 1} if phase else None)
            res[f"rank_{bname}"] = {**{k: v for k, v in r.items() if k != "daily"}, "k": bset["k"], "every_bars": bset["every"], "fee_bps": fee,
                                    "phase": phase_stats,
                                    "core": {k2: v2 for k2, v2 in (bset.get("core") or {}).items() if k2 != "series"} if bset.get("core") else None,
                                    "core_timed_by_policy": bool(bset.get("core") and "series" in bset["core"]),
                                    "core_avg_share": float(bset["core"]["series"].mean()) if bset.get("core") and "series" in bset["core"] else None,
                                    "trend_filter": bool(bset.get("trend")), "reserve": bset.get("reserve", 0.0),
                                    "excess_vs_benchmark": r["total"] - res.get(benchmark, {}).get("total", 0.0)}
        out["results"][wname] = res
    return out


def trailing_ic(ds, inputs: list[str], days: int = 250, horizon: int = 20, min_names: int = 15) -> dict[str, dict]:
    """Mean daily cross-sectional Spearman IC of each input against the ``horizon``-day forward return over the last
    ``days`` bars that have a settled outcome, with its t-statistic."""
    px = closes(ds)
    fwd = px.shift(-horizon) / px - 1.0
    lay = ds.layout
    out = {}
    for key in inputs:
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
        sig = pd.DataFrame(cols).reindex(px.index)
        settled = px.index[: len(px.index) - horizon][-days:]
        ics = []
        for d in settled:
            s, f = sig.loc[d], fwd.loc[d]
            ok = s.notna() & f.notna()
            if ok.sum() >= min_names and s[ok].nunique() > 2:
                ic = s[ok].corr(f[ok], method="spearman")
                if np.isfinite(ic):
                    ics.append(float(ic))
        if ics:
            a = np.asarray(ics)
            out[key] = {"ic": float(a.mean()), "t": float(a.mean() / a.std() * np.sqrt(len(a))) if a.std() > 0 else 0.0, "days": int(len(a))}
    return out


def _previous_report(out_path) -> dict:
    """Last week's report at ``out_path`` (the tuner adopts a change only when two consecutive weekly runs agree on it)."""
    import json
    from pathlib import Path

    if out_path is None or not Path(out_path).exists():
        return {}
    try:
        return dict(json.loads(Path(out_path).read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return {}


def _confirmed(prev: dict, proposal, passed: bool, today: str) -> tuple[bool, int]:
    """(adopt now?, streak): a proposal is adopted on its second consecutive weekly pass with the same content; a rerun on
    the same day does not count as a second week."""
    prev_passed = bool(prev.get("guard_passed", prev.get("accepted", False)))
    prev_prop = prev.get("proposal", prev.get("best") if "best" in prev else prev.get("tuned"))
    if not passed:
        return False, 0
    if prev_passed and prev_prop == proposal and str(prev.get("tuned_at")) != today:
        return True, int(prev.get("streak", 1) or 1) + 1
    if prev_passed and prev_prop == proposal:                                 # same day: keep last week's count
        return bool(prev.get("accepted", False)), int(prev.get("streak", 1) or 1)
    return False, 1


def tune_rank_weights(cfg, ds, out_path=None, days: int = 250, min_t: float = 2.0, anchor_share: float = 0.5) -> dict:
    """Re-weight the blend from the trailing-year ICs (positive, significant inputs only; the ranking head keeps at least
    ``anchor_share`` of the weight) and keep it only when the last year's backtest with it is no worse than the static blend."""
    rk = dict(cfg.get_path("execution.rank", {}) or {})
    static = {str(k): float(v) for k, v in (rk.get("inputs") or DEFAULT_INPUTS).items()}
    ics = trailing_ic(ds, CANDIDATE_INPUTS, days=days)
    good = {k: v["ic"] for k, v in ics.items() if v["ic"] > 0 and v["t"] >= min_t and k != ANCHOR}
    others = sum(good.values())
    tuned = {ANCHOR: 1.0}
    if others > 0:
        budget = (1.0 - anchor_share) / anchor_share            # the others share this much relative to the anchor's 1.0
        tuned.update({k: round(budget * v / others, 4) for k, v in good.items()})
    px = closes(ds)
    start = px.index[-1] - pd.DateOffset(years=1)
    k, every, hyst = int(rk.get("top_k", 20)), every_bars_of(rk.get("every_bars", 10)), int(rk.get("hysteresis", 3))
    fee = round_trip_bps(cfg, 100_000, k)
    elig = eligibility(cfg, px)
    res = {}
    for name, inputs in (("static", static), ("tuned", tuned)):
        try:
            r = simulate(px, blended_scores(ds, inputs), start, k=k, every=every, hysteresis=hyst, fee_bps=fee, eligible=elig)
            res[name] = {kk: vv for kk, vv in r.items() if kk != "daily"}
        except ValueError:
            res[name] = {"total": float("-inf"), "sharpe": float("-inf")}
    same = set(tuned) == set(static) and all(abs(tuned[kk] - static[kk]) < 1e-6 for kk in tuned)
    passed = (not same) and res["tuned"]["sharpe"] >= res["static"]["sharpe"] - 0.05 and res["tuned"]["total"] >= res["static"]["total"] - 0.01
    today = str(date.today())
    # the input set (not the exact weights, which drift with the ICs) must pass two weeks running before it is used live
    proposal = sorted(tuned) if passed else None
    accepted, streak = _confirmed(_previous_report(out_path), proposal, passed, today)
    rep = {"tuned_at": today, "inputs": tuned if accepted else static, "tuned": tuned, "static": static, "ic": ics,
           "backtest_last_year": res, "guard_passed": bool(passed), "proposal": proposal, "streak": streak, "accepted": bool(accepted),
           "reason": "same as static" if same else
           ("tuned blend at least as good over the last year, two weeks running - in use" if accepted else
            ("tuned blend passed this week - in use if it passes again next week" if passed else "tuned blend did worse over the last year - static kept"))}
    if out_path is not None:
        import json
        from pathlib import Path

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    log.info("rank blend tuning: %s (%s)", "accepted" if accepted else "kept static", rep["reason"])
    return rep


# SPY is a regular candidate like every other name (the user's call): no index sleeve is tuned, only slots / cadence / hysteresis
# the opportunistic layer (deals) is in the grid so the evidence decides: 2012-2026 it added nothing for the main book (+0.2 pt/yr,
# a coin flip) and cost the small book 2-25 pt/yr in turnover, so it is off until a two-week pass says otherwise
DEALS_OPTION = {"enter_pct": 0.98, "exit_pct": 0.0, "max_swaps": 1, "min_gap": 0.0}
PROFILE_GRID = {"small": {"top_k": (5, 8), "every_bars": (21, 42), "hysteresis": (3, 8), "core_share": (0.0,), "deals": (None, DEALS_OPTION)},
                "main": {"top_k": (12, 15, 20, 25), "every_bars": (10, 15, 21), "hysteresis": (3, 10), "core_share": (0.0,),
                         "deals": (None, DEALS_OPTION)}}


def deals_of(rk: dict) -> dict | None:
    """The configured opportunistic layer (``execution.rank.deals``) when enabled, else None."""
    dl = dict(rk.get("deals", {}) or {})
    return {k: dl[k] for k in ("enter_pct", "exit_pct", "max_swaps", "min_gap") if k in dl} if bool(dl.get("enabled", False)) else None


def tune_profile(cfg, ds, account: str = "main", out_path=None, years: int = 10, min_gain: float = 0.05) -> dict:
    """Choose an account's structure from the trailing evidence: every candidate (slots x cadence x hysteresis x index sleeve,
    the sleeve timed by the policy) is backtested at the account's fees over the last year and the last ``years``; the best
    last-year Sharpe wins only if it beats the current structure by ``min_gain`` there and is no worse over the long window
    (Sharpe within 0.02, total within 1%).  Otherwise the current structure is kept.  ``rank_profile_<account>.json``.

    The long window is a decade by default: a structure chosen on one year is a coin flip (the tuner once adopted top-25 on a
    one-year Sharpe when top-20 was worth 1.4 points a year more over the full record), and the scores reach back to 2011."""
    from ..config import account_config

    c = account_config(cfg, None if account == "main" else account)
    rk = dict(c.get_path("execution.rank", {}) or {})
    core_cfg = dict(c.get_path("execution.core", {}) or {})
    tf = dict(rk.get("trend_filter", {}) or {})
    trend = {"benchmark": str(tf.get("benchmark", "SPY")), "sma": int(tf.get("sma", 200)), "band": float(tf.get("band", 0.02))} if tf.get("enabled") else None
    reserve = float(c.get_path("execution.cash_reserve", 0.1) or 0.0)
    budget = float(c.get_path("env.initial_cash", 100_000))
    core_t = str(core_cfg.get("ticker", "SPY"))
    inputs = dict(rk.get("inputs") or DEFAULT_INPUTS)
    if rk.get("adaptive", True):
        from ..execution.ranking import load_tuned_inputs

        inputs = load_tuned_inputs(c.path("models_dir", "models"), inputs)
    current = {"top_k": int(rk.get("top_k", 20)), "every_bars": every_bars_of(rk.get("every_bars", 10)), "hysteresis": int(rk.get("hysteresis", 3)),
               "core_share": float(core_cfg.get("share", 0.0) or 0.0), "core_ticker": core_t, "deals": deals_of(rk)}
    config_base = {k: current[k] for k in ("top_k", "every_bars", "hysteresis", "core_share", "deals")}
    prev = load_tuned_profile(c.path("models_dir", "models"), account, config_base)
    if prev:
        current = {**current, **{k: prev[k] for k in ("top_k", "every_bars", "hysteresis", "core_share", "deals") if k in prev}}
    grid = PROFILE_GRID.get("small" if budget < 30_000 else "main", PROFILE_GRID["main"])
    px = closes(ds)
    score = blended_scores(ds, inputs)
    last = px.index[-1]
    windows = {"1y": last - pd.DateOffset(years=1), f"{years}y": last - pd.DateOffset(years=years)}
    series_cache: dict[tuple, pd.Series | None] = {}
    # the structure is judged under the same risk controls production uses (sector cap, volatility targeting)
    max_per_sector = int(rk.get("max_per_sector", 0) or 0)
    vt = dict(rk.get("vol_target", {}) or {})
    vol_target = float(vt.get("target", 0.0) or 0.0) if vt.get("enabled") else 0.0
    vol_window, vol_floor = int(vt.get("window", 20)), float(vt.get("floor", 0.4))
    sectors = None
    if max_per_sector > 0:
        try:
            from ..data.sectors import load_sectors

            sectors = load_sectors(c, list(px.columns), refresh=False) or None
        except Exception as e:  # noqa: BLE001
            log.debug("sectors unavailable for the tuner: %s", e)
    elig = eligibility(cfg, px)

    def run(p: dict, wname: str) -> dict:
        start = windows[wname]
        cs = float(p["core_share"])
        core = None
        if cs > 0:
            key = (wname, cs)
            if key not in series_cache:
                series_cache[key] = core_series_from_policy(cfg, ds, core_t, start, cs / 2.0, cs, 21, 0.05, baseline=cs)
            core = {"ticker": core_t, "share": cs, "series": series_cache[key]}
        fee = round_trip_bps(cfg, budget * max(0.0, 1.0 - cs - reserve), int(p["top_k"]))
        r = simulate(px, score, start, k=int(p["top_k"]), every=int(p["every_bars"]), hysteresis=int(p["hysteresis"]), fee_bps=fee,
                     core=core, trend=trend, reserve=reserve, sectors=sectors, max_per_sector=max_per_sector, vol_target=vol_target,
                     vol_window=vol_window, vol_floor=vol_floor, eligible=elig, deals=p.get("deals"))
        return {k: v for k, v in r.items() if k != "daily"}

    cands = [{"top_k": k, "every_bars": e, "hysteresis": h, "core_share": cs, "core_ticker": core_t, "deals": dl}
             for k in grid["top_k"] for e in grid["every_bars"] for h in grid["hysteresis"] for cs in grid["core_share"]
             for dl in grid.get("deals", (None,))]
    if not any(all(cd[k] == current[k] for k in ("top_k", "every_bars", "hysteresis", "core_share", "deals")) for cd in cands):
        cands.append(dict(current))
    results = []
    for cd in cands:
        r1, r3 = run(cd, "1y"), run(cd, f"{years}y")
        results.append({**cd, "1y": r1, f"{years}y": r3})
    cur = next(r for r in results if all(r[k] == current[k] for k in ("top_k", "every_bars", "hysteresis", "core_share", "deals")))
    best = max(results, key=lambda r: r["1y"]["sharpe"])
    long = f"{years}y"
    ok = (best is not cur and best["1y"]["sharpe"] >= cur["1y"]["sharpe"] + min_gain
          and best[long]["sharpe"] >= cur[long]["sharpe"] - 0.02 and best[long]["total"] >= cur[long]["total"] - 0.01)
    keys = ("top_k", "every_bars", "hysteresis", "core_share", "deals")
    today = str(date.today())
    proposal = {k: best[k] for k in keys} if ok else None
    accepted, streak = _confirmed(_previous_report(out_path), proposal, ok, today)     # two consecutive weekly passes
    chosen = best if accepted else cur
    rep = {"account": account, "tuned_at": today, "guard_passed": bool(ok), "proposal": proposal, "streak": streak, "accepted": bool(accepted),
           "profile": {k: chosen[k] for k in ("top_k", "every_bars", "hysteresis", "core_share", "core_ticker", "deals")},   # in force from now on
           "config_base": config_base,                                        # the config this was tuned from: an edit there resets it
           "current": {k: cur[k] for k in keys},
           "current_result": {"1y": cur["1y"], long: cur[long]}, "best_result": {"1y": best["1y"], long: best[long]},
           "best": {k: best[k] for k in keys},
           "reason": ("a better structure over both windows, two weeks running - in force" if accepted else
                      ("a better structure this week - in force if it wins again next week" if ok else
                       "the current structure is as good or the best one fails the long-window guard - kept")),
           "candidates": [{k: r[k] for k in ("top_k", "every_bars", "hysteresis", "core_share", "deals")} | {"1y_sharpe": r["1y"]["sharpe"], "1y_total": r["1y"]["total"],
                           f"{long}_sharpe": r[long]["sharpe"], f"{long}_total": r[long]["total"], "1y_maxdd": r["1y"]["max_drawdown"]} for r in results]}
    if out_path is not None:
        import json
        from pathlib import Path

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    log.info("rank profile tuning (%s): %s -> %s (%s)", account, rep["current"], rep["profile"], rep["reason"])
    return rep


# the owner's take-profit idea (2026-09-29): candidates tested every weekend against holding (and against whatever is in force)
# what counts as a peak is a hyperparameter: a fixed gain, a move of z times the name's own volatility, or the price beating the
# model's own 20-day projection by fz volatilities - each either re-evaluated (keep while ranking in the top half) or always sold
PEAKS = [{"pct": 0.15}, {"pct": 0.25}, {"pct": 0.40}, {"z": 2.0}, {"z": 3.0}, {"z": 4.0}, {"fz": 1.5}, {"fz": 2.5}]
TP_CANDIDATES = ([None] + [{**p, "keep_pct": kp} for p in PEAKS for kp in (0.5, 1.01)]
                 + [{"fz": 2.5, "keep_pct": 0.5, "rebuy_dip": 0.05}, {"z": 3.0, "keep_pct": 0.5, "replace": True},
                    # practitioners' exits: scale out half at the peak, trail what is left, a plain stop-loss, and combinations
                    {"z": 3.0, "keep_pct": 0.5, "scale": 0.5}, {"fz": 2.5, "keep_pct": 0.5, "scale": 0.5},
                    {"z": 3.0, "keep_pct": 0.5, "trail": 1.0}, {"fz": 2.5, "keep_pct": 0.5, "trail": 1.0},
                    {"fz": 2.5, "keep_pct": 0.5, "scale": 0.5, "trail": 1.0, "rebuy_dip": 0.05},
                    {"pct": 0.25, "keep_pct": 1.01, "rebuy_dip": 0.05}, {"stop": 0.10}, {"stop": 0.15},
                    {"z": 3.0, "keep_pct": 0.5, "trail": 1.0, "stop": 0.15}])


def load_tuned_take_profit(models_dir, account: str = "main") -> dict | None:
    """The take-profit rule in force for an account (``models/take_profit_<account>.json``), or None (plain holding)."""
    import json
    from pathlib import Path

    f = Path(models_dir) / f"take_profit_{account}.json"
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        rule = d.get("in_force")
        return dict(rule) if isinstance(rule, dict) and rule else None
    except Exception as e:  # noqa: BLE001
        log.warning("take-profit file %s unreadable: %s", f, e)
        return None


def _account_book(cfg, account: str) -> dict:
    """The simulate() settings of an account as the runner trades it (structure in force, risk controls, fees)."""
    from ..config import account_config

    c = account_config(cfg, None if account == "main" else account)
    rk = dict(c.get_path("execution.rank", {}) or {})
    k, every, hyst = int(rk.get("top_k", 20)), every_bars_of(rk.get("every_bars", 10)), int(rk.get("hysteresis", 3))
    base = {"top_k": k, "every_bars": every, "hysteresis": hyst, "core_share": 0.0, "deals": deals_of(rk)}
    prof = load_tuned_profile(c.path("models_dir", "models"), account, base) if rk.get("adaptive", True) else None
    if prof:
        k, every, hyst = int(prof.get("top_k", k)), every_bars_of(prof.get("every_bars", every)), int(prof.get("hysteresis", hyst))
    vt = dict(rk.get("vol_target", {}) or {})
    reserve = float(c.get_path("execution.cash_reserve", 0.1) or 0.0)
    budget = float(c.get_path("env.initial_cash", 100_000))
    return {"cfg": c, "rk": rk, "k": k, "every": every, "hysteresis": hyst, "reserve": reserve,
            "fee": round_trip_bps(c, budget * max(0.0, 1.0 - reserve), k), "max_per_sector": int(rk.get("max_per_sector", 0) or 0),
            "vol_target": float(vt.get("target", 0.0) or 0.0) if vt.get("enabled") else 0.0, "vol_window": int(vt.get("window", 20)),
            "vol_floor": float(vt.get("floor", 0.4))}


def tune_take_profit(cfg, ds, account: str = "main", out_path=None, years: int = 10, recent_years: int = 3, min_gain: float = 0.002,
                     offsets=(0, 5, 10), off_loss: float = 0.01, off_dd: float = 0.03, off_weeks: int = 2) -> dict:
    """Test the take-profit candidates (the peak definition is tuned) against the rule in force at the account's structure and
    fees, phase-averaged, over the last ``years`` and the last ``recent_years``.  The owner's asymmetric rule (2026-09-29):

    * ON quickly: the best candidate goes in force after ONE weekly replay in which it beats the rule in force by ``min_gain``
      compounded a year over the long window, is no worse over the recent window, and its drawdown is at most a point deeper
      (the same bar switches from one take-profit rule to a better one);
    * OFF only on a very bad, repeated loss: plain holding must beat the rule in force by ``off_loss`` a year over the long
      window, or the rule's drawdown be ``off_dd`` deeper, on ``off_weeks`` consecutive weekly replays - one bad week never
      turns it off.

    Writes ``take_profit_<account>.json`` (``in_force`` = the rule the runner applies, None = hold)."""
    b = _account_book(cfg, account)
    c = b["cfg"]
    tp_cfg = dict(b["rk"].get("take_profit", {}) or {})
    models_dir = c.path("models_dir", "models")
    prev = _previous_report(out_path)
    current = prev.get("in_force") if isinstance(prev.get("in_force"), dict) else None
    if tp_cfg.get("enabled") is True:
        current = {k: v for k, v in tp_cfg.items() if k != "enabled"}
    px = closes(ds)
    inputs = dict(b["rk"].get("inputs") or DEFAULT_INPUTS)
    if b["rk"].get("adaptive", True):
        from ..execution.ranking import load_tuned_inputs

        inputs = load_tuned_inputs(models_dir, inputs)
    score = blended_scores(ds, inputs)
    elig = eligibility(c, px)
    proj = projection_frame(ds)
    sectors = None
    if b["max_per_sector"] > 0:
        try:
            from ..data.sectors import load_sectors

            sectors = load_sectors(c, list(px.columns), refresh=False) or None
        except Exception as e:  # noqa: BLE001
            log.debug("sectors unavailable for the take-profit test: %s", e)
    last = px.index[-1]
    windows = {"long": (last - pd.DateOffset(years=years), years), "recent": (last - pd.DateOffset(years=recent_years), recent_years)}

    def run(tp) -> dict:
        out = {}
        for wname, (start, yrs) in windows.items():
            idx = px.index[px.index >= start]
            rs = [simulate(px, score, idx[o], k=b["k"], every=b["every"], hysteresis=b["hysteresis"], fee_bps=b["fee"], reserve=b["reserve"],
                           sectors=sectors, max_per_sector=b["max_per_sector"], vol_target=b["vol_target"], vol_window=b["vol_window"],
                           vol_floor=b["vol_floor"], eligible=elig, take_profit=tp, projection=proj) for o in offsets if o < len(idx) - 60]
            tot = float(np.mean([r["total"] for r in rs]))
            out[wname] = {"total": tot, "geo": float((1.0 + tot) ** (1.0 / yrs) - 1.0) if tot > -1 else -1.0,
                          "max_drawdown": float(np.mean([r["max_drawdown"] for r in rs])), "sharpe": float(np.mean([r["sharpe"] for r in rs])),
                          "turnover": float(np.mean([r["turnover_per_year"] for r in rs])),
                          "triggers_per_year": float(np.mean([r.get("take_profit", {}).get("triggers", 0) for r in rs])) / yrs}
        return out

    cands = list(TP_CANDIDATES)
    if current not in cands:
        cands.append(current)
    results = [{"rule": tp, **run(tp)} for tp in cands]
    cur = next(r for r in results if r["rule"] == current)
    hold = next(r for r in results if r["rule"] is None)
    rules = [r for r in results if r["rule"] is not None]
    best = max(rules, key=lambda r: r["long"]["geo"]) if rules else hold
    today = str(date.today())
    same_day = str(prev.get("tuned_at")) == today                     # a rerun the same day never counts as another week
    better = (best is not cur and best["long"]["geo"] >= cur["long"]["geo"] + min_gain and best["recent"]["geo"] >= cur["recent"]["geo"]
              and best["long"]["max_drawdown"] >= cur["long"]["max_drawdown"] - 0.01)
    loss_streak = int(prev.get("loss_streak", 0) or 0)
    if current is None:                                               # holding: turn on after one winning replay
        in_force, loss_streak = (best["rule"] if better else None), 0
        reason = (f"take-profit turned on: {best['rule']} beat holding over both windows" if better else
                  "plain holding is as good as every take-profit rule - still holding")
    else:
        very_bad = (hold["long"]["geo"] >= cur["long"]["geo"] + off_loss or cur["long"]["max_drawdown"] <= hold["long"]["max_drawdown"] - off_dd)
        if very_bad:
            loss_streak = loss_streak if same_day else loss_streak + 1
        else:
            loss_streak = 0
        if very_bad and loss_streak >= off_weeks:
            in_force, reason = None, f"take-profit turned off: plain holding beat it badly {loss_streak} weeks running"
        elif better:
            in_force, reason = best["rule"], f"switched to a better take-profit rule: {best['rule']}"
        else:
            in_force = current
            reason = (f"take-profit stays on (a very bad week, {loss_streak} of {off_weeks} needed to turn it off)" if very_bad else
                      "take-profit stays on")
    rep = {"account": account, "tuned_at": today, "in_force": in_force, "current": current, "loss_streak": loss_streak,
           "accepted": in_force is not None and in_force != current, "best": best["rule"], "better": bool(better),
           "windows": {k: str(v[0].date()) for k, v in windows.items()},
           "structure": {"top_k": b["k"], "every_bars": b["every"], "hysteresis": b["hysteresis"], "fee_bps": b["fee"]},
           "results": results, "reason": reason}
    if out_path is not None:
        import json
        from pathlib import Path

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    log.info("take-profit test (%s): %s; in force: %s", account, rep["reason"], in_force or "plain holding")
    return rep


def format_report(rep: dict) -> str:
    lines = [f"rank inputs: {rep['inputs']}" + ("  (point-in-time: a name is choosable only from its first index membership)" if rep.get("point_in_time") else "")]
    for wname, res in rep["results"].items():
        lines.append(f"\n== {wname} (from {rep['windows'][wname]}) ==")
        lines.append(f"{'strategy':16s} {'total':>8s} {'sharpe':>7s} {'maxdd':>6s} {'turn/yr':>8s} {'fee bps':>8s}")
        for name, r in res.items():
            lines.append(f"{name:16s} {100 * r['total']:+7.1f}% {r['sharpe']:7.2f} {100 * r['max_drawdown']:5.0f}% "
                         f"{r.get('turnover_per_year', float('nan')):8.1f} {r.get('fee_bps', 0.0):8.1f}")
            ph = r.get("phase")
            if ph:
                lines.append(f"{'  phase-averaged':16s} {100 * ph['mean_total']:+7.1f}% {ph['mean_sharpe']:7.2f}        "
                             f"(worst {100 * ph['min_total']:+.1f}%, best {100 * ph['max_total']:+.1f}% over {ph['n']} start dates)")
    return "\n".join(lines)
