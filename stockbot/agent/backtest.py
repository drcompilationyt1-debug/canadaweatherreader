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

from ..execution.ranking import ANCHOR, CANDIDATE_INPUTS, DEFAULT_INPUTS, every_bars_of, select_top
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


def trend_state(px: pd.DataFrame, benchmark: str = "SPY", sma: int = 200, band: float = 0.02) -> pd.Series:
    """Faber-style trend switch: on while the benchmark closes above its ``sma``-day average (a ``band`` either side to
    avoid whipsaw), off below it.  Checked daily; it flips a couple of times a year."""
    c = px[benchmark].astype(float)
    m = c.rolling(int(sma)).mean()
    state = pd.Series(np.nan, index=px.index)
    on = True
    for d in px.index:
        cm, mm = c.get(d), m.get(d)
        if pd.notna(cm) and pd.notna(mm):
            if on and cm < mm * (1.0 - band):
                on = False
            elif not on and cm > mm * (1.0 + band):
                on = True
        state[d] = float(on)
    return state


def simulate(px: pd.DataFrame, score: pd.DataFrame | None, start, k: int = 20, every: int = 10, hysteresis: int = 3,
             fee_bps: float = 8.0, end=None, core: dict | None = None, trend: dict | None = None, reserve: float = 0.0) -> dict:
    """Daily portfolio returns of the rank-core rule (``score`` None = equal-weight everything), fees on turnover.
    ``core`` = {ticker, share}: a buy-and-hold slice bought once and never sold; ``trend`` = {benchmark, sma, band}: the
    rank slots go to cash while the benchmark is below its moving average; ``reserve`` = cash never invested."""
    idx = px.index[(px.index >= pd.Timestamp(start)) & ((px.index <= pd.Timestamp(end)) if end is not None else True)]
    rets = px.pct_change(fill_method=None).reindex(idx).fillna(0.0)
    wts = pd.DataFrame(0.0, index=idx, columns=px.columns)
    core_t, core_share = (str(core.get("ticker", "SPY")), float(core.get("share", 0.0))) if core else (None, 0.0)
    core_series = core.get("series") if core else None                    # the core's weight per day (the model's timing)
    if core_t is not None and core_t not in px.columns:
        core_t, core_share, core_series = None, 0.0, None
    on = trend_state(px, str(trend.get("benchmark", "SPY")), int(trend.get("sma", 200)), float(trend.get("band", 0.02))) if trend else None
    held: list[str] = []
    for i, d in enumerate(idx):
        share_d = float(core_series.get(d, core_share)) if core_series is not None else core_share
        satellite = max(0.0, 1.0 - share_d - float(reserve))
        if core_t is not None:
            wts.loc[d, core_t] = share_d
        if score is None:
            others = [t for t in px.columns if t != core_t]
            wts.loc[d, others] = satellite / len(others)
            continue
        risk_on = True if on is None else bool(on.get(d, 1.0) >= 0.5)
        if not risk_on:
            held = []                                                     # the trend filter is off: satellite in cash
        elif i % every == 0 or not held:
            s = score.loc[d].dropna() if d in score.index else pd.Series(dtype=float)
            if core_t is not None:
                s = s.drop(core_t, errors="ignore")
            held = select_top(s.to_dict(), held, k, hysteresis) if len(s) else held
        if held:
            wts.loc[d, held] = satellite / max(k, 1)                      # a slot is full or empty: no trims
    prev = wts.shift(1).fillna(0.0)
    turnover = (wts - prev).abs().sum(axis=1)
    daily = (prev * rets).sum(axis=1) - turnover * fee_bps / 1e4
    eq = (1.0 + daily).cumprod()
    return {"daily": daily, "total": float(eq.iloc[-1] - 1.0) if len(eq) else 0.0,
            "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 1 and daily.std() > 0 else 0.0,
            "max_drawdown": float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0,
            "turnover_per_year": float(turnover.sum() / max(len(idx), 1) * 252), "days": int(len(idx))}


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
        tf = dict(rk_.get("trend_filter", {}) or {})
        share = float(core.get("share", 0.0) or 0.0)
        core_spec = None
        if share > 0:
            core_spec = {"ticker": str(core.get("ticker", "SPY")), "share": share, "decide": str(core.get("decide", "model")),
                         "min_share": float(core.get("min_share", share * 0.5)), "max_share": float(core.get("max_share", share * 1.3)),
                         "every": every_bars_of(core.get("every_bars", 21)), "band": float(core.get("band", 0.05))}
        return {"budget": float(budget), "k": int(rk_.get("top_k", 20)), "every": every_bars_of(rk_.get("every_bars", 10)),
                "hysteresis": int(rk_.get("hysteresis", 3)), "reserve": float(c.get_path("execution.cash_reserve", 0.1) or 0.0),
                "core": core_spec,
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
    out: dict = {"inputs": inputs, "windows": {k: str(v.date()) for k, v in windows.items()}, "results": {}}
    for wname, start in windows.items():
        res = {}
        if benchmark in px.columns:
            b = px[benchmark].pct_change(fill_method=None).reindex(px.index[px.index >= start]).fillna(0.0)
            eq = (1 + b).cumprod()
            res[benchmark] = {"total": float(eq.iloc[-1] - 1), "sharpe": float(b.mean() / b.std() * np.sqrt(252)) if b.std() > 0 else 0.0,
                              "max_drawdown": float((eq / eq.cummax() - 1).min())}
        ew = simulate(px, None, start)
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
            r = simulate(px, score, start, k=bset["k"], every=bset["every"], hysteresis=bset.get("hysteresis", hyst), fee_bps=fee,
                         core=bset.get("core"), trend=bset.get("trend"), reserve=bset.get("reserve", 0.0))
            res[f"rank_{bname}"] = {**{k: v for k, v in r.items() if k != "daily"}, "k": bset["k"], "every_bars": bset["every"], "fee_bps": fee,
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
    res = {}
    for name, inputs in (("static", static), ("tuned", tuned)):
        try:
            r = simulate(px, blended_scores(ds, inputs), start, k=k, every=every, hysteresis=hyst, fee_bps=fee)
            res[name] = {kk: vv for kk, vv in r.items() if kk != "daily"}
        except ValueError:
            res[name] = {"total": float("-inf"), "sharpe": float("-inf")}
    same = set(tuned) == set(static) and all(abs(tuned[kk] - static[kk]) < 1e-6 for kk in tuned)
    accepted = (not same) and res["tuned"]["sharpe"] >= res["static"]["sharpe"] - 0.05 and res["tuned"]["total"] >= res["static"]["total"] - 0.01
    rep = {"tuned_at": str(date.today()), "inputs": tuned if accepted else static, "tuned": tuned, "static": static, "ic": ics,
           "backtest_last_year": res, "accepted": bool(accepted), "reason": "same as static" if same else
           ("tuned blend is at least as good over the last year" if accepted else "tuned blend did worse over the last year - static kept")}
    if out_path is not None:
        import json
        from pathlib import Path

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    log.info("rank blend tuning: %s (%s)", "accepted" if accepted else "kept static", rep["reason"])
    return rep


def format_report(rep: dict) -> str:
    lines = [f"rank inputs: {rep['inputs']}"]
    for wname, res in rep["results"].items():
        lines.append(f"\n== {wname} (from {rep['windows'][wname]}) ==")
        lines.append(f"{'strategy':16s} {'total':>8s} {'sharpe':>7s} {'maxdd':>6s} {'turn/yr':>8s} {'fee bps':>8s}")
        for name, r in res.items():
            lines.append(f"{name:16s} {100 * r['total']:+7.1f}% {r['sharpe']:7.2f} {100 * r['max_drawdown']:5.0f}% "
                         f"{r.get('turnover_per_year', float('nan')):8.1f} {r.get('fee_bps', 0.0):8.1f}")
    return "\n".join(lines)
