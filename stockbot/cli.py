"""Command line interface: ``python -m stockbot <command>`` (or ``stockbot <command>`` after pip install -e .)."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from .config import env_settings, load_config
from .logging_utils import get_logger
from .paths import ROOT, THIRD_PARTY

log = get_logger("stockbot")


# ---------------------------------------------------------------------- commands
def cmd_doctor(cfg, args) -> int:
    from .agent.policy import PolicyBundle
    from .signals.registry import build_context, build_providers

    print(f"StockBot doctor  (python {platform.python_version()}, {platform.system()})")
    print(f"project root: {ROOT}")
    print("\n[submodules]")
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from setup_submodules import REPOS
    except Exception:  # noqa: BLE001
        REPOS = {}
    for name, (url, group, note) in REPOS.items():
        d = THIRD_PARTY / name
        present = d.is_dir() and any(d.iterdir())
        print(f"  {'OK ' if present else '-- '} {name:32s} {group:9s} {note}")
    print("\n[signal providers]  (tier A native, B optional, C opt-in LLM agents)")
    ctx = build_context(cfg)
    ctx.extra["frames"] = {"_a": None, "_b": None}  # lets cross-sectional providers report availability
    for p in build_providers(cfg, ctx, include_disabled=True):
        ok, why = p.availability()
        flag = "ON " if (p.enabled and ok) else ("off" if not p.enabled else "-- ")
        print(f"  {flag} [{p.tier}] {p.name:16s} {p.size:3d} feats  {why}")
    print("\n[llm backends]  order: " + ", ".join(cfg.get_path("llm.order", [])))
    if ctx.llm:
        for row in ctx.llm.status():
            state = "ready" if row["configured"] else "not configured"
            if row.get("disabled"):
                state = f"disabled: {row['disabled']}"
            elif row.get("cooldown_s"):
                state = f"cooling down {row['cooldown_s']}s"
            print(f"  {row['backend']:10s} model={row['model']!s:40s} {state}  ({row['reason']})")
    print("\n[execution]")
    ex = cfg.section("execution")
    print(f"  mode={ex.get('mode')}  max_position={ex.get('max_position')}  max_gross={ex.get('max_gross_exposure')}  short={cfg.get_path('env.allow_short')}")
    try:
        from .execution.alpaca import alpaca_keys

        k, s = alpaca_keys()
        print(f"  alpaca keys: {'set' if k and s else 'not set'} (paper={ex.get_path('alpaca.paper', True)})")
    except Exception:  # noqa: BLE001
        pass
    ckpt = cfg.path("train.checkpoint_dir")
    print("\n[policy]")
    if PolicyBundle.exists(ckpt):
        meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8")) if (ckpt / "meta.json").exists() else {}
        print(f"  {ckpt}: trained {meta.get('trained_at', '?')}, algo={meta.get('algo')}, timesteps={meta.get('timesteps')}, layout={meta.get('signature')}")
        print(f"  best.zip: {'yes' if (ckpt / 'best.zip').exists() else 'no'}")
    else:
        print(f"  none in {ckpt} - run: stockbot train")
    ds = cfg.path("models_dir", "models") / "dataset"
    print(f"\n[dataset] {'cached at ' + str(ds) if (ds / 'layout.json').exists() else 'not built (stockbot build-dataset)'}")
    news_dir = cfg.path("news.local_csv_dir", "data/news")
    csvs = list(news_dir.glob("*.csv")) if news_dir.is_dir() else []
    print(f"[news history] {len(csvs)} csv files in {news_dir} ({'stockbot news-history' if not csvs else 'ok'})")
    return 0


def cmd_fetch_data(cfg, args) -> int:
    from .agent.train import load_frames

    frames = load_frames(cfg, refresh=args.refresh, tickers=args.tickers)
    for t, df in frames.items():
        print(f"  {t:8s} {len(df):6d} bars  {df.index[0].date()} -> {df.index[-1].date()}  last close {df['close'].iloc[-1]:.2f}")
    return 0


def cmd_build_dataset(cfg, args) -> int:
    from .agent.train import prepare_dataset

    ds, providers, _ = prepare_dataset(cfg, offline=args.offline, synthetic=args.synthetic, fit=not args.no_fit)
    print(f"dataset: {len(ds)} tickers, obs_dim={ds.layout.obs_dim}, signature={ds.layout.signature()}")
    for b in ds.layout.blocks:
        on = sum(float(td.signals[-1, b.offset] > 0.5) for td in ds.data.values())
        print(f"  {b.name:16s} {b.size:3d} feats  available (last bar) for {int(on)}/{len(ds)} tickers")
    return 0


def cmd_train(cfg, args) -> int:
    from .agent.train import train

    path = train(cfg, total_timesteps=args.timesteps, resume=args.resume, offline=args.offline,
                 synthetic=args.synthetic, n_envs=args.n_envs, refresh=args.refresh, seeds=args.seeds, max_minutes=args.max_minutes)
    print(f"policy saved: {path}")
    return 0


def _load_dataset(cfg, args, fit: bool = False):
    from .agent.train import prepare_dataset
    from .env.dataset import MarketDataset

    folder = cfg.path("models_dir", "models") / "dataset"
    if getattr(args, "rebuild", False) or not (folder / "layout.json").exists():
        ds, _, _ = prepare_dataset(cfg, offline=getattr(args, "offline", False), synthetic=getattr(args, "synthetic", False), fit=fit)
        return ds
    return MarketDataset.load(folder)


def cmd_evaluate(cfg, args) -> int:
    from .agent.evaluate import aggregate, evaluate, format_summary
    from .agent.policy import PolicyBundle

    ckpt = cfg.path("train.checkpoint_dir")
    bundle = PolicyBundle.load(ckpt, args.which)
    ds = _load_dataset(cfg, args)
    if bundle.layout.signature() != ds.layout.signature():
        print("WARNING: dataset layout differs from the model's layout - rebuild the dataset or retrain")
    train_end = cfg.get_path("data.train_end")
    _, test = ds.split(train_end) if train_end and not args.all_data else (ds, ds)
    if len(test) == 0:
        test = ds
    summary, curves = evaluate(bundle.model, test, env_settings(cfg), args.tickers, max_bars=args.max_bars,
                               cash_levels=cfg.get_path("train.eval_cash"))
    print(format_summary(summary))
    print("\naggregate:", json.dumps(aggregate(summary), indent=1))
    if args.plot:
        _plot(curves, args.plot)
    return 0


def _plot(curves: dict, out: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot")
        return
    n = len(curves)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), squeeze=False)
    for ax, (t, res) in zip(axes[:, 0], curves.items()):
        ax.plot(res["equity"] / res["equity"][0], label="policy")
        ax.plot(res["bench"] / res["bench"][0], label="buy & hold", alpha=0.7)
        ax.set_title(f"{t}  {res['metrics'].get('start')} -> {res['metrics'].get('end')}")
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"plot saved to {out}")


def cmd_backtest(cfg, args) -> int:
    from .agent.evaluate import run_window
    from .agent.policy import PolicyBundle
    from .backtest.backtrader_runner import run_backtrader

    bundle = PolicyBundle.load(cfg.path("train.checkpoint_dir"), args.which)
    ds = _load_dataset(cfg, args)
    if args.ticker not in ds.data:
        print(f"{args.ticker} not in dataset {ds.tickers}")
        return 1
    td = ds.data[args.ticker]
    import numpy as np
    import pandas as pd

    start = td.min_start
    end = len(td)
    if args.start:
        start = max(start, int(np.searchsorted(td.dates, np.datetime64(pd.Timestamp(args.start), "ns"))))
    if args.end:
        end = min(end, int(np.searchsorted(td.dates, np.datetime64(pd.Timestamp(args.end), "ns"), side="right")))
    env_cfg = env_settings(cfg)
    sim = run_window(bundle.model, ds, args.ticker, env_cfg, start=start, length=end - start - 2)["metrics"]
    print("simulator :", json.dumps({k: sim[k] for k in ("start", "end", "total_return", "bh_return", "sharpe", "max_drawdown", "avg_exposure")}, indent=1))
    try:
        bt = run_backtrader(bundle, td, env_cfg, start, end)
        print("backtrader:", json.dumps(bt, indent=1))
    except ImportError as e:
        print("backtrader unavailable:", e)
    return 0


def cmd_news(cfg, args) -> int:
    from .signals.news_llm import LLMNewsSignal, result_to_vector
    from .signals.registry import build_context
    from .signals.sentiment import _analyzer, score_texts

    ctx = build_context(cfg)
    items = ctx.news.fetch(args.ticker, args.days, use_cache=not args.no_cache)
    print(f"{len(items)} headlines for {args.ticker} (last {args.days} days)")
    for it in items[:15]:
        print(f"  {it.published[:10]} [{it.source or '-'}] {it.title}")
    sia = _analyzer()
    if sia:
        v = score_texts([i.text() for i in items], sia)
        print(f"\nVADER: mean={v[0]:+.2f} pos-neg={v[1]:+.2f} extreme={v[3]:+.2f}")
    if args.no_llm:
        return 0
    sig = LLMNewsSignal(cfg, ctx)
    ok, why = sig.availability()
    print(f"\nLLM news reader: {why}")
    if not ok:
        return 0
    from datetime import datetime, timezone

    res = sig.score_items(args.ticker, items, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    if res is None:
        print("no backend could answer (quota / auth) - the block will be masked out")
        return 0
    print(json.dumps(res, indent=1))
    print("feature vector:", result_to_vector(res, len(items)).round(3).tolist())
    return 0


def cmd_news_history(cfg, args) -> int:
    """Download a headline history (Google News RSS or GDELT, both free and keyless) into data/news/."""
    from datetime import datetime, timedelta

    import pandas as pd

    from .news.gdelt import company_name, fetch_gdelt_history
    from .news.google_history import fetch_google_history

    out_dir = cfg.path("news.local_csv_dir", "data/news")
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = args.tickers or list(cfg.get("universe", []))
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.utcnow()
    names = cfg.get_path("news.company_names", {}) or {}
    sleep = args.sleep if args.sleep is not None else (6.0 if args.source == "gdelt" else 2.0)
    for t in tickers:
        f = out_dir / f"{args.source}_{t}.csv"
        start = datetime.strptime(args.start, "%Y-%m-%d")
        existing = None
        if f.exists():
            existing = pd.read_csv(f)
            if len(existing):
                start = max(start, datetime.strptime(str(existing["date"].max()), "%Y-%m-%d") + timedelta(days=1))
        if start >= end:
            print(f"  {t}: up to date ({len(existing) if existing is not None else 0} headlines)")
            continue
        name = company_name(t, names)
        print(f"  {t} ({name}) via {args.source}: {start.date()} -> {end.date()} ...", flush=True)
        if args.source == "gdelt":
            df = fetch_gdelt_history(t, name, start, end, sleep=sleep, months=args.months)
        else:
            df = fetch_google_history(t, name, start, end, sleep=sleep, window_days=args.window_days, min_days=args.min_days)
        if existing is not None and len(existing):
            df = pd.concat([existing, df]).drop_duplicates(subset=["title"]).sort_values("date")
        df.to_csv(f, index=False)
        print(f"  {t}: {len(df)} headlines in {f}", flush=True)
    return 0


def cmd_patterns(cfg, args) -> int:
    from .features.candles import PatternStats

    import pandas as pd

    names = ["pattern_stats.json"] + (["talib_pattern_stats.json"] if args.talib else [])
    for n in names:
        p = cfg.path("models_dir", "models") / "signals" / n
        if not p.exists():
            print(f"no fitted statistics at {p} - run: stockbot build-dataset")
            continue
        stats = PatternStats.load(p)
        tbl = stats.table().round(2)
        if args.talib:
            tbl = tbl[tbl["n"] >= args.min_n].sort_values("edge_20d_%", ascending=False)
        with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 500):
            print(f"\n== {n} ({stats.n_bars} bars) ==")
            print(tbl.to_string(index=False))
        print("baseline mean return: " + ", ".join(f"{h}d={100*v:+.2f}%" for h, v in stats.base_mean.items()))
    return 0


def cmd_candles(cfg, args) -> int:
    """Show (and optionally plot) the candlestick patterns on a ticker at any timeframe."""
    import numpy as np
    import pandas as pd

    from .data.loader import fetch_ohlcv, resample_ohlcv
    from .features.candles import PATTERN_DIRECTION, detect_patterns

    d = cfg.section("data")
    df = fetch_ohlcv(args.ticker, d.get("start", "2008-01-01"), None, d.get("interval", "1d"),
                     cfg.path("data.cache_dir", "data/cache"), offline=args.offline)
    if args.timeframe.upper() != "D":
        df = resample_ohlcv(df, args.timeframe)
    pat = detect_patterns(df)
    talib_pat = None
    try:
        from .signals.talib_candles import talib_patterns

        talib_pat = talib_patterns(df)
    except Exception:  # noqa: BLE001
        pass
    tail = df.tail(args.bars)
    print(f"{args.ticker} {args.timeframe.upper()} candles, last {len(tail)} bars ({tail.index[0].date()} -> {tail.index[-1].date()})")
    for ts, row in tail.iterrows():
        mine = [p for p in pat.columns if pat.at[ts, p]]
        tal = []
        if talib_pat is not None:
            tal = [f"{c[3:].lower()}{'+' if talib_pat.at[ts, c] > 0 else '-'}" for c in talib_pat.columns if talib_pat.at[ts, c] != 0]
        if mine or tal or args.all:
            print(f"  {ts.date()}  O {row['open']:.2f} H {row['high']:.2f} L {row['low']:.2f} C {row['close']:.2f}  "
                  f"{', '.join(mine) or '-'}  |  talib: {', '.join(tal) or '-'}")
    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import mplfinance as mpf
        except ImportError:
            print("pip install mplfinance to draw candlestick charts")
            return 0
        Path(args.plot).resolve().parent.mkdir(parents=True, exist_ok=True)
        plot_df = tail.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        bull = pd.Series(np.nan, index=tail.index)
        bear = pd.Series(np.nan, index=tail.index)
        for ts in tail.index:
            dirs = [PATTERN_DIRECTION[p] for p in pat.columns if pat.at[ts, p]]
            if any(x > 0 for x in dirs):
                bull[ts] = tail.at[ts, "low"] * 0.99
            if any(x < 0 for x in dirs):
                bear[ts] = tail.at[ts, "high"] * 1.01
        adds = []
        if bull.notna().any():
            adds.append(mpf.make_addplot(bull, type="scatter", markersize=70, marker="^", color="green"))
        if bear.notna().any():
            adds.append(mpf.make_addplot(bear, type="scatter", markersize=70, marker="v", color="red"))
        mpf.plot(plot_df, type="candle", style="yahoo", volume=True, addplot=adds or None,
                 title=f"{args.ticker} ({args.timeframe.upper()}) - green ^ bullish pattern, red v bearish", savefig=args.plot)
        print(f"chart saved to {args.plot}")
    return 0


def _real_money_guard(cfg, mode: str, flag: bool) -> bool:
    real = (mode in ("alpaca", "live") and not cfg.get_path("execution.alpaca.paper", True)) or \
           (mode == "moomoo" and str(cfg.get_path("execution.moomoo.env", "simulate")).lower() == "real")
    if real and not flag:
        print("refusing to trade a REAL money account without --i-understand-real-money")
        return False
    return True


def cmd_trade(cfg, args) -> int:
    from .execution.runner import TradingRunner

    mode = args.mode or cfg.get_path("execution.mode", "paper")
    if not _real_money_guard(cfg, mode, args.i_understand_real_money):
        return 2
    runner = TradingRunner(cfg, mode=mode, offline=args.offline, with_llm=not args.no_llm, allow_closed=args.allow_closed)
    if args.loop:
        runner.loop(args.loop, dry_run=args.dry_run)
        return 0
    decisions = runner.cycle(dry_run=args.dry_run, refresh=not args.offline)
    note = runner.last_cycle_note
    print(f"\n{len(decisions)} decisions ({'DRY RUN - nothing executed' if args.dry_run else (note or mode)})")
    print(json.dumps(runner.broker.summary(), indent=1, default=str))
    if not args.dry_run and not note and not args.no_report:
        from .report import build_dashboard

        print(f"dashboard: {build_dashboard(cfg, mode=mode)}")
    return 0


def cmd_session(cfg, args) -> int:
    """Market-hours session: trade at the open, watch for N hours, train in the background."""
    from .execution.market_hours import MarketClock
    from .execution.session import TradingSession, session_slot

    mode = args.mode or cfg.get_path("execution.mode", "paper")
    if not _real_money_guard(cfg, mode, args.i_understand_real_money):
        return 2
    clock = MarketClock()
    if args.gate_minutes is not None:
        ok, why = session_slot(clock, args.gate_minutes, args.gate_after_minutes)
        print(f"session slot: {why}")
        if not ok:
            print("not a session slot - exiting")
            return 0
    session = TradingSession(cfg, mode=mode, hours=args.hours, dry_run=args.dry_run, offline=args.offline, with_llm=not args.no_llm,
                             train=False if args.no_train else None, clock=clock, train_minutes=args.train_minutes,
                             train_timesteps=args.train_timesteps, train_seeds=args.train_seeds, train_n_envs=args.train_n_envs,
                             snapshot_minutes=args.snapshot_minutes, max_wait_minutes=args.max_wait_minutes,
                             deadline_minutes=args.deadline_minutes, deadline_at=args.deadline_at or None,
                             accounts=False if args.no_accounts else None)
    summary = session.run(force=args.force)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("open_prices",)}, indent=1, default=str))
    return 0


def cmd_review(cfg, args) -> int:
    """Post-trade review: replay a period and score what we did against the alternatives."""
    from datetime import date

    from .feedback.review import Review

    rev = Review(cfg)
    end = date.fromisoformat(args.end) if args.end else None
    results = {}
    if args.learn:
        from .execution.market_hours import MarketClock
        from .feedback.hindsight import learn, learn_due

        budget = args.max_minutes
        if args.before_open_minutes is not None:
            st = MarketClock().status()
            if not st.is_open:
                avail = st.minutes_to_open - float(args.before_open_minutes)
                budget = avail if budget is None else min(budget, avail)
                print(f"market opens in {st.minutes_to_open:.0f} min: {max(0.0, avail):.0f} min available for learning")
        if budget is not None and budget < 2.0:
            print("no time to learn before the open - skipped")
            return 0
        if args.due:
            reps = learn_due(cfg, today=end, force=args.force, refresh=not args.offline, budget_minutes=budget)
            if not reps:
                print("nothing new to learn: every finished day / week / month / year was already learned")
        else:
            reps = {p: learn(cfg, period=p, end=end, force=args.force, refresh=not args.offline, budget_minutes=budget)
                    for p in (args.period or ["day"])}
        for period, rep in reps.items():
            print(f"\n== learn from the {period} ({rep.get('start')} to {rep.get('end')}) ==")
            if rep.get("error"):
                print("  failed:", rep["error"])
                continue
            if rep.get("skipped"):
                print("  skipped:", rep["skipped"])
                continue
            print(f"  {rep['samples']} decisions; the best path held {rep['mean_level_best']:.2f} of a slice on average, we held "
                  f"{rep['mean_level_ours']:.2f}")
            for name, m in rep["members"].items():
                if m.get("error"):
                    print(f"  {name}: failed - {m['error']}")
                    continue
                if m.get("skipped"):
                    print(f"  {name}: {m['skipped']}")
                    continue
                sb, sa = m.get("score_before"), m.get("score_after")
                print(f"  {name}: loss {m['loss_before']:.4f} -> {m['loss_after']:.4f} in {m['epochs']} epochs, score "
                      f"{sb if sb is None else round(sb, 3)} -> {sa if sa is None else round(sa, 3)}: {'kept' if m.get('accepted') else 'rejected'}")
            print(f"  accepted {rep['accepted']} of {len(rep['members'])} members in {rep.get('seconds', 0):.0f}s "
                  f"(yardstick {rep['yardstick']['tickers']} tickers x {rep['yardstick']['bars']} bars, {rep['yardstick']['epochs']} epochs)")
        return 0
    if args.due:
        results = rev.run_due(end)
        if not results:
            print("no review due")
    else:
        for period in (args.period or ["day"]):
            results[period] = rev.review_day(end) if period == "day" else rev.review_period(period, end)
    for period, res in results.items():
        print(f"\n== {period} review ==")
        if not res:
            print("  nothing to review (no session / decisions in that window)")
            continue
        for line in res["lessons"]:
            print("  " + line)
        print("  alternatives: " + ", ".join(f"{r['name']} {100 * r['return']:+.2f}%" for r in res["ranking"][:8]))
        print(f"  saved: {res['file']}")
    return 0


def cmd_market_status(cfg, args) -> int:
    from .execution.market_hours import MarketClock

    st = MarketClock(source=args.source).status()
    print(json.dumps(st.to_dict(), indent=1))
    return 0


def cmd_forecast(cfg, args) -> int:
    """Today's up/down votes of every model per ticker, next to each model's running hit rate."""
    import pandas as pd

    from .feedback.direction import DirectionBoard, consensus

    board = DirectionBoard(cfg.path("feedback.direction_file", "data/experience/direction.jsonl"))
    if not args.no_refresh:
        from .execution.runner import TradingRunner

        mode = args.mode or cfg.get_path("execution.mode", "paper")
        runner = TradingRunner(cfg, mode=mode, offline=args.offline, with_llm=not args.no_llm, allow_closed=True)
        runner.cycle(dry_run=True, refresh=not args.offline)
        rows = {t: {k: v["vote"] for k, v in votes.items()} | {"consensus": round(consensus(votes), 2)}
                for t, votes in runner.last_votes.items()}
        latest = pd.DataFrame(rows).T
        latest.index.name = "fresh (not recorded)"
    else:
        latest = board.latest()
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200):
        if latest is None or len(latest) == 0:
            print("no votes yet - run a session or `stockbot trade` first")
        else:
            up = latest.get("consensus", pd.Series(dtype=float))
            print(f"== votes ({latest.index.name}) : +1 up, -1 down, 0 no opinion ==")
            print(latest.fillna(0).astype(float).round(2).to_string())
            if len(up):
                bulls = ", ".join(f"{t} {v:+.2f}" for t, v in up.sort_values(ascending=False).head(5).items())
                bears = ", ".join(f"{t} {v:+.2f}" for t, v in up.sort_values().head(5).items())
                print(f"\nmost bullish consensus: {bulls}\nmost bearish consensus: {bears}")
        for horizon in ("session", "daily"):
            sc = board.scorecard(horizon)
            print(f"\n== scorecard, {horizon} horizon ({'open -> end of session' if horizon == 'session' else 'decision -> next close'}) ==")
            if sc is None or len(sc) == 0:
                print("  nothing settled yet")
            else:
                print(sc.round({"hit_rate": 3, "edge_bps": 1, "up_share": 2}).to_string(index=False))
    return 0


def cmd_autopilot(cfg, args) -> int:
    from .execution.autopilot import Autopilot

    mode = args.mode or cfg.get_path("execution.mode", "paper")
    if not _real_money_guard(cfg, mode, args.i_understand_real_money):
        return 2
    Autopilot(cfg, mode=mode, dry_run=args.dry_run, offline=args.offline).run(run_now=args.run_now, once=args.once)
    return 0


def cmd_report(cfg, args) -> int:
    from .report import build_dashboard

    mode = args.mode or cfg.get_path("execution.mode", "paper")
    out = build_dashboard(cfg, mode=mode, out=args.out, open_browser=args.open)
    print(f"dashboard written to {out}")
    return 0


def cmd_experience(cfg, args) -> int:
    from .feedback.experience import ExperienceStore

    print(json.dumps(ExperienceStore(cfg.path("feedback.experience_file")).summary(), indent=1, default=str))
    return 0


def cmd_account(cfg, args) -> int:
    """The broker's own record (Alpaca): equity per day, positions, every fill."""
    from datetime import date, timedelta

    from .execution.alpaca_history import AlpacaHistory

    h = AlpacaHistory(cfg)
    if not h.has_keys:
        print(f"no Alpaca keys ({h.keys_env}_API_KEY / {h.keys_env}_SECRET_KEY)")
        return 1
    print(f"account '{cfg.get('account') or 'main'}' ({h.keys_env} keys, {'paper' if h.paper else 'LIVE'})")
    if args.configure_like_moomoo:
        from .execution.alpaca import AlpacaBroker

        b = AlpacaBroker(paper=h.paper, fractional=bool(cfg.get_path("execution.alpaca.fractional", True)), keys_env=h.keys_env)
        print("Alpaca account configured like a moomoo cash account:", json.dumps(b.configure_like_moomoo(), indent=1, default=str))
    from .execution.alpaca import VirtualLedger

    ledger = VirtualLedger(cfg.path("execution.state_file", "data/paper/state.json").with_name("alpaca_ledger.json"))
    print(f"moomoo fees booked virtually so far: {ledger.fees:,.2f} over {ledger.n_fills} fills (deducted from the equity the strategy sees)")
    daily = h.daily(args.period)
    if len(daily):
        d = daily.tail(args.days)
        ups, downs = int((d["direction"] == "up").sum()), int((d["direction"] == "down").sum())
        print(f"== account per day ({args.period}): {ups} up days, {downs} down days, P&L {d['profit_loss'].sum():+,.2f} ==")
        print(d.to_string(index=False, formatters={"equity": "{:,.2f}".format, "profit_loss": "{:+,.2f}".format, "profit_loss_pct": "{:+.2%}".format}))
    try:
        pos = h.positions()
        if len(pos):
            print("\n== positions ==")
            print(pos.to_string(index=False, formatters={"unrealized_plpc": "{:+.2%}".format, "unrealized_pl": "{:+,.2f}".format, "today_pl": "{:+,.2f}".format}))
    except Exception as e:  # noqa: BLE001
        print("positions unavailable:", e)
    fills = h.sync_fills() if args.sync else h.fills(after=date.today() - timedelta(days=45))
    if len(fills):
        print(f"\n== last {min(args.fills, len(fills))} of {len(fills)} fills ==")
        print(fills[["ts", "ticker", "side", "qty", "price", "notional"]].tail(args.fills).to_string(index=False))
    else:
        print("\nno fills in the window")
    if args.sync:
        h.portfolio_history("3M", "1D")
        print(f"\ncached under {h.cache_dir} (saved with the state)")
    return 0


def cmd_intraday_fit(cfg, args) -> int:
    """Fit the intraday exit model on the broker's 15-minute bars."""
    from .feedback.intraday import fit_from_alpaca

    rep = fit_from_alpaca(cfg, days=args.days, timeframe=args.timeframe, holdout_days=args.holdout_days)
    print(json.dumps(rep, indent=1, default=str))
    return 0


def cmd_retrain(cfg, args) -> int:
    from .agent.train import retrain
    from .feedback.experience import ExperienceStore

    print("experience so far:", json.dumps(ExperienceStore(cfg.path("feedback.experience_file")).summary(), default=str))
    path = retrain(cfg, total_timesteps=args.timesteps, n_envs=args.n_envs, offline=args.offline,
                   synthetic=args.synthetic, from_scratch=args.from_scratch, seeds=args.seeds, max_minutes=args.max_minutes,
                   reuse_dataset_days=args.reuse_dataset_days or 0.0)
    print(f"policy updated: {path}")
    return 0


def cmd_ensemble(cfg, args) -> int:
    """Register members trained separately (e.g. in parallel with --set train.checkpoint_dir=models/policy/ensemble/seedN)."""
    from .agent.train import register_ensemble

    path = register_ensemble(cfg, args.min_score)
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=1))
    return 0


def cmd_submodules(cfg, args) -> int:
    script = ROOT / "scripts" / "setup_submodules.py"
    return subprocess.call([sys.executable, str(script)] + args.rest)


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    # --config / --set are accepted both before and after the sub-command
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", "-c", default=argparse.SUPPRESS, help="user config YAML merged over config/default.yaml")
    common.add_argument("--set", "-s", action="append", default=argparse.SUPPRESS, metavar="KEY=VAL",
                        help="override e.g. train.n_envs=4 (repeatable)")
    common.add_argument("--account", default=argparse.SUPPRESS, metavar="NAME",
                        help="act on one of the extra accounts (`accounts:` in the config) instead of the main one")
    ap = argparse.ArgumentParser(prog="stockbot", description="Self-improving long-term stock trading agent", parents=[common])
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add = sub.add_parser
    sub.add_parser = lambda *a, **kw: _add(*a, parents=[common], **kw)  # type: ignore[method-assign]

    sub.add_parser("doctor", help="show what is installed / configured / available").set_defaults(fn=cmd_doctor)

    p = sub.add_parser("fetch-data", help="download / refresh OHLCV cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--tickers", nargs="*")
    p.set_defaults(fn=cmd_fetch_data)

    p = sub.add_parser("build-dataset", help="compute every signal and cache the dataset")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="use synthetic prices (no network)")
    p.add_argument("--no-fit", action="store_true", help="reuse fitted provider state")
    p.set_defaults(fn=cmd_build_dataset)

    p = sub.add_parser("train", help="train the policy on parallel simulators")
    p.add_argument("--timesteps", type=int)
    p.add_argument("--resume", help="checkpoint dir or .zip to continue from")
    p.add_argument("--n-envs", type=int)
    p.add_argument("--seeds", type=int, help="train N seeds and trade their averaged action (default train.seeds)")
    p.add_argument("--max-minutes", type=float, help="stop the PPO updates after this much wall-clock time")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("evaluate", help="out-of-sample evaluation of the trained policy")
    p.add_argument("--which", default="best", choices=["best", "latest"])
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--max-bars", type=int)
    p.add_argument("--all-data", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--plot", help="save equity curves to this PNG")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("backtest", help="cross-check one ticker in the simulator and in backtrader")
    p.add_argument("--ticker", required=True)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--which", default="best", choices=["best", "latest"])
    p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser("news", help="fetch headlines and score them (VADER + LLM)")
    p.add_argument("--ticker", required=True)
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(fn=cmd_news)

    p = sub.add_parser("news-history", help="download a free headline history (Google News RSS or GDELT) for training the news signals")
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--source", choices=["google", "gdelt"], default="google")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end")
    p.add_argument("--sleep", type=float, help="seconds between requests (default 2 for google, 6 for gdelt)")
    p.add_argument("--window-days", type=int, default=30, help="google: days per query, split automatically when full")
    p.add_argument("--min-days", type=int, default=5, help="google: stop splitting full windows below this many days")
    p.add_argument("--months", type=int, default=3, help="gdelt: months per request (250 headlines max each)")
    p.set_defaults(fn=cmd_news_history)

    p = sub.add_parser("patterns", help="print candlestick pattern statistics")
    p.add_argument("--talib", action="store_true", help="also print the TA-Lib recogniser statistics")
    p.add_argument("--min-n", type=int, default=30)
    p.set_defaults(fn=cmd_patterns)

    p = sub.add_parser("candles", help="list / draw candlestick patterns for a ticker at any timeframe")
    p.add_argument("--ticker", required=True)
    p.add_argument("--bars", type=int, default=40)
    p.add_argument("--timeframe", default="D", help="D | W | M | any pandas offset alias")
    p.add_argument("--plot", help="save a candlestick chart PNG (needs mplfinance)")
    p.add_argument("--all", action="store_true", help="print every bar, not only bars with patterns")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(fn=cmd_candles)

    p = sub.add_parser("trade", help="run one paper / Alpaca trading cycle (real brokers: only while the market is open)")
    p.add_argument("--mode", choices=["paper", "alpaca", "moomoo", "live"])
    p.add_argument("--dry-run", action="store_true", help="decide but do not execute or log")
    p.add_argument("--loop", type=float, metavar="MINUTES", help="repeat every N minutes")
    p.add_argument("--allow-closed", action="store_true", help="send orders even while the exchange is closed (queued for the open)")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--i-understand-real-money", action="store_true")
    p.set_defaults(fn=cmd_trade)

    p = sub.add_parser("session", help="market-hours session: trade at the open, watch for N hours, keep training in the background")
    p.add_argument("--mode", choices=["paper", "alpaca", "moomoo", "live"])
    p.add_argument("--hours", type=float, help="length of the session after the open (default session.hours)")
    p.add_argument("--dry-run", action="store_true", help="decide and watch, send nothing, record nothing")
    p.add_argument("--no-train", action="store_true", help="do not train in the background")
    p.add_argument("--train-minutes", type=float, help="trainer time budget (default: the rest of the session)")
    p.add_argument("--train-timesteps", type=int)
    p.add_argument("--train-seeds", type=int)
    p.add_argument("--train-n-envs", type=int)
    p.add_argument("--snapshot-minutes", type=float)
    p.add_argument("--max-wait-minutes", type=float, help="how long to wait for the open when started early")
    p.add_argument("--gate-minutes", type=float, metavar="MIN",
                   help="only run when the market opens within MIN minutes (cron slot filter; see --gate-after-minutes)")
    p.add_argument("--gate-after-minutes", type=float, default=30.0, metavar="MIN",
                   help="with --gate-minutes: also run when the market opened less than MIN minutes ago (late cron)")
    p.add_argument("--force", action="store_true", help="run even if a session already ran today")
    p.add_argument("--no-accounts", action="store_true", help="trade the main account only (skip `accounts:`)")
    p.add_argument("--deadline-minutes", type=float, help="finish everything (watch, review, trainer) within this many minutes of starting")
    p.add_argument("--deadline-at", help="... or by this ISO 8601 time (the workflow anchors it on the job's start); the earlier one wins")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--i-understand-real-money", action="store_true")
    p.set_defaults(fn=cmd_session)

    p = sub.add_parser("review", help="post-trade review: replay a day / week / month / year and score it against the alternatives")
    p.add_argument("--period", nargs="*", choices=["day", "week", "month", "year"], help="default: day (unless only --learn is asked)")
    p.add_argument("--end", help="period end date (YYYY-MM-DD); default today / latest session")
    p.add_argument("--due", action="store_true", help="run whichever week / month / year reviews are due")
    p.add_argument("--learn", action="store_true",
                   help="learn from the paper trades instead of reporting: fine-tune the policy on the hindsight labels of one finished unit "
                        "(--period day = yesterday, week = last week, ...; --due = every finished unit not learned yet), guarded by the OOS score")
    p.add_argument("--force", action="store_true", help="with --learn: run even if that unit was already learned")
    p.add_argument("--max-minutes", type=float, help="with --learn: wall-clock budget")
    p.add_argument("--before-open-minutes", type=float, metavar="MIN",
                   help="with --learn: stop MIN minutes before the next market open (the morning run leaves room for the pre-open warm-up)")
    p.add_argument("--offline", action="store_true", help="with --learn: do not refresh the bars first")
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser("market-status", help="is the exchange open? next open / close (Alpaca clock or built-in NYSE calendar)")
    p.add_argument("--source", choices=["auto", "alpaca", "builtin"], default="auto")
    p.set_defaults(fn=cmd_market_status)

    p = sub.add_parser("forecast", help="every model's up/down vote per ticker and each model's running hit rate")
    p.add_argument("--mode", choices=["paper", "alpaca", "moomoo", "live"])
    p.add_argument("--no-refresh", action="store_true", help="show the votes recorded by the last cycle instead of computing fresh ones")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--no-llm", action="store_true")
    p.set_defaults(fn=cmd_forecast)

    p = sub.add_parser("autopilot", help="keep running: trade every weekday after the close, retrain monthly")
    p.add_argument("--mode", choices=["paper", "alpaca", "moomoo", "live"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-now", action="store_true", help="run the first cycle immediately")
    p.add_argument("--once", action="store_true", help="one cycle (and a retrain if due), then exit")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--i-understand-real-money", action="store_true")
    p.set_defaults(fn=cmd_autopilot)

    p = sub.add_parser("report", help="write the HTML performance dashboard")
    p.add_argument("--mode", choices=["paper", "alpaca", "moomoo", "live"])
    p.add_argument("--out")
    p.add_argument("--open", action="store_true", help="open it in the browser")
    p.set_defaults(fn=cmd_report)

    sub.add_parser("experience", help="summary of logged paper / live decisions and outcomes").set_defaults(fn=cmd_experience)

    p = sub.add_parser("account", help="the broker's own record (Alpaca): equity per day (ups and downs), positions, every fill")
    p.add_argument("--period", default="1M", help="1D | 1W | 1M | 3M | 1A | all")
    p.add_argument("--days", type=int, default=30, help="how many of the latest days to print")
    p.add_argument("--fills", type=int, default=30)
    p.add_argument("--sync", action="store_true", help="cache the history and every fill under data/paper/alpaca (saved with the state)")
    p.add_argument("--configure-like-moomoo", action="store_true",
                   help="set the Alpaca account itself to no margin, no shorting and (unless execution.alpaca.fractional) whole shares")
    p.set_defaults(fn=cmd_account)

    p = sub.add_parser("intraday-fit", help="fit the intraday exit model on the broker's 15-minute bars: when should a held name have been sold?")
    p.add_argument("--days", type=int, default=40)
    p.add_argument("--timeframe", default="15Min")
    p.add_argument("--holdout-days", type=int, default=5)
    p.set_defaults(fn=cmd_intraday_fit)

    p = sub.add_parser("retrain", help="refresh data, refit sub-models and continue training (feedback loop)")
    p.add_argument("--timesteps", type=int)
    p.add_argument("--n-envs", type=int)
    p.add_argument("--seeds", type=int)
    p.add_argument("--max-minutes", type=float, help="stop the PPO updates after this much wall-clock time")
    p.add_argument("--reuse-dataset-days", type=float, help="reuse models/dataset when younger than this (skip the sub-model refit)")
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.set_defaults(fn=cmd_retrain)

    p = sub.add_parser("ensemble", help="register models/policy/ensemble/seed* members (trained in parallel) as one ensemble")
    p.add_argument("--min-score", type=float, help="drop members below this out-of-sample score (default train.ensemble_min_score)")
    p.set_defaults(fn=cmd_ensemble)

    p = sub.add_parser("submodules", help="manage third_party submodules (args passed through)")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_submodules)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(getattr(args, "config", None), getattr(args, "set", None) or [])
    account = getattr(args, "account", None)
    if account:
        from .config import account_config

        cfg = account_config(cfg, account)
    return int(args.fn(cfg, args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
