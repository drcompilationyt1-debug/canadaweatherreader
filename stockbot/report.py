"""Self-contained HTML dashboard: equity curve, positions, recent decisions, training history."""
from __future__ import annotations

import html
import json
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .feedback.experience import ExperienceStore
from .logging_utils import get_logger

log = get_logger(__name__)

CSS = """
body{font-family:Segoe UI,Helvetica,Arial,sans-serif;margin:0;padding:20px 24px;background:#f7f7f5;color:#1a1a1a}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:24px 0 8px;color:#333}
.muted{color:#777;font-size:12px}.tiles{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.tile{background:#fff;border:1px solid #e3e3df;border-radius:8px;padding:10px 14px;min-width:140px}
.tile .v{font-size:20px;font-weight:600}.tile .k{font-size:11px;color:#777;text-transform:uppercase}
.pos{color:#1b7f3b}.neg{color:#b3261e}table{border-collapse:collapse;background:#fff;font-size:13px;width:100%;max-width:1100px}
th,td{border-bottom:1px solid #eee;padding:5px 8px;text-align:right}th:first-child,td:first-child{text-align:left}
th{background:#f0f0ec;font-weight:600}svg{background:#fff;border:1px solid #e3e3df;border-radius:8px;max-width:100%}
"""


def _svg_line(xs: list[float], ys: list[float], ys2: list[float] | None = None, w: int = 900, h: int = 260,
              labels: tuple[str, str] = ("", "")) -> str:
    if len(ys) < 2:
        return "<p class='muted'>not enough history for a chart yet</p>"
    allv = ys + (ys2 or [])
    lo, hi = min(allv), max(allv)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pad = 36

    def pt(i, v):
        x = pad + (w - 2 * pad) * i / (len(ys) - 1)
        y = h - pad - (h - 2 * pad) * (v - lo) / (hi - lo)
        return f"{x:.1f},{y:.1f}"

    line1 = " ".join(pt(i, v) for i, v in enumerate(ys))
    parts = [f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}'>"]
    parts.append(f"<polyline fill='none' stroke='#2b6cb0' stroke-width='2' points='{line1}'/>")
    if ys2 and len(ys2) == len(ys):
        parts.append("<polyline fill='none' stroke='#999' stroke-width='1.5' stroke-dasharray='4 3' points='"
                     + " ".join(pt(i, v) for i, v in enumerate(ys2)) + "'/>")
    parts.append(f"<text x='{pad}' y='{pad - 10}' font-size='11' fill='#555'>{hi:,.0f}</text>")
    parts.append(f"<text x='{pad}' y='{h - pad + 14}' font-size='11' fill='#555'>{lo:,.0f}</text>")
    parts.append(f"<text x='{pad}' y='{h - 6}' font-size='11' fill='#555'>{html.escape(labels[0])}</text>")
    parts.append(f"<text x='{w - pad}' y='{h - 6}' font-size='11' fill='#555' text-anchor='end'>{html.escape(labels[1])}</text>")
    parts.append("</svg>")
    return "".join(parts)


def _table(df: pd.DataFrame, cols: list[str] | None = None, fmt: dict | None = None) -> str:
    if df is None or len(df) == 0:
        return "<p class='muted'>nothing yet</p>"
    cols = [c for c in (cols or list(df.columns)) if c in df.columns]
    fmt = fmt or {}
    out = ["<table><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in cols) + "</tr>"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt and v is not None and not (isinstance(v, float) and np.isnan(v)):
                s = fmt[c].format(v)
            else:
                s = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
            cls = ""
            if isinstance(v, (int, float, np.floating)) and c in fmt and ("pnl" in c or "return" in c or "excess" in c):
                cls = " class='pos'" if v > 0 else (" class='neg'" if v < 0 else "")
            cells.append(f"<td{cls}>{html.escape(s)}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</table>")
    return "".join(out)


def _paper_state(cfg: Config) -> dict:
    f = cfg.path("execution.state_file", "data/paper/state.json")
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _alpaca_state(cfg: Config) -> dict:
    try:
        from .execution.alpaca import AlpacaBroker

        b = AlpacaBroker(paper=bool(cfg.get_path("execution.alpaca.paper", True)))
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        hist = b.client.get_portfolio_history(GetPortfolioHistoryRequest(period="3M", timeframe="1D"))
        eq = [{"ts": datetime.fromtimestamp(t, tz=timezone.utc).isoformat(), "equity": float(e)}
              for t, e in zip(hist.timestamp, hist.equity) if e is not None]
        acct = b.client.get_account()
        positions = {t: {"shares": p.shares, "avg_price": p.avg_price} for t, p in b.positions().items()}
        extra: dict = {"fills": []}
        try:
            from .execution.alpaca_history import AlpacaHistory

            h = AlpacaHistory(cfg)
            extra["daily"] = h.daily("1M").tail(30).iloc[::-1].to_dict("records")
            f = h.sync_fills()
            extra["fills"] = f.tail(30).iloc[::-1].assign(ts=lambda d: d["ts"].astype(str).str[:16]).to_dict("records") if len(f) else []
        except Exception as e:  # noqa: BLE001
            log.debug("alpaca history unavailable: %s", e)
        return {"cash": float(acct.cash), "initial_cash": float(getattr(acct, "last_equity", acct.equity)),
                "equity_history": eq, "positions": positions, "last_prices": {}, "broker": b.name, **extra}
    except Exception as e:  # noqa: BLE001
        log.warning("alpaca dashboard data unavailable: %s", e)
        return {}


def _moomoo_state(cfg: Config) -> dict:
    try:
        from .execution.moomoo import MoomooBroker

        mm = cfg.section("execution.moomoo")
        b = MoomooBroker(env=str(mm.get("env", "simulate")), host=str(mm.get("host", "127.0.0.1")), port=int(mm.get("port", 11111)),
                         security_firm=str(mm.get("security_firm", "FUTUINC")), market=str(mm.get("market", "US")))
        summary = b.summary()
        prices = {}
        for t in summary["positions"]:
            try:
                prices[t] = b.price(t)
            except Exception:  # noqa: BLE001
                pass
        store = ExperienceStore(cfg.path("feedback.experience_file", "data/experience/trades.jsonl"))
        recs = store.load()
        hist = []
        if recs is not None and len(recs):
            dec = recs[(recs["type"] == "decision") & (recs["mode"] == "moomoo")]
            if len(dec):
                hist = [{"ts": d, "equity": float(g["equity"].iloc[-1])} for d, g in dec.groupby("date")]
        b.close()
        return {"cash": summary["cash"], "initial_cash": hist[0]["equity"] if hist else summary["equity"], "equity_history": hist,
                "positions": summary["positions"], "fills": [], "last_prices": prices, "broker": summary["broker"]}
    except Exception as e:  # noqa: BLE001
        log.warning("moomoo dashboard data unavailable: %s", e)
        return {}


def build_dashboard(cfg: Config, mode: str = "paper", out: str | Path | None = None, open_browser: bool = False) -> Path:
    if mode in ("alpaca", "live"):
        state = _alpaca_state(cfg)
    elif mode == "moomoo":
        state = _moomoo_state(cfg)
    else:
        state = _paper_state(cfg)
    out = Path(out) if out else cfg.path("report.out", "reports/dashboard.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    eq_hist = state.get("equity_history") or []
    equity = float(eq_hist[-1]["equity"]) if eq_hist else float(state.get("cash", 0.0))
    initial = float(state.get("initial_cash") or (eq_hist[0]["equity"] if eq_hist else equity) or 1.0)
    ret = equity / initial - 1.0 if initial else 0.0
    positions = state.get("positions") or {}
    prices = state.get("last_prices") or {}
    pos_rows = []
    for t, p in positions.items():
        px = float(prices.get(t, 0.0))
        val = float(p["shares"]) * px
        cost = float(p["shares"]) * float(p.get("avg_price", 0.0))
        pos_rows.append({"ticker": t, "shares": float(p["shares"]), "avg_price": float(p.get("avg_price", 0.0)),
                         "last": px, "value": val, "unrealized_pnl": val - cost if px else float("nan")})
    pos_df = pd.DataFrame(pos_rows)
    gross = float(pos_df["value"].abs().sum()) if len(pos_df) else 0.0

    store = ExperienceStore(cfg.path("feedback.experience_file", "data/experience/trades.jsonl"))
    recs = store.load()
    dec_df = out_df = None
    avail_txt = ""
    if recs is not None and len(recs):
        dec = recs[recs["type"] == "decision"].copy()
        if len(dec):
            dec_df = dec[["date", "ticker", "decision", "target_exposure", "weight", "price", "mode"]].tail(40).iloc[::-1]
            last_av = dec.iloc[-1].get("availability") or {}
            if isinstance(last_av, dict):
                on = [k for k, v in last_av.items() if v]
                off = [k for k, v in last_av.items() if not v]
                avail_txt = f"signals on: {', '.join(on) or '-'} | off: {', '.join(off) or '-'}"
        outc = recs[recs["type"] == "outcome"].copy()
        if len(outc):
            out_df = outc[["decision_date", "ticker", "weight", "asset_log_return", "pnl_log_return"]].tail(40).iloc[::-1]
    summary = store.summary()

    ckpt = cfg.path("train.checkpoint_dir", "models/policy")
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8")) if (ckpt / "meta.json").exists() else {}
    train_log = json.loads((ckpt / "train_log.json").read_text(encoding="utf-8")) if (ckpt / "train_log.json").exists() else []
    tl_df = pd.DataFrame(train_log)[["timesteps", "mean_sharpe", "mean_bh_sharpe", "median_excess_return", "win_rate_vs_bh", "mean_avg_exposure", "mean_short_share"]] \
        if train_log and "median_excess_return" in train_log[0] else pd.DataFrame(train_log)

    # direction scorecard: which models called the moves (session = open -> end of session, daily = close -> next close)
    from .feedback.direction import DirectionBoard

    board = DirectionBoard(cfg.path("feedback.direction_file", "data/experience/direction.jsonl"))
    score_parts = []
    for horizon in ("session", "daily"):
        sc = board.scorecard(horizon)
        if sc is not None and len(sc):
            score_parts.append(f"<h3>{horizon} horizon</h3>" + _table(sc, fmt={"hit_rate": "{:.1%}", "edge_bps": "{:+.1f}", "up_share": "{:.0%}"}))
    votes_tbl = board.latest()
    if votes_tbl is not None and len(votes_tbl):
        vt = votes_tbl.reset_index().rename(columns={votes_tbl.index.name or "index": "ticker"})
        score_parts.append(f"<h3>latest votes ({html.escape(str(votes_tbl.index.name))})</h3>" + _table(vt, fmt={c: "{:+.0f}" for c in vt.columns if c not in ("ticker", "consensus")} | {"consensus": "{:+.2f}"}))
    fees_paid = float(summary.get("fees_paid", 0.0) or 0.0)

    # latest market-hours session: equity path and who moved
    sess_parts = []
    sess_dir = cfg.path("session.log_dir", "data/paper/sessions")
    sess_files = sorted(sess_dir.glob("session_*.json")) if sess_dir.is_dir() else []
    if sess_files:
        try:
            s = json.loads(sess_files[-1].read_text(encoding="utf-8"))
            snaps = []
            jl = sess_files[-1].with_suffix(".jsonl")
            if jl.exists():
                snaps = [json.loads(line) for line in jl.read_text(encoding="utf-8").splitlines() if line.strip()]
                snaps = [r for r in snaps if r.get("type") == "snapshot"]
            sess_parts.append(f"<div class='muted'>{html.escape(str(s.get('date', '')))} {html.escape(str(s.get('mode', '')))}: "
                              f"equity {float(s.get('equity_open', 0) or 0):,.0f} -> {float(s.get('equity_end', 0) or 0):,.0f} "
                              f"({100 * float(s.get('session_return', 0) or 0):+.2f}%), {s.get('orders', 0)} orders, avg move "
                              f"{100 * float(s.get('mean_move', 0) or 0):+.2f}%, consensus right "
                              f"{('%.0f%%' % (100 * s['consensus_hit_rate'])) if s.get('consensus_hit_rate') is not None else '-'}, "
                              f"trainer {html.escape(str((s.get('trainer') or {}).get('returncode', '-')))}</div>")
            if snaps:
                sess_parts.append(_svg_line(list(range(len(snaps))), [float(r["equity"]) for r in snaps],
                                            labels=(str(snaps[0]["ts"])[11:16], str(snaps[-1]["ts"])[11:16])))
                last = snaps[-1].get("tickers") or {}
                mv = pd.DataFrame([{"ticker": t, "move": v.get("move"), "consensus": v.get("consensus"), "held": v.get("held"), "price": v.get("price")}
                                   for t, v in last.items()]).sort_values("move", ascending=False)
                sess_parts.append(_table(mv, fmt={"move": "{:+.2%}", "consensus": "{:+.2f}", "held": "{:.3f}", "price": "{:.2f}"}))
        except Exception as e:  # noqa: BLE001
            log.debug("session section failed: %s", e)

    ys = [float(e["equity"]) for e in eq_hist]
    labels = (str(eq_hist[0]["ts"])[:10], str(eq_hist[-1]["ts"])[:10]) if eq_hist else ("", "")
    cls = "pos" if ret >= 0 else "neg"
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='300'>",
             f"<title>StockBot dashboard</title><style>{CSS}</style></head><body>",
             f"<h1>StockBot dashboard <span class='muted'>({html.escape(state.get('broker', mode))})</span></h1>",
             f"<div class='muted'>generated {now} - refreshes every 5 minutes - {html.escape(avail_txt)}</div>",
             "<div class='tiles'>",
             f"<div class='tile'><div class='k'>equity</div><div class='v'>{equity:,.0f}</div></div>",
             f"<div class='tile'><div class='k'>return since start</div><div class='v {cls}'>{100 * ret:+.2f}%</div></div>",
             f"<div class='tile'><div class='k'>cash</div><div class='v'>{float(state.get('cash', 0.0)):,.0f}</div></div>",
             f"<div class='tile'><div class='k'>gross exposure</div><div class='v'>{(gross / equity if equity else 0):.2f}</div></div>",
             f"<div class='tile'><div class='k'>positions</div><div class='v'>{len(pos_df)}</div></div>",
             f"<div class='tile'><div class='k'>decisions / settled</div><div class='v'>{summary.get('decisions', 0)} / {summary.get('outcomes', 0)}</div></div>",
             f"<div class='tile'><div class='k'>realised pnl (log)</div><div class='v'>{100 * summary.get('realized_pnl_log_return', 0.0):+.2f}%</div></div>",
             f"<div class='tile'><div class='k'>hit rate</div><div class='v'>{100 * summary.get('hit_rate', 0.0):.0f}%</div></div>",
             f"<div class='tile'><div class='k'>fees (moomoo schedule)</div><div class='v'>{fees_paid:,.2f}</div></div>",
             "</div>",
             "<h2>Equity</h2>", _svg_line(list(range(len(ys))), ys, labels=labels),
             "<h2>Latest session</h2>", "".join(sess_parts) or "<div class='muted'>no session yet</div>",
             "<h2>Account: per-day ups and downs (broker)</h2>",
             _table(pd.DataFrame(state["daily"]), fmt={"equity": "{:,.0f}", "profit_loss": "{:+,.0f}", "profit_loss_pct": "{:+.2%}"})
             if state.get("daily") else "<div class='muted'>no broker history</div>",
             "<h2>Transactions (broker fills)</h2>",
             _table(pd.DataFrame(state["fills"])[["ts", "ticker", "side", "qty", "price", "notional"]], fmt={"qty": "{:.3f}", "price": "{:.2f}", "notional": "{:,.0f}"})
             if state.get("fills") else "<div class='muted'>no fills yet</div>",
             "<h2>Who predicts the direction?</h2>", "".join(score_parts) or "<div class='muted'>nothing settled yet</div>",
             "<h2>Positions</h2>", _table(pos_df, fmt={"shares": "{:.3f}", "avg_price": "{:.2f}", "last": "{:.2f}", "value": "{:,.0f}", "unrealized_pnl": "{:+,.0f}"}),
             "<h2>Recent decisions</h2>", _table(dec_df, fmt={"target_exposure": "{:+.2f}", "weight": "{:+.3f}", "price": "{:.2f}"}),
             "<h2>Settled outcomes</h2>", _table(out_df, fmt={"weight": "{:+.3f}", "asset_log_return": "{:+.4f}", "pnl_log_return": "{:+.5f}"}),
             "<h2>Policy</h2>",
             f"<div class='muted'>trained {html.escape(str(meta.get('trained_at', '?')))} - {html.escape(str(meta.get('algo', '?')))} - "
             f"{meta.get('timesteps', '?')} steps - layout {html.escape(str(meta.get('signature', '?')))}</div>",
             _table(tl_df, fmt={"mean_sharpe": "{:.2f}", "mean_bh_sharpe": "{:.2f}", "median_excess_return": "{:+.1%}", "win_rate_vs_bh": "{:.0%}", "mean_avg_exposure": "{:.2f}", "mean_short_share": "{:.0%}"}),
             "</body></html>"]
    out.write_text("".join(parts), encoding="utf-8")
    log.info("dashboard written to %s", out)
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out
