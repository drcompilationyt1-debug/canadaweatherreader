"""Post-trade review: what we did, what else we could have done, and what to learn from it.

Trading desks run this as a matter of routine (post-trade / transaction-cost analysis, attribution,
regret against an oracle).  Every period the review replays the paper account on the same prices
with the same fee schedule and scores the actual decisions against alternatives:

* ``hold``        - keep the positions of the start of the period, trade nothing
* ``equal``       - equal-weight the whole universe
* ``benchmark``   - the index ETF (SPY)
* ``oracle``      - perfect foresight within our own constraints (long-only, one slice per name, fees):
                    the upper bound; ``regret`` = oracle - actual, ``captured`` = actual / oracle
* ``follow:<m>``  - each input model followed alone (long the names it called up that day)
* ``half`` / ``double`` / ``top_consensus`` - sizing variants of what we actually did
* ``path_oracle`` - (week / month / year) the fee-aware best exposure *path*: the best week is not
                    five best days strung together, fees make holding through noise the better play
* ``what_if``     - (day) a parallel search over every move that was available on the session's
                    price path (exposure level x entry x exit), with the regret split into sizing and timing

Each review also attributes the regret (missed / wrong side / under-sized / cost), scores the
execution (fills vs the decision price), rates every input by information coefficient, summarises the
round trips (pyfolio) and tags the market regime; ``stockbot review --learn`` then fine-tunes the
policy on the hindsight labels (see ``feedback/hindsight.py``).

Periods: ``day`` uses the session's intraday snapshots (open -> end of the watch window); ``week``,
``month`` and ``year`` replay the recorded daily weights on daily bars and, when quantstats is
installed, also write a tearsheet against SPY.  Results land in ``data/experience/reviews/`` as JSON
(plus ``lessons``: plain sentences) and the oracle's per-name actions are kept as hindsight labels for
the reliability block and future imitation experiments.  ``stockbot review --period week`` prints one.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..execution.fees import FeeBook
from ..execution.markets import market_of
from ..logging_utils import get_logger
from .attribution import (LEVELS, attribute_regret, attribute_regret_daily, ic_by_voter, portfolio_path_oracle,
                          regime, round_trips, what_if_day)
from .direction import DirectionBoard
from .experience import ExperienceStore

log = get_logger(__name__)

PERIODS = ("day", "week", "month", "year")


def period_bounds(period: str, end: date) -> tuple[date, date]:
    """Calendar window ending on ``end`` (inclusive)."""
    if period == "day":
        return end, end
    if period == "week":
        start = end - timedelta(days=end.weekday())          # Monday of that week
        return start, end
    if period == "month":
        return end.replace(day=1), end
    if period == "year":
        return end.replace(month=1, day=1), end
    raise ValueError(period)


def _pct(x: float | None) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:+.2f}%"


class Review:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.max_position = float(cfg.get_path("execution.max_position", 0.10))
        self.fees = FeeBook.from_config(cfg)
        self.store = ExperienceStore(cfg.path("feedback.experience_file", "data/experience/trades.jsonl"))
        self.board = DirectionBoard(cfg.path("feedback.direction_file", "data/experience/direction.jsonl"))
        self.session_dir = cfg.path("session.log_dir", "data/paper/sessions")
        self.out_dir = cfg.path("feedback.review_dir", "data/experience/reviews")
        self.benchmark = str(cfg.get_path("feedback.review_benchmark", "SPY"))
        self.gross_cap = float(cfg.get_path("execution.max_gross_exposure", 1.0) or 1.0)
        wi = cfg.get_path("feedback.what_if", {}) or {}
        self.levels = tuple(float(x) for x in (wi.get("levels") or LEVELS))
        self.workers = int(wi.get("workers", 4) or 4)

    # ------------------------------------------------------------------ helpers
    def _fee_frac(self, ticker: str, notional: float, price: float, side: str = "buy") -> float:
        """Fees as a fraction of the notional for a trade of that size (0 when the fee book is empty)."""
        sched = self.fees.for_ticker(ticker)
        if sched is None or notional <= 0 or price <= 0:
            return 0.0
        return float(sched.cost(notional / price, price, side)) / notional

    def _fee_cost(self, ticker: str, notional: float, price: float, side: str = "buy") -> float:
        sched = self.fees.for_ticker(ticker)
        if sched is None or notional <= 0 or price <= 0:
            return 0.0
        return float(sched.cost(notional / price, price, side))

    def _records_between(self, start: date, end: date) -> pd.DataFrame | None:
        recs = self.store.load()
        if recs is None or len(recs) == 0:
            return None
        dec = recs[recs["type"] == "decision"].copy()
        dec["_d"] = pd.to_datetime(dec["date"]).dt.date
        dec = dec[(dec["_d"] >= start) & (dec["_d"] <= end)]
        return dec if len(dec) else None

    def _execution(self, dec: pd.DataFrame | None) -> tuple[dict, list[dict], float]:
        """Fill quality of the recorded decisions: implementation shortfall vs the decision price (bps, positive =
        paid more than planned), fees in bps of the traded notional; also the raw fills and the fees paid."""
        if dec is None or len(dec) == 0:
            return {}, [], 0.0
        fills, short, notional, fees = [], [], 0.0, 0.0
        for _, r in dec.iterrows():
            fees += float(r.get("fees") or 0.0)
            p0 = float(r.get("price") or 0.0)
            for f in (r.get("fills") or []):
                f = dict(f)
                f.setdefault("ticker", r["ticker"])
                fills.append(f)
                q, px = float(f.get("qty") or 0.0), float(f.get("price") or 0.0)
                if p0 > 0 and px > 0 and q > 0:
                    sign = 1.0 if f.get("side", "buy") == "buy" else -1.0
                    short.append(sign * (px / p0 - 1.0) * 1e4)
                    notional += q * px
        out = {"fills": len(fills), "traded_notional": notional, "fees": fees,
               "shortfall_bps": float(np.mean(short)) if short else None,
               "fees_bps": float(fees / notional * 1e4) if notional > 0 else None}
        return out, fills, fees

    def _full_day(self, day: date, names: list[str], dec_by: dict, held_now: dict[str, float], slice_cap: float, equity0: float,
                  mode: str) -> tuple[dict, list[str], float | None]:
        """The whole 09:30-16:00 session from the broker's 15-minute bars (Alpaca): every move that was available on the
        full day, the best exit of every name we held, what the intraday exit model would have done, and the fills."""
        if mode not in ("alpaca", "live"):
            return {}, [], None
        try:
            from ..execution.alpaca_history import AlpacaHistory
            from .intraday import ExitModel, what_if_exit

            if not AlpacaHistory.available():
                return {}, [], None
            hist = AlpacaHistory(self.cfg)
            paths = hist.day_paths(names, day)
        except Exception as e:  # noqa: BLE001
            log.debug("broker bars unavailable for %s: %s", day, e)
            return {}, [], None
        if not paths:
            return {}, [], None
        n_steps = max(len(s) for s in paths.values())
        paths = {t: s for t, s in paths.items() if len(s) == n_steps}
        labels = ["open"] + [ts.strftime("%H:%M") for ts in next(iter(paths.values())).index]
        prices = {t: [float(dec_by.get(t, {}).get("price") or s.iloc[0])] + s.to_numpy(float).tolist() for t, s in paths.items()}
        wi = what_if_day(prices, held_now, slice_cap, equity0, self._fee_cost, labels=labels, levels=self.levels, gross_cap=self.gross_cap,
                         workers=self.workers)
        moves_close = {t: p[-1] / p[0] - 1.0 for t, p in prices.items()}
        hold_to_close = sum(held_now.get(t, 0.0) * slice_cap * moves_close[t] for t in prices) / equity0 if equity0 else None
        block = {"source": "alpaca 15Min bars", "n_names": len(prices), "steps": n_steps, "what_if": wi, "moves_close": moves_close,
                 "hold_to_close_return": hold_to_close}
        lessons = [f"the whole day from the broker's bars ({len(prices)} names, {n_steps} steps): best set of moves {_pct(wi['best_return'])}, "
                   f"sizing regret {_pct(wi['sizing_regret'])}, timing regret {_pct(wi['timing_regret'])}"]
        ie = self.cfg.get_path("session.intraday_exit", {}) or {}
        model = ExitModel.load(self.cfg.path("session.intraday_exit.model_dir", "models/intraday_exit"))
        exit_total = hold_total = 0.0
        held_lines, per_held = [], {}
        for t in sorted(prices, key=lambda t: -held_now.get(t, 0.0)):
            e = held_now.get(t, 0.0)
            if e <= 1e-6:
                continue
            c = np.asarray(prices[t][1:], dtype=float)
            p0 = float(prices[t][0])
            fee_rt = 2.0 * self._fee_frac(t, max(e * slice_cap, 1.0), p0)
            w = what_if_exit(c, p0, fee_rt)
            row = {"exposure": e, **w}
            line = f"{t}: high-water {_pct(w['high_water'])}, best exit {labels[w['best_exit_k'] + 1]} for {_pct(w['best_exit_return'])}, close {_pct(w['hold_return'])}"
            if model is not None:
                rp = model.replay(c, p0, None, fee_rt, min_prob=float(ie.get("min_prob", 0.6)), min_gain=float(ie.get("min_gain", 0.005)),
                                  stop_loss=float(ie.get("stop_loss", 0.02)))
                exit_total += e * slice_cap * rp["exit_return"]
                hold_total += e * slice_cap * rp["hold_return"]
                row["exit_model"] = rp
                line += (f"; the exit model would have sold at {labels[rp['exit_k'] + 1]} for {_pct(rp['exit_return'])} (p={rp['prob']:.2f})"
                         if rp["exit_k"] is not None else "; the exit model would have held")
            per_held[t] = row
            if w["gain_vs_hold"] > 0.002:
                held_lines.append(line)
        block["held"] = per_held
        if model is not None and per_held and equity0:
            block["exit_model_return"] = exit_total / equity0
            lessons.append(f"following the exit model on the names we held: {_pct(exit_total / equity0)} vs holding to the close {_pct(hold_total / equity0)}")
        lessons.extend(held_lines[:3])
        try:
            fills = hist.fills(after=day, until=day + timedelta(days=1))
            block["broker_fills"] = [{"ts": str(r["ts"]), "ticker": r["ticker"], "side": r["side"], "qty": r["qty"], "price": r["price"]}
                                     for _, r in fills.iterrows()] if len(fills) else []
        except Exception as e:  # noqa: BLE001
            log.debug("broker fills unavailable: %s", e)
        return block, lessons, hold_to_close

    @staticmethod
    def _rank_alternatives(actual: float, alts: dict[str, float]) -> list[dict]:
        rows = [{"name": "actual", "return": actual}] + [{"name": k, "return": v} for k, v in alts.items()]
        rows.sort(key=lambda r: -r["return"] if np.isfinite(r["return"]) else 1e9)
        return rows

    def _save(self, period: str, end: date, result: dict) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        p = self.out_dir / f"{period}_{end.isoformat()}.json"
        p.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
        return p

    # ------------------------------------------------------------------ day: from the session snapshots
    def review_day(self, day: date | None = None) -> dict | None:
        files = sorted(self.session_dir.glob("session_*.jsonl")) if self.session_dir.is_dir() else []
        if day is not None:
            files = [f for f in files if f.stem == f"session_{day.isoformat()}"]
        if not files:
            return None
        f = files[-1]
        day = date.fromisoformat(f.stem.replace("session_", ""))
        recs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        decisions = next((r for r in recs if r.get("type") == "decisions"), None)
        snaps = [r for r in recs if r.get("type") == "snapshot"]
        if decisions is None or not snaps:
            return None
        end_snap = snaps[-1]
        tickers = end_snap.get("tickers") or {}
        dec_by = {d["ticker"]: d for d in decisions.get("decisions", [])}
        votes_by = decisions.get("votes") or {}
        summary_file = f.with_suffix(".json")
        summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
        equity0 = float(summary.get("equity_open") or 0.0) or float(snaps[0]["equity"])
        slice_cap = equity0 * self.max_position

        moves, held_now, prev_exp, targets = {}, {}, {}, {}
        for t, row in tickers.items():
            mv = row.get("move")
            if mv is None:
                continue
            moves[t] = float(mv)
            d = dec_by.get(t, {})
            price = float(d.get("price") or row.get("price") or 0.0)
            held_now[t] = float(row.get("held") or 0.0) * price / slice_cap if slice_cap > 0 and price > 0 else 0.0
            prev_exp[t] = float(d.get("current_exposure", 0.0))
            targets[t] = float(d.get("target_exposure", 0.0))
        if not moves:
            return None
        names = sorted(moves)

        def pnl_of(exposure: dict[str, float], entry_fees: bool = True) -> float:
            """Return on the account for a set of exposures (fractions of a slice), fees on the change from prev."""
            total = 0.0
            for t in names:
                e = float(np.clip(exposure.get(t, 0.0), 0.0, 1.0))
                total += e * slice_cap * moves[t]
                if entry_fees:
                    delta = abs(e - prev_exp.get(t, 0.0)) * slice_cap
                    if delta > 0:
                        d = dec_by.get(t, {})
                        price = float(d.get("price") or tickers[t].get("price") or 0.0)
                        total -= delta * self._fee_frac(t, delta, price, "buy" if e > prev_exp.get(t, 0.0) else "sell")
            return total / equity0 if equity0 > 0 else 0.0

        actual = pnl_of(held_now)
        alts = {
            "hold": pnl_of(prev_exp, entry_fees=False),
            "equal": pnl_of({t: min(1.0, (1.0 / len(names)) / self.max_position) for t in names}),
            "oracle": pnl_of({t: 1.0 if moves[t] > 0 else 0.0 for t in names}),
            "half": pnl_of({t: 0.5 * held_now[t] for t in names}),
            "double": pnl_of({t: min(1.0, 2.0 * held_now[t]) for t in names}),
        }
        if self.benchmark in moves:
            alts["benchmark"] = moves[self.benchmark]
        voters = sorted({v for t in names for v in (votes_by.get(t) or {})})
        for v in voters:
            if v in ("policy", "momentum"):
                continue
            alts[f"follow:{v}"] = pnl_of({t: 1.0 if (votes_by.get(t) or {}).get(v, {}).get("vote", 0) > 0 else 0.0 for t in names})
        cons = {t: (votes_by.get(t) or {}).get("consensus") for t in names}
        cons = {t: float(np.mean([vv["vote"] for k, vv in (votes_by.get(t) or {}).items() if k not in ("policy", "momentum") and vv["vote"] != 0]) or 0.0)
                if votes_by.get(t) else 0.0 for t in names}
        top = sorted(names, key=lambda t: -cons[t])[: max(1, len(names) // 5)]
        alts["top_consensus"] = pnl_of({t: 1.0 if t in top else 0.0 for t in names})

        # per-name attribution: what each held name contributed and what the oracle would have done
        per_name = []
        for t in names:
            contrib = held_now[t] * slice_cap * moves[t] / equity0 if equity0 else 0.0
            oracle_e = 1.0 if moves[t] > 0 else 0.0
            up = [k for k, vv in (votes_by.get(t) or {}).items() if vv["vote"] > 0]
            down = [k for k, vv in (votes_by.get(t) or {}).items() if vv["vote"] < 0]
            per_name.append({"ticker": t, "move": moves[t], "exposure": held_now[t], "target": targets.get(t), "contribution": contrib,
                             "oracle_exposure": oracle_e, "action": dec_by.get(t, {}).get("action", "?"),
                             "voted_up": up, "voted_down": down, "consensus": cons[t]})
        per_name.sort(key=lambda r: r["contribution"])
        extra: dict = {"session_file": str(f), "equity_open": equity0, "n_names": len(names)}
        more: list[str] = []
        execution, _fills, fees_paid = self._execution(self._records_between(day, day))
        extra["attribution"] = attribute_regret(held_now, moves, slice_cap / equity0 if equity0 else 0.0, fees_paid / equity0 if equity0 else 0.0)
        a = extra["attribution"]
        if a["total"] > 0:
            more.append(f"regret came mostly from {a['biggest'].replace('_', ' ')}: missed {_pct(a['missed'])}, wrong side {_pct(a['wrong_side'])}, "
                        f"under-sized {_pct(a['under_sized'])}, fees {_pct(a['cost'])}")
        try:
            prices = {t: [float(dec_by.get(t, {}).get("price") or snaps[0]["tickers"][t]["price"])] +
                        [float(s["tickers"][t]["price"]) for s in snaps if t in s.get("tickers", {}) and s["tickers"][t].get("price")]
                      for t in names}
            labels = ["open"] + [str(s.get("label", k)) for k, s in enumerate(snaps)]
            wi = what_if_day(prices, held_now, slice_cap, equity0, self._fee_cost, labels=labels, levels=self.levels,
                             gross_cap=self.gross_cap, workers=self.workers)
            extra["what_if"] = wi
            more.append(f"searched {wi['moves_searched']} moves: the best set would have made {_pct(wi['best_return'])} vs actual "
                        f"{_pct(actual)} (sizing regret {_pct(wi['sizing_regret'])}, timing regret {_pct(wi['timing_regret'])})")
            for r in wi["per_name"][:2]:
                if r["regret"] <= 0:
                    continue
                more.append(f"{r['ticker']}: best move {100 * r['best_level']:.0f}% at {r['best_entry']} -> {r['best_exit']} ({_pct(r['best_pnl'])}); "
                            f"we held {100 * r['level_us']:.0f}% to the end ({_pct(r['pnl_us'])}) - sizing {_pct(r['sizing_regret'])}, timing {_pct(r['timing_regret'])}")
        except Exception as e:  # noqa: BLE001 - the search is advisory
            log.debug("what-if search failed: %s", e)
        if execution:
            extra["execution"] = execution
            if execution.get("shortfall_bps") is not None:
                more.append(f"execution: {execution['fills']} fills, {execution['shortfall_bps']:+.1f} bps vs the decision price, "
                            f"fees {execution['fees_bps']:.1f} bps of notional")
        try:
            full, full_lessons, hold_to_close = self._full_day(day, names, dec_by, held_now, slice_cap, equity0, str(summary.get("mode", "")))
            if full:
                extra["full_day"] = full
                more.extend(full_lessons)
                if hold_to_close is not None:
                    alts["hold_to_close"] = hold_to_close
        except Exception as e:  # noqa: BLE001
            log.debug("full-day review failed: %s", e)
        if self.benchmark in moves:
            extra["regime"] = {"benchmark_return": moves[self.benchmark],
                               "label": "up" if moves[self.benchmark] > 0.002 else "down" if moves[self.benchmark] < -0.002 else "flat"}
        result = self._finish("day", day, day, actual, alts, per_name, extra=extra, more_lessons=more)
        return result

    # ------------------------------------------------------------------ week / month / year: daily replay
    def _weights_and_returns(self, start: date, end: date):
        """(W, R, votes): daily weights the bot held (fraction of equity per name, ffilled from the recorded
        decisions), daily close-to-close returns, and the votes recorded per (date, ticker)."""
        from ..agent.train import load_frames

        recs = self.store.load()
        if recs is None or len(recs) == 0:
            return None, None, None
        dec = recs[recs["type"] == "decision"].copy()
        dec["date"] = pd.to_datetime(dec["date"])
        dec = dec[(dec["date"] >= pd.Timestamp(start) - pd.Timedelta(days=10)) & (dec["date"] <= pd.Timestamp(end))]
        if len(dec) == 0:
            return None, None, None
        tickers = sorted(dec["ticker"].unique().tolist())
        if self.benchmark not in tickers:
            tickers.append(self.benchmark)
        frames = load_frames(self.cfg, offline=True, tickers=tickers)
        closes = pd.DataFrame({t: df["close"].astype(float) for t, df in frames.items()}).sort_index()
        closes = closes[(closes.index >= pd.Timestamp(start) - pd.Timedelta(days=10)) & (closes.index <= pd.Timestamp(end))]
        R = closes.pct_change()
        W = pd.DataFrame(0.0, index=closes.index, columns=[t for t in tickers if t in closes.columns])
        last = dec.sort_values("date").groupby(["date", "ticker"])["weight"].last().unstack().reindex(closes.index).ffill().fillna(0.0)
        for t in W.columns:
            if t in last.columns:
                W[t] = last[t]
        votes: dict[tuple, dict] = {}
        board = self.board.load()
        if board is not None and len(board):
            b = board[board["type"] == "votes"]
            for _, r in b.iterrows():
                votes[(pd.Timestamp(r["date"]).normalize(), r["ticker"])] = r["votes"]
        window = closes.index[(closes.index >= pd.Timestamp(start)) & (closes.index <= pd.Timestamp(end))]
        return W.loc[window], R.loc[window], votes

    def _replay(self, W: pd.DataFrame, R: pd.DataFrame, fee_on_turnover: bool = True) -> pd.Series:
        """Daily portfolio returns of a weight path: yesterday's weights earn today's returns, turnover pays fees."""
        prev = W.shift(1).fillna(0.0)
        gross = (prev * R.reindex(columns=W.columns).fillna(0.0)).sum(axis=1)
        if not fee_on_turnover:
            return gross
        fee = pd.Series(0.0, index=W.index)
        turnover = (W - prev).abs()
        for t in W.columns:
            sched = self.fees.for_ticker(t)
            if sched is None:
                continue
            # per-order minimums: assume a $100k book so the fee fraction of a slice is realistic
            notional = turnover[t] * 100_000.0
            fee += notional.apply(lambda n, tt=t: sched.cost(max(n, 0.0) / 100.0, 100.0, "buy") if n > 0 else 0.0) / 100_000.0
        return gross - fee

    def review_period(self, period: str, end: date | None = None) -> dict | None:
        end = end or date.today()
        start, end = period_bounds(period, end)
        W, R, votes = self._weights_and_returns(start, end)
        if W is None or len(W) < 2:
            return None
        names = [t for t in W.columns if t != self.benchmark or W[t].abs().sum() > 0]
        actual_daily = self._replay(W, R)
        actual = float((1.0 + actual_daily).prod() - 1.0)
        n = len(names)
        alts_daily: dict[str, pd.Series] = {}
        hold = pd.DataFrame(np.tile(W.iloc[0].to_numpy(), (len(W), 1)), index=W.index, columns=W.columns)
        alts_daily["hold"] = self._replay(hold, R, fee_on_turnover=False)
        equal = pd.DataFrame(1.0 / max(n, 1), index=W.index, columns=W.columns)
        alts_daily["equal"] = self._replay(equal, R)
        if self.benchmark in R.columns:
            alts_daily["benchmark"] = R[self.benchmark].fillna(0.0)
        # oracle: each day, one slice in every name that goes up the next day (perfect one-day foresight)
        fut = R.reindex(columns=W.columns).shift(-1).fillna(0.0)
        oracle = (fut > 0).astype(float) * self.max_position
        oracle = oracle.div(oracle.sum(axis=1).clip(lower=1.0), axis=0)   # gross <= 1
        alts_daily["oracle"] = self._replay(oracle, R)
        # period oracle: hold the names that end the period higher, all period long
        total = (1.0 + R.reindex(columns=W.columns).fillna(0.0)).prod() - 1.0
        winners = [t for t in W.columns if total.get(t, 0.0) > 0]
        po = pd.DataFrame(0.0, index=W.index, columns=W.columns)
        if winners:
            po[winners] = min(self.max_position, 1.0 / len(winners))
        alts_daily["period_oracle"] = self._replay(po, R)
        for name, factor in (("half", 0.5), ("double", 2.0)):
            alts_daily[name] = self._replay((W * factor).clip(upper=self.max_position), R)
        voters = sorted({v for vv in votes.values() for v in vv} - {"policy", "momentum"}) if votes else []
        for v in voters:
            Wv = pd.DataFrame(0.0, index=W.index, columns=W.columns)
            for (d, t), vv in votes.items():
                if d in Wv.index and t in Wv.columns and vv.get(v, {}).get("vote", 0) > 0:
                    Wv.at[d, t] = self.max_position
            Wv = Wv.replace(0.0, np.nan).ffill().fillna(0.0)          # a vote stands until the next recorded vote
            Wv = Wv.div(Wv.sum(axis=1).clip(lower=1.0), axis=0)
            alts_daily[f"follow:{v}"] = self._replay(Wv, R)
        alts = {k: float((1.0 + s.fillna(0.0)).prod() - 1.0) for k, s in alts_daily.items()}
        per_name = []
        for t in names:
            contrib = float((W[t].shift(1).fillna(0.0) * R[t].fillna(0.0)).sum())
            per_name.append({"ticker": t, "move": float(total.get(t, 0.0)), "avg_weight": float(W[t].mean()), "contribution": contrib,
                             "oracle_exposure": 1.0 if total.get(t, 0.0) > 0 else 0.0})
        per_name.sort(key=lambda r: r["contribution"])
        stats = self._stats(actual_daily, alts_daily.get("benchmark"))
        extra = {"days": int(len(W)), "n_names": n, "stats": stats}
        more: list[str] = []
        # the fee-aware best path over the whole period (the best week is not five best days in a row)
        fee_fracs = {t: self._fee_frac(t, 100_000.0 * self.max_position, 100.0) for t in names}
        po_path = portfolio_path_oracle(R.reindex(columns=names).fillna(0.0), fee_fracs, self.max_position, self.gross_cap,
                                        levels=self.levels, start_weights=W.iloc[0].to_dict())
        alts["path_oracle"] = po_path["return"]
        extra["path_oracle"] = {"return": po_path["return"], "names": po_path["names"], "switches": po_path["switches"],
                                "per_name": {t: {"pnl_per_slice": v["pnl_per_slice"], "switches": v["switches"], "avg_level": v["avg_level"]}
                                             for t, v in po_path["per_name"].items() if t in po_path["names"]}}
        more.append(f"fee-aware best path this {period}: {_pct(po_path['return'])} with {po_path['switches']} switches across "
                    f"{len(po_path['names'])} names, vs {_pct(alts.get('oracle'))} if every day were played perfectly - "
                    f"the best {period} is not the best days strung together")
        gross_daily = self._replay(W, R, fee_on_turnover=False)
        fees_frac = float((gross_daily - actual_daily).fillna(0.0).sum())
        extra["attribution"] = attribute_regret_daily(W[names], R, self.max_position, cost=max(0.0, fees_frac))
        a = extra["attribution"]
        if a["total"] > 0:
            more.append(f"regret came mostly from {a['biggest'].replace('_', ' ')}: missed {_pct(a['missed'])}, wrong side {_pct(a['wrong_side'])}, "
                        f"under-sized {_pct(a['under_sized'])}, fees {_pct(a['cost'])}")
        dec = self._records_between(start, end)
        execution, fills, _fees = self._execution(dec)
        if execution:
            extra["execution"] = execution
        try:
            eq = float(dec["equity"].astype(float).iloc[-1]) if dec is not None and "equity" in dec.columns else None
            rt = round_trips(fills, eq)
        except Exception as e:  # noqa: BLE001
            log.debug("round trips failed: %s", e)
            rt = None
        if rt:
            extra["round_trips"] = rt
            if rt.get("n"):
                more.append(f"{rt['n']} round trips: win rate {100 * rt['win_rate']:.0f}%, profit factor {rt['profit_factor']:.2f}, "
                            f"average hold {rt['avg_holding_days']:.1f} days; best {rt['best']['symbol']} {rt['best']['pnl']:+,.0f}, "
                            f"worst {rt['worst']['symbol']} {rt['worst']['pnl']:+,.0f}")
        ic = ic_by_voter(votes, R) if votes else {}
        if ic:
            extra["ic"] = ic
            good = [f"{k} {v['ic']:+.2f}" for k, v in ic.items() if v["ic"] > 0.02][:5]
            bad = [f"{k} {v['ic']:+.2f}" for k, v in ic.items() if v["ic"] < -0.02][-3:]
            if good or bad:
                more.append("inputs by information coefficient - useful: " + (", ".join(good) or "none") +
                            "; harmful: " + (", ".join(bad) or "none"))
        rg = regime(alts_daily.get("benchmark"))
        if rg:
            extra["regime"] = rg
            more.append(f"market context: {rg['label']} ({self.benchmark} {_pct(rg['benchmark_return'])}, vol {100 * rg['benchmark_vol']:.0f}%)")
        tear = self._tearsheet(period, end, actual_daily, alts_daily.get("benchmark"))
        if tear:
            extra["tearsheet"] = str(tear)
        return self._finish(period, start, end, actual, alts, per_name, extra=extra, more_lessons=more)

    @staticmethod
    def _stats(daily: pd.Series, bench: pd.Series | None) -> dict:
        d = daily.fillna(0.0)
        out = {"return": float((1 + d).prod() - 1), "volatility": float(d.std() * np.sqrt(252)) if len(d) > 1 else 0.0,
               "max_drawdown": float(((1 + d).cumprod() / (1 + d).cumprod().cummax() - 1).min()) if len(d) else 0.0,
               "win_days": float((d > 0).mean()) if len(d) else 0.0}
        out["sharpe"] = float(d.mean() / d.std() * np.sqrt(252)) if len(d) > 1 and d.std() > 0 else 0.0
        if bench is not None:
            b = bench.fillna(0.0)
            out["benchmark_return"] = float((1 + b).prod() - 1)
            out["excess_return"] = out["return"] - out["benchmark_return"]
        return out

    def _tearsheet(self, period: str, end: date, daily: pd.Series, bench: pd.Series | None) -> Path | None:
        if period == "day" or len(daily) < 5:
            return None
        try:
            import quantstats as qs
        except ImportError:
            return None
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            out = self.out_dir / f"{period}_{end.isoformat()}_tearsheet.html"
            qs.reports.html(daily.fillna(0.0), benchmark=bench.fillna(0.0) if bench is not None else None, output=str(out),
                            title=f"{period} review to {end.isoformat()}", download_filename=str(out))
            return out
        except Exception as e:  # noqa: BLE001
            log.debug("quantstats tearsheet failed: %s", e)
            return None

    # ------------------------------------------------------------------ verdict
    def _finish(self, period: str, start: date, end: date, actual: float, alts: dict[str, float], per_name: list[dict], extra: dict,
                more_lessons: list[str] | None = None) -> dict:
        ranking = self._rank_alternatives(actual, alts)
        oracle = alts.get("oracle")
        regret = (oracle - actual) if oracle is not None else None
        captured = (actual / oracle) if oracle not in (None, 0.0) and oracle > 0 else None
        best_alt = next((r for r in ranking if r["name"] not in ("actual", "oracle", "period_oracle", "path_oracle")), None)
        follows = {k[len("follow:"):]: v for k, v in alts.items() if k.startswith("follow:")}
        best_model = max(follows, key=follows.get) if follows else None
        lessons = [f"{period} {start.isoformat()} to {end.isoformat()}: actual {_pct(actual)}, oracle {_pct(oracle)}, "
                   f"regret {_pct(regret)}" + (f", captured {100 * captured:.0f}% of what was available" if captured is not None else "")]
        if best_alt is not None and best_alt["return"] > actual:
            lessons.append(f"'{best_alt['name']}' would have done better: {_pct(best_alt['return'])} vs actual {_pct(actual)}")
        elif best_alt is not None:
            lessons.append(f"the actual plan beat every simple alternative (best: '{best_alt['name']}' {_pct(best_alt['return'])})")
        if best_model is not None:
            lessons.append(f"most useful input this {period}: {best_model} ({_pct(follows[best_model])} if followed alone)")
        losers = [r for r in per_name if r["contribution"] < 0][:3]
        for r in losers:
            why = ""
            if r.get("voted_down"):
                why = f"; {', '.join(r['voted_down'][:4])} had voted down"
            lessons.append(f"mistake: {r['ticker']} {r.get('action', '')} cost {_pct(r['contribution'])} (move {_pct(r['move'])}{why})")
        missed = [r for r in per_name if r.get("oracle_exposure", 0) > 0 and r.get("exposure", r.get("avg_weight", 0)) < 1e-6]
        missed.sort(key=lambda r: -r["move"])
        if missed:
            lessons.append("missed: " + ", ".join(f"{r['ticker']} {_pct(r['move'])}" for r in missed[:5]) + " (not held)")
        lessons.extend(more_lessons or [])
        result = {"period": period, "start": start.isoformat(), "end": end.isoformat(), "generated": datetime.now(timezone.utc).isoformat(),
                  "actual_return": actual, "oracle_return": oracle, "regret": regret, "captured": captured, "alternatives": alts,
                  "ranking": ranking, "best_alternative": best_alt, "best_model": best_model, "per_name": per_name, "lessons": lessons, **extra}
        result["file"] = str(self._save(period, end, result))
        for line in lessons:
            log.info("review: %s", line)
        return result

    # ------------------------------------------------------------------ scheduling
    def last_review_end(self, period: str) -> date | None:
        files = sorted(self.out_dir.glob(f"{period}_*.json")) if self.out_dir.is_dir() else []
        if not files:
            return None
        try:
            return date.fromisoformat(files[-1].stem.split("_", 1)[1])
        except ValueError:
            return None

    def due(self, today: date | None = None) -> list[str]:
        """Which longer reviews are due: week after each trading week, month / year once their last day has passed."""
        today = today or date.today()
        out = []
        last_w = self.last_review_end("week")
        week_end = today - timedelta(days=(today.weekday() - 4) % 7)          # most recent Friday
        if week_end <= today and (last_w is None or last_w < week_end):
            out.append("week")
        first = today.replace(day=1)
        prev_month_end = first - timedelta(days=1)
        last_m = self.last_review_end("month")
        if last_m is None or last_m < prev_month_end:
            out.append("month")
        prev_year_end = date(today.year - 1, 12, 31)
        last_y = self.last_review_end("year")
        if (last_y is None or last_y < prev_year_end) and self.store.load() is not None:
            out.append("year")
        return out

    def run_due(self, today: date | None = None) -> dict[str, dict | None]:
        today = today or date.today()
        results = {}
        for period in self.due(today):
            if period == "week":
                end = today - timedelta(days=(today.weekday() - 4) % 7)
            elif period == "month":
                end = today.replace(day=1) - timedelta(days=1)
            else:
                end = date(today.year - 1, 12, 31)
            try:
                results[period] = self.review_period(period, end)
            except Exception as e:  # noqa: BLE001
                log.warning("%s review failed: %s", period, e)
                results[period] = None
        return results
