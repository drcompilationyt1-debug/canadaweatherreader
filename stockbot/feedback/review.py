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

    # ------------------------------------------------------------------ helpers
    def _fee_frac(self, ticker: str, notional: float, price: float, side: str = "buy") -> float:
        """Fees as a fraction of the notional for a trade of that size (0 when the fee book is empty)."""
        sched = self.fees.for_ticker(ticker)
        if sched is None or notional <= 0 or price <= 0:
            return 0.0
        return float(sched.cost(notional / price, price, side)) / notional

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
        result = self._finish("day", day, day, actual, alts, per_name, extra={"session_file": str(f), "equity_open": equity0,
                                                                                "n_names": len(names)})
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
        tear = self._tearsheet(period, end, actual_daily, alts_daily.get("benchmark"))
        if tear:
            extra["tearsheet"] = str(tear)
        return self._finish(period, start, end, actual, alts, per_name, extra=extra)

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
    def _finish(self, period: str, start: date, end: date, actual: float, alts: dict[str, float], per_name: list[dict], extra: dict) -> dict:
        ranking = self._rank_alternatives(actual, alts)
        oracle = alts.get("oracle")
        regret = (oracle - actual) if oracle is not None else None
        captured = (actual / oracle) if oracle not in (None, 0.0) and oracle > 0 else None
        best_alt = next((r for r in ranking if r["name"] not in ("actual", "oracle", "period_oracle")), None)
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
