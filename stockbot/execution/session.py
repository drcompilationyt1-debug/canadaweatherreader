"""The daily market-hours session: trade at the open, watch the positions, keep training meanwhile.

    stockbot session --mode alpaca --hours 4

1. Wait for the exchange to open (``session.max_wait_minutes`` at most), let the opening auction
   settle (``after_open_minutes``), then run one trading cycle: last night's complete bars ->
   signals -> policy -> orders at live prices.  Every model's up/down vote is recorded.
2. Start the background trainer (``stockbot retrain`` in a subprocess with a time budget): the
   policy keeps learning on the parallel simulators while the market moves; ``best.zip`` only
   changes when the out-of-sample score improves, so a bad training day cannot hurt tomorrow.
3. Every ``snapshot_minutes``: prices, equity, move since the open per ticker versus the votes -
   the log shows what went up and down and who called it.
4. At the end (``hours`` after the open or the close, whichever first): settle the session-horizon
   votes with the last prices, wait for the trainer, write the dashboard and a JSON summary under
   ``session.log_dir``.  The daily horizon is settled by the next session's cycle.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import Config
from ..logging_utils import get_logger
from ..paths import ROOT
from .market_hours import MarketClock, NY
from .runner import TradingRunner

log = get_logger(__name__)


class TradingSession:
    def __init__(self, cfg: Config, mode: str = "alpaca", hours: float | None = None, dry_run: bool = False, offline: bool = False,
                 with_llm: bool = True, train: bool | None = None, clock: MarketClock | None = None,
                 sleep: Callable[[float], None] = time.sleep, runner: TradingRunner | None = None, **overrides):
        self.cfg = cfg
        self.mode = mode
        self.dry_run = dry_run
        self.offline = offline
        self.with_llm = with_llm
        s = dict(cfg.section("session"))
        s.update({k: v for k, v in overrides.items() if v is not None})
        self.hours = float(hours if hours is not None else s.get("hours", 4))
        self.after_open_minutes = float(s.get("after_open_minutes", 5))
        self.max_wait_minutes = float(s.get("max_wait_minutes", 80))
        self.snapshot_minutes = float(s.get("snapshot_minutes", 15))
        self.end_margin_minutes = float(s.get("end_margin_minutes", 15))
        self.train = bool(s.get("train", True)) if train is None else bool(train)
        self.train_minutes = float(s.get("train_minutes", 0) or 0)
        self.train_timesteps = int(s.get("train_timesteps", 2_000_000))
        self.train_n_envs = int(s.get("train_n_envs", 3))
        self.train_seeds = s.get("train_seeds")
        self.reuse_dataset_days = float(s.get("reuse_dataset_days", 7))
        self.log_dir: Path = cfg.path("session.log_dir", "data/paper/sessions")
        self.clock = clock or MarketClock()
        self.sleep = sleep
        self.runner = runner
        self.trainer: subprocess.Popen | None = None
        self.snapshots: list[dict] = []
        self.summary: dict = {}

    # ------------------------------------------------------------------ helpers
    def now(self) -> datetime:
        return self.clock.now()

    def _sleep_until(self, when: datetime, step: float = 30.0) -> None:
        while True:
            remaining = (when - self.now()).total_seconds()
            if remaining <= 0:
                return
            self.sleep(min(step, remaining))

    def _log_file(self, date: str) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self.log_dir / f"session_{date}.jsonl"

    def _append(self, rec: dict) -> None:
        with open(self._log_file(self.summary["date"]), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    # ------------------------------------------------------------------ trainer
    def trainer_command(self, minutes: float) -> list[str]:
        # the split (data.train_end, e.g. rolling:12) and the fee model come from the same config the runner uses
        cmd = [sys.executable, "-m", "stockbot", "retrain", "--max-minutes", f"{minutes:.1f}", "--timesteps", str(self.train_timesteps),
               "--n-envs", str(self.train_n_envs), "--reuse-dataset-days", str(self.reuse_dataset_days)]
        if self.train_seeds:
            cmd += ["--seeds", str(int(self.train_seeds))]
        if self.offline:
            cmd.append("--offline")
        return cmd

    def start_trainer(self, end: datetime) -> None:
        minutes = self.train_minutes or max(1.0, (end - self.now()).total_seconds() / 60.0 - self.end_margin_minutes)
        cmd = self.trainer_command(minutes)
        log.info("background trainer: %s", " ".join(cmd[2:]))
        self.trainer = subprocess.Popen(cmd, cwd=str(ROOT))
        self.summary["trainer"] = {"cmd": cmd[2:], "minutes": round(minutes, 1), "started": self.now().isoformat(timespec="seconds")}

    def finish_trainer(self, grace_minutes: float) -> None:
        if self.trainer is None:
            return
        deadline = time.time() + grace_minutes * 60.0
        while self.trainer.poll() is None and time.time() < deadline:
            self.sleep(10.0)
        if self.trainer.poll() is None:
            log.warning("trainer still running %.0f minutes after the session - terminating it", grace_minutes)
            self.trainer.terminate()
            try:
                self.trainer.wait(60)
            except Exception:  # noqa: BLE001
                self.trainer.kill()
        rc = self.trainer.returncode
        self.summary["trainer"]["returncode"] = rc
        self.summary["trainer"]["finished"] = self.now().isoformat(timespec="seconds")
        log.info("trainer finished with code %s", rc)

    # ------------------------------------------------------------------ snapshots
    def snapshot(self, label: str) -> dict:
        r = self.runner
        assert r is not None
        equity = float(r.broker.equity())
        cash = float(r.broker.cash())
        positions = {t: p.shares for t, p in r.broker.positions().items()}
        rows = {}
        for t in r.frames:
            try:
                px = float(r.price(t))
            except Exception as e:  # noqa: BLE001
                log.debug("price %s: %s", t, e)
                continue
            base = self.summary["open_prices"].get(t)
            move = (px / base - 1.0) if base else None
            votes = r.last_votes.get(t, {})
            from ..feedback.direction import consensus

            rows[t] = {"price": px, "move": move, "consensus": consensus(votes), "held": positions.get(t, 0.0)}
        moves = [v["move"] for v in rows.values() if v["move"] is not None]
        called = [np.sign(v["move"]) == np.sign(v["consensus"]) for v in rows.values()
                  if v["move"] is not None and abs(v["consensus"]) > 1e-9 and abs(v["move"]) > 1e-9]
        snap = {"type": "snapshot", "label": label, "ts": self.now().isoformat(timespec="seconds"), "equity": equity, "cash": cash,
                "n_positions": len(positions), "mean_move": float(np.mean(moves)) if moves else 0.0,
                "consensus_hit_rate": float(np.mean(called)) if called else None, "tickers": rows}
        self.snapshots.append(snap)
        self._append(snap)
        ups = sorted(((v["move"], t) for t, v in rows.items() if v["move"] is not None), reverse=True)
        top = ", ".join(f"{t} {100 * m:+.2f}%" for m, t in ups[:3])
        bottom = ", ".join(f"{t} {100 * m:+.2f}%" for m, t in ups[-3:][::-1]) if len(ups) > 3 else ""
        log.info("[%s] equity %.2f (%+.2f%% today) cash %.0f  %d positions  avg move %+.2f%%  votes right %s  | up: %s | down: %s",
                 label, equity, 100 * (equity / self.summary["equity_open"] - 1.0) if self.summary.get("equity_open") else 0.0, cash,
                 len(positions), 100 * snap["mean_move"], f"{100 * snap['consensus_hit_rate']:.0f}%" if called else "-", top, bottom)
        return snap

    # ------------------------------------------------------------------ main
    def run(self, force: bool = False) -> dict:
        st = self.clock.status()
        log.info("market %s (source %s) - now %s NY", "OPEN" if st.is_open else "closed", st.source, st.now.strftime("%a %Y-%m-%d %H:%M"))
        today = (st.now if st.is_open else st.next_open).strftime("%Y-%m-%d")
        done = self.log_dir / f"session_{today}.json"
        if done.exists() and not force and not self.dry_run:
            self.summary = {"skipped": f"a session already ran on {today} ({done})", "date": today}
            log.info("no session: %s", self.summary["skipped"])
            return self.summary
        if not st.is_open:
            st = self.clock.wait_for_open(self.max_wait_minutes, sleep=self.sleep)
            if not st.is_open:
                self.summary = {"skipped": f"market closed until {st.next_open.isoformat(timespec='minutes')}", "date": st.now.strftime("%Y-%m-%d")}
                log.info("no session: %s", self.summary["skipped"])
                return self.summary
        start = self.now()
        date = start.strftime("%Y-%m-%d")
        end = min(start + timedelta(hours=self.hours), st.next_close - timedelta(minutes=2))
        self.summary = {"date": date, "mode": self.mode, "dry_run": self.dry_run, "start": start.isoformat(timespec="seconds"),
                        "planned_end": end.isoformat(timespec="seconds"), "clock": st.source, "open_prices": {}}
        if st.minutes_since_open < self.after_open_minutes:
            wait = self.after_open_minutes - st.minutes_since_open
            log.info("letting the opening auction settle (%.1f min)", wait)
            self._sleep_until(self.now() + timedelta(minutes=wait))

        if self.runner is None:
            self.runner = TradingRunner(self.cfg, mode=self.mode, offline=self.offline, with_llm=self.with_llm, clock=self.clock)
        r = self.runner
        self.summary["equity_open"] = float(r.broker.equity())
        decisions = r.cycle(dry_run=self.dry_run, refresh=not self.offline)
        self.summary["equity_after_orders"] = float(r.broker.equity())
        for t in r.frames:
            try:
                self.summary["open_prices"][t] = float(r.price(t))
            except Exception as e:  # noqa: BLE001
                log.debug("open price %s: %s", t, e)
        self.summary["decisions"] = {d.ticker: d.action for d in decisions}
        self.summary["orders"] = sum(1 for d in decisions if d.action != "HOLD")
        self.summary["note"] = r.last_cycle_note
        self._append({"type": "decisions", "ts": self.now().isoformat(timespec="seconds"),
                      "decisions": [d.to_dict() for d in decisions], "votes": r.last_votes})
        log.info("%d decisions, %d orders; watching until %s", len(decisions), self.summary["orders"], end.strftime("%H:%M"))

        if self.train:
            try:
                self.start_trainer(end)
            except Exception as e:  # noqa: BLE001
                log.error("could not start the trainer: %s", e)
                self.summary["trainer"] = {"error": str(e)}

        k = 0
        while self.now() < end:
            nxt = min(self.now() + timedelta(minutes=self.snapshot_minutes), end)
            self._sleep_until(nxt)
            k += 1
            try:
                self.snapshot(f"t+{k * self.snapshot_minutes:.0f}m")
            except Exception as e:  # noqa: BLE001
                log.warning("snapshot failed: %s", e)
            if self.trainer is not None and self.trainer.poll() is not None and "returncode" not in self.summary.get("trainer", {}):
                self.summary["trainer"]["returncode"] = self.trainer.returncode
                log.info("trainer finished early with code %s", self.trainer.returncode)

        final = self.snapshot("end")
        settled = 0
        for t, row in final["tickers"].items():
            if self.dry_run:
                continue
            try:
                if r.board.settle(t, "session", row["price"], date, decision_date=None):
                    settled += 1
            except Exception as e:  # noqa: BLE001
                log.debug("settle session %s: %s", t, e)
        self.summary.update({"end": self.now().isoformat(timespec="seconds"), "equity_end": final["equity"],
                             "session_return": final["equity"] / self.summary["equity_open"] - 1.0 if self.summary.get("equity_open") else 0.0,
                             "mean_move": final["mean_move"], "consensus_hit_rate": final["consensus_hit_rate"], "session_votes_settled": settled,
                             "snapshots": len(self.snapshots)})
        r.broker.save()
        if self.trainer is not None:
            self.finish_trainer(self.end_margin_minutes)
        if not self.dry_run:
            try:
                from ..report import build_dashboard

                self.summary["dashboard"] = str(build_dashboard(self.cfg, mode=self.mode))
            except Exception as e:  # noqa: BLE001
                log.warning("dashboard failed: %s", e)
        out = self.log_dir / (f"session_{date}.json" if not self.dry_run else f"session_{date}_dryrun.json")
        out.write_text(json.dumps(self.summary, indent=1, default=str), encoding="utf-8")
        log.info("session done: equity %.2f -> %.2f (%+.2f%%), avg move %+.2f%%, summary %s", self.summary.get("equity_open", 0.0),
                 final["equity"], 100 * self.summary["session_return"], 100 * final["mean_move"], out)
        return self.summary


def session_slot(clock: MarketClock, before_minutes: float, after_minutes: float = 30.0) -> tuple[bool, str]:
    """Should a scheduled run start a session now?  Yes when the market opens within ``before_minutes``
    or opened less than ``after_minutes`` ago (cron jobs start late sometimes) - so of the two
    daylight-saving cron slots only the one near the open runs, and holidays / weekends are skipped."""
    st = clock.status()
    if st.is_open:
        ok = st.minutes_since_open <= after_minutes
        return ok, f"market open for {st.minutes_since_open:.0f} min ({'within' if ok else 'past'} the {after_minutes:.0f} min window)"
    ok = st.minutes_to_open <= before_minutes
    return ok, f"market opens in {st.minutes_to_open:.0f} min ({'within' if ok else 'outside'} the {before_minutes:.0f} min window)"
