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

from ..config import Config, account_config, account_names
from ..logging_utils import get_logger
from ..paths import ROOT
from .alpaca import alpaca_keys
from .base import Order
from .fees import FeeBook
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
        # the whole session (wait, watch, review, trainer wind-down) must end within this many minutes of
        # starting: GitHub kills a job at 6 h, so the workflow passes ~300 and everything is planned to fit
        self.deadline_minutes = float(s.get("deadline_minutes", 0) or 0)
        # ... or an absolute time (ISO 8601 with offset) computed by the workflow from the job's start, so the
        # learning / warm-up steps before the session can take as long as they like without moving the end
        self.deadline_at = datetime.fromisoformat(str(s["deadline_at"])) if s.get("deadline_at") else None
        self.review_after = bool(s.get("review_after", True))
        # before the open: learn from the finished days / weeks not learned yet (hindsight fine-tune) in a
        # background process while the LLM agents warm up; it must end learn_margin_minutes before the open
        self.learn_before_open = bool(s.get("learn_before_open", True))
        self.learn_margin_minutes = float(s.get("learn_margin_minutes", 8))
        # during the watch window: sell a held name when the intraday exit model (fitted on the broker's 15-minute
        # bars, `stockbot intraday-fit`) says the close will most likely be lower than where we could sell now
        ie = dict(s.get("intraday_exit", {}) or {})
        self.exit_enabled = bool(ie.get("enabled", True))
        self.exit_min_prob = float(ie.get("min_prob", 0.6))
        self.exit_min_gain = float(ie.get("min_gain", 0.005))
        self.exit_stop_loss = float(ie.get("stop_loss", 0.02))
        self.exit_min_auc = float(ie.get("min_auc", 0.55))
        self.exit_min_edge = float(ie.get("min_edge", 0.0005))
        self.exit_model_dir = cfg.path("session.intraday_exit.model_dir", "models/intraday_exit")
        self.exit_model = None
        self.exit_same_day = bool(ie.get("same_day", True))
        self.exited: dict[str, set[str]] = {}
        self.bought_today: dict[str, set[str]] = {}
        # extra accounts (``accounts:`` in the config) are traded from the same signals with their own rules and books
        self.accounts_enabled = bool(s.get("accounts", True))
        self.extra: dict[str, TradingRunner] = {}
        self.account_cfg: dict[str, Config] = {}
        self.account_dirs: dict[str, Path] = {}
        self.account_snaps: dict[str, list[dict]] = {}
        self.fees = FeeBook.from_config(cfg)
        self.clock = clock or MarketClock()
        self.started_at = self.now()
        self.sleep = sleep
        self.runner = runner
        self.trainer: subprocess.Popen | None = None
        self.learner: subprocess.Popen | None = None
        self.learn_info: dict = {}
        self.snapshots: list[dict] = []
        self.summary: dict = {}
        self.prewarmed: dict = {}

    # ------------------------------------------------------------------ helpers
    def now(self) -> datetime:
        return self.clock.now()

    def _sleep_until(self, when: datetime, step: float = 30.0) -> None:
        while True:
            remaining = (when - self.now()).total_seconds()
            if remaining <= 0:
                return
            self.sleep(min(step, remaining))

    def _log_file(self, date: str, account: str | None = None) -> Path:
        d = self.account_dirs.get(account, self.log_dir) if account else self.log_dir
        d.mkdir(parents=True, exist_ok=True)
        return d / f"session_{date}.jsonl"

    def _append(self, rec: dict, account: str | None = None) -> None:
        with open(self._log_file(self.summary["date"], account), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    # ------------------------------------------------------------------ accounts
    def _ensure_runners(self) -> None:
        """The main runner, then one runner per extra account (``accounts:``) sharing its models, bars and signals."""
        if self.runner is None:
            self.runner = TradingRunner(self.cfg, mode=self.mode, offline=self.offline, with_llm=self.with_llm, clock=self.clock)
        if not self.accounts_enabled:
            return
        for name in account_names(self.cfg):
            if name in self.extra:
                continue
            try:
                cfg_a = account_config(self.cfg, name)
                mode_a = str(cfg_a.get_path("execution.mode", self.mode) or self.mode)
                if mode_a in ("alpaca", "live"):
                    prefix = str(cfg_a.get_path("execution.alpaca.keys_env", "ALPACA") or "ALPACA")
                    k, sec = alpaca_keys(prefix)
                    if not (k and sec):
                        log.info("account %s: no %s_API_KEY / %s_SECRET_KEY in the environment - skipped", name, prefix, prefix)
                        continue
                r2 = TradingRunner(cfg_a, mode=mode_a, offline=self.offline, with_llm=False, clock=self.clock, shared=self.runner)
                self.extra[name] = r2
                self.account_cfg[name] = cfg_a
                self.account_dirs[name] = cfg_a.path("session.log_dir", f"data/paper/sessions/{name}")
                log.info("account %s: %s, max_position %.2f, max_names %d, cadence %s, %s", name, r2.broker.name, r2.max_position,
                         r2.max_names, r2.cadence, "whole shares" if r2.whole_shares else "fractional shares")
            except Exception as e:  # noqa: BLE001
                log.warning("account %s unavailable: %s", name, e)

    def _finish_accounts(self, date: str) -> None:
        """End of the watch window for every extra account: equity, its own session summary, day review and dashboard."""
        for name, r2 in self.extra.items():
            info = self.summary.setdefault("accounts", {}).setdefault(name, {})
            try:
                eq_end = float(r2.broker.equity())
                r2.broker.save()
                info.update({"equity_end": eq_end, "session_return": eq_end / info["equity_open"] - 1.0 if info.get("equity_open") else 0.0,
                             "exits": len(self.exited.get(name, set()))})
                if self.dry_run:
                    continue
                cfg_a = self.account_cfg[name]
                self.account_dirs[name].mkdir(parents=True, exist_ok=True)
                (self.account_dirs[name] / f"session_{date}.json").write_text(
                    json.dumps({"date": date, "mode": r2.mode, "account": name, **info}, indent=1, default=str), encoding="utf-8")
                if self.review_after:
                    from ..feedback.review import Review

                    rev = Review(cfg_a).review_day(self.now().date())
                    if rev:
                        info["review"] = {k: rev.get(k) for k in ("actual_return", "oracle_return", "regret", "captured", "best_model", "lessons", "file")}
                try:
                    from ..report import build_dashboard

                    info["dashboard"] = str(build_dashboard(cfg_a, mode=r2.mode))
                except Exception as e:  # noqa: BLE001
                    log.warning("account %s dashboard failed: %s", name, e)
                log.info("account %s done: equity %.2f -> %.2f (%+.2f%%), %d exits", name, info.get("equity_open", 0.0), eq_end,
                         100 * info["session_return"], info["exits"])
            except Exception as e:  # noqa: BLE001
                log.warning("account %s wrap-up failed: %s", name, e)

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

    def hard_deadline(self) -> datetime | None:
        """When everything (watch, review, trainer) must be over: the earlier of the relative and the absolute deadline."""
        cands = []
        if self.deadline_minutes > 0:
            cands.append(self.started_at + timedelta(minutes=self.deadline_minutes))
        if self.deadline_at is not None:
            cands.append(self.deadline_at.astimezone(self.started_at.tzinfo) if self.deadline_at.tzinfo else self.deadline_at)
        return min(cands) if cands else None

    def _exit_model_has_edge(self, metrics: dict) -> bool:
        """Act on the exit model only when it proved itself on its held-out days: enough discrimination and the
        moments it flagged were really better sold than held."""
        auc = metrics.get("auc")
        if auc is None or not np.isfinite(float(auc)) or float(auc) < self.exit_min_auc:
            return False
        return float(metrics.get("gain_when_exit") or 0.0) > float(metrics.get("gain_all") or 0.0) + self.exit_min_edge

    def _prev_close(self, ticker: str) -> float | None:
        df = self.runner.frames.get(ticker) if self.runner is not None else None
        if df is None or len(df) < 2:
            return None
        last = df.index[-1]
        return float(df["close"].iloc[-2]) if last.date() == self.now().date() else float(df["close"].iloc[-1])

    def _intraday_exits(self, snap: dict, runner: TradingRunner | None = None, account: str = "main") -> int:
        """Ask the exit model about every held name on the path since the open; sell the ones it flags (per account)."""
        if self.exit_model is None or self.dry_run or self.runner is None:
            return 0
        r = runner or self.runner
        exited = self.exited.setdefault(account, set())
        bought = self.bought_today.get(account, set())
        same_day = self.exit_same_day if account == "main" else \
            bool(self.account_cfg.get(account, self.cfg).get_path("session.intraday_exit.same_day", self.exit_same_day))
        n = 0
        for t, row in (snap.get("tickers") or {}).items():
            shares = float(row.get("held") or 0.0)
            open_px = self.summary.get("open_prices", {}).get(t)
            if shares <= 0 or not open_px or t in exited:
                continue
            if not same_day and t in bought:
                continue                                              # no same-day round trips (cash account / day-trade rules)
            path = [float(open_px)] + [float(s["tickers"][t]["price"]) for s in self.snapshots if t in (s.get("tickers") or {}) and s["tickers"][t].get("price")]
            price = float(row.get("price") or path[-1])
            sched = self.fees.for_ticker(t)
            notional = max(shares * price, 1.0)
            fee_rt = 2.0 * float(sched.cost(shares, price, "sell")) / notional if sched is not None else 0.0
            adv = self.exit_model.advice(np.asarray(path), float(open_px), self._prev_close(t), fee_rt, min_prob=self.exit_min_prob,
                                        min_gain=self.exit_min_gain, stop_loss=self.exit_stop_loss)
            if not adv.get("exit"):
                continue
            try:
                fill = r.broker.submit(Order(t, "sell", shares, note="intraday exit"))
            except Exception as e:  # noqa: BLE001
                log.warning("intraday exit %s failed: %s", t, e)
                continue
            if fill is None:
                continue
            n += 1
            exited.add(t)
            rec = {"type": "exit", "account": account, "ts": self.now().isoformat(timespec="seconds"), "ticker": t, "qty": float(fill.qty), "price": float(fill.price),
                   "fees": float(fill.cost), "prob": adv.get("prob"), "ret_open": adv.get("ret_open"), "ret_max": adv.get("ret_max"),
                   "reason": adv.get("reason")}
            self._append(rec, None if account == "main" else account)
            log.info("intraday exit [%s]: sold %.3f %s @ %.2f (%s: %+.2f%% since the open, high %+.2f%%, p=%.2f)", account, fill.qty, t, fill.price,
                     adv.get("reason"), 100 * float(adv.get("ret_open") or 0.0), 100 * float(adv.get("ret_max") or 0.0), float(adv.get("prob") or 0.0))
        if n:
            try:
                r.broker.save()
            except Exception as e:  # noqa: BLE001
                log.debug("broker save after exits: %s", e)
        return n

    def learn_command(self, minutes: float) -> list[str]:
        cmd = [sys.executable, "-m", "stockbot", "review", "--learn", "--due", "--max-minutes", f"{minutes:.0f}",
               "--before-open-minutes", f"{self.learn_margin_minutes:.0f}"]
        if self.offline:
            cmd.append("--offline")
        return cmd

    def start_learner(self, minutes_to_open: float) -> None:
        minutes = minutes_to_open - self.learn_margin_minutes
        if minutes < 3.0:
            log.info("no time to learn before the open (%.0f min left)", minutes_to_open)
            return
        cmd = self.learn_command(minutes)
        log.info("pre-open learner (up to %.0f min, alongside the warm-up): %s", minutes, " ".join(cmd[2:]))
        self.learner = subprocess.Popen(cmd, cwd=str(ROOT))
        self.learn_info = {"cmd": cmd[2:], "minutes": round(minutes, 1), "started": self.now().isoformat(timespec="seconds"), "t0": time.time()}

    def finish_learner(self) -> None:
        """Wait for the pre-open learner (until the margin before the open), then use what it learned."""
        if self.learner is None:
            return
        st = self.clock.status()
        budget = 0.0 if st.is_open else max(0.0, (st.minutes_to_open - self.learn_margin_minutes) * 60.0)
        killed = False
        try:
            self.learner.wait(timeout=budget + 30.0)
        except subprocess.TimeoutExpired:
            log.warning("pre-open learner still running at the margin - stopping it")
            self.learner.terminate()
            try:
                self.learner.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                self.learner.kill()
            killed = True
        self.learn_info.update({"returncode": self.learner.returncode, "killed": killed,
                                "seconds": round(time.time() - self.learn_info.pop("t0", time.time()), 1)})
        state = self.cfg.path("train.checkpoint_dir", "models/policy") / "hindsight.json"
        try:
            self.learn_info["learned"] = json.loads(state.read_text(encoding="utf-8")).get("learned", {}) if state.exists() else {}
        except Exception:  # noqa: BLE001
            self.learn_info["learned"] = {}
        if self.runner is not None:
            try:
                self.learn_info["reloaded"] = bool(self.runner.reload_policy())
                for r2 in self.extra.values():
                    r2.bundle = self.runner.bundle
            except Exception as e:  # noqa: BLE001
                log.warning("could not reload the policy after learning: %s", e)
                self.learn_info["reloaded"] = False
        log.info("pre-open learner done in %.0fs (code %s%s), policy %s", self.learn_info["seconds"], self.learn_info["returncode"],
                 ", stopped" if killed else "", "reloaded" if self.learn_info.get("reloaded") else "not reloaded")
        self.learner = None

    def start_trainer(self, end: datetime) -> None:
        until = end
        hard = self.hard_deadline()
        if hard is not None:            # the trainer may run past the watch window, up to the job deadline
            until = max(end, hard)
        minutes = self.train_minutes or max(1.0, (until - self.now()).total_seconds() / 60.0 - self.end_margin_minutes)
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
        for name, r2 in self.extra.items():
            try:
                pos2 = {t: p.shares for t, p in r2.broker.positions().items()}
                snap2 = {**snap, "account": name, "equity": float(r2.broker.equity()), "cash": float(r2.broker.cash()), "n_positions": len(pos2),
                         "tickers": {t: {**v, "held": pos2.get(t, 0.0)} for t, v in rows.items()}}
                self.account_snaps.setdefault(name, []).append(snap2)
                self._append(snap2, account=name)
                log.info("[%s] account %s: equity %.2f cash %.0f  %d positions", label, name, snap2["equity"], snap2["cash"], len(pos2))
            except Exception as e:  # noqa: BLE001
                log.warning("account %s snapshot failed: %s", name, e)
        ups = sorted(((v["move"], t) for t, v in rows.items() if v["move"] is not None), reverse=True)
        top = ", ".join(f"{t} {100 * m:+.2f}%" for m, t in ups[:3])
        bottom = ", ".join(f"{t} {100 * m:+.2f}%" for m, t in ups[-3:][::-1]) if len(ups) > 3 else ""
        log.info("[%s] equity %.2f (%+.2f%% today) cash %.0f  %d positions  avg move %+.2f%%  votes right %s  | up: %s | down: %s",
                 label, equity, 100 * (equity / self.summary["equity_open"] - 1.0) if self.summary.get("equity_open") else 0.0, cash,
                 len(positions), 100 * snap["mean_move"], f"{100 * snap['consensus_hit_rate']:.0f}%" if called else "-", top, bottom)
        return snap

    def _prewarm(self, minutes_to_open: float) -> None:
        """Run the LLM agent frameworks on the top-N consensus tickers before the open, so the
        cycle at the open finds their answers cached and the orders are not delayed."""
        try:
            self._ensure_runners()
            if not self.runner.has_agent_frameworks():
                return
            budget = max(1.0, min(self.runner.agent_settings()[1], minutes_to_open - 3.0))
            log.info("pre-open: computing the agent frameworks (up to %.0f min before the open)", budget)
            t0 = time.time()
            tickers = self.runner.prewarm_agents(refresh=not self.offline, budget_minutes=budget)
            self.prewarmed = {"tickers": tickers, "minutes": round((time.time() - t0) / 60.0, 1)}
            log.info("pre-open done in %.1f min: %s", self.prewarmed["minutes"], tickers)
        except Exception as e:  # noqa: BLE001 - never let the warm-up stop the session
            log.warning("pre-open agent warm-up failed: %s", e)

    # ------------------------------------------------------------------ main
    def run(self, force: bool = False) -> dict:
        self.started_at = self.now()
        st = self.clock.status()
        log.info("market %s (source %s) - now %s NY", "OPEN" if st.is_open else "closed", st.source, st.now.strftime("%a %Y-%m-%d %H:%M"))
        today = (st.now if st.is_open else st.next_open).strftime("%Y-%m-%d")
        done = self.log_dir / f"session_{today}.json"
        if done.exists() and not force and not self.dry_run:
            self.summary = {"skipped": f"a session already ran on {today} ({done})", "date": today}
            log.info("no session: %s", self.summary["skipped"])
            return self.summary
        if not st.is_open:
            if st.minutes_to_open <= self.max_wait_minutes:
                if self.learn_before_open and not self.dry_run:
                    self.start_learner(st.minutes_to_open)              # learns yesterday (and any missed unit) ...
                self._prewarm(st.minutes_to_open)                       # ... while the LLM agents warm up
                self.finish_learner()
            st = self.clock.wait_for_open(self.max_wait_minutes, sleep=self.sleep)
            if not st.is_open:
                self.summary = {"skipped": f"market closed until {st.next_open.isoformat(timespec='minutes')}", "date": st.now.strftime("%Y-%m-%d")}
                log.info("no session: %s", self.summary["skipped"])
                return self.summary
        start = self.now()
        date = start.strftime("%Y-%m-%d")
        end = min(start + timedelta(hours=self.hours), st.next_close - timedelta(minutes=2))
        hard_deadline = self.hard_deadline()
        if hard_deadline is not None and end > hard_deadline - timedelta(minutes=self.end_margin_minutes):
            end = hard_deadline - timedelta(minutes=self.end_margin_minutes)
            log.warning("watch window shortened to %s to respect the job deadline", end.strftime("%H:%M"))
        self.summary = {"date": date, "mode": self.mode, "dry_run": self.dry_run, "start": start.isoformat(timespec="seconds"),
                        "planned_end": end.isoformat(timespec="seconds"), "clock": st.source, "open_prices": {}}
        if self.learn_info:
            self.summary["learn"] = {k: v for k, v in self.learn_info.items() if k != "t0"}
        if st.minutes_since_open < self.after_open_minutes:
            wait = self.after_open_minutes - st.minutes_since_open
            log.info("letting the opening auction settle (%.1f min)", wait)
            self._sleep_until(self.now() + timedelta(minutes=wait))

        self._ensure_runners()
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
        self.bought_today["main"] = {d.ticker for d in decisions if d.action == "BUY"}
        self.summary["orders"] = sum(1 for d in decisions if d.action != "HOLD")
        self.summary["note"] = r.last_cycle_note
        self.summary["agent_tickers"] = list(r.agent_tickers)
        if self.prewarmed:
            self.summary["prewarm"] = self.prewarmed
        self._append({"type": "decisions", "ts": self.now().isoformat(timespec="seconds"),
                      "decisions": [d.to_dict() for d in decisions], "votes": r.last_votes})
        log.info("%d decisions, %d orders; watching until %s", len(decisions), self.summary["orders"], end.strftime("%H:%M"))
        for name, r2 in self.extra.items():                       # the other accounts: same signals, their own books and rules
            info: dict = {"broker": r2.broker.name}
            self.summary.setdefault("accounts", {})[name] = info
            try:
                info["equity_open"] = float(r2.broker.equity())
                decs = r2.cycle(dry_run=self.dry_run, refresh=False)
                info.update({"equity_after_orders": float(r2.broker.equity()), "decisions": {d.ticker: d.action for d in decs},
                             "orders": sum(1 for d in decs if d.action != "HOLD"), "note": r2.last_cycle_note})
                self.bought_today[name] = {d.ticker for d in decs if d.action == "BUY"}
                self._append({"type": "decisions", "ts": self.now().isoformat(timespec="seconds"), "decisions": [d.to_dict() for d in decs],
                              "votes": r2.last_votes}, account=name)
                log.info("account %s: %d decisions, %d orders (equity %.2f)", name, len(decs), info["orders"], info["equity_open"])
            except Exception as e:  # noqa: BLE001
                log.error("account %s cycle failed: %s", name, e)
                info["error"] = str(e)[:200]

        if self.train:
            try:
                self.start_trainer(end)
            except Exception as e:  # noqa: BLE001
                log.error("could not start the trainer: %s", e)
                self.summary["trainer"] = {"error": str(e)}

        if self.exit_enabled and not self.dry_run and self.exit_model is None:
            try:
                from ..feedback.intraday import ExitModel

                self.exit_model = ExitModel.load(self.exit_model_dir)
            except Exception as e:  # noqa: BLE001
                log.warning("intraday exit model unavailable: %s", e)
                self.exit_model = None
            m = (self.exit_model.meta.get("metrics") or {}) if self.exit_model is not None else {}
            if self.exit_model is not None and not self._exit_model_has_edge(m):
                log.info("intraday exit model: OFF - no edge on its held-out days (auc %.2f, exits worth %+.2f%% vs %+.2f%% for holding)",
                         float(m.get("auc") or 0.0), 100 * float(m.get("gain_when_exit") or 0.0), 100 * float(m.get("gain_all") or 0.0))
                self.exit_model = None
            else:
                log.info("intraday exit model: %s", f"on (holdout auc {m.get('auc', float('nan')):.2f}, p>={self.exit_min_prob:.2f}, gain>={100 * self.exit_min_gain:.1f}%)"
                         if self.exit_model is not None else "none fitted yet (stockbot intraday-fit)")
        k = 0
        while self.now() < end:
            nxt = min(self.now() + timedelta(minutes=self.snapshot_minutes), end)
            self._sleep_until(nxt)
            k += 1
            snap = None
            try:
                snap = self.snapshot(f"t+{k * self.snapshot_minutes:.0f}m")
            except Exception as e:  # noqa: BLE001
                log.warning("snapshot failed: %s", e)
            if snap is not None:
                try:
                    self._intraday_exits(snap)
                    for name, r2 in self.extra.items():
                        if self.account_snaps.get(name):
                            self._intraday_exits(self.account_snaps[name][-1], r2, name)
                except Exception as e:  # noqa: BLE001
                    log.warning("intraday exit check failed: %s", e)
            if self.trainer is not None and self.trainer.poll() is not None and "returncode" not in self.summary.get("trainer", {}):
                self.summary["trainer"]["returncode"] = self.trainer.returncode
                log.info("trainer finished early with code %s", self.trainer.returncode)

        final = self.snapshot("end")
        self.summary["exits"] = sum(len(v) for v in self.exited.values())
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
        out = self.log_dir / (f"session_{date}.json" if not self.dry_run else f"session_{date}_dryrun.json")
        out.write_text(json.dumps(self.summary, indent=1, default=str), encoding="utf-8")   # the day review reads equity_open / mode from it
        self._finish_accounts(date)
        if self.review_after and not self.dry_run:
            try:
                from ..feedback.review import Review

                rev = Review(self.cfg).review_day(self.now().date())
                if rev:
                    self.summary["review"] = {k: rev.get(k) for k in ("actual_return", "oracle_return", "regret", "captured", "best_model", "lessons", "file")}
            except Exception as e:  # noqa: BLE001
                log.warning("daily review failed: %s", e)
        if self.trainer is not None:
            grace = self.end_margin_minutes
            hard = self.hard_deadline()
            if hard is not None:
                grace = max(1.0, (hard - self.now()).total_seconds() / 60.0)
            self.finish_trainer(grace)
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
