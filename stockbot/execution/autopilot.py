"""Autopilot: keep the bot running unattended.

Every weekday at ``autopilot.trade_time`` (exchange time) it runs one trading cycle and refreshes
the dashboard; every ``autopilot.retrain_every_days`` days it refreshes the data, refits every
sub-model (candlestick statistics, factor model, ...) and continues training the policy.  If no
policy exists yet it trains one first.  Meant for a machine that stays on (a PC, a small VPS) -
Colab sessions are too short for it.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from ..agent.policy import PolicyBundle
from ..config import Config
from ..logging_utils import get_logger

log = get_logger(__name__)


def _tz(name: str):
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 - missing tzdata on Windows -> local time
        log.warning("timezone %s unavailable (pip install tzdata) - using local time", name)
        return None


class Autopilot:
    def __init__(self, cfg: Config, mode: str = "paper", dry_run: bool = False, offline: bool = False):
        self.cfg = cfg
        self.mode = mode
        self.dry_run = dry_run
        self.offline = offline
        ap = cfg.section("autopilot")
        self.trade_time = str(ap.get("trade_time", "16:30"))
        self.tz = _tz(str(ap.get("timezone", "America/New_York")))
        self.retrain_every_days = int(ap.get("retrain_every_days", 30))
        self.retrain_timesteps = int(ap.get("retrain_timesteps", 100_000))
        self.initial_timesteps = int(ap.get("initial_timesteps", 300_000))
        self.report = bool(ap.get("report", True))
        self.state_file: Path = cfg.path("autopilot.state_file", "data/autopilot.json")
        self.state = self._load()

    # ------------------------------------------------------------------ state
    def _load(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
        return {"last_trade_date": None, "last_retrain": None, "cycles": 0, "retrains": 0}

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ scheduling
    def now(self) -> datetime:
        return datetime.now(self.tz) if self.tz else datetime.now()

    def next_trade_time(self, now: datetime | None = None) -> datetime:
        now = now or self.now()
        hh, mm = (int(x) for x in self.trade_time.split(":"))
        candidate = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        while candidate.weekday() >= 5:  # skip Saturday / Sunday
            candidate += timedelta(days=1)
        return candidate

    def retrain_due(self) -> bool:
        last = self.state.get("last_retrain")
        if not last:
            return True
        return (self.now().replace(tzinfo=None) - datetime.fromisoformat(last)).days >= self.retrain_every_days

    # ------------------------------------------------------------------ actions
    def ensure_policy(self) -> None:
        from ..agent.train import train

        ckpt = self.cfg.path("train.checkpoint_dir", "models/policy")
        if PolicyBundle.exists(ckpt):
            return
        log.info("autopilot: no policy yet - initial training (%d steps)", self.initial_timesteps)
        train(self.cfg, total_timesteps=self.initial_timesteps, offline=self.offline)
        self.state["last_retrain"] = self.now().replace(tzinfo=None).isoformat(timespec="seconds")
        self._save()

    def do_retrain(self) -> None:
        from ..agent.train import retrain

        log.info("autopilot: retraining (%d steps, refreshed data, sub-models refit)", self.retrain_timesteps)
        retrain(self.cfg, total_timesteps=self.retrain_timesteps, offline=self.offline)
        self.state["last_retrain"] = self.now().replace(tzinfo=None).isoformat(timespec="seconds")
        self.state["retrains"] = int(self.state.get("retrains", 0)) + 1
        self._save()

    def do_cycle(self) -> None:
        from .runner import TradingRunner

        runner = TradingRunner(self.cfg, mode=self.mode, offline=self.offline)  # fresh: picks up new checkpoints
        runner.cycle(dry_run=self.dry_run, refresh=not self.offline)
        self.state["last_trade_date"] = self.now().strftime("%Y-%m-%d")
        self.state["cycles"] = int(self.state.get("cycles", 0)) + 1
        self._save()
        if self.report:
            try:
                from ..report import build_dashboard

                build_dashboard(self.cfg, mode=self.mode)
            except Exception as e:  # noqa: BLE001
                log.warning("dashboard failed: %s", e)

    # ------------------------------------------------------------------ main loop
    def run(self, run_now: bool = False, once: bool = False) -> None:
        self.ensure_policy()
        first = True
        while True:
            try:
                if self.retrain_due():
                    self.do_retrain()
                if not (first and run_now):
                    target = self.next_trade_time()
                    log.info("autopilot: next trading cycle at %s (%s)", target.strftime("%Y-%m-%d %H:%M"), self.mode)
                    while True:
                        remaining = (target - self.now()).total_seconds()
                        if remaining <= 0:
                            break
                        time.sleep(min(remaining, 300.0))
                self.do_cycle()
            except KeyboardInterrupt:
                log.info("autopilot stopped")
                return
            except Exception as e:  # noqa: BLE001 - never die unattended
                log.exception("autopilot iteration failed: %s", e)
                time.sleep(600.0)
            first = False
            if once:
                return
