"""Daily trading loop: fresh data -> live signals -> policy -> allocation -> orders -> experience log.

The same runner drives paper trading (``PaperBroker``) and live trading (``AlpacaBroker``); the
policy never knows which one it is talking to.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..agent.decide import Decision, decide
from ..agent.policy import PolicyBundle
from ..agent.sizing import conviction_to_exposure, realized_vol, size_exposure
from ..config import Config, env_settings
from ..data.loader import load_universe
from ..feedback.direction import DirectionBoard, consensus, votes_from_vector
from ..feedback.experience import ExperienceStore
from ..logging_utils import get_logger
from ..signals.base import SignalProvider
from ..signals.layout import PORTFOLIO_FEATURES
from ..signals.registry import build_context, providers_for_layout
from .allocator import allocate
from .base import Broker, Order
from .paper import PaperBroker

log = get_logger(__name__)


class TradingRunner:
    def __init__(self, cfg: Config, mode: str = "paper", bundle: PolicyBundle | None = None,
                 frames_loader: Callable[[bool], dict[str, pd.DataFrame]] | None = None,
                 broker: Broker | None = None, offline: bool = False, with_llm: bool = True,
                 allow_closed: bool = False, clock=None):
        self.cfg = cfg
        self.mode = mode
        self.offline = offline
        self.ex = cfg.section("execution")
        # real brokers only get orders while the exchange is open (execution.require_market_open)
        self.require_open = bool(self.ex.get("require_market_open", True)) and not allow_closed
        self.clock = clock
        self.board = DirectionBoard(cfg.path("feedback.direction_file", "data/experience/direction.jsonl"))
        self.last_votes: dict[str, dict[str, dict[str, float]]] = {}
        self.last_cycle_note = ""
        self.env_cfg = env_settings(cfg)
        self.max_position = float(self.ex.get("max_position", 0.25))
        self.deadband = float(self.env_cfg.get("deadband", 0.05))
        self.allow_short = bool(self.env_cfg.get("allow_short", True))
        from .fees import FeeSchedule

        self.fees = FeeSchedule.from_config(cfg)  # moomoo's schedule; None = legacy proportional commission
        # an order too small for the per-order minimums is not worth sending
        self.min_trade_usd = float(self.ex.get("min_trade_usd", 50))
        if self.fees is not None:
            self.min_trade_usd = max(self.min_trade_usd, self.fees.min_trade_usd())
            log.info("fees: %s -> orders below $%.0f are skipped", self.fees.describe(), self.min_trade_usd)
        ckpt = cfg.path("train.checkpoint_dir", "models/policy")
        if bundle is None:
            if not PolicyBundle.exists(ckpt):
                raise FileNotFoundError(f"no trained policy in {ckpt} - run `stockbot train` first")
            bundle = PolicyBundle.load(ckpt, "best" if (ckpt / "best.zip").exists() else "latest")
        self.bundle = bundle
        self.ctx = build_context(cfg, with_llm=with_llm, with_news=True)
        self.providers: list[SignalProvider] = providers_for_layout(cfg, self.ctx, bundle.layout)
        for p in self.providers:
            try:
                p.load_state()
            except Exception as e:  # noqa: BLE001
                log.warning("could not load state for %s: %s", p.name, e)
        self.store = ExperienceStore(cfg.path("feedback.experience_file", "data/experience/trades.jsonl"))
        self.frames_loader = frames_loader or self._default_loader
        self.frames: dict[str, pd.DataFrame] = {}
        self.state_file = cfg.path("execution.state_file", "data/paper/state.json").with_name(f"runner_{mode}.json")
        self.state = self._load_state()
        self.broker = broker or self._make_broker()

    # ------------------------------------------------------------------ setup
    def _default_loader(self, refresh: bool) -> dict[str, pd.DataFrame]:
        d = self.cfg.section("data")
        return load_universe(list(self.cfg.get("universe", [])), d.get("start", "2008-01-01"), None, d.get("interval", "1d"),
                             self.cfg.path("data.cache_dir", "data/cache"), float(d.get("refresh_days", 1)),
                             offline=self.offline, refresh=refresh and not self.offline)

    def _make_broker(self) -> Broker:
        if self.mode in ("alpaca", "live"):
            from .alpaca import AlpacaBroker

            return AlpacaBroker(paper=bool(self.ex.get_path("alpaca.paper", True)), fractional=bool(self.ex.get_path("alpaca.fractional", True)),
                                fees=self.fees)
        if self.mode == "moomoo":
            from .moomoo import MoomooBroker

            mm = self.ex.section("moomoo")
            return MoomooBroker(env=str(mm.get("env", "simulate")), host=str(mm.get("host", "127.0.0.1")), port=int(mm.get("port", 11111)),
                                security_firm=str(mm.get("security_firm", "FUTUCA")), market=str(mm.get("market", "US")),
                                allow_short=bool(mm.get("allow_short", False)), fees=self.fees)
        return PaperBroker(self.cfg.path("execution.state_file", "data/paper/state.json"), self.last_close,
                           float(self.env_cfg.get("initial_cash", 100_000)), float(self.env_cfg.get("commission", 0.0005)),
                           float(self.env_cfg.get("slippage", 0.0005)), self.allow_short, fees=self.fees)

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
        return {"pos_age": {}, "peak_equity": None, "initial_equity": None, "cycles": 0}

    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    def last_close(self, ticker: str) -> float:
        df = self.frames.get(ticker)
        if df is None or len(df) == 0:
            raise KeyError(f"no data for {ticker}")
        return float(df["close"].iloc[-1])

    def market_is_open(self) -> bool:
        """The built-in paper simulator fills at the last close and never needs an open exchange."""
        if isinstance(self.broker, PaperBroker):
            return True
        if self.clock is None:
            from .market_hours import MarketClock

            self.clock = MarketClock()
        return bool(self.clock.is_open())

    def price(self, ticker: str) -> float:
        """Broker quote, falling back to the latest daily close (e.g. no quote subscription)."""
        try:
            p = float(self.broker.price(ticker))
            if p > 0:
                return p
        except Exception as e:  # noqa: BLE001
            log.debug("broker quote for %s unavailable (%s) - using last close", ticker, e)
        return self.last_close(ticker)

    # ------------------------------------------------------------------ signals
    def latest_vectors(self) -> tuple[dict[str, dict[str, np.ndarray | None]], dict[str, str]]:
        frames = self.frames
        self.ctx.extra["frames"] = frames
        vectors: dict[str, dict[str, np.ndarray | None]] = {t: {} for t in frames}
        reasons: dict[str, str] = {}
        for p in self.providers:
            if not p.enabled:
                continue
            ok, why = p.availability()
            reasons[p.name] = why
            if not ok:
                for t in frames:
                    vectors[t][p.name] = None
                continue
            custom_latest = type(p).compute_latest is not SignalProvider.compute_latest
            if p.needs_universe:
                try:
                    arrs = p.compute_history_all(frames)
                except Exception as e:  # noqa: BLE001
                    log.warning("signal %s failed: %s", p.name, e)
                    arrs = {t: None for t in frames}
                for t in frames:
                    a = arrs.get(t)
                    row = None if a is None or len(a) == 0 or np.isnan(a[-1]).any() else a[-1]
                    vectors[t][p.name] = row
            elif p.live_only or custom_latest:
                for t in frames:
                    vectors[t][p.name] = p.safe_latest(t, frames[t])
            else:
                for t in frames:
                    a = p.safe_history(t, frames[t])
                    vectors[t][p.name] = None if a is None or len(a) == 0 or np.isnan(a[-1]).any() else a[-1]
        return vectors, reasons

    def portfolio_state(self, ticker: str, price: float, equity: float) -> tuple[np.ndarray, float]:
        slice_cap = max(equity * self.max_position, 1e-9)
        pos = self.broker.position(ticker)
        exposure = float(np.clip(pos.shares * price / slice_cap, -1.0, 1.0))
        init = self.state.get("initial_equity") or equity
        peak = self.state.get("peak_equity") or equity
        age = int(self.state.get("pos_age", {}).get(ticker, 0))
        vec = np.array([exposure, np.clip(equity / init - 1.0, -1.0, 3.0) * 2.0, (1.0 - equity / max(peak, 1e-9)) * 5.0,
                        min(age / 252.0, 2.0), float(exposure > 0.01), float(exposure < -0.01)], dtype=np.float32)
        assert len(vec) == len(PORTFOLIO_FEATURES)
        return vec, exposure

    # ------------------------------------------------------------------ one trading cycle
    def cycle(self, dry_run: bool = False, refresh: bool = True) -> list[Decision]:
        self.frames = self.frames_loader(refresh)
        tickers = [t for t in self.cfg.get("universe", []) if t in self.frames]
        as_of = max(str(df.index[-1].date()) for df in self.frames.values())
        log.info("=== %s cycle as of %s (%d tickers, broker=%s) ===", self.mode, as_of, len(tickers), self.broker.name)

        for t in tickers:  # settle yesterday's decisions (and direction votes) with today's prices
            try:
                self.store.settle(t, as_of, self.last_close(t))
                self.board.settle(t, "daily", self.last_close(t), as_of)
            except Exception as e:  # noqa: BLE001
                log.debug("settle %s: %s", t, e)

        self.last_cycle_note = ""
        if not dry_run and self.require_open and not self.market_is_open():
            self.last_cycle_note = "market closed - decided only, no orders sent (--allow-closed overrides)"
            log.warning("%s", self.last_cycle_note)
            dry_run = True

        equity = float(self.broker.equity())
        if not self.state.get("initial_equity"):
            self.state["initial_equity"] = equity
        self.state["peak_equity"] = max(float(self.state.get("peak_equity") or equity), equity)

        # position context for the LLM trader block (nofx-style: exposure, PnL, peak PnL, holding days)
        positions = {}
        peaks = self.state.setdefault("peak_pnl", {})
        for t in tickers:
            pos = self.broker.position(t)
            if abs(pos.shares) < 1e-9 or pos.avg_price <= 0:
                peaks.pop(t, None)
                continue
            price = self.price(t)
            pnl = (price / pos.avg_price - 1.0) * 100.0 * (1.0 if pos.shares > 0 else -1.0)
            peaks[t] = max(float(peaks.get(t, pnl)), pnl)
            positions[t] = {"exposure": float(np.clip(pos.shares * price / max(equity * self.max_position, 1e-9), -1, 1)),
                            "pnl_pct": pnl, "peak_pnl_pct": peaks[t], "days": int(self.state.get("pos_age", {}).get(t, 0))}
        self.ctx.extra["positions"] = positions

        vectors, reasons = self.latest_vectors()
        targets, obs_by, avail_by = {}, {}, {}
        for t in tickers:
            price = self.price(t)
            sig = self.bundle.layout.assemble_latest(vectors[t])
            port, _ = self.portfolio_state(t, price, equity)
            obs = np.concatenate([sig, port]).astype(np.float32)
            obs_by[t] = obs
            avail_by[t] = self.bundle.layout.availability_of(sig)
            conviction = conviction_to_exposure(self.bundle.predict(obs), self.allow_short)
            vol_target = float(self.env_cfg.get("vol_target", 0.0) or 0.0)
            targets[t] = size_exposure(conviction, realized_vol(self.frames[t]["close"].to_numpy(float), int(self.env_cfg.get("vol_window", 20))),
                                       vol_target, float(self.env_cfg.get("max_leverage", 1.0)), float(self.env_cfg.get("vol_max_scale", 1.5))) \
                if vol_target > 0 else conviction
        on = sorted(k for k, v in (avail_by[tickers[0]].items() if tickers else []) if v)
        off = sorted(k for k, v in (avail_by[tickers[0]].items() if tickers else []) if not v)
        log.info("signals ON: %s | OFF: %s", ", ".join(on) or "-", ", ".join(off) or "-")

        allow_short = self.allow_short and self.broker.supports_short
        weights = allocate(targets, self.max_position, float(self.ex.get("max_gross_exposure", 1.0)), allow_short)
        decisions: list[Decision] = []
        for t in tickers:
            price = self.price(t)
            slice_cap = equity * self.max_position
            _, current = self.portfolio_state(t, price, equity)
            target = weights[t] / self.max_position if self.max_position > 0 else 0.0
            dec = decide(t, target, current, slice_cap, price, self.deadband, allow_short, self.min_trade_usd)
            fills = []
            fees_paid = 0.0
            if dec.action != "HOLD" and not dry_run:
                try:
                    fill = self.broker.submit(Order(t, "buy" if dec.shares > 0 else "sell", abs(dec.shares), note=dec.action))
                except Exception as e:  # noqa: BLE001 - one rejected order must not abort the cycle
                    log.error("order for %s failed: %s", t, e)
                    fill = None
                    dec.note = f"order failed: {str(e)[:120]}"
                if fill is not None:
                    fills.append(fill.to_dict())
                    fees_paid = float(fill.cost)
                    dec.note = f"filled {fill.qty:.3f} @ {fill.price:.2f} fees {fill.cost:.2f}"
                    self.state["fees_paid"] = float(self.state.get("fees_paid", 0.0)) + fees_paid
            new_exp = self.portfolio_state(t, price, equity)[1]
            ages = self.state.setdefault("pos_age", {})
            ages[t] = ages.get(t, 0) + 1 if abs(new_exp) > 0.01 and np.sign(new_exp) == np.sign(current if abs(current) > 0.01 else new_exp) else (1 if abs(new_exp) > 0.01 else 0)
            decisions.append(dec)
            votes = votes_from_vector(self.bundle.layout, obs_by[t][: self.bundle.layout.signal_dim],
                                      dec.target_exposure - dec.current_exposure, self.deadband)
            self.last_votes[t] = votes
            log.info("%s  | up/down votes: consensus %+.2f (%s)", dec, consensus(votes),
                     " ".join(f"{k}{'+' if v['vote'] > 0 else '-' if v['vote'] < 0 else '0'}" for k, v in votes.items()))
            if not dry_run:
                self.store.record(mode=self.mode, ticker=t, date=as_of, obs=obs_by[t], action=targets[t],
                                  target_exposure=dec.target_exposure, weight=weights[t], decision=dec.action,
                                  price=price, equity=equity, availability=avail_by[t], fills=fills, fees=fees_paid)
                self.board.record(ticker=t, date=as_of, price=price, votes=votes, mode=self.mode)
        if isinstance(self.broker, PaperBroker):
            self.broker.mark(as_of)
        self.broker.save()
        self.state["cycles"] = int(self.state.get("cycles", 0)) + 1
        self.state["last_cycle"] = datetime.now(timezone.utc).isoformat()
        if not dry_run:
            self._save_state()
        log.info("equity %.2f  cash %.2f  gross exposure %.2f", self.broker.equity(), self.broker.cash(),
                 sum(abs(w) for w in weights.values()))
        return decisions

    def loop(self, every_minutes: float, dry_run: bool = False) -> None:
        while True:
            try:
                self.cycle(dry_run=dry_run)
            except Exception as e:  # noqa: BLE001
                log.exception("cycle failed: %s", e)
            time.sleep(max(60.0, every_minutes * 60.0))
