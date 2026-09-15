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
                 allow_closed: bool = False, clock=None, shared: "TradingRunner | None" = None):
        """``shared`` = another runner whose policy, signal providers, bars and computed signals this one reuses: the
        same market view, another account's book and rules (``accounts:`` in the config)."""
        self.cfg = cfg
        self.mode = mode
        self.offline = offline
        self.ex = cfg.section("execution")
        self.account = str(cfg.get("account") or "main")
        self.shared = shared
        self.last_vectors: dict | None = None
        self.last_reasons: dict = {}
        # a small account's rules (moomoo's per-order minimums make many small orders ruinous)
        self.max_names = int(self.ex.get("max_names", 0) or 0)
        self.name_hysteresis = int(self.ex.get("max_names_hysteresis", 3) or 0)
        self.cadence = str(self.ex.get("cadence", "daily") or "daily").lower()
        self.whole_shares = bool(self.ex.get("whole_shares", False)) or \
            (mode in ("alpaca", "live") and not bool(self.ex.get_path("alpaca.fractional", True)))
        # the rank-core decision layer (execution.rank): top-K by blended cross-sectional rank, periodic rebalance
        from .ranking import DEFAULT_INPUTS, every_bars_of

        rk = dict(self.ex.get("rank", {}) or {})
        self.rank_enabled = bool(rk.get("enabled", False))
        self.rank_top_k = int(rk.get("top_k", 20) or 20)
        self.rank_every = every_bars_of(rk.get("every_bars", 10))
        self.rank_hysteresis = int(rk.get("hysteresis", 3) or 0)
        self.rank_inputs = dict(rk.get("inputs") or DEFAULT_INPUTS)
        self.rank_floor = float(rk.get("policy_floor", 0.5))
        self.rank_veto = float(rk.get("policy_veto", 0.05))
        self.last_rank: dict = {}
        self._rebalanced = False
        # real brokers only get orders while the exchange is open (execution.require_market_open)
        self.require_open = bool(self.ex.get("require_market_open", True)) and not allow_closed
        self.clock = clock
        self.board = DirectionBoard(cfg.path("feedback.direction_file", "data/experience/direction.jsonl"))
        self.last_votes: dict[str, dict[str, dict[str, float]]] = {}
        self.last_cycle_note = ""
        self.agent_tickers: list[str] = []
        self.env_cfg = env_settings(cfg)
        self.max_position = float(self.ex.get("max_position", 0.25))
        self.deadband = float(self.env_cfg.get("deadband", 0.05))
        self.allow_short = bool(self.env_cfg.get("allow_short", True))
        from .fees import FeeBook

        self.fee_book = FeeBook.from_config(cfg)  # one broker schedule per market; None entries = legacy proportional commission
        self.fees = self.fee_book.default
        # an order too small for the per-order minimums is not worth sending (per market)
        self.min_trade_usd = float(self.ex.get("min_trade_usd", 50))
        self.cash_reserve = float(self.ex.get("cash_reserve", 0.10))   # share of own cash never spent (no margin, 10% buffer)
        for market, sched in [("default", self.fees)] + list(self.fee_book.by_market.items()):
            if sched is not None:
                log.info("fees %s: %s -> orders below %.0f are skipped", market, sched.describe(), max(self.min_trade_usd, sched.min_trade_usd()))
        ckpt = cfg.path("train.checkpoint_dir", "models/policy")
        if bundle is None and shared is not None:
            bundle = shared.bundle
        if bundle is None:
            if not PolicyBundle.exists(ckpt):
                raise FileNotFoundError(f"no trained policy in {ckpt} - run `stockbot train` first")
            bundle = PolicyBundle.load(ckpt, "best" if (ckpt / "best.zip").exists() else "latest")
        self.bundle = bundle
        if shared is not None:                      # same models and signals, no second copy in memory
            self.ctx = shared.ctx
            self.providers: list[SignalProvider] = shared.providers
        else:
            self.ctx = build_context(cfg, with_llm=with_llm, with_news=True)
            self.providers = providers_for_layout(cfg, self.ctx, bundle.layout)
            for p in self.providers:
                try:
                    p.load_state()
                except Exception as e:  # noqa: BLE001
                    log.warning("could not load state for %s: %s", p.name, e)
        self.store = ExperienceStore(cfg.path("feedback.experience_file", "data/experience/trades.jsonl"))
        self.frames_loader = frames_loader or ((lambda refresh: shared.frames) if shared is not None else self._default_loader)
        self.frames: dict[str, pd.DataFrame] = {}
        self.state_file = cfg.path("execution.state_file", "data/paper/state.json").with_name(f"runner_{mode}.json")
        self.state = self._load_state()
        self.broker = broker or self._make_broker()
        self._seed_ledgers()

    # ------------------------------------------------------------------ setup
    def _seed_ledgers(self) -> None:
        """A new virtual fee ledger (Alpaca) starts from the moomoo fees already recorded for this account, so the equity
        the strategy sees is net of every fee it has ever paid, not only of those from today on."""
        from .alpaca import AlpacaBroker

        brokers = list(getattr(self.broker, "sleeves", {}).values()) or [self.broker]
        for b in brokers:
            if not isinstance(b, AlpacaBroker) or b.ledger.path is None or b.ledger.path.exists():
                continue
            recs = self.store.load()
            if recs is None or len(recs) == 0:
                continue
            dec = recs[recs["type"] == "decision"]
            if "mode" in dec.columns:
                dec = dec[dec["mode"].isin(["alpaca", "live"])]
            fees = float(pd.to_numeric(dec["fees"], errors="coerce").fillna(0.0).sum()) if "fees" in dec.columns and len(dec) else 0.0
            n = int(dec["fills"].apply(lambda f: len(f) if isinstance(f, list) else 0).sum()) if "fills" in dec.columns and len(dec) else 0
            if fees > 0:
                b.ledger.fees, b.ledger.n_fills = fees, n
                b.ledger.save()
                log.info("virtual fee ledger seeded from the experience store: %.2f over %d fills", fees, n)

    def _default_loader(self, refresh: bool) -> dict[str, pd.DataFrame]:
        d = self.cfg.section("data")
        return load_universe(list(self.cfg.get("universe", [])), d.get("start", "2008-01-01"), None, d.get("interval", "1d"),
                             self.cfg.path("data.cache_dir", "data/cache"), float(d.get("refresh_days", 1)),
                             offline=self.offline, refresh=refresh and not self.offline)

    def _make_broker(self) -> Broker:
        """The broker for ``mode``; when the universe spans markets the main broker cannot trade
        (``execution.routes``), a ``RoutedBroker`` sends those tickers to their own sleeve."""
        from .markets import group_by_market
        from .routed import RoutedBroker

        main = self._single_broker(self.mode, "us")
        routes = {str(m): str(b) for m, b in (self.ex.get("routes") or {}).items()}
        markets = group_by_market(self.cfg.get("universe", []))
        sleeves = {}
        for market, broker_mode in routes.items():
            if market in markets and market != "us" and broker_mode != self.mode:
                sleeves[market] = self._single_broker(broker_mode, market)
        if not sleeves or self.mode == "paper":
            if self.mode == "paper" and len(markets) > 1:   # one simulator per market so each pays its own fees
                sleeves = {m: self._single_broker("paper", m) for m in markets if m != "us"}
                if sleeves:
                    return RoutedBroker({"us": main, **sleeves}, default="us")
            return main
        return RoutedBroker({"us": main, **sleeves}, default="us")

    def _single_broker(self, mode: str, market: str) -> Broker:
        fees = self.fee_book.for_market(market)
        if mode in ("alpaca", "live"):
            from .alpaca import AlpacaBroker

            return AlpacaBroker(paper=bool(self.ex.get_path("alpaca.paper", True)), fractional=bool(self.ex.get_path("alpaca.fractional", True)),
                                fees=fees, keys_env=str(self.ex.get_path("alpaca.keys_env", "ALPACA") or "ALPACA"),
                                ledger_file=self.cfg.path("execution.state_file", "data/paper/state.json").with_name("alpaca_ledger.json"))
        if mode == "moomoo":
            from .moomoo import MoomooBroker

            mm = self.ex.section("moomoo")
            return MoomooBroker(env=str(mm.get("env", "simulate")), host=str(mm.get("host", "127.0.0.1")), port=int(mm.get("port", 11111)),
                                security_firm=str(mm.get("security_firm", "FUTUCA")), market=str(mm.get("market", "US")),
                                allow_short=bool(mm.get("allow_short", False)), fees=fees)
        if market != "us":   # a sleeve keeps its own paper account (own currency, own file)
            sleeve = self.ex.section("sleeves").section(market)
            state_file = self.cfg.path(f"execution.sleeves.{market}.state_file", f"data/paper/state_{market}.json")
            cash = float(sleeve.get("initial_cash", self.env_cfg.get("initial_cash", 100_000)))
        else:
            state_file = self.cfg.path("execution.state_file", "data/paper/state.json")
            cash = float(self.env_cfg.get("initial_cash", 100_000))
        return PaperBroker(state_file, self.last_close, cash, float(self.env_cfg.get("commission", 0.0005)),
                           float(self.env_cfg.get("slippage", 0.0005)), self.allow_short, fees=fees)

    def allocate_by_market(self, targets: dict[str, float], allow_short: bool) -> dict[str, float]:
        """Gross-exposure cap per sleeve: a Canadian book and a US book are separate accounts."""
        from .markets import group_by_market

        gross = float(self.ex.get("max_gross_exposure", 1.0))
        weights: dict[str, float] = {}
        for market, names in group_by_market(targets).items():
            weights.update(allocate({t: targets[t] for t in names}, self.max_position, gross, allow_short))
        return weights

    def equity_for(self, ticker: str) -> float:
        """Equity of the sleeve that trades ``ticker`` (the whole account for a single broker)."""
        eq = getattr(self.broker, "equity_for", None)
        return float(eq(ticker)) if eq is not None else float(self.broker.equity())

    def cash_for(self, ticker: str) -> float:
        """Cash on hand in the sleeve that trades ``ticker``."""
        fn = getattr(self.broker, "cash_for", None)
        return float(fn(ticker)) if fn is not None else float(self.broker.cash())

    def sleeve_of(self, ticker: str) -> str:
        fn = getattr(self.broker, "market_for", None)
        return str(fn(ticker)) if fn is not None else "main"

    def min_trade_for(self, ticker: str) -> float:
        sched = self.fee_book.for_ticker(ticker)
        return max(self.min_trade_usd, sched.min_trade_usd()) if sched is not None else self.min_trade_usd

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
        sleeves = getattr(self.broker, "sleeves", None)
        if isinstance(self.broker, PaperBroker) or (sleeves and all(isinstance(b, PaperBroker) for b in sleeves.values())):
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
    def agent_settings(self) -> tuple[int, float]:
        """(top_n, budget_minutes) for the tier-C agent frameworks (signals.agents)."""
        a = self.cfg.section("signals.agents")
        return int(a.get("top_n", 5) or 0), float(a.get("budget_minutes", 20) or 0)

    def agent_providers(self) -> list[SignalProvider]:
        """The LLM multi-agent frameworks: live only, tier C, run on the top-N consensus tickers."""
        return [p for p in self.providers if p.enabled and p.live_only and p.tier == "C" and p.name in ("trading_agents", "ai_hedge_fund")]

    def has_agent_frameworks(self) -> bool:
        return self.agent_settings()[0] > 0 and any(p.availability()[0] for p in self.agent_providers())

    def _compute_provider(self, p: SignalProvider, frames: dict[str, pd.DataFrame], vectors: dict[str, dict]) -> None:
        custom_latest = type(p).compute_latest is not SignalProvider.compute_latest
        if p.needs_universe:
            try:
                arrs = p.compute_history_all(frames)
            except Exception as e:  # noqa: BLE001
                log.warning("signal %s failed: %s", p.name, e)
                arrs = {t: None for t in frames}
            for t in frames:
                a = arrs.get(t)
                vectors[t][p.name] = None if a is None or len(a) == 0 or np.isnan(a[-1]).any() else a[-1]
        elif p.live_only or custom_latest:
            for t in frames:
                vectors[t][p.name] = p.safe_latest(t, frames[t])
        else:
            for t in frames:
                a = p.safe_history(t, frames[t])
                vectors[t][p.name] = None if a is None or len(a) == 0 or np.isnan(a[-1]).any() else a[-1]

    def select_agent_tickers(self, vectors: dict[str, dict], top_n: int) -> list[str]:
        """The ``top_n`` tickers where the other models agree most (largest |consensus|, bullish first on ties)."""
        scores = {}
        for t, vecs in vectors.items():
            sig = self.bundle.layout.assemble_latest(vecs)
            scores[t] = consensus(votes_from_vector(self.bundle.layout, sig))
        return sorted(scores, key=lambda t: (-abs(scores[t]), -scores[t]))[:max(0, top_n)]

    def latest_vectors(self, budget_minutes: float | None = None, skip: set[str] | None = None
                       ) -> tuple[dict[str, dict[str, np.ndarray | None]], dict[str, str]]:
        """Every provider's latest vector per ticker.  The LLM agent frameworks are computed last and
        only for the top-N consensus tickers (``signals.agents``), within a time budget."""
        frames = self.frames
        self.ctx.extra["frames"] = frames
        vectors: dict[str, dict[str, np.ndarray | None]] = {t: {} for t in frames}
        self.ctx.extra["latest_vectors"] = vectors   # filled as providers run: the ranking head reads the other blocks
        self.ctx.extra.pop("per_block", None)
        reasons: dict[str, str] = {}
        top_n, budget = self.agent_settings()
        if budget_minutes is not None:
            budget = float(budget_minutes)
        agents = self.agent_providers() if top_n > 0 else []
        skip = skip or set()
        for p in self.providers:
            if not p.enabled or p in agents:
                continue
            if p.name in skip:
                for t in frames:
                    vectors[t][p.name] = None
                continue
            ok, why = p.availability()
            reasons[p.name] = why
            if not ok:
                for t in frames:
                    vectors[t][p.name] = None
                continue
            self._compute_provider(p, frames, vectors)
        self.agent_tickers = []
        if agents:
            selected = self.select_agent_tickers(vectors, top_n)
            self.agent_tickers = selected
            deadline = time.time() + budget * 60.0 if budget > 0 else None
            log.info("agent frameworks %s on the top-%d consensus tickers %s (budget %.0f min)",
                     [p.name for p in agents], top_n, selected, budget)
            max_fail = int(self.cfg.get_path("signals.agents.max_failures", 2) or 0)
            alive: dict[str, bool] = {}
            for p in agents:
                ok, why = p.availability()
                reasons[p.name] = why
                alive[p.name] = ok
                for t in frames:
                    vectors[t][p.name] = None
            failures = {p.name: 0 for p in agents}
            stopped: set[str] = set()
            for t in selected:                      # ticker by ticker: one framework's outage cannot starve the other
                for p in agents:
                    if not alive[p.name] or p.name in stopped:
                        continue
                    if deadline is not None and time.time() >= deadline:
                        log.warning("agent %s: budget exhausted before %s", p.name, t)
                        continue
                    v = p.safe_latest(t, frames[t])
                    vectors[t][p.name] = v
                    if v is None:
                        failures[p.name] += 1
                        if max_fail and failures[p.name] >= max_fail:
                            stopped.add(p.name)
                            log.warning("agent %s: %d failures in a row - skipped for the rest of this run", p.name, failures[p.name])
                    else:
                        failures[p.name] = 0
        return vectors, reasons

    def prewarm_agents(self, refresh: bool = True, budget_minutes: float | None = None) -> list[str]:
        """Before the open: run the agent frameworks on the top-N tickers so the cycle finds their answers cached.

        The LLM news / trader blocks are skipped here (they would spend quota twice); the consensus
        that picks the tickers comes from the quantitative blocks."""
        self.frames = self.frames_loader(refresh)
        self.ctx.extra["positions"] = {}
        self.latest_vectors(budget_minutes=budget_minutes, skip={"news_llm", "llm_trader"})
        return list(self.agent_tickers)

    def reload_policy(self) -> bool:
        """Re-read the policy from disk (the pre-open learner may have fine-tuned it); same layout only."""
        ckpt = self.cfg.path("train.checkpoint_dir", "models/policy")
        if not PolicyBundle.exists(ckpt):
            return False
        bundle = PolicyBundle.load(ckpt, "best" if (ckpt / "best.zip").exists() else "latest")
        if bundle.layout.signature() != self.bundle.layout.signature():
            log.warning("policy on disk has another layout (%s vs %s) - keeping the loaded one", bundle.layout.signature(), self.bundle.layout.signature())
            return False
        self.bundle = bundle
        return True

    def portfolio_state(self, ticker: str, price: float, equity: float) -> tuple[np.ndarray, float]:
        slice_cap = max(equity * self.max_position, 1e-9)
        pos = self.broker.position(ticker)
        exposure = float(np.clip(pos.shares * price / slice_cap, -1.0, 1.0))
        init = self.state.get("initial_equity") or equity
        peak = self.state.get("peak_equity") or equity
        age = int(self.state.get("pos_age", {}).get(ticker, 0))
        vec = np.array([exposure, np.clip(equity / init - 1.0, -1.0, 3.0) * 2.0, (1.0 - equity / max(peak, 1e-9)) * 5.0,
                        min(age / 252.0, 2.0), float(exposure > 0.01), float(exposure < -0.01),
                        self.fee_drag(ticker, price, slice_cap)], dtype=np.float32)
        assert len(vec) == len(PORTFOLIO_FEATURES)
        return vec, exposure

    def fee_drag(self, ticker: str, price: float, slice_cap: float) -> float:
        """Round-trip fee of a full slice as a share of the slice, x100 (the same feature the simulator shows the policy)."""
        sched = self.fee_book.for_ticker(ticker)
        if sched is None or price <= 0 or slice_cap <= 0:
            return float(min(2.0 * float(self.env_cfg.get("commission", 0.0)) * 100.0, 2.0))
        return float(min(2.0 * float(sched.cost(slice_cap / price, price, "buy")) / slice_cap * 100.0, 2.0))

    @staticmethod
    def _week_key(as_of: str) -> str:
        y, w, _ = pd.Timestamp(as_of).isocalendar()
        return f"{int(y)}-W{int(w):02d}"

    def apply_account_rules(self, weights: dict[str, float], tickers: list[str], eq_of: dict[str, float], as_of: str) -> dict[str, float]:
        """A small account's rules: with a weekly cadence nothing is rebalanced outside the first session of the week;
        with ``max_names`` only that many names are held (a held name keeps its slot unless it drops more than
        ``max_names_hysteresis`` places below the cut, so the book does not churn on small ranking moves)."""
        current: dict[str, float] = {}
        for t in tickers:
            try:
                current[t] = self.portfolio_state(t, self.last_close(t), eq_of[t])[1]
            except Exception:  # noqa: BLE001
                current[t] = 0.0
        if self.cadence == "weekly" and self.state.get("last_rebalance_week") == self._week_key(as_of):
            self.last_cycle_note = (self.last_cycle_note + "; " if self.last_cycle_note else "") + \
                "weekly cadence: positions kept until the first session of next week"
            log.info("weekly cadence: already rebalanced this week - holding every position")
            return {t: current[t] * self.max_position for t in tickers}
        if self.max_names > 0:
            ranked = sorted(tickers, key=lambda t: (-weights.get(t, 0.0), t))
            rank = {t: i for i, t in enumerate(ranked)}
            keep = sorted([t for t in tickers if abs(current[t]) > 0.05 and weights.get(t, 0.0) > 0
                           and rank[t] < self.max_names + self.name_hysteresis], key=lambda t: rank[t])[: self.max_names]
            for t in ranked:
                if len(keep) >= self.max_names:
                    break
                if t not in keep and weights.get(t, 0.0) > 0:
                    keep.append(t)
            dropped = [t for t in tickers if t not in keep and weights.get(t, 0.0) > 0]
            for t in dropped:
                weights[t] = 0.0
            if dropped:
                log.info("at most %d names: keeping %s; %d others set to zero", self.max_names, ", ".join(keep), len(dropped))
        return weights

    def apply_rank_core(self, targets: dict[str, float], obs_by: dict[str, np.ndarray], tickers: list[str], eq_of: dict[str, float],
                        as_of: str) -> dict[str, float]:
        """Weights (fractions of equity) from the rank-core rule: the top-K names by blended rank each get 1/K of the
        book scaled by the policy's conviction; outside a rebalance day every position is simply kept."""
        from .ranking import rank_scores, rebalance_due, select_top, slot_weights

        current: dict[str, float] = {}
        for t in tickers:
            try:
                current[t] = self.portfolio_state(t, self.last_close(t), eq_of[t])[1]
            except Exception:  # noqa: BLE001
                current[t] = 0.0
        self._rebalanced = False
        index = self.frames["SPY"].index if "SPY" in self.frames else next(iter(self.frames.values())).index
        if not rebalance_due(index, self.state.get("last_rebalance_date"), as_of, self.rank_every):
            self.last_cycle_note = (self.last_cycle_note + "; " if self.last_cycle_note else "") + \
                f"rank core: next rebalance {self.rank_every} bars after {self.state.get('last_rebalance_date')} - positions kept"
            log.info("rank core: not a rebalance day (every %d bars, last %s) - holding", self.rank_every, self.state.get("last_rebalance_date"))
            return {t: current[t] * self.max_position for t in tickers}
        sig_dim = self.bundle.layout.signal_dim
        scores = rank_scores(self.bundle.layout, {t: obs_by[t][:sig_dim] for t in tickers}, self.rank_inputs)
        held = [t for t in tickers if abs(current[t]) > 0.05]
        chosen = select_top(scores, held, self.rank_top_k, self.rank_hysteresis)
        ppo_frac = {t: float(np.clip(targets.get(t, 0.0), 0.0, 1.0)) for t in tickers}
        slots = slot_weights(chosen, self.rank_top_k, ppo_frac, self.rank_floor, self.rank_veto)
        weights = {t: min(slots.get(t, 0.0), self.max_position) for t in tickers}
        self.last_rank = {"scores": scores, "chosen": chosen, "as_of": as_of}
        self._rebalanced = True
        dropped = [t for t in held if t not in chosen]
        log.info("rank core: top-%d by %s -> %s%s", self.rank_top_k, "+".join(self.rank_inputs), ", ".join(chosen),
                 f"; leaving {', '.join(dropped)}" if dropped else "")
        return weights

    # ------------------------------------------------------------------ one trading cycle
    def cycle(self, dry_run: bool = False, refresh: bool = True) -> list[Decision]:
        self.frames = self.frames_loader(refresh)
        tickers = [t for t in self.cfg.get("universe", []) if t in self.frames]
        as_of = max(str(df.index[-1].date()) for df in self.frames.values())
        log.info("=== %s cycle as of %s (%d tickers, broker=%s, account=%s) ===", self.mode, as_of, len(tickers), self.broker.name, self.account)

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
        eq_of = {t: self.equity_for(t) for t in tickers}   # each ticker is sized against its own sleeve
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
            positions[t] = {"exposure": float(np.clip(pos.shares * price / max(eq_of[t] * self.max_position, 1e-9), -1, 1)),
                            "pnl_pct": pnl, "peak_pnl_pct": peaks[t], "days": int(self.state.get("pos_age", {}).get(t, 0))}
        self.ctx.extra["positions"] = positions

        if self.shared is not None and self.shared.last_vectors is not None:
            vectors, reasons = self.shared.last_vectors, self.shared.last_reasons     # the same signals, this account's book
            self.agent_tickers = list(self.shared.agent_tickers)
        else:
            vectors, reasons = self.latest_vectors()
        self.last_vectors, self.last_reasons = vectors, reasons
        targets, obs_by, avail_by, conv_by = {}, {}, {}, {}
        for t in tickers:
            price = self.price(t)
            sig = self.bundle.layout.assemble_latest(vectors[t])
            port, _ = self.portfolio_state(t, price, eq_of[t])
            obs = np.concatenate([sig, port]).astype(np.float32)
            obs_by[t] = obs
            avail_by[t] = self.bundle.layout.availability_of(sig)
            conv_by[t] = self.bundle.predict(obs)
            conviction = conviction_to_exposure(conv_by[t], self.allow_short)
            vol_target = float(self.env_cfg.get("vol_target", 0.0) or 0.0)
            targets[t] = size_exposure(conviction, realized_vol(self.frames[t]["close"].to_numpy(float), int(self.env_cfg.get("vol_window", 20))),
                                       vol_target, float(self.env_cfg.get("max_leverage", 1.0)), float(self.env_cfg.get("vol_max_scale", 1.5))) \
                if vol_target > 0 else conviction
        on = sorted(k for k, v in (avail_by[tickers[0]].items() if tickers else []) if v)
        off = sorted(k for k, v in (avail_by[tickers[0]].items() if tickers else []) if not v)
        log.info("signals ON: %s | OFF: %s", ", ".join(on) or "-", ", ".join(off) or "-")

        allow_short = self.allow_short and self.broker.supports_short
        weights = self.allocate_by_market(targets, allow_short)
        if self.rank_enabled:
            weights = self.apply_rank_core(targets, obs_by, tickers, eq_of, as_of)
        else:
            weights = self.apply_account_rules(weights, tickers, eq_of, as_of)
        decisions: list[Decision] = []
        # decide everything first, then execute sells before buys: the cash on hand is the hard limit for
        # buys, and sale proceeds are not reused in the same cycle (they settle T+1 in a cash account)
        planned: dict[str, tuple[Decision, float, float]] = {}
        for t in tickers:
            price = self.price(t)
            slice_cap = eq_of[t] * self.max_position
            _, current = self.portfolio_state(t, price, eq_of[t])
            target = weights[t] / self.max_position if self.max_position > 0 else 0.0
            dec = decide(t, target, current, slice_cap, price, self.deadband, allow_short, self.min_trade_for(t))
            if self.whole_shares and dec.action != "HOLD":                     # moomoo Canada: whole shares only
                whole = float(int(abs(dec.shares)))
                if whole < 1.0:
                    dec = Decision(t, "HOLD", dec.target_exposure, dec.current_exposure, 0.0, 0.0, 0.0, price, "less than one share")
                else:
                    dec.shares = whole * (1.0 if dec.shares > 0 else -1.0)
                    dec.amount_usd = dec.shares * price
            planned[t] = (dec, price, current)
        cash_left: dict[str, float] = {}
        order = sorted(tickers, key=lambda t: 0 if planned[t][0].shares < 0 else 1)   # sells / covers first
        results: dict[str, tuple[Decision, list, float, float, float]] = {}
        for t in order:
            dec, price, current = planned[t]
            fills = []
            fees_paid = 0.0
            if dec.action != "HOLD" and dec.shares > 0 and not dry_run:
                sleeve = self.sleeve_of(t)
                if sleeve not in cash_left:
                    cash_left[sleeve] = self.cash_for(t)
                # own cash only, and a reserve of cash_reserve x the sleeve's equity always stays untouched (a floor,
                # not a haircut per order: many small buys must not eat the reserve away)
                avail = max(0.0, cash_left[sleeve] - self.cash_reserve * self.equity_for(t))
                notional = dec.shares * price
                if notional > avail:
                    if avail < self.min_trade_for(t):
                        log.warning("%s: %s of $%.0f skipped - only $%.0f cash left in the %s sleeve", t, dec.action, notional, avail, sleeve)
                        dec = Decision(t, "HOLD", dec.target_exposure, dec.current_exposure, 0.0, 0.0, 0.0, price,
                                       f"insufficient cash (${avail:,.0f} left)")
                    else:
                        cut = float(int(avail / price)) if self.whole_shares else avail / price
                        if cut < 1e-9:
                            dec = Decision(t, "HOLD", dec.target_exposure, dec.current_exposure, 0.0, 0.0, 0.0, price,
                                           f"insufficient cash for one share (${avail:,.0f} left)")
                        else:
                            log.info("%s: buy cut from %.3f to %.3f shares to stay within $%.0f cash", t, dec.shares, cut, avail)
                            dec.shares, dec.amount_usd = cut, cut * price
                            dec.note = f"cut to cash (${avail:,.0f} left)"
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
                    dec.note = (dec.note + "; " if dec.note else "") + f"filled {fill.qty:.3f} @ {fill.price:.2f} fees {fill.cost:.2f}"
                    self.state["fees_paid"] = float(self.state.get("fees_paid", 0.0)) + fees_paid
                    if dec.shares > 0:
                        cash_left[self.sleeve_of(t)] -= fill.qty * fill.price + fill.cost
            results[t] = (dec, fills, fees_paid, price, current)
        for t in tickers:
            dec, fills, fees_paid, price, current = results[t]
            new_exp = self.portfolio_state(t, price, eq_of[t])[1]
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
                                  price=price, equity=eq_of[t], availability=avail_by[t], fills=fills, fees=fees_paid,
                                  conviction=conv_by.get(t))
                self.board.record(ticker=t, date=as_of, price=price, votes=votes, mode=self.mode)
        if isinstance(self.broker, PaperBroker):
            self.broker.mark(as_of)
        self.broker.save()
        if not dry_run and self.cadence == "weekly":
            self.state["last_rebalance_week"] = self._week_key(as_of)
        if not dry_run and self.rank_enabled and self._rebalanced:
            self.state["last_rebalance_date"] = as_of
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
