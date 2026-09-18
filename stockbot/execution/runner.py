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
        if bool(rk.get("adaptive", True)):                          # the weekend tuner's blend, when it passed its guard
            from .ranking import load_tuned_inputs

            self.rank_inputs = load_tuned_inputs(cfg.path("models_dir", "models"), self.rank_inputs)
        self.rank_floor = float(rk.get("policy_floor", 0.5))
        self.rank_veto = float(rk.get("policy_veto", 0.05))
        self.max_per_sector = int(rk.get("max_per_sector", 0) or 0)
        tu = dict(rk.get("top_up", {}) or {})       # between rebalances: buy held names back up to their slot while cash sits idle
        self.top_up = bool(tu.get("enabled", True))
        self.top_up_min_cash = float(tu.get("min_cash", 0.02))      # only when this much of the book's equity is idle above the reserve
        self.top_up_band = float(tu.get("band", 0.15))              # only names this far below their slot
        self.top_up_max_names = int(tu.get("max_names", 5))         # at most this many orders a day (each one pays a fee)
        dl = dict(rk.get("deals", {}) or {})                                # the opportunistic layer between rebalances (off unless the evidence says so)
        self.rank_deals = {k: dl[k] for k in ("enter_pct", "exit_pct", "max_swaps", "min_gap") if k in dl} if bool(dl.get("enabled", False)) else None
        vt = dict(rk.get("vol_target", {}) or {})
        self.vol_target = float(vt.get("target", 0.0) or 0.0) if bool(vt.get("enabled", False)) else 0.0
        self.vol_window = int(vt.get("window", 20))
        self.vol_floor = float(vt.get("floor", 0.4))
        self._sectors: dict[str, str] | None = None
        self._vec_cache: dict[str, np.ndarray] = {}                       # provider|ticker|last bar -> vector (one process, one day)
        self.compute_workers = int(cfg.get_path("signals.compute_workers", 1) or 1)
        self.portfolio_agent = None
        if self.rank_enabled and bool((rk.get("portfolio_rl") or {}).get("enabled", False)):
            from ..agent.portfolio_rl import PortfolioAgent

            self.portfolio_agent = PortfolioAgent.load(cfg.path("models_dir", "models"), self.account)
            if self.portfolio_agent is not None:
                log.info("portfolio agent in force for %s: it decides rebalance timing and exposure", self.account)
        self.llm_trader_top_n = int(cfg.get_path("signals.llm_trader.top_n", 0) or 0)
        self.llm_trader_budget = float(cfg.get_path("signals.llm_trader.budget_minutes", 0) or 0)
        from ..signals.pruning import load_block_mask

        self.block_mask = list(load_block_mask(cfg.path("models_dir", "models"))) if bool(cfg.get_path("signals.pruning.enabled", True)) else []
        if self.block_mask:
            log.info("block mask: %s masked (no value for six months)", ", ".join(self.block_mask))
        self.last_rank: dict = {}
        self._rebalanced = False
        # the core sleeve (execution.core): a broad index the model times between min_share and max_share around its
        # baseline share - trimmed when the model's conviction is low (sell high), rebuilt when it is high (buy low, with the
        # proceeds) - re-decided every core_every bars, traded only when the change clears the band; never sold intraday
        core = dict(self.ex.get("core", {}) or {})
        self.core_ticker = (str(core.get("ticker") or "").strip() or None) if float(core.get("share", 0.0) or 0.0) > 0 else None
        self.core_share = float(core.get("share", 0.0) or 0.0) if self.core_ticker else 0.0
        self.core_decide = str(core.get("decide", "model") or "model").lower()
        self.core_min = float(core.get("min_share", self.core_share * 0.5))
        self.core_max = float(core.get("max_share", min(1.0 - float(self.ex.get("cash_reserve", 0.10)), self.core_share * 1.3)))
        self.core_every = every_bars_of(core.get("every_bars", 21))
        self.core_band = float(core.get("band", 0.05))
        self._core_due = False
        if self.rank_enabled and bool(rk.get("adaptive", True)):        # the weekend tuner's structure, when it passed its guard
            from .ranking import load_tuned_profile

            base = {"top_k": self.rank_top_k, "every_bars": self.rank_every, "hysteresis": self.rank_hysteresis, "core_share": self.core_share}
            prof = load_tuned_profile(cfg.path("models_dir", "models"), self.account, base)
            if prof:
                self.rank_top_k = int(prof.get("top_k", self.rank_top_k))
                self.rank_every = every_bars_of(prof.get("every_bars", self.rank_every))
                self.rank_hysteresis = int(prof.get("hysteresis", self.rank_hysteresis))
                if "deals" in prof:
                    self.rank_deals = dict(prof["deals"]) if prof["deals"] else None
                share = float(prof.get("core_share", self.core_share))
                if share > 0:
                    self.core_ticker = str(prof.get("core_ticker") or core.get("ticker") or "SPY")
                    self.core_share, self.core_min, self.core_max, self.core_decide = share, share / 2.0, share, "model"
                else:
                    self.core_ticker, self.core_share = None, 0.0
                log.info("rank profile (tuned): top_k %d, every %d bars, hysteresis %d, core %.0f%%", self.rank_top_k, self.rank_every,
                         self.rank_hysteresis, 100 * self.core_share)
        # Faber-style trend filter on the rank slots: to cash while the benchmark sits below its moving average
        tf = dict(rk.get("trend_filter", {}) or {})
        self.trend_filter = tf if bool(tf.get("enabled", False)) else None
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
        # execution timing: sells go at the decision cycle, buys wait until buy_at (ET, "HH:MM"; empty = at once).  With a
        # buy_limit_discount a limit order rests from the cycle on and whatever is unfilled goes market at buy_at
        self.buy_at = str(self.ex.get("buy_at", "") or "").strip()
        self.buy_limit_discount = float(self.ex.get("buy_limit_discount", 0.0) or 0.0)
        self.pending_buys: list[dict] = []
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
        if self.mode == "moomoo" and str(self.ex.get_path("moomoo.market", "ALL")).upper() == "ALL":
            return main                               # one moomoo Canada account trades both markets: no simulated sleeve
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
                    return RoutedBroker({"us": main, **sleeves}, default="us", seeds=self._sleeve_seeds(sleeves))
            return main
        proxied = set()
        if self.mode in ("alpaca", "live"):          # Canadian names with a US listing trade on Alpaca itself (execution.alpaca.proxies)
            proxied = {t for t in (self.ex.get_path("alpaca.proxies", {}) or {}) if t in set(self.cfg.get("universe", []))}
        return RoutedBroker({"us": main, **sleeves}, default="us", seeds=self._sleeve_seeds(sleeves), proxied=proxied)

    @staticmethod
    def _sleeve_seeds(sleeves: dict[str, Broker]) -> dict[str, float]:
        """The simulated sleeves' starting cash: they are funded from the same book as the real broker, not on top of it."""
        return {m: float(b.initial_cash) for m, b in sleeves.items() if hasattr(b, "initial_cash")}

    def _single_broker(self, mode: str, market: str) -> Broker:
        fees = self.fee_book.for_market(market)
        if mode in ("alpaca", "live"):
            from .alpaca import AlpacaBroker, set_proxies

            set_proxies(self.ex.get_path("alpaca.proxies", {}) or {})
            return AlpacaBroker(paper=bool(self.ex.get_path("alpaca.paper", True)), fractional=bool(self.ex.get_path("alpaca.fractional", True)),
                                fees=fees, keys_env=str(self.ex.get_path("alpaca.keys_env", "ALPACA") or "ALPACA"),
                                ledger_file=self.cfg.path("execution.state_file", "data/paper/state.json").with_name("alpaca_ledger.json"),
                                fee_book=self.fee_book)
        if mode == "moomoo":
            from .moomoo import MoomooBroker

            mm = self.ex.section("moomoo")
            return MoomooBroker(env=str(mm.get("env", "simulate")), host=str(mm.get("host", "127.0.0.1")), port=int(mm.get("port", 11111)),
                                security_firm=str(mm.get("security_firm", "FUTUCA")), market=str(mm.get("market", "ALL")),
                                allow_short=bool(mm.get("allow_short", False)), fees=fees, fee_book=self.fee_book)
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
        else:
            if p.live_only or custom_latest:
                def compute(t):
                    return p.safe_latest(t, frames[t])
            else:
                def compute(t):
                    a = p.safe_history(t, frames[t])
                    return None if a is None or len(a) == 0 or np.isnan(a[-1]).any() else a[-1]

            todo = []
            for t in frames:
                key = self._cache_key(p, t, frames[t])
                if key is not None and key in self._vec_cache:
                    vectors[t][p.name] = self._vec_cache[key]
                else:
                    todo.append(t)
            workers = self.compute_workers if (getattr(p, "parallel_ok", True) and p.name not in self.FRESH_EVERY_CYCLE) else 1
            if workers > 1 and len(todo) > 1:                              # per-ticker work in parallel threads (the runner has 4 cores)
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as pool:
                    results = list(pool.map(lambda t: (t, compute(t)), todo))
            else:
                results = [(t, compute(t)) for t in todo]
            for t, v in results:
                vectors[t][p.name] = v
                key = self._cache_key(p, t, frames[t])
                if v is not None and key is not None:                      # a failure is retried next cycle
                    if len(self._vec_cache) > 50_000:
                        self._vec_cache.clear()
                    self._vec_cache[key] = v

    FRESH_EVERY_CYCLE = {"news_llm", "sentiment", "finbert", "llm_trader"}        # they read the news, not the bars

    def _cache_key(self, p: SignalProvider, t: str, df: pd.DataFrame) -> str | None:
        """A ticker's vector is reused within the process while its last bar is unchanged: the pre-open warm-up and every
        cycle of the session see the same daily bars, so the slow per-ticker models run once a day, not once a cycle."""
        if p.name in self.FRESH_EVERY_CYCLE or len(df) == 0:
            return None
        return f"{p.name}|{t}|{df.index[-1]}|{len(df)}"

    def select_agent_tickers(self, vectors: dict[str, dict], top_n: int) -> list[str]:
        """The ``top_n`` tickers where the other models agree most (largest |consensus|, bullish first on ties)."""
        scores = {}
        for t, vecs in vectors.items():
            sig = self.bundle.layout.apply_mask(self.bundle.layout.assemble_latest(vecs), self.block_mask)
            scores[t] = consensus(votes_from_vector(self.bundle.layout, sig))
        return sorted(scores, key=lambda t: (-abs(scores[t]), -scores[t]))[:max(0, top_n)]

    def latest_vectors(self, budget_minutes: float | None = None, skip: set[str] | None = None
                       ) -> tuple[dict[str, dict[str, np.ndarray | None]], dict[str, str]]:
        """Every provider's latest vector per ticker.  The LLM agent frameworks are computed last and
        only for the top-N consensus tickers (``signals.agents``), within a time budget."""
        frames = self.frames
        self.ctx.extra["frames"] = frames
        self.ctx.extra["latest_only"] = True         # a live cycle: an uncached name gets today's window, not five years of history
        vectors: dict[str, dict[str, np.ndarray | None]] = {t: {} for t in frames}
        self.ctx.extra["latest_vectors"] = vectors   # filled as providers run: the ranking head reads the other blocks
        self.ctx.extra.pop("per_block", None)
        reasons: dict[str, str] = {}
        top_n, budget = self.agent_settings()
        if budget_minutes is not None:
            budget = float(budget_minutes)
        agents = self.agent_providers() if top_n > 0 else []
        skip = skip or set()
        deferred = None
        for p in self.providers:
            if not p.enabled or p in agents:
                continue
            if p.name in skip:
                for t in frames:
                    vectors[t][p.name] = None
                continue
            if p.name == "llm_trader" and self.llm_trader_top_n > 0:         # one LLM call per name: only where it can matter, below
                deferred = p
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
        if deferred is not None:
            ok, why = deferred.availability()
            reasons[deferred.name] = why
            if ok:
                held = [t for t in self.ctx.extra.get("positions", {}) or {} if t in frames]
                picks = held + [t for t in self.select_agent_tickers(vectors, self.llm_trader_top_n) if t not in held]
                deadline = time.time() + self.llm_trader_budget * 60.0 if self.llm_trader_budget > 0 else None
                log.info("llm_trader on %d names (%d held + top-%d consensus, budget %.0f min)", len(picks), len(held), self.llm_trader_top_n,
                         self.llm_trader_budget)
                for t in picks:
                    if deadline is not None and time.time() >= deadline:
                        log.warning("llm_trader: budget exhausted before %s", t)
                        break
                    vectors[t][deferred.name] = deferred.safe_latest(t, frames[t])
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
        self.ctx.extra.pop("latest_only", None)
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

    def core_now(self, ticker: str, equity: float) -> float:
        """The core's current weight (fraction of the account's equity)."""
        try:
            pos = self.broker.position(ticker)
            return float(pos.shares * self.last_close(ticker) / equity) if equity > 0 else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def core_target(self, ticker: str, ppo_target: float, equity: float, as_of: str) -> float:
        """The core's target weight.  ``decide: model``: on a core decision day the model's conviction places it between
        min_share (no conviction: trimmed, cash kept for the next dip) and max_share (full conviction: bought back); in
        between, the weight is kept unless it fell below the floor.  ``decide: hold``: bought to share, topped up, never sold."""
        from .ranking import rebalance_due

        now = self.core_now(ticker, equity)
        self._core_due = False
        if self.core_decide != "model":
            return max(self.core_share, now)
        index = self.frames[ticker].index if ticker in self.frames else next(iter(self.frames.values())).index
        due = rebalance_due(index, self.state.get("core_last_date"), as_of, self.core_every) or now < self.core_min - 1e-9
        if not due:
            return now
        frac = float(np.clip(ppo_target, 0.0, 1.0))
        target = self.core_min + (self.core_max - self.core_min) * frac
        self._core_due = True
        if abs(target - now) < self.core_band:
            log.info("core %s: conviction %.2f -> %.0f%%, within %.0f%% of the current %.0f%% - kept", ticker, frac, 100 * target,
                     100 * self.core_band, 100 * now)
            return now
        log.info("core %s: conviction %.2f -> %s to %.0f%% of the book (now %.0f%%)", ticker, frac, "add" if target > now else "trim",
                 100 * target, 100 * now)
        return target

    def trend_on(self, as_of: str) -> bool:
        """The trend filter's state (persisted): off once the benchmark closes below its moving average by the band, on
        again above it by the band.  The bar dated ``as_of`` may be today's partial bar and is not used."""
        tf = self.trend_filter
        if not tf:
            return True
        df = self.frames.get(str(tf.get("benchmark", "SPY")))
        n, band = int(tf.get("sma", 200)), float(tf.get("band", 0.02))
        if df is None or len(df) < n + 2:
            return bool(self.state.get("trend_on", True))
        c = df["close"].astype(float)
        if str(c.index[-1].date()) == as_of and as_of == datetime.now().strftime("%Y-%m-%d"):
            c = c.iloc[:-1]
        close, sma = float(c.iloc[-1]), float(c.iloc[-n:].mean())
        on = bool(self.state.get("trend_on", True))
        if on and close < sma * (1.0 - band):
            on = False
        elif not on and close > sma * (1.0 + band):
            on = True
        if on != bool(self.state.get("trend_on", True)):
            log.warning("trend filter %s: %s %.2f vs %d-day average %.2f", "ON" if on else "OFF", tf.get("benchmark", "SPY"), close, n, sma)
        self.state["trend_on"] = on
        return on

    def _rank_position(self, ticker: str) -> int | None:
        scores = self.last_rank.get("scores") or {}
        if ticker not in scores or not np.isfinite(scores[ticker]):
            return None
        ranked = sorted((t for t, s in scores.items() if np.isfinite(s)), key=lambda t: -scores[t])
        return ranked.index(ticker) + 1

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
        core_t = self.core_ticker if self.core_ticker in tickers else None
        core_w: dict[str, float] = {}
        core_weight = 0.0
        if core_t:
            core_weight = self.core_target(core_t, targets.get(core_t, 0.0), eq_of[core_t], as_of)
            core_w = {core_t: core_weight}
        universe = [t for t in tickers if t != core_t]
        satellite = max(0.0, 1.0 - core_weight - self.cash_reserve)              # what the rank slots share (the core's trims fund dips)
        was_on = bool(self.state.get("trend_on", True))
        if not self.trend_on(as_of):
            self.last_cycle_note = (self.last_cycle_note + "; " if self.last_cycle_note else "") + "trend filter off: rank slots in cash"
            self._rebalanced = True
            self.last_rank = {"scores": {}, "chosen": [], "as_of": as_of, "trend_on": False}
            return {**{t: 0.0 for t in universe}, **core_w}
        index = self.frames["SPY"].index if "SPY" in self.frames else next(iter(self.frames.values())).index
        sig_dim = self.bundle.layout.signal_dim
        scores = rank_scores(self.bundle.layout, {t: obs_by[t][:sig_dim] for t in universe}, self.rank_inputs)
        agent_exposure = None
        due = rebalance_due(index, self.state.get("last_rebalance_date"), as_of, self.rank_every)
        if was_on and self.portfolio_agent is not None:                    # the adopted agent decides timing and exposure
            try:
                from .ranking import bars_since

                since = bars_since(index, self.state.get("last_rebalance_date"), as_of)
                exposure_now = float(sum(abs(current[t]) for t in universe) * self.max_position)
                eq_now = float(self.broker.equity())
                dd = eq_now / float(self.state.get("peak_equity") or eq_now or 1.0) - 1.0
                agent_exposure, agent_reb = self.portfolio_agent.decide(
                    self.frames, scores, None, exposure_now, since, self.rank_every, list(self.state.get("book_returns", [])), dd,
                    float(self.state.get("last_turnover", 0.0)), self.rank_top_k)
                due = bool(agent_reb) or since is None
                log.info("portfolio agent: %s, exposure %.0f%%", "rebalance" if due else "hold", 100 * agent_exposure)
            except Exception as e:  # noqa: BLE001
                log.warning("portfolio agent failed (%s) - the cadence rule decides", e)
                agent_exposure = None
        if was_on and not due:
            kept = {t: current[t] * self.max_position for t in universe}
            if self.rank_deals:                                               # the opportunistic layer, when the evidence switched it on
                from ..agent.backtest import apply_deals

                held_now = [t for t in universe if abs(current[t]) > 0.05]
                new_held = apply_deals(scores, held_now, self.rank_top_k, self.rank_deals, sectors=self._sector_map(universe),
                                       max_per_sector=self.max_per_sector)
                entered, left = [t for t in new_held if t not in held_now], [t for t in held_now if t not in new_held]
                if entered or left:
                    ppo_frac = {t: float(np.clip(targets.get(t, 0.0), 0.0, 1.0)) for t in universe}
                    slots = slot_weights(entered, self.rank_top_k, ppo_frac, self.rank_floor, self.rank_veto)
                    for t in left:
                        kept[t] = 0.0
                    for t in entered:
                        kept[t] = min(slots.get(t, 0.0) * satellite, self.max_position)
                    note = "opportunistic: " + ", ".join(([f"in {', '.join(entered)}"] if entered else []) + ([f"out {', '.join(left)}"] if left else []))
                    self.last_cycle_note = (self.last_cycle_note + "; " if self.last_cycle_note else "") + note
                    log.info("rank core: %s (between rebalances)", note)
                    return {**kept, **core_w}
            topped = self._top_up(kept, current, universe, satellite) if self.top_up else []
            note = f"rank core: next rebalance {self.rank_every} bars after {self.state.get('last_rebalance_date')} - positions kept"
            if topped:
                note += "; idle cash put to work in " + ", ".join(topped)
            self.last_cycle_note = (self.last_cycle_note + "; " if self.last_cycle_note else "") + note
            log.info("rank core: not a rebalance day (every %d bars, last %s) - holding%s", self.rank_every, self.state.get("last_rebalance_date"),
                     f"; topping up {', '.join(topped)}" if topped else "")
            return {**kept, **core_w}
        held = [t for t in universe if abs(current[t]) > 0.05]
        sectors = self._sector_map(universe)
        chosen = select_top(scores, held, self.rank_top_k, self.rank_hysteresis, sectors=sectors, max_per_sector=self.max_per_sector)
        ppo_frac = {t: float(np.clip(targets.get(t, 0.0), 0.0, 1.0)) for t in universe}
        slots = slot_weights(chosen, self.rank_top_k, ppo_frac, self.rank_floor, self.rank_veto)
        scale = 1.0
        if agent_exposure is not None:                                       # the agent's exposure replaces volatility targeting
            scale = float(agent_exposure)
        elif self.vol_target > 0:                                           # volatility targeting on the whole book
            from .ranking import vol_scale

            scale = vol_scale(self.state.get("book_returns", []), self.vol_target, self.vol_window, self.vol_floor, 1.0)
            if scale < 0.999:
                log.info("vol targeting: recent book volatility above %.0f%% - gross exposure scaled to %.0f%%", 100 * self.vol_target, 100 * scale)
        weights = {**{t: min(slots.get(t, 0.0) * satellite * scale, self.max_position) for t in universe}, **core_w}
        self.state["last_turnover"] = float(sum(abs(weights.get(t, 0.0) - current[t] * self.max_position) for t in universe))
        self.last_rank = {"scores": scores, "chosen": chosen, "as_of": as_of, "trend_on": True}
        self._rebalanced = True
        dropped = [t for t in held if t not in chosen]
        log.info("rank core: top-%d by %s -> %s%s", self.rank_top_k, "+".join(self.rank_inputs), ", ".join(chosen),
                 f"; leaving {', '.join(dropped)}" if dropped else "")
        return weights

    def _top_up(self, kept: dict[str, float], current: dict[str, float], universe: list[str], satellite: float) -> list[str]:
        """Bring held names back up to their slot with the cash sitting above the reserve (the backtested rule holds full slots;
        a live book drifts below them after conviction-scaled entries and intraday exits).  Buy only - drift is never sold, and
        at most ``top_up_max_names`` orders a day.  ``kept`` is modified in place; returns the names topped up."""
        equity = float(self.broker.equity()) if self.top_up else 0.0
        if equity <= 0:
            return []
        idle = float(self.broker.cash()) - self.cash_reserve * equity
        if idle < self.top_up_min_cash * equity:
            return []
        target = min(satellite / max(self.rank_top_k, 1), self.max_position)
        held = [t for t in universe if abs(current[t]) > 0.05]
        gaps = sorted(((target - kept[t], t) for t in held if kept[t] < target * (1.0 - self.top_up_band)), reverse=True)
        out: list[str] = []
        for gap, t in gaps:
            if len(out) >= self.top_up_max_names or idle <= 0:
                break
            spend = min(gap * equity, idle)
            if spend < self.min_trade_for(t):
                continue
            kept[t] = kept[t] + spend / equity
            idle -= spend
            out.append(t)
        if out:
            log.info("idle cash %.0f%% of equity above the reserve: topping up %s toward %.2f%% slots", 100 * (float(self.broker.cash()) /
                     equity - self.cash_reserve), ", ".join(out), 100 * target)
        return out

    def _sector_map(self, universe: list[str]) -> dict[str, str] | None:
        """Yahoo sectors for the sector cap (fetched once per process, cached weekly on disk); None without a cap."""
        if self.max_per_sector <= 0:
            return None
        if self._sectors is None:
            try:
                from ..data.sectors import load_sectors

                self._sectors = load_sectors(self.cfg, universe, refresh=not self.offline)
            except Exception as e:  # noqa: BLE001
                log.warning("sectors unavailable (%s) - no sector cap this cycle", e)
                self._sectors = {}
        return self._sectors or None

    def execute_pending_buys(self) -> list[dict]:
        """Send the buys the cycle deferred: cancel a resting limit order first and buy only what is still unfilled, at market,
        within the cash the sleeve still has (the reserve floor holds).  Returns the fills."""
        from .base import Order

        fills: list[dict] = []
        cash_left: dict[str, float] = {}
        for item in self.pending_buys:
            t, shares = item["ticker"], float(item["shares"])
            cancel = getattr(self.broker, "cancel_open", None)
            if item.get("limit") and cancel is not None:
                try:
                    shares = float(cancel(t))
                except Exception as e:  # noqa: BLE001
                    log.warning("%s: could not cancel the resting limit order (%s) - no market order sent", t, e)
                    continue
                if shares <= 0:
                    log.info("%s: the resting limit order filled in full", t)
                    continue
            try:
                price = float(self.price(t))
            except Exception:  # noqa: BLE001
                price = float(item["price"])
            sleeve = self.sleeve_of(t)
            if sleeve not in cash_left:
                cash_left[sleeve] = self.cash_for(t)
            avail = max(0.0, cash_left[sleeve] - self.cash_reserve * self.equity_for(t))
            if shares * price > avail:
                shares = float(int(avail / price)) if self.whole_shares else avail / price
            if shares <= 0 or shares * price < self.min_trade_for(t):
                log.warning("%s: deferred buy skipped - $%.0f cash left in the %s sleeve", t, avail, sleeve)
                continue
            try:
                fill = self.broker.submit(Order(t, "buy", shares, note=item.get("action", "BUY")))
            except Exception as e:  # noqa: BLE001
                log.error("deferred buy for %s failed: %s", t, e)
                continue
            if fill is not None:
                fills.append({**fill.to_dict(), "deferred": True})
                cash_left[sleeve] -= fill.qty * fill.price + fill.cost
                self.state["fees_paid"] = float(self.state.get("fees_paid", 0.0)) + float(fill.cost)
                log.info("%s: deferred buy filled %.3f @ %.2f (fees %.2f)", t, fill.qty, fill.price, fill.cost)
        self.pending_buys = []
        if fills:
            self.broker.save()
        return fills

    # ------------------------------------------------------------------ one trading cycle
    def cycle(self, dry_run: bool = False, refresh: bool = True) -> list[Decision]:
        self.frames = self.frames_loader(refresh)
        tickers = [t for t in self.cfg.get("universe", []) if t in self.frames]
        as_of = max(str(df.index[-1].date()) for df in self.frames.values())
        if getattr(self.broker, "proxied", None) and not self.state.get("proxies_migrated"):
            from .routed import migrate_proxied

            moved = migrate_proxied(self.broker)
            self.state["proxies_migrated"] = True
            if moved:
                log.info("simulated positions closed - these names now trade through their US listings on Alpaca: %s", ", ".join(moved))
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
        eq_of = {t: self.equity_for(t) for t in tickers}   # each ticker is sized against its book (a real sleeve of its own, else the whole book)
        if getattr(self.broker, "seeds", None) and self.state.get("equity_basis") != "book":
            log.info("equity basis is now the whole book (simulated sleeves count only their P&L): baseline %s -> %.2f",
                     self.state.get("initial_equity"), equity)
            self.state.update({"initial_equity": equity, "peak_equity": equity, "equity_basis": "book", "book_returns": []})
            self.state.pop("last_equity", None)
        last_eq = self.state.get("last_equity")
        if last_eq and self.state.get("last_equity_date") != as_of and float(last_eq) > 0:
            book = list(self.state.get("book_returns", []))[-120:] + [equity / float(last_eq) - 1.0]
            self.state["book_returns"] = book
        self.state["last_equity"], self.state["last_equity_date"] = equity, as_of
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
            if abs(pos.shares) < 1.0 and abs(pos.shares * price) < self.min_trade_for(t):   # a fractional remnant, not a position
                peaks.pop(t, None)
                continue
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
            sig = self.bundle.layout.apply_mask(self.bundle.layout.assemble_latest(vectors[t]), self.block_mask)
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
            if self.core_ticker and t == self.core_ticker and self.core_share > 0:      # the core: its own slice and band
                cap = self.core_max if self.core_decide == "model" else self.core_share
                slice_cap = eq_of[t] * cap
                current = self.core_now(t, eq_of[t]) / cap
                target = weights.get(t, 0.0) / cap
                if self.core_decide != "model":
                    target = max(target, current)                                        # hold: never sold
                core_dec = decide(t, target, current, slice_cap, price, self.core_band / cap, allow_short, self.min_trade_for(t))
                planned[t] = (core_dec, price, current)
                continue
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
            if dec.action != "HOLD" and dec.shares > 0 and not dry_run and self.buy_at:
                # a buy waits for buy_at (intraday prices drift down from the morning on average): reserve the cash now,
                # rest a limit order meanwhile when asked, send the market order later
                item = {"ticker": t, "shares": float(dec.shares), "price": float(price), "action": dec.action, "limit": None}
                if self.buy_limit_discount > 0:
                    item["limit"] = float(price) * (1.0 - self.buy_limit_discount)
                    try:
                        self.broker.submit(Order(t, "buy", abs(dec.shares), note=dec.action, limit_price=item["limit"]))
                    except Exception as e:  # noqa: BLE001
                        log.error("limit order for %s failed: %s", t, e)
                        item["limit"] = None
                self.pending_buys.append(item)
                cash_left[self.sleeve_of(t)] -= dec.shares * price
                dec.note = (dec.note + "; " if dec.note else "") + (f"limit {item['limit']:.2f} resting, market at {self.buy_at} if unfilled"
                                                                    if item["limit"] else f"buy scheduled for {self.buy_at}")
                results[t] = (dec, [], 0.0, price, current)
                continue
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
                                  conviction=conv_by.get(t),
                                  chosen=((t in self.last_rank.get("chosen", [])) or t == self.core_ticker) if self.rank_enabled else None,
                                  rank=self._rank_position(t))
                self.board.record(ticker=t, date=as_of, price=price, votes=votes, mode=self.mode)
        if isinstance(self.broker, PaperBroker):
            self.broker.mark(as_of)
        self.broker.save()
        if not dry_run and self.cadence == "weekly":
            self.state["last_rebalance_week"] = self._week_key(as_of)
        if not dry_run and self.rank_enabled and self._rebalanced:
            self.state["last_rebalance_date"] = as_of
        if not dry_run and self._core_due:
            self.state["core_last_date"] = as_of
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
