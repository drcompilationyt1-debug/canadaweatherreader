"""Builds the ordered list of signal providers (= the observation layout) from the config."""
from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..logging_utils import get_logger
from ..paths import resolve
from .alpha_factors import AlphaFactorSignal
from .base import SignalContext, SignalProvider
from .candles import CandleSignal
from .layout import ObservationLayout
from .market_regime import MarketRegimeSignal
from .news_llm import LLMNewsSignal
from .sentiment import SentimentSignal
from .technical import TechnicalSignal
from .trend import TrendSignal

log = get_logger(__name__)


def _optional(module: str, klass: str) -> type[SignalProvider] | None:
    """Import a third-party adapter; a broken adapter must never take the whole bot down."""
    import importlib

    try:
        return getattr(importlib.import_module(module, __package__), klass)
    except Exception as e:  # noqa: BLE001
        log.warning("adapter %s.%s unavailable: %s", module, klass, e)
        return None


# Order matters: it defines the observation layout.  Add new providers at the end.
PROVIDER_CLASSES: list[type[SignalProvider]] = [k for k in [
    TechnicalSignal, CandleSignal, TrendSignal, AlphaFactorSignal, MarketRegimeSignal,
    _optional(".talib_candles", "TalibCandleSignal"),
    _optional(".thirdparty.trendet_signal", "TrendetSignal"),
    SentimentSignal, LLMNewsSignal,
    _optional(".thirdparty.qlib_signal", "QlibSignal"),
    _optional(".thirdparty.freqtrade_signal", "FreqtradeSignal"),
    _optional(".thirdparty.trading_agents_signal", "TradingAgentsSignal"),
    _optional(".thirdparty.ai_hedge_fund_signal", "AIHedgeFundSignal"),
    _optional(".strategy_zoo", "StrategyZooSignal"),
    _optional(".es_agent", "ESAgentSignal"),
    _optional(".dl_forecast", "DLForecastSignal"),
    _optional(".dqn_agent", "DQNAgentSignal"),
    _optional(".llm_trader", "LLMTraderSignal"),
    _optional(".thirdparty.ta_keras_signal", "TAKerasSignal"),
    # 2026-09-13: seven more third-party inputs (all CPU, all cached)
    _optional(".thirdparty.pandas_ta_signal", "PandasTASignal"),
    _optional(".thirdparty.portfolio_opt_signal", "PortfolioOptSignal"),
    _optional(".thirdparty.alpha101_signal", "Alpha101Signal"),
    _optional(".thirdparty.finbert_signal", "FinBERTSignal"),
    _optional(".thirdparty.chronos_signal", "ChronosSignal"),
    _optional(".thirdparty.kronos_signal", "KronosSignal"),
    _optional(".thirdparty.timesfm_signal", "TimesFMSignal"),
    _optional(".fundamentals", "FundamentalsSignal"),   # statements-based value / growth / quality / size
    _optional(".xs_rank", "XSRankSignal"),               # ranks the universe from all the blocks above
    _optional(".reliability", "ReliabilitySignal"),      # LAST: trailing hit rate of every input (the review's lesson, as an input)
] if k is not None]


def build_context(cfg: Config, with_llm: bool = True, with_news: bool = True, models_dir: str | Path | None = None) -> SignalContext:
    from ..llm.router import LLMRouter
    from ..news.fetcher import NewsFetcher

    mdir = resolve(models_dir) if models_dir else cfg.path("models_dir", "models")
    ctx = SignalContext(cfg=cfg, models_dir=mdir)
    if with_news:
        try:
            ctx.news = NewsFetcher(cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("news fetcher unavailable: %s", e)
    if with_llm:
        try:
            ctx.llm = LLMRouter.from_config(cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("llm router unavailable: %s", e)
    return ctx


def build_providers(cfg: Config, ctx: SignalContext, include_disabled: bool = False) -> list[SignalProvider]:
    providers = []
    for klass in PROVIDER_CLASSES:
        p = klass(cfg, ctx)
        if p.enabled or include_disabled:
            providers.append(p)
    return providers


def build_layout(providers: list[SignalProvider]) -> ObservationLayout:
    return ObservationLayout([(p.name, list(p.feature_names)) for p in providers if p.enabled])


def providers_for_layout(cfg: Config, ctx: SignalContext, layout: ObservationLayout) -> list[SignalProvider]:
    """Instantiate exactly the providers a saved layout expects (for inference with an old model)."""
    by_name = {k.name: k for k in PROVIDER_CLASSES}
    out = []
    for b in layout.blocks:
        klass = by_name.get(b.name)
        if klass is None:
            log.warning("layout block %s has no provider in this version - it will be masked", b.name)
            continue
        p = klass(cfg, ctx)
        if list(p.feature_names) != list(b.feature_names):
            log.warning("provider %s feature list changed since the model was trained - block masked", b.name)
            continue
        out.append(p)
    return out
