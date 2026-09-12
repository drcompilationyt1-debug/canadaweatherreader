"""StockBot - a self-improving long-term stock trading agent.

Architecture (see README.md):

    data/        -> OHLCV loading (yfinance + parquet cache, synthetic data for tests)
    features/    -> technical indicators, candlestick patterns + pattern statistics, trend
    signals/     -> every "input model" is a SignalProvider that emits a fixed-size block of
                    features plus an availability mask (0 when the provider cannot run)
    llm/         -> swappable LLM backends (Claude / OpenAI-compatible / Ollama) behind a router
    news/        -> news fetching + on-disk cache
    env/         -> gymnasium trading environment (parallel simulators) + portfolio accounting
    agent/       -> PPO/SAC policy training, evaluation, decision mapping
    execution/   -> paper broker, Alpaca live broker, daily trading runner
    feedback/    -> experience store of paper / live trades used for retraining
    backtest/    -> backtrader cross-check of the trained policy
"""

__version__ = "0.1.0"
