"""The direction voters: which feature of which block carries each input's up/down opinion.

Kept in a module without imports so both the scorecard and the reliability block can use it.
"""
from __future__ import annotations

# voter name -> (block, feature): the signed feature whose sign is the block's opinion about the next move
VOTERS: dict[str, tuple[str, str]] = {
    "dl_forecast": ("dl_forecast", "dl_pred"),          # GRU predicted 5-day log return
    "alpha_factors": ("alpha_factors", "af_pred"),      # qlib-lite factor model
    "qlib": ("qlib", "qlib_score"),                     # qlib model score
    "es_agent": ("es_agent", "es_action"),              # evolution-strategy agent: buy / sell / hold
    "dqn_agent": ("dqn_agent", "dqn_action"),           # DQN agent
    "ta_keras": ("ta_keras", "akm_action"),             # akurgat's Keras classifiers
    "llm_trader": ("llm_trader", "nofx_direction"),     # nofx-style LLM decision
    "news_llm": ("news_llm", "llm_direction"),          # LLM news reader: bullish / bearish
    "sentiment": ("sentiment", "vader_mean"),           # VADER headline sentiment
    "candles": ("candles", "direction_score"),          # candlestick patterns weighted by their historical edge
    "talib_candles": ("talib_candles", "direction_score"),
    "strategy_zoo": ("strategy_zoo", "vote"),           # majority of the ported rule-based strategies
    "trend": ("trend", "slope_30"),                     # 30-day regression slope
    "trading_agents": ("trading_agents", "ta_decision"),
    "ai_hedge_fund": ("ai_hedge_fund", "ahf_conviction"),
    "chronos": ("chronos", "chr_ret_5"),                # Chronos-Bolt median 5-day forecast
    "kronos": ("kronos", "kr_ret_5"),                   # Kronos K-line model 5-bar path
    "timesfm": ("timesfm", "tfm_ret_5"),                # TimesFM 5-day forecast
    "finbert": ("finbert", "fb_net"),                   # FinBERT headline sentiment
    "alpha101": ("alpha101", "a101_mean"),              # WorldQuant alphas composite (cross-sectional)
    "supertrend": ("pandas_ta", "pta_supertrend"),      # pandas-ta SuperTrend direction
    "xs_rank": ("xs_rank", "xs_score"),                 # cross-sectional ranking head (relative to the universe)
    "momentum": ("technical", "ret_20"),                # baseline: last 20-day return, no model at all
}
