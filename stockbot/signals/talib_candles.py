"""TA-Lib's 61 candlestick recognisers as an optional block (``pip install TA-Lib``).

Each recogniser returns +100 (bullish occurrence), -100 (bearish) or 0.  Bullish and bearish
occurrences are treated as separate patterns for the forward-return statistics, and the policy
receives the six statistics-weighted summaries plus the signed flag of every recogniser.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.candles import STAT_FEATURES, PatternStats, pattern_features
from ..logging_utils import get_logger
from ..paths import import_optional
from .base import SignalProvider

log = get_logger(__name__)

TALIB_PATTERNS = [
    "CDL2CROWS", "CDL3BLACKCROWS", "CDL3INSIDE", "CDL3LINESTRIKE", "CDL3OUTSIDE", "CDL3STARSINSOUTH",
    "CDL3WHITESOLDIERS", "CDLABANDONEDBABY", "CDLADVANCEBLOCK", "CDLBELTHOLD", "CDLBREAKAWAY", "CDLCLOSINGMARUBOZU",
    "CDLCONCEALBABYSWALL", "CDLCOUNTERATTACK", "CDLDARKCLOUDCOVER", "CDLDOJI", "CDLDOJISTAR", "CDLDRAGONFLYDOJI",
    "CDLENGULFING", "CDLEVENINGDOJISTAR", "CDLEVENINGSTAR", "CDLGAPSIDESIDEWHITE", "CDLGRAVESTONEDOJI", "CDLHAMMER",
    "CDLHANGINGMAN", "CDLHARAMI", "CDLHARAMICROSS", "CDLHIGHWAVE", "CDLHIKKAKE", "CDLHIKKAKEMOD", "CDLHOMINGPIGEON",
    "CDLIDENTICAL3CROWS", "CDLINNECK", "CDLINVERTEDHAMMER", "CDLKICKING", "CDLKICKINGBYLENGTH", "CDLLADDERBOTTOM",
    "CDLLONGLEGGEDDOJI", "CDLLONGLINE", "CDLMARUBOZU", "CDLMATCHINGLOW", "CDLMATHOLD", "CDLMORNINGDOJISTAR",
    "CDLMORNINGSTAR", "CDLONNECK", "CDLPIERCING", "CDLRICKSHAWMAN", "CDLRISEFALL3METHODS", "CDLSEPARATINGLINES",
    "CDLSHOOTINGSTAR", "CDLSHORTLINE", "CDLSPINNINGTOP", "CDLSTALLEDPATTERN", "CDLSTICKSANDWICH", "CDLTAKURI",
    "CDLTASUKIGAP", "CDLTHRUSTING", "CDLTRISTAR", "CDLUNIQUE3RIVER", "CDLUPSIDEGAP2CROWS", "CDLXSIDEGAP3METHODS",
]
SPLIT_PATTERNS = [f"{n}_bull" for n in TALIB_PATTERNS] + [f"{n}_bear" for n in TALIB_PATTERNS]
SPLIT_DIRECTIONS = {**{f"{n}_bull": 1 for n in TALIB_PATTERNS}, **{f"{n}_bear": -1 for n in TALIB_PATTERNS}}


def talib_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Signed flags (-1 / 0 / +1) for every TA-Lib recogniser, aligned to ``df.index``."""
    import talib

    o, h, l, c = (np.ascontiguousarray(df[k].to_numpy(np.float64)) for k in ("open", "high", "low", "close"))
    out = {}
    for name in TALIB_PATTERNS:
        fn = getattr(talib, name, None)
        if fn is None:
            out[name] = np.zeros(len(df), dtype=np.int8)
            continue
        out[name] = np.sign(fn(o, h, l, c)).astype(np.int8)
    return pd.DataFrame(out, index=df.index)


def split_signed(pat: pd.DataFrame) -> pd.DataFrame:
    cols = {f"{n}_bull": (pat[n] > 0).astype(np.int8) for n in TALIB_PATTERNS}
    cols.update({f"{n}_bear": (pat[n] < 0).astype(np.int8) for n in TALIB_PATTERNS})
    return pd.DataFrame(cols, index=pat.index)


class TalibCandleSignal(SignalProvider):
    name = "talib_candles"
    feature_names = STAT_FEATURES + [n.lower() for n in TALIB_PATTERNS]
    tier = "B"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.horizons = [int(h) for h in self.cfg.get("horizons", [1, 5, 20])]
        self.decay = [float(d) for d in self.cfg.get("decay", [1.0, 0.6, 0.3])]
        self.stats = PatternStats(horizons=self.horizons, patterns=list(SPLIT_PATTERNS), directions=dict(SPLIT_DIRECTIONS))

    def availability(self) -> tuple[bool, str]:
        if import_optional("talib") is None:
            return False, "pip install TA-Lib (61 candlestick recognisers)"
        return True, "61 TA-Lib recognisers + forward-return statistics"

    def fit(self, frames: dict[str, pd.DataFrame], train_end: str | None = None) -> None:
        pats = {t: split_signed(talib_patterns(df)) for t, df in frames.items()}
        self.stats = PatternStats(horizons=self.horizons, patterns=list(SPLIT_PATTERNS), directions=dict(SPLIT_DIRECTIONS))
        self.stats.fit(frames, patterns=pats, end=train_end)
        log.info("TA-Lib pattern stats fitted on %d tickers (%d bars)", len(frames), self.stats.n_bars)

    def save_state(self) -> None:
        if self.stats.fitted:
            self.stats.save(self.state_path("talib_pattern_stats.json"))

    def load_state(self) -> bool:
        p = self.state_path("talib_pattern_stats.json")
        if p.exists():
            try:
                self.stats = PatternStats.load(p)
                return True
            except Exception as e:  # noqa: BLE001
                log.warning("could not load TA-Lib pattern stats: %s", e)
        return False

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        if not self.stats.fitted:
            self.load_state()
        pat = talib_patterns(df)
        feats = pattern_features(split_signed(pat), self.stats if self.stats.fitted else None, self.decay,
                                 patterns=SPLIT_PATTERNS, directions=SPLIT_DIRECTIONS, include_flags=False)[STAT_FEATURES]
        out = np.column_stack([feats.to_numpy(np.float32), pat[TALIB_PATTERNS].to_numpy(np.float32)])
        out[:20] = np.nan  # TA-Lib warm-up
        return out
