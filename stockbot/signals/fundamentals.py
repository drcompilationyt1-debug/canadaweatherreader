"""Fundamentals: value, growth, profitability and leverage from the financial statements.

The bread and butter of long-horizon quant funds, and the one thing the price-based blocks cannot
see.  Statements come from Yahoo Finance (free): the last four annual reports plus the newest
quarters, per ticker, cached under ``models/signals/fundamentals`` and refreshed weekly by the
retrain job.  Ratios are computed point-in-time: a report only counts from ``lag_days`` after its
period end (roughly when it was filed), and price-based ratios use each bar's own close.  Before
the first usable report the block is simply masked (about 5 years of history).

Features per bar: earnings yield, book / price, sales / price, revenue growth (yoy), net margin,
return on equity, debt / equity, and size (log market cap).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .base import SignalProvider

log = get_logger(__name__)

FEATURES = ["f_ey", "f_bp", "f_sp", "f_growth", "f_margin", "f_roe", "f_leverage", "f_size"]
ROWS = {"revenue": ["Total Revenue", "Operating Revenue"], "net_income": ["Net Income", "Net Income Common Stockholders"],
        "equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
        "debt": ["Total Debt", "Long Term Debt"], "shares": ["Ordinary Shares Number", "Share Issued"]}


def _pick(frame: pd.DataFrame | None, names: list[str]) -> pd.Series | None:
    if frame is None or len(frame) == 0:
        return None
    for n in names:
        if n in frame.index:
            s = pd.to_numeric(frame.loc[n], errors="coerce")
            if s.notna().any():
                return s
    return None


def statements_from_yfinance(ticker: str) -> pd.DataFrame:
    """One row per reporting period: period_end, annual (bool), revenue, net_income, equity, debt, shares."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    rows = []
    for annual, inc, bal in ((True, tk.income_stmt, tk.balance_sheet), (False, tk.quarterly_income_stmt, tk.quarterly_balance_sheet)):
        rev, ni = _pick(inc, ROWS["revenue"]), _pick(inc, ROWS["net_income"])
        eq, debt, sh = _pick(bal, ROWS["equity"]), _pick(bal, ROWS["debt"]), _pick(bal, ROWS["shares"])
        periods = set()
        for s in (rev, ni, eq, sh):
            if s is not None:
                periods |= set(pd.to_datetime(s.index))
        for p in sorted(periods):
            def val(s):
                if s is None:
                    return np.nan
                s2 = s.copy()
                s2.index = pd.to_datetime(s2.index)
                return float(s2.get(p, np.nan))
            rows.append({"period_end": p, "annual": annual, "revenue": val(rev), "net_income": val(ni), "equity": val(eq),
                         "debt": val(debt), "shares": val(sh)})
    df = pd.DataFrame(rows, columns=["period_end", "annual", "revenue", "net_income", "equity", "debt", "shares"])
    return df.sort_values(["period_end", "annual"]).reset_index(drop=True)


class FundamentalsSignal(SignalProvider):
    name = "fundamentals"
    feature_names = list(FEATURES)
    tier = "A"

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.lag_days = int(self.cfg.get("lag_days", 60))
        self.refresh_days = float(self.cfg.get("refresh_days", 7))
        self.sleep = float(self.cfg.get("sleep", 0.3))
        self.folder = ctx.state_dir() / "fundamentals"
        self.fetcher = statements_from_yfinance

    def availability(self) -> tuple[bool, str]:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            return False, "pip install yfinance"
        return True, f"annual + quarterly statements, {self.lag_days}-day filing lag, refreshed every {self.refresh_days:.0f} days"

    # ------------------------------------------------------------------ statements cache
    def _path(self, ticker: str):
        return self.folder / f"{ticker.replace('/', '_')}.parquet"

    def statements(self, ticker: str, refresh: bool = False) -> pd.DataFrame | None:
        p = self._path(ticker)
        if p.exists() and not refresh:
            age = (time.time() - p.stat().st_mtime) / 86400.0
            if age <= self.refresh_days:
                try:
                    return pd.read_parquet(p)
                except Exception:  # noqa: BLE001
                    pass
        try:
            df = self.fetcher(ticker)
        except Exception as e:  # noqa: BLE001
            log.warning("fundamentals %s: fetch failed (%s)%s", ticker, e, " - using the cached statements" if p.exists() else "")
            return pd.read_parquet(p) if p.exists() else None
        if df is None or len(df) == 0:
            return pd.read_parquet(p) if p.exists() else None
        self.folder.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)
        return df

    def fit(self, frames: dict[str, pd.DataFrame], train_end: str | None = None) -> None:
        """Refresh the statements of every ticker (weekly retrain) - the only network step."""
        n = 0
        for t in frames:
            df = self.statements(t)
            n += int(df is not None and len(df) > 0)
            if self.sleep > 0:
                time.sleep(self.sleep)
        log.info("fundamentals: statements for %d/%d tickers", n, len(frames))

    # ------------------------------------------------------------------ features
    def _ttm(self, st: pd.DataFrame) -> pd.DataFrame:
        """Per reporting period: trailing-twelve-month revenue / net income, latest equity, debt, shares."""
        q = st[~st["annual"]].sort_values("period_end")
        a = st[st["annual"]].sort_values("period_end")
        rows = []
        for _, r in a.iterrows():
            rows.append({"period_end": r["period_end"], "revenue": r["revenue"], "net_income": r["net_income"], "equity": r["equity"],
                         "debt": r["debt"], "shares": r["shares"], "prev_revenue": np.nan, "lag": self.lag_days + 15})
        if len(a) >= 2:
            for i in range(1, len(rows)):
                rows[i]["prev_revenue"] = rows[i - 1]["revenue"]
        if len(q) >= 4:
            for i in range(3, len(q)):
                w = q.iloc[i - 3:i + 1]
                rev, ni = float(w["revenue"].sum(skipna=False)), float(w["net_income"].sum(skipna=False))
                prev = float(q.iloc[i - 7:i - 3]["revenue"].sum(skipna=False)) if i >= 7 else np.nan
                last = q.iloc[i]
                rows.append({"period_end": last["period_end"], "revenue": rev, "net_income": ni, "equity": last["equity"], "debt": last["debt"],
                             "shares": last["shares"], "prev_revenue": prev, "lag": self.lag_days})
        out = pd.DataFrame(rows)
        if len(out) == 0:
            return out
        out["effective"] = (pd.to_datetime(out["period_end"]) + pd.to_timedelta(out["lag"], unit="D")).astype("datetime64[ns]")
        return out.sort_values("effective").drop_duplicates("effective", keep="last").reset_index(drop=True)

    def compute_history(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        st = self.statements(ticker)
        if st is None or len(st) == 0:
            return None
        ttm = self._ttm(st)
        if len(ttm) == 0:
            return None
        bars = pd.DataFrame({"date": pd.to_datetime(df.index).astype("datetime64[ns]"), "close": df["close"].astype(float).to_numpy()})
        merged = pd.merge_asof(bars.sort_values("date"), ttm.sort_values("effective"), left_on="date", right_on="effective", direction="backward")
        price = merged["close"].to_numpy(float)
        shares = merged["shares"].to_numpy(float)
        mcap = price * shares
        with np.errstate(divide="ignore", invalid="ignore"):
            eps = merged["net_income"].to_numpy(float) / shares
            ey = eps / price
            bp = merged["equity"].to_numpy(float) / mcap
            sp = merged["revenue"].to_numpy(float) / mcap
            growth = merged["revenue"].to_numpy(float) / merged["prev_revenue"].to_numpy(float) - 1.0
            margin = merged["net_income"].to_numpy(float) / merged["revenue"].to_numpy(float)
            roe = merged["net_income"].to_numpy(float) / merged["equity"].to_numpy(float)
            lev = merged["debt"].to_numpy(float) / merged["equity"].to_numpy(float)
            size = np.log10(np.where(mcap > 0, mcap, np.nan)) - 10.5
        out = np.column_stack([
            np.clip(ey * 10.0, -5, 5), np.clip(bp * 2.0, -1, 5), np.clip(sp, 0, 5), np.clip(np.nan_to_num(growth, nan=0.0) * 2.0, -3, 3),
            np.clip(margin * 5.0, -5, 5), np.clip(roe * 2.0, -5, 5), np.clip(np.nan_to_num(lev, nan=0.0) / 2.0, 0, 5), np.clip(size, -3, 3),
        ]).astype(np.float32)
        core = np.isnan(ey) | np.isnan(bp) | np.isnan(size)     # no usable report yet -> masked
        out[core] = np.nan
        return out

    def compute_latest(self, ticker: str, df: pd.DataFrame) -> np.ndarray | None:
        hist = self.compute_history(ticker, df.tail(400))
        if hist is None or len(hist) == 0:
            return None
        row = hist[-1]
        return None if np.isnan(row).any() else row
