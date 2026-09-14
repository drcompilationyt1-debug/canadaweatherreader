"""Run ai-hedge-fund's staffed alpha models for one ticker/date and print a JSON line (inside .venv-aihf).

    python scripts/agents/run_ai_hedge_fund.py NVDA 2026-09-11 --mandate third_party/ai-hedge-fund/hedge_fund/fund/example.yaml

Data: by default a Yahoo Finance client (``--data yfinance``, free, no key) that implements the
framework's ``DataClient`` protocol with point-in-time fundamentals, prices, news, insider trades
and an earnings beat/miss history; ``--data fd`` uses the framework's own financialdatasets.ai
client (needs FINANCIAL_DATASETS_API_KEY and paid credits).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta

QUARTER_LAG_DAYS = 45      # a quarterly report is public roughly 45 days after the period end
ANNUAL_LAG_DAYS = 75


def _f(x):
    try:
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def _div(a, b):
    a, b = _f(a), _f(b)
    return None if a is None or b in (None, 0.0) else a / b


def _row(frame, names):
    """First matching statement row (a Series indexed by period end) or None."""
    if frame is None or len(frame) == 0:
        return None
    for n in names:
        if n in frame.index:
            return frame.loc[n]
    return None


class YFDataClient:
    """Yahoo Finance implementation of hedge_fund.data.protocol.DataClient.

    Infrastructure failures raise (the protocol's contract); genuinely missing data is an empty
    list / None.  Fundamentals are point-in-time: a period counts from its estimated filing date.
    """

    def __init__(self):
        import yfinance as yf

        self._yf = yf
        self._tk = {}

    def _ticker(self, ticker):
        if ticker not in self._tk:
            self._tk[ticker] = self._yf.Ticker(ticker)
        return self._tk[ticker]

    # ------------------------------------------------------------------ prices
    def get_prices(self, ticker, start_date, end_date, **kwargs):
        from hedge_fund.data.models import Price

        hist = self._ticker(ticker).history(start=start_date, end=(datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
                                            auto_adjust=True)
        out = []
        for ts, r in hist.iterrows():
            if _f(r.get("Close")) is None:
                continue
            out.append(Price(open=float(r["Open"]), close=float(r["Close"]), high=float(r["High"]), low=float(r["Low"]),
                             volume=int(r.get("Volume", 0) or 0), time=ts.strftime("%Y-%m-%d")))
        return out

    def _price_at(self, ticker, date):
        end = datetime.strptime(date, "%Y-%m-%d")
        hist = self._ticker(ticker).history(start=(end - timedelta(days=10)).strftime("%Y-%m-%d"), end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                                            auto_adjust=True)
        return _f(hist["Close"].iloc[-1]) if len(hist) else None

    # ------------------------------------------------------------------ fundamentals
    def _periods(self, ticker):
        """Statement rows per period end, newest first: annual reports and trailing-twelve-month quarterly sums."""
        tk = self._ticker(ticker)
        out = []
        for annual, inc, bal, cf in ((True, tk.income_stmt, tk.balance_sheet, tk.cashflow),
                                     (False, tk.quarterly_income_stmt, tk.quarterly_balance_sheet, tk.quarterly_cashflow)):
            rev = _row(inc, ["Total Revenue", "Operating Revenue"])
            if rev is None:
                continue
            cols = sorted(rev.index)
            get = lambda fr, names, c: _f(_row(fr, names).get(c)) if _row(fr, names) is not None else None  # noqa: E731
            flow = {"revenue": (inc, ["Total Revenue", "Operating Revenue"]), "net_income": (inc, ["Net Income", "Net Income Common Stockholders"]),
                    "gross_profit": (inc, ["Gross Profit"]), "operating_income": (inc, ["Operating Income", "EBIT"]),
                    "interest": (inc, ["Interest Expense"]), "ocf": (cf, ["Operating Cash Flow"]), "capex": (cf, ["Capital Expenditure"]),
                    "fcf": (cf, ["Free Cash Flow"]), "dividends": (cf, ["Common Stock Dividend Paid", "Cash Dividends Paid"])}
            stock = {"equity": (bal, ["Stockholders Equity", "Common Stock Equity"]), "debt": (bal, ["Total Debt", "Long Term Debt"]),
                     "assets": (bal, ["Total Assets"]), "cur_assets": (bal, ["Current Assets"]), "cur_liab": (bal, ["Current Liabilities"]),
                     "shares": (bal, ["Ordinary Shares Number", "Share Issued"]), "cash": (bal, ["Cash And Cash Equivalents"])}
            if annual:
                for c in cols:
                    d = {k: get(fr, names, c) for k, (fr, names) in {**flow, **stock}.items()}
                    out.append({"period_end": c, "period": "annual", "lag": ANNUAL_LAG_DAYS, **d})
            else:
                for i in range(3, len(cols)):
                    window = cols[i - 3:i + 1]
                    d = {}
                    for k, (fr, names) in flow.items():
                        vals = [get(fr, names, c) for c in window]
                        d[k] = None if any(v is None for v in vals) else float(sum(vals))
                    for k, (fr, names) in stock.items():
                        d[k] = get(fr, names, cols[i])
                    out.append({"period_end": cols[i], "period": "ttm", "lag": QUARTER_LAG_DAYS, **d})
        for p in out:
            p["filing_date"] = (p["period_end"].to_pydatetime() + timedelta(days=p["lag"])).strftime("%Y-%m-%d")
            p["report_period"] = p["period_end"].strftime("%Y-%m-%d")
        return sorted(out, key=lambda p: p["filing_date"], reverse=True)

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        from hedge_fund.data.models import FinancialMetrics

        periods = [p for p in self._periods(ticker) if p["filing_date"] <= end_date]
        out = []
        prev_by_kind = {}
        for p in reversed(periods):        # oldest first to compute growth vs the previous period of the same kind
            prev = prev_by_kind.get(p["period"])
            prev_by_kind[p["period"]] = p
            price = self._price_at(ticker, p["filing_date"])
            shares = _f(p["shares"])
            mcap = price * shares if price is not None and shares else None
            fcf = _f(p["fcf"])
            if fcf is None and _f(p["ocf"]) is not None and _f(p["capex"]) is not None:
                fcf = p["ocf"] + p["capex"]         # capex is negative in Yahoo's cash-flow statement
            m = FinancialMetrics(
                ticker=ticker, report_period=p["report_period"], period=p["period"], filing_date=p["filing_date"], currency=None,
                market_cap=mcap, price_to_earnings_ratio=_div(mcap, p["net_income"]), price_to_book_ratio=_div(mcap, p["equity"]),
                price_to_sales_ratio=_div(mcap, p["revenue"]), free_cash_flow_yield=_div(fcf, mcap),
                gross_margin=_div(p["gross_profit"], p["revenue"]), operating_margin=_div(p["operating_income"], p["revenue"]),
                net_margin=_div(p["net_income"], p["revenue"]), return_on_equity=_div(p["net_income"], p["equity"]),
                return_on_assets=_div(p["net_income"], p["assets"]), current_ratio=_div(p["cur_assets"], p["cur_liab"]),
                cash_ratio=_div(p["cash"], p["cur_liab"]), debt_to_equity=_div(p["debt"], p["equity"]), debt_to_assets=_div(p["debt"], p["assets"]),
                interest_coverage=(_div(p["operating_income"], abs(p["interest"])) if _f(p["interest"]) not in (None, 0.0) else None),
                revenue_growth=(_div(p["revenue"], prev["revenue"]) - 1.0 if prev and _div(p["revenue"], prev["revenue"]) is not None else None),
                earnings_growth=(_div(p["net_income"], prev["net_income"]) - 1.0 if prev and _div(p["net_income"], prev["net_income"]) is not None
                                 and (prev["net_income"] or 0) > 0 else None),
                book_value_growth=(_div(p["equity"], prev["equity"]) - 1.0 if prev and _div(p["equity"], prev["equity"]) is not None else None),
                payout_ratio=(_div(abs(p["dividends"]), p["net_income"]) if _f(p["dividends"]) is not None else None),
                earnings_per_share=_div(p["net_income"], shares), book_value_per_share=_div(p["equity"], shares),
                free_cash_flow_per_share=_div(fcf, shares),
            )
            out.append(m)
        out.reverse()                       # newest first, as the framework expects
        return out[:limit]

    def get_market_cap(self, ticker, end_date):
        ms = self.get_financial_metrics(ticker, end_date, limit=1)
        return ms[0].market_cap if ms else None

    def get_company_facts(self, ticker):
        from hedge_fund.data.models import CompanyFacts

        try:
            info = self._ticker(ticker).info or {}
        except Exception:  # noqa: BLE001
            info = {}
        return CompanyFacts(ticker=ticker, name=info.get("longName") or info.get("shortName"), sector=info.get("sector"),
                            industry=info.get("industry"), exchange=info.get("exchange"), is_active=True)

    # ------------------------------------------------------------------ news / insiders / earnings
    def get_news(self, ticker, end_date, start_date=None, limit=1000):
        from hedge_fund.data.models import CompanyNews

        out = []
        for it in (self._ticker(ticker).news or [])[:limit]:
            c = it.get("content", it)
            title = c.get("title") or it.get("title")
            if not title:
                continue
            date = (c.get("pubDate") or c.get("displayTime") or "")[:10] or None
            if date and date > end_date:
                continue
            src = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else (it.get("publisher") or "yahoo")
            url = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else it.get("link")
            out.append(CompanyNews(ticker=ticker, title=title, source=src or "yahoo", date=date, url=url))
        return out

    def get_insider_trades(self, ticker, end_date, start_date=None, limit=1000):
        from hedge_fund.data.models import InsiderTrade

        df = self._ticker(ticker).insider_transactions
        out = []
        if df is None or len(df) == 0:
            return out
        for _, r in df.head(limit).iterrows():
            d = r.get("Start Date")
            date = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else (str(d)[:10] if d is not None else None)
            if date and (date > end_date or (start_date and date < start_date)):
                continue
            out.append(InsiderTrade(ticker=ticker, name=str(r.get("Insider", "")), filing_date=date or end_date, title=str(r.get("Position", "") or ""),
                                    transaction_date=date, transaction_type=str(r.get("Transaction", r.get("Text", "")) or ""),
                                    transaction_shares=_f(r.get("Shares")), transaction_value=_f(r.get("Value"))))
        return out

    def get_earnings_history(self, ticker, limit=12):
        from hedge_fund.data.models import EarningsData, EarningsRecord

        df = self._ticker(ticker).earnings_dates
        out = []
        if df is None or len(df) == 0:
            return out
        for ts, r in df.iterrows():
            reported, est = _f(r.get("Reported EPS")), _f(r.get("EPS Estimate"))
            if reported is None:
                continue
            filing = ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts
            # the fiscal quarter the call is about: the last calendar quarter end at least 15 days before the date
            ref = filing - timedelta(days=15)
            q_end = datetime(ref.year, 3 * ((ref.month - 1) // 3) + 3, 1) + timedelta(days=31)
            q_end = q_end.replace(day=1) - timedelta(days=1)
            if est is None:
                surprise = None
            else:
                pct = (reported - est) / abs(est) if est else (0.0 if reported == est else (1.0 if reported > est else -1.0))
                surprise = "BEAT" if pct > 0.005 else "MISS" if pct < -0.005 else "MEET"
            out.append(EarningsRecord(ticker=ticker, report_period=q_end.strftime("%Y-%m-%d"), source_type="8-K",
                                      filing_date=filing.strftime("%Y-%m-%d"), fiscal_period=None,
                                      quarterly=EarningsData(earnings_per_share=reported, estimated_earnings_per_share=est, eps_surprise=surprise)))
        out.sort(key=lambda r: r.filing_date, reverse=True)
        return out[:limit]

    def get_earnings(self, ticker):
        from hedge_fund.data.models import Earnings

        hist = self.get_earnings_history(ticker, limit=1)
        if not hist:
            return None
        return Earnings(ticker=ticker, report_period=hist[0].report_period, quarterly=hist[0].quarterly)

    def close(self):
        return None


def _register_model() -> None:
    """Let HEDGE_FUND_LLM_MODEL name a Gemini / OpenAI model the framework's registry does not list yet.

    ``make_llm`` routes by the registry's provider and falls back to Anthropic for unknown ids; the
    free Gemini flash models are not listed, so they are added here (in our process only)."""
    import os

    model = os.environ.get("HEDGE_FUND_LLM_MODEL")
    if not model:
        return
    from hedge_fund.llm import registry

    if registry.provider_for(model) is not None:
        return
    provider = "Google" if model.lower().startswith("gemini") else "OpenAI" if model.lower().startswith(("gpt", "o1", "o3", "o4")) else None
    if provider is None:
        return
    original = registry.load_api_models
    registry.load_api_models = lambda: original() + [(model, model, provider)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("date")
    ap.add_argument("--mandate", required=True)
    ap.add_argument("--data", choices=["yfinance", "fd"], default="yfinance")
    args = ap.parse_args()

    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    signals, errors = [], []
    try:
        from hedge_fund.data import CachedDataClient
        from hedge_fund.fund import Fund, load_spec

        _register_model()
        fund = Fund(load_spec(args.mandate))
        if args.data == "fd":
            from hedge_fund.data import FDClient

            raw = FDClient()
        else:
            raw = YFDataClient()
        try:
            fd = CachedDataClient(raw)
            for _strategy, staff in fund.strategies:
                for model in staff:
                    try:
                        sig = model.predict(args.ticker, args.date, fd)
                        signals.append({"model": getattr(sig, "model_name", type(model).__name__), "value": float(sig.value),
                                        "reasoning": (getattr(sig, "reasoning", None) or "")[:400]})
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"{type(model).__name__}: {type(e).__name__}: {e}")
        finally:
            try:
                raw.close()
            except Exception:  # noqa: BLE001
                pass
        result = {"ticker": args.ticker, "date": args.date, "signals": signals, "errors": errors, "data": args.data}
    except Exception as e:  # noqa: BLE001
        result = {"ticker": args.ticker, "date": args.date, "signals": [], "error": f"{type(e).__name__}: {e}"}
    finally:
        sys.stdout = real_stdout
    print(json.dumps(result))
    return 0 if signals else 1


if __name__ == "__main__":
    sys.exit(main())
