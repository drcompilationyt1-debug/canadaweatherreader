"""Fundamentals block: point-in-time alignment, ratio maths and the statements cache (fake fetcher)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockbot.data.loader import synthetic_universe
from stockbot.signals.fundamentals import FundamentalsSignal
from stockbot.signals.registry import build_context


def _fake_statements(ticker: str) -> pd.DataFrame:
    rows = []
    for year, rev, ni, eq, debt, sh in ((2016, 100.0, 10.0, 50.0, 20.0, 10.0), (2017, 110.0, 12.0, 55.0, 22.0, 10.0),
                                        (2018, 121.0, 15.0, 60.0, 24.0, 10.0), (2019, 133.0, 16.0, 66.0, 26.0, 10.0)):
        rows.append({"period_end": pd.Timestamp(f"{year}-12-31"), "annual": True, "revenue": rev * 1e9, "net_income": ni * 1e9,
                     "equity": eq * 1e9, "debt": debt * 1e9, "shares": sh * 1e9})
    for q in ("2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31"):
        rows.append({"period_end": pd.Timestamp(q), "annual": False, "revenue": 33.0e9, "net_income": 4.0e9, "equity": 66.0e9,
                     "debt": 26.0e9, "shares": 10.0e9})
    return pd.DataFrame(rows)


def test_fundamentals_point_in_time(cfg, tmp_path):
    frames = synthetic_universe(["AAA"], n=1300, seed=1)          # 2015-01-01 -> 2019-12-25
    ctx = build_context(cfg, with_llm=False, with_news=False)
    p = FundamentalsSignal(cfg, ctx)
    calls = []

    def fetcher(t):
        calls.append(t)
        return _fake_statements(t)

    p.fetcher = fetcher
    df = frames["AAA"]
    a = p.safe_history("AAA", df)
    assert a is not None and a.shape == (len(df), 8) and calls == ["AAA"]
    valid = ~np.isnan(a).any(axis=1)
    first_valid = df.index[valid][0]
    # FY2016 (period end 2016-12-31) counts from 75 days later: nothing before mid-March 2017
    assert first_valid >= pd.Timestamp("2017-03-15") and first_valid <= pd.Timestamp("2017-03-20")
    assert not valid[df.index < pd.Timestamp("2017-03-15")].any()
    # ratios on the last bar (2019-12-25): FY2019 is not filed yet (period end + 75 days) and the four 2019
    # quarters only complete a trailing year on 2019-12-31 (+60 days), so the FY2018 report is in force:
    # revenue 121bn, net income 15bn, equity 60bn, debt 24bn, 10bn shares
    px = float(df["close"].iloc[-1])
    ey = (15.0e9 / 10.0e9) / px
    assert abs(a[-1, 0] - np.clip(ey * 10.0, -5, 5)) < 1e-4                  # earnings yield
    assert abs(a[-1, 4] - np.clip(15.0 / 121.0 * 5.0, -5, 5)) < 1e-4          # net margin
    assert abs(a[-1, 5] - np.clip(15.0 / 60.0 * 2.0, -5, 5)) < 1e-4           # ROE
    assert abs(a[-1, 6] - 24.0 / 60.0 / 2.0) < 1e-4                           # debt / equity
    assert abs(a[-1, 3] - np.clip((121.0 / 110.0 - 1.0) * 2.0, -3, 3)) < 1e-4  # revenue growth vs FY2017
    assert abs(a[-1, 7] - (np.log10(px * 10.0e9) - 10.5)) < 1e-4              # size
    # cached: a second pass does not refetch; latest equals the last history row
    a2 = p.safe_history("AAA", df)
    assert calls == ["AAA"] and np.array_equal(np.nan_to_num(a), np.nan_to_num(a2))
    assert (p.folder / "AAA.parquet").exists()
    lat = p.safe_latest("AAA", df)
    assert lat is not None and np.allclose(lat, a[-1])
    # a fetch failure falls back to the cached statements
    p.fetcher = lambda t: (_ for _ in ()).throw(RuntimeError("offline"))
    p.refresh_days = -1
    assert p.safe_history("AAA", df) is not None
