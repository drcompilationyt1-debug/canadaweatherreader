"""Staged entries, pullback entry, sector tilt and hold-period opportunities vs the current rule; both books, moomoo's minimum
fee charged per order, 2012-2026 and 2019-2026, phase-averaged."""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\Home\Downloads\StockBot")
import stockbot.agent  # noqa
from stockbot.config import load_config
from stockbot.env.dataset import MarketDataset
from stockbot.agent.backtest import blended_scores, closes, eligibility, round_trip_bps, simulate
from stockbot.data.sectors import load_sectors
cfg = load_config(); ds = MarketDataset.load("models/dataset"); px = closes(ds); score = blended_scores(ds); elig = eligibility(cfg, px)
sectors = load_sectors(cfg, list(px.columns), refresh=False)
BOOKS = {"100k": dict(k=20, every=10, hysteresis=10, budget=90_000, offs=(0, 4, 8, 12), vt=dict(vol_target=0.25, vol_window=10, vol_floor=0.4)),
         "10k": dict(k=5, every=21, hysteresis=8, budget=9_000, offs=(0, 5, 10, 15), vt={})}
VARIANTS = {
    "current rule": {},
    "stages 50%, +25% at -3%, -6%": {"scale_in": {"first": 0.5, "dips": [0.03, 0.06], "adds": [0.25, 0.25], "expire": 10}},
    "stages 50%, rest only on dips": {"scale_in": {"first": 0.5, "dips": [0.03, 0.06], "adds": [0.25, 0.25], "expire": 10, "complete": False}},
    "stages 34/33/33 at -4%, -8%": {"scale_in": {"first": 0.34, "dips": [0.04, 0.08], "adds": [0.33, 0.33], "expire": 15}},
    "wait for a 2% pullback (10d)": {"scale_in": {"first": 0.0, "dips": [0.02], "adds": [1.0], "expire": 10}},
    "wait for a 4% pullback (15d)": {"scale_in": {"first": 0.0, "dips": [0.04], "adds": [1.0], "expire": 15}},
    "sector tilt 0.5": {"sector_tilt": 0.5},
    "sector tilt 1.0": {"sector_tilt": 1.0},
    "hold-period deals (98th pct in)": {"deals": {"enter_pct": 0.98, "exit_pct": 0.0, "max_swaps": 1}},
}
for bname, b in BOOKS.items():
    fee = round_trip_bps(cfg, b["budget"], b["k"])
    oc = 2.0 / (b["budget"] / 0.9)                                    # moomoo's ~US$2 minimum per order, as a share of the book
    print(f"\n=== {bname} book (min fee per order {1e4*oc:.1f} bps of the book)")
    print(f"{'variant':32s} {'2012-26/yr':>10s} {'maxdd':>7s} {'sharpe':>6s} | {'2019-26/yr':>10s} {'maxdd':>7s} {'turn':>6s}")
    for vname, kw in VARIANTS.items():
        row = []
        for start, yrs in (("2012-01-03", 14.7), ("2019-01-02", 7.7)):
            idx = px.index[px.index >= pd.Timestamp(start)]
            rs = [simulate(px, score, idx[o], k=b["k"], every=b["every"], hysteresis=b["hysteresis"], fee_bps=fee, reserve=0.1, eligible=elig,
                           sectors=sectors, order_cost=oc, **b["vt"], **kw) for o in b["offs"]]
            tot = np.mean([r["total"] for r in rs])
            row += [(1 + tot) ** (1 / yrs) - 1, np.mean([r["max_drawdown"] for r in rs]), np.mean([r["sharpe"] for r in rs]),
                    np.mean([r["turnover_per_year"] for r in rs])]
        print(f"{vname:32s} {100*row[0]:+9.1f}% {100*row[1]:+6.0f}% {row[2]:6.2f} | {100*row[4]:+9.1f}% {100*row[5]:+6.0f}% {row[3]:6.1f}")
