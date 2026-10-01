"""Risk-based sizing, conviction sizing and the Halloween effect vs equal slots: both books, 2012-2026 and 2019-2026, phase-averaged."""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\Home\Downloads\StockBot")
import stockbot.agent  # noqa
from stockbot.config import load_config
from stockbot.env.dataset import MarketDataset
from stockbot.agent.backtest import blended_scores, closes, eligibility, round_trip_bps, simulate
cfg = load_config(); ds = MarketDataset.load("models/dataset"); px = closes(ds); score = blended_scores(ds); elig = eligibility(cfg, px)
BOOKS = {"100k": dict(k=20, every=15, hysteresis=10, budget=90_000, offs=(0, 4, 8, 12), vt=dict(vol_target=0.25, vol_window=10, vol_floor=0.4)),
         "10k": dict(k=5, every=21, hysteresis=8, budget=9_000, offs=(0, 5, 10, 15), vt={})}
SUMMER = [5, 6, 7, 8, 9, 10]
VARIANTS = {"equal (current)": {}, "inverse-vol sizing": {"weighting": "inv_vol"}, "conviction sizing": {"weighting": "score"},
            "sell in May: 50% summer": {"season": {"months": SUMMER, "scale": 0.5}},
            "sell in May: 75% summer": {"season": {"months": SUMMER, "scale": 0.75}},
            "inv-vol + 75% summer": {"weighting": "inv_vol", "season": {"months": SUMMER, "scale": 0.75}}}
for bname, b in BOOKS.items():
    fee = round_trip_bps(cfg, b["budget"], b["k"])
    print(f"\n=== {bname} book")
    print(f"{'variant':26s} {'2012-26/yr':>10s} {'maxdd':>7s} {'sharpe':>6s} | {'2019-26/yr':>10s} {'maxdd':>7s}")
    for vname, kw in VARIANTS.items():
        row = []
        for start, yrs in (("2012-01-03", 14.7), ("2019-01-02", 7.7)):
            idx = px.index[px.index >= pd.Timestamp(start)]
            rs = [simulate(px, score, idx[o], k=b["k"], every=b["every"], hysteresis=b["hysteresis"], fee_bps=fee, reserve=0.1, eligible=elig,
                           **b["vt"], **kw) for o in b["offs"]]
            tot = np.mean([r["total"] for r in rs])
            row += [(1 + tot) ** (1 / yrs) - 1, np.mean([r["max_drawdown"] for r in rs]), np.mean([r["sharpe"] for r in rs])]
        print(f"{vname:26s} {100*row[0]:+9.1f}% {100*row[1]:+6.0f}% {row[2]:6.2f} | {100*row[3]:+9.1f}% {100*row[4]:+6.0f}%")
