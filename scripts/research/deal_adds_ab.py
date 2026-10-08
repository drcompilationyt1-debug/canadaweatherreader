"""Staged buying where every add-on is a fee-worthy order (>= the ~$800 moomoo minimum worth paying $1.99 on) and fires only on a
very good deal: the price dipped AND the name still ranks in the top 10%.  Both books, per-order fee charged, phase-averaged."""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\Home\Downloads\StockBot")
import stockbot.agent  # noqa
from stockbot.config import load_config
from stockbot.env.dataset import MarketDataset
from stockbot.agent.backtest import blended_scores, closes, eligibility, round_trip_bps, simulate
cfg = load_config(); ds = MarketDataset.load("models/dataset"); px = closes(ds); score = blended_scores(ds); elig = eligibility(cfg, px)
BOOKS = {
    "100k": (dict(k=20, every=10, hysteresis=10, budget=90_000, offs=(0, 4, 8, 12), vt=dict(vol_target=0.25, vol_window=10, vol_floor=0.4)), {
        "current rule": None,
        "50% now, +25% at -3% & -6%, top 10% only": {"first": 0.5, "dips": [0.03, 0.06], "adds": [0.25, 0.25], "min_pct": 0.9, "expire": 10},
        "75% now, +25% at -3%, top 10% only": {"first": 0.75, "dips": [0.03], "adds": [0.25], "min_pct": 0.9, "expire": 10},
        "75% now, +25% at -5%, top 10%, else never": {"first": 0.75, "dips": [0.05], "adds": [0.25], "min_pct": 0.9, "expire": 10, "complete": False},
    }),
    "10k": (dict(k=5, every=21, hysteresis=8, budget=9_000, offs=(0, 5, 10, 15), vt={}), {
        "current rule": None,
        "50% now, +50% at -4%, top 10% only": {"first": 0.5, "dips": [0.04], "adds": [0.5], "min_pct": 0.9, "expire": 10},
        "50% now, +50% at -6%, top 10% only": {"first": 0.5, "dips": [0.06], "adds": [0.5], "min_pct": 0.9, "expire": 15},
        "50% now, +50% at -4%, top 10%, else never": {"first": 0.5, "dips": [0.04], "adds": [0.5], "min_pct": 0.9, "expire": 10, "complete": False},
    }),
}
for bname, (b, variants) in BOOKS.items():
    fee = round_trip_bps(cfg, b["budget"], b["k"])
    oc = 2.0 / (b["budget"] / 0.9)
    slot = b["budget"] / b["k"]
    print(f"\n=== {bname} book (slot ~${slot:,.0f})")
    print(f"{'variant':44s} {'2012-26/yr':>10s} {'maxdd':>7s} | {'2019-26/yr':>10s} {'maxdd':>7s}")
    for vname, si in variants.items():
        row = []
        for start, yrs in (("2012-01-03", 14.7), ("2019-01-02", 7.7)):
            idx = px.index[px.index >= pd.Timestamp(start)]
            rs = [simulate(px, score, idx[o], k=b["k"], every=b["every"], hysteresis=b["hysteresis"], fee_bps=fee, reserve=0.1, eligible=elig,
                           order_cost=oc, scale_in=si, **b["vt"]) for o in b["offs"]]
            tot = np.mean([r["total"] for r in rs])
            row += [(1 + tot) ** (1 / yrs) - 1, np.mean([r["max_drawdown"] for r in rs])]
        print(f"{vname:44s} {100*row[0]:+9.1f}% {100*row[1]:+6.0f}% | {100*row[2]:+9.1f}% {100*row[3]:+6.0f}%")
