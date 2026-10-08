"""Sector weighting re-set weekly (or at each re-rank) rather than daily: more money to holdings in sectors with the best
3-month return.  Both books, per-order fee, phase-averaged."""
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
for bname, b in BOOKS.items():
    fee = round_trip_bps(cfg, b["budget"], b["k"]); oc = 2.0 / (b["budget"] / 0.9)
    variants = {"current (equal slots)": {}, "tilt 0.5, reweighted weekly": {"sector_tilt": 0.5, "sector_every": 5},
                "tilt 1.0, reweighted weekly": {"sector_tilt": 1.0, "sector_every": 5},
                "tilt 0.5, at each re-rank": {"sector_tilt": 0.5, "sector_every": b["every"]},
                "tilt 1.0, at each re-rank": {"sector_tilt": 1.0, "sector_every": b["every"]}}
    print(f"\n=== {bname} book")
    print(f"{'variant':30s} {'3y/yr':>7s} {'10y/yr':>7s} {'blend':>7s} {'3y dd':>6s} {'turn':>5s}")
    last = px.index[-1]
    for vname, kw in variants.items():
        row = {}
        for label, yrs in (("3y", 3), ("10y", 10)):
            idx = px.index[px.index >= last - pd.DateOffset(years=yrs)]
            rs = [simulate(px, score, idx[o], k=b["k"], every=b["every"], hysteresis=b["hysteresis"], fee_bps=fee, reserve=0.1, eligible=elig,
                           sectors=sectors, order_cost=oc, **b["vt"], **kw) for o in b["offs"]]
            tot = np.mean([r["total"] for r in rs])
            row[label] = ((1 + tot) ** (1 / yrs) - 1, np.mean([r["max_drawdown"] for r in rs]), np.mean([r["turnover_per_year"] for r in rs]))
        blend = 2 / 3 * row["3y"][0] + 1 / 3 * row["10y"][0]
        print(f"{vname:30s} {100*row['3y'][0]:+6.1f}% {100*row['10y'][0]:+6.1f}% {100*blend:+6.1f}% {100*row['3y'][1]:+5.0f}% {row['10y'][2]:5.1f}")
