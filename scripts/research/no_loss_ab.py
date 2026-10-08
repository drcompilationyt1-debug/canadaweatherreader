"""The owner's rule: never sell at a loss unless it is very bad.  A name dropping out of the top-K at a rebalance is kept while
it is below its buy price and not down more than `floor`; the weakest newcomers make room.  Both books, per-order fee."""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\Home\Downloads\StockBot")
import stockbot.agent  # noqa
from stockbot.config import load_config
from stockbot.env.dataset import MarketDataset
from stockbot.agent.backtest import blended_scores, closes, eligibility, round_trip_bps, simulate
cfg = load_config(); ds = MarketDataset.load("models/dataset"); px = closes(ds); score = blended_scores(ds); elig = eligibility(cfg, px)
BOOKS = {"100k": dict(k=20, every=10, hysteresis=10, budget=90_000, offs=(0, 4, 8, 12), vt=dict(vol_target=0.25, vol_window=10, vol_floor=0.4)),
         "10k": dict(k=5, every=21, hysteresis=8, budget=9_000, offs=(0, 5, 10, 15), vt={})}
VARIANTS = {"current rule (sells losers at the re-rank)": None, "never sell a loser unless down > 15%": {"floor": -0.15},
            "never sell a loser unless down > 25%": {"floor": -0.25}, "never sell a loser unless down > 40%": {"floor": -0.40}}
for bname, b in BOOKS.items():
    fee = round_trip_bps(cfg, b["budget"], b["k"]); oc = 2.0 / (b["budget"] / 0.9)
    print(f"\n=== {bname} book")
    print(f"{'variant':44s} {'2012-26/yr':>10s} {'maxdd':>7s} | {'2019-26/yr':>10s} {'maxdd':>7s}")
    for vname, nl in VARIANTS.items():
        row = []
        for start, yrs in (("2012-01-03", 14.7), ("2019-01-02", 7.7)):
            idx = px.index[px.index >= pd.Timestamp(start)]
            rs = [simulate(px, score, idx[o], k=b["k"], every=b["every"], hysteresis=b["hysteresis"], fee_bps=fee, reserve=0.1, eligible=elig,
                           order_cost=oc, no_loss_sells=nl, **b["vt"]) for o in b["offs"]]
            tot = np.mean([r["total"] for r in rs])
            row += [(1 + tot) ** (1 / yrs) - 1, np.mean([r["max_drawdown"] for r in rs])]
        print(f"{vname:44s} {100*row[0]:+9.1f}% {100*row[1]:+6.0f}% | {100*row[2]:+9.1f}% {100*row[3]:+6.0f}%")
