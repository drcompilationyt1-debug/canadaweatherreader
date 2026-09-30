"""Take-profit re-evaluation vs plain holding, both books, 2012-2026, phase-averaged, production fees and settings."""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\Home\Downloads\StockBot")
import stockbot.agent  # noqa
from stockbot.config import load_config
from stockbot.env.dataset import MarketDataset
from stockbot.agent.backtest import blended_scores, closes, eligibility, round_trip_bps, simulate

cfg = load_config(); ds = MarketDataset.load("models/dataset"); px = closes(ds); score = blended_scores(ds)
elig = eligibility(cfg, px)
print("names:", len(px.columns), "from", px.index[0].date(), "to", px.index[-1].date())
BOOKS = {"100k": dict(k=20, every=15, hysteresis=10, budget=90_000, offs=(0, 4, 8, 12), vt=dict(vol_target=0.25, vol_window=10, vol_floor=0.4)),
         "10k": dict(k=5, every=21, hysteresis=8, budget=9_000, offs=(0, 5, 10, 15), vt={})}
VARIANTS = {
    "hold (current)": None,
    "sell at +15%": {"pct": 0.15, "keep_pct": 1.01},
    "sell at +25%": {"pct": 0.25, "keep_pct": 1.01},
    "sell at 2 sigma": {"z": 2.0, "keep_pct": 1.01},
    "sell at 3 sigma": {"z": 3.0, "keep_pct": 1.01},
    "3s, keep if top 30%": {"z": 3.0, "keep_pct": 0.70},
    "3s, keep if top half": {"z": 3.0, "keep_pct": 0.50},
    "2s, keep if top half": {"z": 2.0, "keep_pct": 0.50},
    "3s, keep half, refill": {"z": 3.0, "keep_pct": 0.50, "replace": True},
    "3s, sell, rebuy -5%": {"z": 3.0, "keep_pct": 1.01, "rebuy_dip": 0.05},
    "3s, keep half, rebuy": {"z": 3.0, "keep_pct": 0.50, "rebuy_dip": 0.05},
    "+25%, keep half, refill": {"pct": 0.25, "keep_pct": 0.50, "replace": True},
}
for bname, b in BOOKS.items():
    fee = round_trip_bps(cfg, b["budget"], b["k"])
    print(f"\n=== {bname} book: top-{b['k']}, every {b['every']}, hysteresis {b['hysteresis']}, {fee:.1f} bps round trip")
    print(f"{'variant':24s} {'geo/yr':>7s} {'worst':>7s} {'>hold':>6s} {'mean dd':>8s} {'worst dd':>9s} {'sharpe':>6s} {'turn':>6s} {'trig/yr':>8s} {'sold%':>6s}")
    base = None
    for vname, tp in VARIANTS.items():
        tot, dd, sh, tn, trig, sold = [], [], [], [], 0, 0
        for y in range(2012, 2027):
            start, end = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
            idx = px.index[(px.index >= start) & (px.index <= end)]
            if len(idx) < 60:
                continue
            rs = [simulate(px, score, idx[o], end=end, k=b["k"], every=b["every"], hysteresis=b["hysteresis"], fee_bps=fee, reserve=0.1,
                           eligible=elig, take_profit=tp, **b["vt"]) for o in b["offs"]]
            tot.append(np.mean([r["total"] for r in rs])); dd.append(np.mean([r["max_drawdown"] for r in rs]))
            sh.append(np.mean([r["sharpe"] for r in rs])); tn.append(np.mean([r["turnover_per_year"] for r in rs]))
            if tp:
                trig += np.mean([r["take_profit"]["triggers"] for r in rs]); sold += np.mean([r["take_profit"]["sold"] for r in rs])
        geo = float(np.prod([1 + x for x in tot]) ** (1 / len(tot)) - 1)
        if base is None:
            base = tot
        beat = sum(a > bb + 1e-9 for a, bb in zip(tot, base)) if tp else 0
        print(f"{vname:24s} {100*geo:+6.1f}% {100*min(tot):+6.1f}% {beat:3d}/{len(tot)} {100*np.mean(dd):+7.1f}% {100*min(dd):+8.1f}% "
              f"{np.mean(sh):6.2f} {np.mean(tn):6.1f} {trig/len(tot):8.1f} {100*sold/max(trig,1e-9) if tp else 0:5.0f}%")
