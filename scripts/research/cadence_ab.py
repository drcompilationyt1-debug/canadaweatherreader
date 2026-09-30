"""How often should the book re-rank?  Every cadence at each account's real fees, 2012-2026, phase-averaged over start dates.

Reported per variant: the compounded growth rate (what a lifetime book actually earns), the mean year, the worst year, how
often it beat SPY, the mean and worst drawdown, the Sharpe and the yearly turnover.
"""
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
YEARS = range(2012, 2027)


def run(name, k, every, hyst, budget, offs, **kw):
    fee = round_trip_bps(cfg, budget * 0.9, k)
    tot, dd, sh, turn, wins = [], [], [], [], 0
    for y in YEARS:
        start, end = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
        idx = px.index[(px.index >= start) & (px.index <= end)]
        if len(idx) < 60:
            continue
        spy = float((1 + px["SPY"].pct_change(fill_method=None).reindex(idx).fillna(0.0)).prod() - 1)
        rs = [simulate(px, score, idx[o], end=end, k=k, every=every, hysteresis=hyst, fee_bps=fee, reserve=0.1, eligible=elig, **kw) for o in offs]
        tot.append(np.mean([r["total"] for r in rs])); dd.append(np.mean([r["max_drawdown"] for r in rs]))
        sh.append(np.mean([r["sharpe"] for r in rs])); turn.append(np.mean([r["turnover_per_year"] for r in rs]))
        wins += tot[-1] > spy
    geo = float(np.prod([1 + x for x in tot]) ** (1 / len(tot)) - 1)
    print(f"{name:22s} {100*geo:+6.1f}% {100*np.mean(tot):+7.1f}% {100*min(tot):+6.1f}% {wins:3d}/{len(tot)} {100*np.mean(dd):+7.1f}% "
          f"{100*min(dd):+8.1f}% {np.mean(sh):6.2f} {np.mean(turn):7.1f} {fee:6.1f}")
    return geo


hdr = f"{'variant':22s} {'geo/yr':>7s} {'mean/yr':>8s} {'worst':>7s} {'>SPY':>5s} {'mean dd':>8s} {'worst dd':>9s} {'sharpe':>6s} {'turn/yr':>7s} {'bps':>6s}"
print("=== the 100k book: top-25, hysteresis 3, 10% reserve, 25% vol target")
print(hdr)
vt = dict(vol_target=0.25, vol_window=10, vol_floor=0.4)
for every in (1, 3, 5, 10, 15, 21, 42, 63):
    run(f"every {every} bars", 25, every, 3, 90_000, (0, 3, 6, 9), **vt)
print("\n  the same without volatility targeting (the plain rule)")
print(hdr)
for every in (3, 5, 10, 21, 42):
    run(f"every {every} bars plain", 25, every, 3, 90_000, (0, 3, 6, 9))
print("\n  hysteresis at the best cadences (a held name keeps its slot while it ranks within k + h)")
print(hdr)
for every in (5, 10):
    for h in (0, 3, 6, 10):
        run(f"every {every}, hyst {h}", 25, every, h, 90_000, (0, 3, 6, 9), **vt)

print("\n=== the 10k book: top-5, hysteresis 3, 10% reserve, no vol target")
print(hdr)
for every in (5, 10, 21, 42, 63):
    run(f"every {every} bars", 5, every, 3, 9_000, (0, 5, 10, 15))
