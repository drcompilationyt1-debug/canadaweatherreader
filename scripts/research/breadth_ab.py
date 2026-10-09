"""Does a wider universe help?  The same price-based ranker built on today's 150 names and on 300 (the next 150 largest S&P 500
members of 2011 that still trade), walk-forward LightGBM on the factor + technical blocks, top-20 every 10 bars, hysteresis 10,
fees, point-in-time eligibility from the S&P 500 membership history.  Usage: python scripts/research/breadth_ab.py <next150.json>
<sp500_ticker_start_end.csv>"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import stockbot.agent  # noqa
from stockbot.config import load_config
from stockbot.data.loader import load_universe
from stockbot.signals.factors import ranked_factors, raw_factors
from stockbot.features.technical import compute_technical
from stockbot.data.sectors import load_sectors
from stockbot.agent.backtest import round_trip_bps, simulate
import lightgbm as lgb

cfg = load_config()
extra = json.load(open(sys.argv[1]))
members = pd.read_csv(sys.argv[2], parse_dates=["start_date", "end_date"])
base = [t for t in cfg.get("universe", []) if not t.endswith(".TO")]
wide = base + [t for t in extra if t not in base]
d = cfg.section("data")
frames = load_universe(wide, d.get("start", "2008-01-01"), None, "1d", cfg.path("data.cache_dir", "data/cache"), 7.0)
wide = [t for t in wide if t in frames and len(frames[t]) > 600]
base = [t for t in base if t in wide]
print(f"universes: base {len(base)}, wide {len(wide)}", flush=True)
frames = {t: f[~f.index.duplicated(keep="last")].sort_index() for t, f in frames.items()}
px = pd.DataFrame({t: frames[t]["close"] for t in wide}).sort_index()
px.index.name = "date"
vol = pd.DataFrame({t: frames[t]["volume"] for t in wide}).reindex(px.index)
first = members.groupby("ticker")["start_date"].min()
alias = {"META": "FB", "GOOGL": "GOOG", "BRK-B": "BRK.B"}
elig = pd.DataFrame(True, index=px.index, columns=px.columns)
for t in px.columns:
    k = t if t in first.index else alias.get(t, t.replace("-", "."))
    if k in first.index:
        elig.loc[elig.index < first[k], t] = False
H = 20


def scores_for(names: list[str]) -> pd.DataFrame:
    sectors = load_sectors(cfg, names, refresh=True)
    fac = ranked_factors(raw_factors(px[names], vol[names], sectors, px["SPY"].pct_change(fill_method=None) if "SPY" in names else None))
    feats = {k: v for k, v in fac.items()}
    tech = {t: compute_technical(frames[t]).reindex(px.index) for t in names}
    cols = list(next(iter(tech.values())).columns)
    rows = []
    for t in names:
        f = pd.DataFrame({k: feats[k][t] for k in feats}, index=px.index)
        f = f.join(tech[t].add_prefix("tech_"))
        f["ticker"] = t
        rows.append(f)
    panel = pd.concat(rows)
    panel["date"] = panel.index
    fwd = (px[names].shift(-H) / px[names] - 1.0).stack().rename("fwd")
    fwd.index.names = ["date", "ticker"]
    panel = panel.set_index(["date", "ticker"])
    panel = panel[~panel.index.duplicated(keep="last")]
    panel["fwd"] = fwd.reindex(panel.index).to_numpy()
    panel["y"] = panel.groupby(level=0)["fwd"].rank(pct=True) - 0.5
    X = panel.drop(columns=["fwd", "y"]).astype(np.float32)
    years = panel.index.get_level_values(0).year
    pred = pd.Series(np.nan, index=panel.index)
    for yr in range(2011, int(years.max()) + 1):
        tr = (years < yr) & panel["y"].notna().to_numpy() & (panel.index.get_level_values(0) < pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=35))
        te = years == yr
        if tr.sum() < 20000 or te.sum() == 0:
            continue
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.7, subsample_freq=1,
                              colsample_bytree=0.5, reg_lambda=5.0, n_jobs=4, verbose=-1)
        idx = np.flatnonzero(tr)
        if len(idx) > 400_000:
            idx = np.random.default_rng(0).choice(idx, 400_000, replace=False)
        m.fit(X.iloc[idx], panel["y"].iloc[idx])
        pred[te] = m.predict(X[te])
        print(f"  {len(names)} names: refit {yr}", flush=True)
    return pred.unstack("ticker").reindex(index=px.index)


out = {}
for label, names in (("150 names (today)", base), ("300 names (wider)", wide)):
    sc = scores_for(names)
    fee = round_trip_bps(cfg, 90_000, 20)
    res = {}
    for start, yrs in (("2012-01-03", 14.7), ("2019-01-02", 7.7)):
        idx = px.index[px.index >= pd.Timestamp(start)]
        rs = [simulate(px[names], sc[names], idx[o], k=20, every=10, hysteresis=10, fee_bps=fee, reserve=0.1, eligible=elig[names],
                       vol_target=0.25, vol_window=10, vol_floor=0.4) for o in (0, 4, 8, 12)]
        tot = np.mean([r["total"] for r in rs])
        res[start[:4]] = {"geo": (1 + tot) ** (1 / yrs) - 1, "maxdd": float(np.mean([r["max_drawdown"] for r in rs]))}
    ew = simulate(px[names], None, px.index[px.index >= "2012-01-03"][0], eligible=elig[names])
    res["equal_weight_2012"] = (1 + ew["total"]) ** (1 / 14.7) - 1
    out[label] = res
    print(label, json.dumps(res), flush=True)
Path("reports").mkdir(exist_ok=True)
Path("reports/breadth_ab.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
