"""Does a convolutional network on recent price action add information?  (Jiang, Kelly and Xiu 2023, "(Re-)Imag(in)ing Price
Trends": a CNN on OHLC-volume charts predicts stock returns out of sample.)  A 1D CNN over each name's last 20 bars of open /
high / low / close / volume, trained walk-forward (fit on all years before, score the next two), target the cross-sectional
rank of the 20-day forward return - the same target as the ranking head.  Reports the out-of-sample rank IC next to the
ranking head's, and the backtest with the CNN score in the blend.  Writes models/research/cnn_preds.parquet."""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\Home\Downloads\StockBot")
import stockbot.agent  # noqa
import torch
from torch import nn
from stockbot.config import load_config
from stockbot.env.dataset import MarketDataset
from stockbot.agent.backtest import blended_scores, closes, eligibility, round_trip_bps, simulate

torch.set_num_threads(4)
WIN, H = 20, 20
cfg = load_config(); ds = MarketDataset.load("models/dataset")


def windows(td):
    o, h, l, c, v = (np.asarray(x, float) for x in (td.open, td.high, td.low, td.close, td.volume))
    n = len(c)
    lc = np.log(np.clip(c, 1e-9, None))
    vm = pd.Series(v).rolling(WIN, min_periods=5).mean().to_numpy()
    feats = np.stack([lc, np.log(np.clip(h, 1e-9, None)) - lc, np.log(np.clip(l, 1e-9, None)) - lc, np.log(np.clip(o, 1e-9, None)) - lc,
                      np.log1p(np.clip(v, 0, None)) - np.log1p(np.clip(vm, 0, None))], axis=0)            # (5, n)
    X = np.full((n, 5, WIN), np.nan, np.float32)
    for t in range(WIN, n):
        w = feats[:, t - WIN + 1:t + 1].copy()
        w[0] = w[0] - w[0, -1]                                                                          # the path relative to today
        X[t] = w
    fwd = np.full(n, np.nan)
    fwd[:-H] = lc[H:] - lc[:-H]
    return X, fwd


t0 = time.time()
Xs, rows = [], []
for tk, td in ds.data.items():
    X, f = windows(td)
    Xs.append(X)
    rows.append(pd.DataFrame({"ticker": tk, "date": pd.DatetimeIndex(td.dates), "fwd": f}))
X = np.concatenate(Xs)
panel = pd.concat(rows, ignore_index=True)
panel["y"] = panel.groupby("date")["fwd"].rank(pct=True) - 0.5
ok = np.isfinite(X).all(axis=(1, 2))
X = np.nan_to_num(np.clip(X, -1.0, 1.0))
years = panel["date"].dt.year.to_numpy()
print(f"panel {len(panel)} rows, {ok.sum()} usable windows ({time.time()-t0:.0f}s)")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(5, 32, 5, padding=2), nn.BatchNorm1d(32), nn.LeakyReLU(), nn.MaxPool1d(2),
                                 nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU(), nn.MaxPool1d(2),
                                 nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU(), nn.AdaptiveMaxPool1d(1),
                                 nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit(idx, seed=0, epochs=3, max_rows=300_000):
    rng = np.random.default_rng(seed)
    if len(idx) > max_rows:
        idx = rng.choice(idx, max_rows, replace=False)
    torch.manual_seed(seed)
    m = CNN()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(X[idx]), torch.tensor(panel["y"].to_numpy(np.float32)[idx])
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(idx))
        for b in range(0, len(idx), 2048):
            j = perm[b:b + 2048]
            opt.zero_grad()
            loss = ((m(Xt[j]) - yt[j]) ** 2).mean()
            loss.backward()
            opt.step()
    m.eval()
    return m


def predict(m, idx):
    out = np.empty(len(idx), np.float32)
    with torch.no_grad():
        for b in range(0, len(idx), 8192):
            out[b:b + 8192] = m(torch.tensor(X[idx[b:b + 8192]])).numpy()
    return out


pred = np.full(len(panel), np.nan)
yv = panel["y"].to_numpy()
for start in range(2012, 2027, 2):
    t1 = time.time()
    tr = np.flatnonzero((years < start - 0) & ok & np.isfinite(yv) & (panel["date"] < pd.Timestamp(f"{start}-01-01") - pd.Timedelta(days=35)).to_numpy())
    te = np.flatnonzero((years >= start) & (years < start + 2) & ok)
    if len(tr) < 20000 or len(te) == 0:
        continue
    m = fit(tr, seed=start)
    pred[te] = predict(m, te)
    print(f"  refit {start}: trained on {len(tr)} rows, scored {len(te)} ({time.time()-t1:.0f}s)")
panel["cnn"] = pred
out = Path("models/research"); out.mkdir(parents=True, exist_ok=True)
panel[["ticker", "date", "cnn"]].dropna().to_parquet(out / "cnn_preds.parquet")

# ---------------------------------------------------------------- information: daily rank IC, the CNN vs the ranking head
px = closes(ds)
cnn = panel.pivot_table(index="date", columns="ticker", values="cnn").reindex(px.index)
fwd = (px.shift(-H) / px - 1.0)
head = blended_scores(ds, {"xs_rank.xs_score": 1.0})


def ic(score, start):
    s, f = score[score.index >= start], fwd[fwd.index >= start]
    vals = [s.loc[d].corr(f.loc[d], method="spearman") for d in s.index[::5] if s.loc[d].notna().sum() > 30 and f.loc[d].notna().sum() > 30]
    a = np.array([v for v in vals if np.isfinite(v)])
    return a.mean(), a.mean() / a.std() * np.sqrt(len(a) / 4) if len(a) > 2 else 0.0   # /4: overlapping 20-day windows sampled weekly


for start in ("2012-01-01", "2019-01-01"):
    c_ic, c_t = ic(cnn, start)
    h_ic, h_t = ic(head, start)
    both = cnn.rank(axis=1, pct=True).add(head.rank(axis=1, pct=True), fill_value=np.nan) / 2
    b_ic, b_t = ic(both, start)
    corr = np.nanmean([cnn.loc[d].corr(head.loc[d], method="spearman") for d in cnn.index[cnn.index >= start][::20]])
    print(f"from {start[:4]}: CNN IC {c_ic:+.3f} (t {c_t:.1f}) | ranking head IC {h_ic:+.3f} (t {h_t:.1f}) | 50/50 IC {b_ic:+.3f} (t {b_t:.1f}) | corr {corr:+.2f}")

# ---------------------------------------------------------------- the backtest with the CNN in the blend
elig = eligibility(cfg, px)
base = blended_scores(ds)
cr = cnn.rank(axis=1, pct=True)
for bname, k, every, h, budget, offs, vt in (("100k", 20, 10, 10, 90_000, (0, 4, 8, 12), dict(vol_target=0.25, vol_window=10, vol_floor=0.4)),
                                              ("10k", 5, 21, 8, 9_000, (0, 5, 10, 15), {})):
    fee = round_trip_bps(cfg, budget, k)
    print(f"\n=== {bname}")
    for name, sc in (("current blend", base), ("blend + CNN (2:1)", (base * 2 + cr.fillna(base)) / 3), ("blend + CNN (1:1)", (base + cr.fillna(base)) / 2)):
        row = []
        for start, yrs in (("2012-01-03", 14.7), ("2019-01-02", 7.7)):
            idx = px.index[px.index >= pd.Timestamp(start)]
            rs = [simulate(px, sc, idx[o], k=k, every=every, hysteresis=h, fee_bps=fee, reserve=0.1, eligible=elig, **vt) for o in offs]
            tot = np.mean([r["total"] for r in rs])
            row += [(1 + tot) ** (1 / yrs) - 1, np.mean([r["max_drawdown"] for r in rs])]
        print(f"  {name:22s} 2012-26 {100*row[0]:+5.1f}%/yr dd {100*row[1]:+4.0f}% | 2019-26 {100*row[2]:+5.1f}%/yr dd {100*row[3]:+4.0f}%")
