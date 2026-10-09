"""A MASTER-style stock transformer (Li et al., AAAI 2024, "MASTER: Market-Guided Stock Transformer") as a ranking head, on the
same signal panel the LightGBM head ranks.  GPU: run through scripts/kaggle_deep.py on Kaggle.

Per date the model sees every name's last T bars of block features; a market-guided gate (from the cross-section's average
features) re-weights the features, an intra-stock transformer reads each name's T bars, an inter-stock transformer lets the names
attend to one another, and a temporal attention pool gives one score per name, trained on the cross-sectional rank of the 20-day
forward return.  Walk-forward: fitted on every date before a two-year block (a 20-bar gap), early stopping on the last tenth.
Writes reports/master_ab.json and models/research/master_preds.parquet.   SMOKE=1: a tiny CPU run to check the plumbing."""
import json, os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import stockbot.agent  # noqa
import torch
from torch import nn
from stockbot.config import load_config
from stockbot.env.dataset import MarketDataset
from stockbot.agent.gpu_train import frames_from_dataset, per_block_from_dataset, rank_ic
from stockbot.signals.registry import build_context
from stockbot.signals.xs_rank import XSRankSignal

SMOKE = os.environ.get("SMOKE") == "1"
T, H = 8, 20
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = load_config()
ds = MarketDataset.load(cfg.path("models_dir", "models") / "dataset")
if SMOKE:
    ds = ds.subset(list(ds.data)[:25])
frames = frames_from_dataset(ds)
head = XSRankSignal(cfg, build_context(cfg, with_llm=False, with_news=False))
per_block = per_block_from_dataset(ds, "xs_nn")
head.spec = head._spec_from(per_block)
names = list(frames)
dates = sorted(set().union(*[set(f.index) for f in frames.values()]))
if SMOKE:
    dates = dates[-900:]
D, N = len(dates), len(names)
pos = {d: i for i, d in enumerate(dates)}
F = sum(s + 1 for _, s in head.spec)
X = np.zeros((D, N, F), np.float32)
valid = np.zeros((D, N), bool)
fwd = np.full((D, N), np.nan, np.float32)
for j, t in enumerate(names):
    f = frames[t]
    m = head._matrix({b: per_block[b].get(t) for b, _ in head.spec}, len(f))
    rows = [pos.get(d) for d in f.index]
    keep = [k for k, r in enumerate(rows) if r is not None]
    idx = np.array([rows[k] for k in keep])
    X[idx, j] = m[keep]
    valid[idx, j] = m[keep].any(axis=1)
    lc = np.log(f["close"].astype(float).clip(lower=1e-9)).to_numpy()
    fr = np.full(len(f), np.nan)
    fr[:-H] = lc[H:] - lc[:-H]
    fwd[idx, j] = fr[keep]
y = pd.DataFrame(fwd).rank(axis=1, pct=True).to_numpy(np.float32) - 0.5
mu, sd = X[valid].mean(axis=0), X[valid].std(axis=0) + 1e-6
X = np.where(valid[..., None], (X - mu) / sd, 0.0).astype(np.float32)
X = np.clip(X, -6, 6)
print(f"panel {D} dates x {N} names x {F} features on {dev}", flush=True)


class Master(nn.Module):
    def __init__(self, f, d=64, heads=4, drop=0.1, beta=5.0):
        super().__init__()
        self.beta = beta
        self.gate = nn.Linear(f, f)
        self.inp = nn.Linear(f, d)
        self.pos = nn.Parameter(torch.zeros(T, d))
        self.intra = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True)
        self.inter = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True)
        self.q = nn.Linear(d, d)
        self.out = nn.Linear(d, 1)

    def forward(self, x, mask):                       # x (B,T,N,F), mask (B,N) valid names
        B, T_, N_, F_ = x.shape
        m = (x[:, -1] * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)       # the market: mean features
        g = F_ * torch.softmax(self.gate(m) / self.beta, dim=-1)                              # market-guided feature gate
        h = self.inp(x * g[:, None, None, :]) + self.pos[None, :, None, :]
        h = self.intra(h.permute(0, 2, 1, 3).reshape(B * N_, T_, -1)).reshape(B, N_, T_, -1)  # each name over its T bars
        hl = h.permute(0, 2, 1, 3).reshape(B * T_, N_, -1)
        km = (~mask.bool()).repeat_interleave(T_, dim=0)
        hl = self.inter(hl, src_key_padding_mask=km).reshape(B, T_, N_, -1).permute(0, 2, 1, 3)  # names attend to one another
        a = torch.softmax((self.q(hl[:, :, -1:]) * hl).sum(-1) / hl.shape[-1] ** 0.5, dim=-1)     # temporal attention pool
        z = (a[..., None] * hl).sum(2)
        return self.out(z).squeeze(-1)


def batch(ix):
    xb = np.stack([X[i - T + 1:i + 1] for i in ix])
    return (torch.tensor(xb, device=dev), torch.tensor(valid[ix].astype(np.float32), device=dev),
            torch.tensor(np.nan_to_num(y[ix]), device=dev), torch.tensor(np.isfinite(y[ix]) & valid[ix], device=dev))


def fit(train_ix, epochs=30 if not SMOKE else 2, bs=16, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_val = max(1, len(train_ix) // 10)
    tr, va = train_ix[:-n_val], train_ix[-n_val:]
    model = Master(F).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(tr))
        for b in range(0, len(tr), bs):
            xb, mb, yb, wb = batch(tr[order[b:b + bs]])
            opt.zero_grad()
            loss = (((model(xb, mb) - yb) ** 2) * wb).sum() / wb.sum().clamp(min=1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v, n = 0.0, 0.0
            for b in range(0, len(va), 64):
                xb, mb, yb, wb = batch(va[b:b + 64])
                v += float((((model(xb, mb) - yb) ** 2) * wb).sum())
                n += float(wb.sum())
        v /= max(n, 1)
        if v < best - 1e-6:
            best, bad, best_state = v, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


years = np.array([d.year for d in dates])
pred = np.full((D, N), np.nan, np.float32)
first = int(years.min()) + 3
for start in range(first, int(years.max()) + 1, 2):
    t0 = time.time()
    te = np.flatnonzero((years >= start) & (years < start + 2))
    te = te[te >= T - 1]
    cut = te.min() - H if len(te) else 0
    tr = np.arange(T - 1, max(T - 1, cut))
    if len(tr) < 200 or len(te) == 0:
        continue
    model = fit(tr)
    with torch.no_grad():
        for b in range(0, len(te), 64):
            xb, mb, _, _ = batch(te[b:b + 64])
            pred[te[b:b + 64]] = model(xb, mb).cpu().numpy()
    pred[te] = np.where(valid[te], pred[te], np.nan)
    print(f"  refit {start}: {len(tr)} train dates, {len(te)} scored ({time.time()-t0:.0f}s)", flush=True)

long = pd.DataFrame({"date": np.repeat(np.array(dates), N), "ticker": np.tile(np.array(names), D), "pred": pred.ravel()}).dropna()
Path("models/research").mkdir(parents=True, exist_ok=True)
long.to_parquet("models/research/master_preds.parquet")
ic_m, t_m = rank_ic(long, frames, "2019-01-01")
rep = {"device": str(dev), "dates": D, "names": N, "features": F, "master_ic_2019": ic_m, "master_t_2019": t_m}
xp = Path(cfg.path("models_dir", "models")) / "signals" / "xs_rank_preds.parquet"
if xp.exists():
    xr = pd.read_parquet(xp)
    xr["date"] = pd.to_datetime(xr["date"])
    rep["xs_rank_ic_2019"], rep["xs_rank_t_2019"] = rank_ic(xr, frames, "2019-01-01")
    a = long.pivot_table(index="date", columns="ticker", values="pred")
    b = xr.pivot_table(index="date", columns="ticker", values="pred").reindex(index=a.index, columns=a.columns)
    both = pd.DataFrame((a.rank(axis=1, pct=True) + b.rank(axis=1, pct=True)) / 2).stack().rename("pred").reset_index()
    both.columns = ["date", "ticker", "pred"]
    rep["blend_ic_2019"], rep["blend_t_2019"] = rank_ic(both, frames, "2019-01-01")
    rep["corr_with_xs_rank"] = float(np.nanmean([a.loc[d].corr(b.loc[d], method="spearman") for d in a.index[a.index >= "2019-01-01"][::20]]))
Path("reports").mkdir(exist_ok=True)
Path("reports/master_ab.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
print(json.dumps(rep, indent=1, default=str))
