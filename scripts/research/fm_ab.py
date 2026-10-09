"""Bigger forecasting models vs the ones in the bot (GPU: run through scripts/kaggle_deep.py on Kaggle).

Kronos-base (4x Kronos-small) and Chronos-Bolt-base (vs -small) forecast every 5th bar since 2018 for every name; their
cross-sectional rank IC against the realised 5- and 20-day returns is compared with the current models' features stored in the
dataset on the same dates.  Writes reports/fm_ab.json; the new models' caches land in models/research_fm/<variant>/ so a winner
can be shipped without recomputing."""
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import stockbot.agent  # noqa
from stockbot.config import load_config
from stockbot.env.dataset import MarketDataset
from stockbot.agent.gpu_train import frames_from_dataset
from stockbot.signals.base import SignalContext

cfg = load_config()
ds = MarketDataset.load(cfg.path("models_dir", "models") / "dataset")
frames = frames_from_dataset(ds)
px = pd.DataFrame({t: f["close"] for t, f in frames.items()})
FWD = {h: px.shift(-h) / px - 1.0 for h in (5, 20)}
START = pd.Timestamp("2019-01-01")
VARIANTS = [("kronos", "NeoQuasar/Kronos-base", "kr_ret_5", 5), ("chronos", "amazon/chronos-bolt-base", "chr_ret_20", 20)]
CURRENT = {"kronos": ("kronos.kr_ret_5", 5), "chronos": ("chronos.chr_ret_20", 20), "chronos2": ("chronos2.c2_ret_20", 20),
           "timesfm": ("timesfm.tfm_ret_20", 20)}


def ic(frame: pd.DataFrame, h: int, dates) -> tuple[float, float, int]:
    vals = []
    for d in dates:
        a, b = frame.loc[d], FWD[h].loc[d]
        if a.notna().sum() > 30 and b.notna().sum() > 30:
            v = a.corr(b, method="spearman")
            if np.isfinite(v):
                vals.append(v)
    v = np.array(vals)
    if len(v) < 5:
        return float("nan"), float("nan"), len(v)
    return float(v.mean()), float(v.mean() / v.std() * np.sqrt(len(v) / max(1, h / 5))), len(v)


def ds_feature(key: str) -> pd.DataFrame:
    block, feat = key.split(".", 1)
    b = ds.layout.block(block)
    j = b.start + list(b.feature_names).index(feat)
    cols = {}
    for t, td in ds.data.items():
        v = td.signals[:, j].astype(float)
        v[td.signals[:, b.offset] < 0.5] = np.nan
        cols[t] = pd.Series(v, index=pd.DatetimeIndex(td.dates))
    return pd.DataFrame(cols).reindex(px.index)


report = {"device": None, "variants": {}, "current": {}}
try:
    import torch
    report["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
except Exception:  # noqa: BLE001
    pass
from stockbot.signals.registry import PROVIDER_CLASSES
by_name = {k.name: k for k in PROVIDER_CLASSES}
for block, model, feat, h in VARIANTS:
    t0 = time.time()
    cfg.set_path(f"signals.{block}.model", model)
    cfg.set_path(f"signals.{block}.history_years", 8)
    cfg.set_path(f"signals.{block}.stride", 5)
    ctx = SignalContext(cfg=cfg, models_dir=Path("models/research_fm") / model.split("/")[-1])
    ctx.extra["frames"] = frames
    p = by_name[block](cfg, ctx)
    ok, why = p.availability()
    if not ok:
        report["variants"][model] = {"error": why}
        continue
    arrs = {}
    for t, df in frames.items():
        try:
            arrs[t] = p.compute_history(t, df)
        except Exception as e:  # noqa: BLE001
            print(t, "failed:", e, flush=True)
    j = list(p.feature_names).index(feat)
    fr = pd.DataFrame({t: pd.Series(a[:, j], index=frames[t].index) for t, a in arrs.items() if a is not None}).reindex(px.index)
    dates = [d for d in fr.index[fr.index >= START][::5] if fr.loc[d].notna().sum() > 30]
    res = {"seconds": round(time.time() - t0), "dates": len(dates)}
    for hh in (5, 20):
        m, tt, n = ic(fr, hh, dates)
        res[f"ic_{hh}d"], res[f"t_{hh}d"] = m, tt
    cur_key, _ = CURRENT[block]
    cur = ds_feature(cur_key)
    for hh in (5, 20):
        m, tt, n = ic(cur, hh, dates)
        res[f"current_ic_{hh}d"], res[f"current_t_{hh}d"] = m, tt
    res["corr_with_current"] = float(np.nanmean([fr.loc[d].corr(cur.loc[d], method="spearman") for d in dates[::4]]))
    report["variants"][model] = res
    print(model, json.dumps(res), flush=True)
dates_all = [d for d in px.index[px.index >= START][::5]]
for block, (key, h) in CURRENT.items():
    try:
        f = ds_feature(key)
        report["current"][key] = {f"ic_{hh}d": ic(f, hh, dates_all)[0] for hh in (5, 20)}
    except Exception as e:  # noqa: BLE001
        report["current"][key] = {"error": str(e)}
Path("reports").mkdir(exist_ok=True)
Path("reports/fm_ab.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
print(json.dumps(report, indent=1, default=str))
