"""Train the heavy ranking heads on a GPU (Kaggle, Colab or a PC) from the saved dataset, for the runners to reuse.

The heads' inputs are rebuilt from ``models/dataset`` exactly as the weekly build hands them over (every block computed
before the head, without the heads themselves and the live-only blocks), the walk-forward runs on the GPU, and the state the
runner reuses is written to ``models/signals`` (``<head>.joblib`` + ``<head>_preds.parquet``).  The out-of-sample rank IC of
each head is reported next to the LightGBM head's, so a GPU run that does not help is visible straight away.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)

HEADS = ("xs_nn", "xs_tabpfn")


def per_block_from_dataset(ds, head: str) -> dict[str, dict[str, np.ndarray | None]]:
    """{block: {ticker: (T, size) array, NaN where absent | None}} for the blocks a head reads (those before it in the layout)."""
    from ..signals.xs_rank import EXCLUDE

    names = ds.layout.names
    upto = names.index(head) if head in names else len(names)
    out: dict[str, dict[str, np.ndarray | None]] = {}
    for b in ds.layout.blocks[:upto]:
        if b.name in EXCLUDE:
            continue
        arrs = {}
        for t, td in ds.data.items():
            flag = td.signals[:, b.offset] > 0.5
            if not flag.any():
                arrs[t] = None
                continue
            a = td.signals[:, b.start:b.end].astype(np.float32).copy()
            a[~flag] = np.nan
            arrs[t] = a
        if any(v is not None for v in arrs.values()):
            out[b.name] = arrs
    return out


def frames_from_dataset(ds) -> dict[str, pd.DataFrame]:
    return {t: pd.DataFrame({"open": td.open, "high": td.high, "low": td.low, "close": td.close, "volume": td.volume},
                            index=pd.DatetimeIndex(td.dates)) for t, td in ds.data.items()}


def rank_ic(preds: pd.DataFrame, frames: dict[str, pd.DataFrame], start: str = "2019-01-01", horizon: int = 20) -> tuple[float, float]:
    """Mean weekly-sampled cross-sectional Spearman IC of ``preds`` (ticker, date, pred) against the next ``horizon``-bar return."""
    px = pd.DataFrame({t: f["close"] for t, f in frames.items()})
    fwd = px.shift(-horizon) / px - 1.0
    p = preds.pivot_table(index="date", columns="ticker", values="pred").reindex(px.index)
    days = [d for d in p.index[p.index >= pd.Timestamp(start)][::5] if p.loc[d].notna().sum() > 30 and fwd.loc[d].notna().sum() > 30]
    ics = np.array([p.loc[d].corr(fwd.loc[d], method="spearman") for d in days])
    ics = ics[np.isfinite(ics)]
    if len(ics) < 3:
        return float("nan"), float("nan")
    return float(ics.mean()), float(ics.mean() / ics.std() * np.sqrt(len(ics) / 4))     # /4: overlapping 20-day windows


def gpu_train(cfg, heads=HEADS, ic_start: str = "2019-01-01") -> dict:
    """Train ``heads`` on the saved dataset on whatever device is present; returns {head: {device, rows, seconds, ic, t}}."""
    from ..env.dataset import MarketDataset
    from ..signals.registry import PROVIDER_CLASSES, build_context

    ds = MarketDataset.load(cfg.path("models_dir", "models") / "dataset")
    frames = frames_from_dataset(ds)
    ctx = build_context(cfg, with_llm=False, with_news=False)
    by_name = {k.name: k for k in PROVIDER_CLASSES}
    out: dict = {}
    try:
        import torch

        device = "cuda:" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    log.info("gpu-train on %s: %d names, %s to %s", device, len(frames), min(f.index[0] for f in frames.values()).date(),
             max(f.index[-1] for f in frames.values()).date())
    for head in heads:
        klass = by_name.get(head)
        if klass is None:
            out[head] = {"error": "not in this version"}
            continue
        p = klass(cfg, ctx)
        ok, why = p.availability()
        if not ok:
            out[head] = {"error": why}
            log.warning("%s unavailable: %s", head, why)
            continue
        p.pretrained = False                                            # this run IS the training
        per_block = per_block_from_dataset(ds, head)
        t0 = time.time()
        p._history(frames, per_block)
        secs = time.time() - t0
        ic, t = rank_ic(p.preds, frames, ic_start) if p.preds is not None and len(p.preds) else (float("nan"), float("nan"))
        out[head] = {"device": device, "rows": int(len(p.preds) if p.preds is not None else 0), "seconds": round(secs, 1), "ic": ic, "t": t,
                     "trained_on": p.trained_on}
        log.info("%s trained on %s in %.0fs: %d out-of-sample rows, IC %+.3f (t %.1f) since %s", head, device, secs, out[head]["rows"], ic, t, ic_start)
    ref = per_block_from_dataset(ds, "xs_nn").get("xs_rank")
    if ref is None:                                                     # the LightGBM head's own stored predictions, for comparison
        from ..signals.xs_rank import XSRankSignal

        lgb = XSRankSignal(cfg, ctx)
        pp = lgb.state_path(lgb.PREDS_FILE)
        if pp.exists():
            pr = pd.read_parquet(pp)
            pr["date"] = pd.to_datetime(pr["date"])
            ic, t = rank_ic(pr, frames, ic_start)
            out["xs_rank (LightGBM, for comparison)"] = {"ic": ic, "t": t}
    return out
