"""Block pruning: a signal block that has shown no value for six months is masked until it shows value again.

Masked, not removed: its availability flag reads 0 and its features 0 wherever observations are built (the training
dataset, the live cycle, the evaluation, the tuners), so the observation layout and every trained model stay valid,
and the policy - trained with random block masking from the start - simply sees one more absent block.

Value has two readings, both on the raw (unmasked) dataset so a masked block can earn its way back:
  * the largest |t| of any feature's trailing ``days``-day cross-sectional rank IC against the 20-day forward return
    (a feature with a consistently negative IC still carries information: the sign is the model's to learn), and
  * the block's share of the ranking head's split gain (``models/signals/xs_rank.joblib``).
A block is masked when both are small (|t| < ``mask_t`` and gain share < ``mask_gain``) and unmasked when either
recovers past a higher bar (``unmask_t`` / ``unmask_gain``): hysteresis, so the mask does not flip week to week.
The core price blocks (they define where an episode may start), the ranking head and the reliability scoreboard are
never masked, and the live-only LLM blocks have no history here - the reliability board judges those.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from ..logging_utils import get_logger

log = get_logger(__name__)

PROTECTED = {"technical", "trend", "candles", "xs_rank", "reliability"}
LIVE_ONLY = {"trading_agents", "ai_hedge_fund", "llm_trader", "news_llm"}
MASK_FILE = "block_mask.json"


def load_block_mask(models_dir) -> dict[str, dict]:
    """{block: {since, max_abs_t, gain_share}} currently masked (``models/block_mask.json``), or {}."""
    f = Path(models_dir) / MASK_FILE
    if not f.exists():
        return {}
    try:
        return dict(json.loads(f.read_text(encoding="utf-8")).get("masked", {}))
    except Exception as e:  # noqa: BLE001
        log.warning("block mask %s unreadable: %s", f, e)
        return {}


def with_block_mask(cfg, ds):
    """The dataset with the currently masked blocks zeroed (flag and features); the dataset itself when nothing is masked."""
    if ds is None or not bool(cfg.get_path("signals.pruning.enabled", True)):
        return ds
    masked = load_block_mask(cfg.path("models_dir", "models"))
    names = [n for n in masked if n in ds.layout.names]
    if not names:
        return ds
    log.info("block mask: %s masked (no value for six months - models/%s)", ", ".join(names), MASK_FILE)
    return ds.masked(names)


def ranker_gain_share(models_dir) -> dict[str, float]:
    """Each block's share of the ranking head's total split gain, from the saved LightGBM model; {} without one."""
    p = Path(models_dir) / "signals" / "xs_rank.joblib"
    if not p.exists():
        return {}
    try:
        import joblib

        d = joblib.load(p)
        model, spec = d["model"], [tuple(x) for x in d["spec"]]
        booster = model.booster_ if hasattr(model, "booster_") else model
        imp = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    except Exception as e:  # noqa: BLE001
        log.warning("ranking head importances unavailable: %s", e)
        return {}
    out, i = {}, 0
    for block, size in spec:
        out[str(block)] = float(imp[i:i + int(size) + 1].sum())       # [flag, features...] per block
        i += int(size) + 1
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()} if tot > 0 else {}


def block_values(ds, days: int = 126, horizon: int = 20, min_names: int = 15, models_dir=None) -> dict[str, dict]:
    """Per block: the largest |t| of its features' trailing rank IC, the ranking head's gain share, features scored."""
    from ..agent.backtest import trailing_ic

    inputs = [f"{b.name}.{f}" for b in ds.layout.blocks for f in b.feature_names]
    ics = trailing_ic(ds, inputs, days=days, horizon=horizon, min_names=min_names)
    gain = ranker_gain_share(models_dir) if models_dir is not None else {}
    out = {}
    for b in ds.layout.blocks:
        ts = [abs(float(ics[f"{b.name}.{f}"]["t"])) for f in b.feature_names if f"{b.name}.{f}" in ics]
        out[b.name] = {"max_abs_t": (max(ts) if ts else None), "gain_share": gain.get(b.name), "features_scored": len(ts),
                       "best_feature": (max(((abs(float(ics[k]["t"])), k) for k in (f"{b.name}.{f}" for f in b.feature_names) if k in ics),
                                            default=(0.0, None))[1])}
    return out


def prune_blocks(cfg, ds, out_path=None, days: int = 126, mask_t: float = 1.0, mask_gain: float = 0.01, unmask_t: float = 1.5,
                 unmask_gain: float = 0.02, min_names: int = 15, horizon: int = 20, protect=None) -> dict:
    """Decide the mask from the raw dataset and write ``models/block_mask.json``; returns the report.  ``protect`` (or
    ``signals.pruning.protect``) names blocks the owner wants in regardless of the evidence."""
    models_dir = cfg.path("models_dir", "models")
    prev = load_block_mask(models_dir)
    values = block_values(ds, days=days, horizon=horizon, min_names=min_names, models_dir=models_dir)
    today = str(date.today())
    protected = PROTECTED | set(protect if protect is not None else (cfg.get_path("signals.pruning.protect", []) or []))
    masked: dict[str, dict] = {}
    would: list[str] = []
    for name, v in values.items():
        if name in LIVE_ONLY or v["max_abs_t"] is None:
            continue
        if name in protected:
            if float(v["max_abs_t"]) < mask_t and float(v["gain_share"] or 0.0) < mask_gain:
                would.append(name)                                        # reported, never masked
            continue
        t, g = float(v["max_abs_t"]), float(v["gain_share"] or 0.0)
        was = name in prev
        stays = (t < unmask_t and g < unmask_gain) if was else (t < mask_t and g < mask_gain)
        if stays:
            masked[name] = {"since": (prev[name].get("since") if was else today) or today, "max_abs_t": round(t, 3), "gain_share": round(g, 4),
                            "best_feature": v.get("best_feature")}
    newly, back = sorted(set(masked) - set(prev)), sorted(set(prev) - set(masked))
    rep = {"updated": today, "days": days, "thresholds": {"mask_t": mask_t, "mask_gain": mask_gain, "unmask_t": unmask_t, "unmask_gain": unmask_gain},
           "masked": masked, "newly_masked": newly, "unmasked": back, "protected_without_value": sorted(would), "values": values}
    if out_path is None:
        out_path = Path(models_dir) / MASK_FILE
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    log.info("block pruning: %d masked %s%s%s", len(masked), sorted(masked), f"; newly masked {newly}" if newly else "",
             f"; back in use {back}" if back else "")
    return rep


def format_report(rep: dict) -> str:
    lines = [f"block pruning ({rep['days']} days): {len(rep['masked'])} masked - {', '.join(sorted(rep['masked'])) or 'none'}"]
    if rep.get("newly_masked"):
        lines.append(f"  newly masked: {', '.join(rep['newly_masked'])}")
    if rep.get("unmasked"):
        lines.append(f"  back in use:  {', '.join(rep['unmasked'])}")
    lines.append(f"{'block':18s} {'max |t|':>8s} {'gain %':>7s} {'best feature':28s} status")
    for name, v in sorted(rep["values"].items(), key=lambda kv: -(kv[1]["max_abs_t"] or 0.0)):
        status = ("protected (no value lately)" if name in rep.get("protected_without_value", []) else
                  "protected" if name in PROTECTED else "live-only (reliability board)" if name in LIVE_ONLY else
                  "MASKED" if name in rep["masked"] else "in use")
        t = f"{v['max_abs_t']:8.2f}" if v["max_abs_t"] is not None else f"{'n/a':>8s}"
        g = f"{100 * v['gain_share']:7.2f}" if v.get("gain_share") is not None else f"{'n/a':>7s}"
        lines.append(f"{name:18s} {t} {g} {str(v.get('best_feature') or ''):28s} {status}")
    return "\n".join(lines)
