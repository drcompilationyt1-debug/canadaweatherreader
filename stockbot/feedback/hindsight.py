"""Learn from the paper trades: hindsight fine-tuning of the policy, guarded by the out-of-sample score.

Every paper / live decision is stored with the exact observation the policy saw.  Once the bars
after it have settled we know what the best conviction would have been - the hindsight label:

* for each horizon (``1`` = that day's close, ``5`` = the week, ``21`` = the month) the realised
  return from the decision price, net of the round-trip fee (a move smaller than the fee is not
  worth trading, so it votes for "keep what you had");
* each net return is scaled by the name's volatility over that horizon (a one-sigma move = full
  conviction) and the horizons are blended with the configured weights - the label reflects the
  holding horizon the agent is built for, not just what happened by the close;
* samples are weighted by how big the move was (regret) and by recency.

The fine-tune is a few epochs of supervised regression of the policy's action mean toward those
labels (actor parameters only), anchored to the policy's own answers on simulator states so it
does not forget what the simulations taught it (hindsight experience replay / DAgger-style
imitation with a KL-style anchor).  Each ensemble member is then re-scored on the held-out
simulator window and the update is kept only if the score did not drop by more than
``max_score_drop`` - the same yardstick the trainer uses to pick checkpoints.  Runs after the day
review in the session workflow and after the weekend retrain (``stockbot review --learn``).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, env_settings
from ..execution.fees import FeeBook
from ..logging_utils import get_logger
from .experience import ExperienceStore

log = get_logger(__name__)

DEFAULTS = {"horizons": {1: 0.2, 5: 0.5, 21: 0.3}, "z_scale": 1.0, "min_samples": 40, "epochs": 8, "lr": 5e-5, "batch_size": 256,
            "anchor_weight": 1.0, "ref_steps": 2048, "recency_half_life_days": 60, "guard": True, "max_score_drop": 0.02,
            "eval_tickers": 8, "eval_bars": 500, "vol_window": 20}


def settings(cfg: Config) -> dict:
    s = dict(DEFAULTS)
    s.update({k: v for k, v in (cfg.get_path("feedback.hindsight", {}) or {}).items() if v is not None})
    s["horizons"] = {int(k): float(v) for k, v in dict(s["horizons"]).items()}
    return s


# ---------------------------------------------------------------------- samples
def build_samples(cfg: Config, obs_dim: int, frames: dict[str, pd.DataFrame] | None = None, today: pd.Timestamp | None = None) -> dict:
    """Hindsight-labelled samples from the experience store: ``X`` observations, ``y`` target conviction in
    -1..1, ``w`` sample weights, plus per-sample metadata.  Only decisions whose observation matches the
    current layout (``obs_dim``) and that have at least one settled horizon are used."""
    s = settings(cfg)
    store = ExperienceStore(cfg.path("feedback.experience_file", "data/experience/trades.jsonl"))
    recs = store.load()
    empty = {"X": np.zeros((0, obs_dim), np.float32), "y": np.zeros(0, np.float32), "w": np.zeros(0, np.float32), "meta": []}
    if recs is None or len(recs) == 0:
        return empty
    dec = recs[recs["type"] == "decision"].copy()
    dec = dec[dec["obs"].apply(lambda o: isinstance(o, list) and len(o) == obs_dim)]
    if len(dec) == 0:
        return empty
    tickers = sorted(dec["ticker"].unique().tolist())
    if frames is None:
        from ..agent.train import load_frames

        frames = load_frames(cfg, offline=True, tickers=tickers)
    fees = FeeBook.from_config(cfg)
    max_position = float(cfg.get_path("execution.max_position", 0.10))
    allow_short = bool(env_settings(cfg).get("allow_short", False))
    horizons = s["horizons"]
    today = pd.Timestamp(today or datetime.now(timezone.utc).date())
    X, y, w, meta = [], [], [], []
    for _, r in dec.iterrows():
        t = r["ticker"]
        df = frames.get(t)
        if df is None or len(df) < s["vol_window"] + 2:
            continue
        closes = df["close"].astype(float)
        d = pd.Timestamp(r["date"])
        i0 = int(np.searchsorted(closes.index.values, np.datetime64(d, "ns"), side="left"))
        if i0 >= len(closes):
            continue
        price = float(r["price"]) if float(r.get("price") or 0.0) > 0 else float(closes.iloc[i0])
        hist = closes.iloc[max(0, i0 - s["vol_window"] - 1): i0 + 1].to_numpy(float)
        lr = np.diff(np.log(np.maximum(hist, 1e-9)))
        vol = float(np.std(lr, ddof=1)) if len(lr) > 2 else float("nan")
        if not np.isfinite(vol) or vol <= 1e-6:
            vol = 0.01
        sched = fees.for_ticker(t)
        slice_notional = max(float(r.get("equity") or 0.0) * max_position, 1.0)
        fee_rt = 2.0 * float(sched.cost(slice_notional / price, price, "buy")) / slice_notional if sched is not None else 0.0
        obs = np.asarray(r["obs"], dtype=np.float32)
        exposure_now = float(obs[-6])                                  # portfolio.exposure is the first portfolio feature
        keep = exposure_now if allow_short else 2.0 * float(np.clip(exposure_now, 0.0, 1.0)) - 1.0
        zs, ws, rets = [], [], {}
        for h, wh in horizons.items():
            j = i0 + h - 1
            if j >= len(closes):
                continue
            ret = float(closes.iloc[j] / price - 1.0)
            rets[h] = ret
            net = np.sign(ret) * max(0.0, abs(ret) - fee_rt)
            if net == 0.0:
                zs.append(keep)
                ws.append(wh * 0.25)                                  # "not worth trading": a weak vote for what we had
                continue
            z = float(np.clip(net / (s["z_scale"] * vol * np.sqrt(h)), -1.0, 1.0))
            zs.append(z)
            ws.append(wh)
        if not zs:
            continue
        target = float(np.average(zs, weights=ws)) if sum(ws) > 0 else keep
        strength = float(np.average(np.abs(zs), weights=ws)) if sum(ws) > 0 else 0.1
        age_days = max(0.0, (today - d).days)
        recency = 0.5 ** (age_days / max(1.0, float(s["recency_half_life_days"])))
        X.append(obs)
        y.append(target)
        w.append(max(0.1, strength) * recency)
        meta.append({"ticker": t, "date": r["date"], "returns": rets, "target": target, "exposure": exposure_now,
                     "settled": len(rets), "action": float(r.get("action") or 0.0)})
    if not X:
        return empty
    return {"X": np.asarray(X, np.float32), "y": np.asarray(y, np.float32), "w": np.asarray(w, np.float32), "meta": meta}


# ---------------------------------------------------------------------- fine-tune
def policy_mean(model, X: np.ndarray) -> np.ndarray:
    import torch

    policy = model.policy
    with torch.no_grad():
        obs = torch.as_tensor(np.asarray(X, np.float32), device=policy.device)
        return policy.get_distribution(obs).distribution.mean.detach().cpu().numpy().reshape(len(X), -1)[:, 0]


def reference_states(model, dataset, env_cfg: dict, n_steps: int, seed: int = 0) -> np.ndarray:
    """Observations from the simulator under the current policy - the anchor set: whatever the
    fine-tune changes on the live states, it must keep answering these the same way."""
    from ..env.trading_env import TradingEnv

    rng = np.random.default_rng(seed)
    tickers = [t for t in dataset.tickers if t in dataset.data]
    if not tickers or n_steps <= 0:
        return np.zeros((0, dataset.layout.obs_dim), np.float32)
    env = TradingEnv(dataset, env_cfg, tickers=tickers, seed=seed, eval_mode=True)
    out = []
    while len(out) < n_steps:
        t = tickers[int(rng.integers(len(tickers)))]
        td = dataset.data[t]
        length = min(128, max(10, len(td) - 2 - td.min_start))
        if length < 10:
            break
        start = int(rng.integers(td.min_start, max(td.min_start + 1, len(td) - 2 - length)))
        obs, _ = env.reset(options={"ticker": t, "start": start, "length": length})
        done = False
        while not done and len(out) < n_steps:
            out.append(np.asarray(obs, np.float32))
            action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
            obs, _, terminated, truncated, _ = env.step(np.asarray(action).reshape(-1))
            done = terminated or truncated
    return np.asarray(out, np.float32)


def finetune(model, X: np.ndarray, y: np.ndarray, w: np.ndarray, X_ref: np.ndarray, epochs: int = 8, lr: float = 5e-5,
             batch_size: int = 256, anchor_weight: float = 1.0, seed: int = 0) -> dict:
    """Move the actor's mean toward the hindsight targets on the live states while holding its answers on
    the reference states.  Returns the weighted loss before / after and the anchor drift."""
    import torch

    policy = model.policy
    device = policy.device
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    wt = torch.as_tensor(w / max(float(np.mean(w)), 1e-9), dtype=torch.float32, device=device)
    Xr = torch.as_tensor(X_ref, dtype=torch.float32, device=device) if len(X_ref) else None
    with torch.no_grad():
        ref_mean = policy.get_distribution(Xr).distribution.mean[:, 0].detach() if Xr is not None else None

    def live_loss(idx=None):
        xb = Xt if idx is None else Xt[idx]
        m = policy.get_distribution(xb).distribution.mean[:, 0]
        err = (torch.clamp(m, -1.0, 1.0) - (yt if idx is None else yt[idx])) ** 2
        return (err * (wt if idx is None else wt[idx])).mean()

    params = [p for p in policy.mlp_extractor.policy_net.parameters()] + [p for p in policy.action_net.parameters()]
    params = [p for p in params if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(lr))
    policy.set_training_mode(True)
    with torch.no_grad():
        loss0 = float(live_loss())
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    n = len(X)
    for _ in range(int(epochs)):
        perm = torch.randperm(n, generator=g)
        for k in range(0, n, int(batch_size)):
            idx = perm[k: k + int(batch_size)].to(device)
            loss = live_loss(idx)
            if Xr is not None and anchor_weight > 0:
                ridx = torch.randint(0, len(Xr), (min(len(Xr), int(batch_size)),), generator=g).to(device)
                m_ref = policy.get_distribution(Xr[ridx]).distribution.mean[:, 0]
                loss = loss + float(anchor_weight) * ((m_ref - ref_mean[ridx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    policy.set_training_mode(False)
    with torch.no_grad():
        loss1 = float(live_loss())
        drift = float(((policy.get_distribution(Xr).distribution.mean[:, 0] - ref_mean) ** 2).mean()) if Xr is not None else 0.0
    return {"loss_before": loss0, "loss_after": loss1, "anchor_drift": drift, "samples": int(n), "epochs": int(epochs)}


# ---------------------------------------------------------------------- orchestration
def _score(model, ds_test, env_cfg: dict, tickers: list[str], max_bars: int) -> float:
    from ..agent.evaluate import aggregate, evaluate

    summary, _ = evaluate(model, ds_test, env_cfg, tickers, max_bars=max_bars)
    agg = aggregate(summary)
    return float(agg.get("mean_sharpe", -np.inf) + agg.get("median_excess_return", 0.0))


def _samples_key(samples: dict) -> str:
    dates = sorted({m["date"] for m in samples["meta"]})
    settled = sum(m["settled"] for m in samples["meta"])
    return hashlib.sha1(f"{len(samples['meta'])}|{settled}|{dates[-1] if dates else ''}".encode()).hexdigest()[:12]


def learn(cfg: Config, force: bool = False, frames: dict[str, pd.DataFrame] | None = None, dataset=None, today=None) -> dict:
    """Fine-tune every member of the current policy on the hindsight labels and keep the ones that pass
    the out-of-sample guard.  Idempotent: skips when nothing new has settled since the last run."""
    from ..agent.policy import EnsemblePolicy, PolicyBundle, load_model
    from ..agent.train import cached_dataset

    s = settings(cfg)
    ckpt = cfg.path("train.checkpoint_dir", "models/policy")
    report: dict = {"ran_at": datetime.now(timezone.utc).isoformat(), "checkpoint": str(ckpt)}
    if not s.get("enabled", True) or not PolicyBundle.exists(ckpt):
        report["skipped"] = "disabled" if not s.get("enabled", True) else f"no policy in {ckpt}"
        return report
    bundle = PolicyBundle.load(ckpt)
    samples = build_samples(cfg, bundle.layout.obs_dim, frames=frames, today=today)
    n = len(samples["meta"])
    report.update({"samples": n, "settled_horizons": int(sum(m["settled"] for m in samples["meta"])),
                   "dates": sorted({m["date"] for m in samples["meta"]})[-5:]})
    state_file = ckpt / "hindsight.json"
    prev = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    key = _samples_key(samples) if n else ""
    if n < int(s["min_samples"]):
        report["skipped"] = f"only {n} labelled samples (need {s['min_samples']})"
        log.info("hindsight: %s", report["skipped"])
        return report
    if not force and prev.get("samples_key") == key and prev.get("signature") == bundle.layout.signature():
        report["skipped"] = "nothing new has settled since the last run"
        log.info("hindsight: %s", report["skipped"])
        return report

    members = bundle.model.members if isinstance(bundle.model, EnsemblePolicy) else [bundle.model]
    paths = bundle.model.paths if isinstance(bundle.model, EnsemblePolicy) else [str(ckpt / "latest.zip")]
    env_cfg = env_settings(cfg)
    ds = dataset if dataset is not None else cached_dataset(cfg, max_age_days=1e9)
    guard = bool(s["guard"])
    ds_train = ds_test = None
    if ds is not None:
        train_end = cfg.get_path("data.train_end")
        ds_train, ds_test = ds.split(train_end) if train_end else (ds, ds)
        if len(ds_test) == 0:
            ds_test = ds_train
        if ds.layout.signature() != bundle.layout.signature():
            ds_train = ds_test = None
    if guard and ds_test is None:
        report["skipped"] = "no simulator dataset for the out-of-sample guard - not touching the policy"
        log.warning("hindsight: %s", report["skipped"])
        return report
    X_ref = reference_states(members[0], ds_train, env_cfg, int(s["ref_steps"])) if ds_train is not None else np.zeros((0, bundle.layout.obs_dim), np.float32)
    eval_tickers = ds_test.tickers[: int(s["eval_tickers"])] if ds_test is not None else []
    per_member = {}
    accepted = 0
    for model, path in zip(members, paths):
        rec: dict = {"path": path}
        try:
            before = _score(model, ds_test, env_cfg, eval_tickers, int(s["eval_bars"])) if guard else None
            fit = finetune(model, samples["X"], samples["y"], samples["w"], X_ref, epochs=int(s["epochs"]), lr=float(s["lr"]),
                           batch_size=int(s["batch_size"]), anchor_weight=float(s["anchor_weight"]))
            rec.update(fit)
            after = _score(model, ds_test, env_cfg, eval_tickers, int(s["eval_bars"])) if guard else None
            rec.update({"score_before": before, "score_after": after})
            ok = fit["loss_after"] <= fit["loss_before"] and (not guard or after >= before - float(s["max_score_drop"]))
            if ok:
                model.save(path)
                accepted += 1
                rec["accepted"] = True
                log.info("hindsight: %s updated (loss %.4f -> %.4f, score %s -> %s)", Path(path).name, fit["loss_before"], fit["loss_after"],
                         f"{before:.3f}" if before is not None else "-", f"{after:.3f}" if after is not None else "-")
            else:
                rec["accepted"] = False
                reloaded = load_model(path, bundle.algo)
                model.policy.load_state_dict(reloaded.policy.state_dict())
                log.info("hindsight: %s rejected (loss %.4f -> %.4f, score %s -> %s) - kept the previous weights", Path(path).name,
                         fit["loss_before"], fit["loss_after"], f"{before:.3f}" if before is not None else "-", f"{after:.3f}" if after is not None else "-")
        except Exception as e:  # noqa: BLE001 - one member failing must not lose the others
            rec.update({"accepted": False, "error": str(e)[:200]})
            log.warning("hindsight: %s failed: %s", path, e)
        per_member[Path(path).parent.name + "/" + Path(path).name] = rec
    report.update({"members": per_member, "accepted": accepted, "samples_key": key, "signature": bundle.layout.signature(),
                   "mean_target": float(np.mean(samples["y"])), "mean_action": float(np.mean([m["action"] for m in samples["meta"]]))})
    history = prev.get("history", [])[-30:]
    history.append({"ran_at": report["ran_at"], "samples": n, "accepted": accepted})
    state_file.write_text(json.dumps({**report, "history": history}, indent=1, default=str), encoding="utf-8")
    return report
