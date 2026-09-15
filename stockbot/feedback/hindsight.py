"""Learn from the paper trades: hindsight fine-tuning of the policy, one period at a time, guarded by the
out-of-sample score.

Every paper / live decision is stored with the exact observation the policy saw.  Once a period is over
we know the best play for it, and the best day play is not the best week play, which is not the best
month play - so each period is learned as ONE unit with its own labels:

* ``day``   (every morning)      - yesterday's decisions, labelled by yesterday's close;
* ``week``  (the weekend)        - the week that just ended, all of it as one unit;
* ``month`` (after month end)    - the month as one unit;  ``year`` likewise.

For each name the label is the exposure level that the fee-aware best path over that unit
(:func:`~stockbot.feedback.attribution.path_oracle`, paying the fee on every switch, starting from the
exposure we actually had) holds on the day of the decision, turned into a conviction in -1..1.  Samples
are weighted by how far our size was from that level (the regret of the decision).

The fine-tune is a few epochs of supervised regression of the policy's action mean toward those labels
(actor parameters only), anchored to the policy's own answers on simulator states so it does not forget
what the simulations taught it (hindsight experience replay / DAgger-style imitation with an anchor).
Each ensemble member is then re-scored on the held-out simulator window and the update is kept only if
the score did not drop by more than ``max_score_drop`` - the trainer's own yardstick.  ``stockbot review
--learn --period day`` runs before the open in the session workflow (with a time budget that ends before
the pre-open warm-up); ``--learn --due`` runs the week / month / year units on the weekend.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, env_settings
from ..execution.fees import FeeBook
from ..logging_utils import get_logger
from .attribution import LEVELS, path_oracle
from .experience import ExperienceStore

log = get_logger(__name__)

PERIODS = ("day", "week", "month", "year")
DEFAULTS = {"min_samples": 40, "epochs": 8, "lr": 5e-5, "batch_size": 256, "anchor_weight": 1.0, "ref_steps": 2048,
            "guard": True, "max_score_drop": 0.02, "eval_tickers": 8, "eval_bars": 500,
            # with a generous budget (the morning run) the guard uses the trainer's full yardstick
            "big_budget_minutes": 20, "eval_tickers_full": 12, "eval_bars_full": 750, "epochs_full": 16}


def settings(cfg: Config) -> dict:
    s = dict(DEFAULTS)
    s.update({k: v for k, v in (cfg.get_path("feedback.hindsight", {}) or {}).items() if v is not None})
    return s


def experience_stores(cfg: Config) -> list[ExperienceStore]:
    """The main account's store plus every extra account's (``accounts:``): all of them are lessons for the one policy."""
    from ..config import account_config, account_names

    paths = [cfg.path("feedback.experience_file", "data/experience/trades.jsonl")]
    for name in account_names(cfg):
        try:
            p = account_config(cfg, name).path("feedback.experience_file", "data/experience/trades.jsonl")
        except Exception:  # noqa: BLE001
            continue
        if p not in paths:
            paths.append(p)
    return [ExperienceStore(p) for p in paths]


def load_all_records(cfg: Config) -> pd.DataFrame | None:
    frames = []
    for store in experience_stores(cfg):
        df = store.load()
        if df is not None and len(df):
            df = df.copy()
            df["_store"] = str(store.path)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def unit_bounds(period: str, end: date) -> tuple[date, date]:
    from .review import period_bounds

    return period_bounds(period, end)


def default_unit_end(period: str, today: date, last_decision: date | None) -> date | None:
    """The most recent finished unit: yesterday's session for ``day`` (the latest decision date before
    today), the last Friday for ``week``, the previous month end / year end otherwise."""
    if period == "day":
        return last_decision if last_decision is not None and last_decision < today else None
    if period == "week":
        friday = today - timedelta(days=(today.weekday() - 4) % 7)
        return friday if friday < today else friday - timedelta(days=7)
    if period == "month":
        return today.replace(day=1) - timedelta(days=1)
    if period == "year":
        return date(today.year - 1, 12, 31)
    raise ValueError(period)


# ---------------------------------------------------------------------- samples
def unit_paths(cfg: Config, period: str, end: date | None, recs: pd.DataFrame | None, frames: dict[str, pd.DataFrame] | None = None,
               today: date | None = None, refresh: bool = False, levels: tuple[float, ...] = LEVELS) -> tuple[date | None, date | None, list[dict]]:
    """Every recorded decision of one unit (``period`` ending ``end``) next to the fee-aware best path of its name over
    that unit: our level vs the best level on the decision day, the move to the unit's end, the whole best path.
    The same rows feed the review's decision ledger (what was wrong, what the better move was) and the hindsight labels."""
    today = today or datetime.now(timezone.utc).date()
    if recs is None or len(recs) == 0:
        return None, None, []
    dec = recs[recs["type"] == "decision"].copy()
    if len(dec) == 0:
        return None, None, []
    dec["_d"] = pd.to_datetime(dec["date"]).dt.date
    if end is None:
        end = default_unit_end(period, today, dec["_d"].max())
        if end is None:
            return None, None, []
    start, end = unit_bounds(period, end)
    dec = dec[(dec["_d"] >= start) & (dec["_d"] <= end)]
    if len(dec) == 0:
        return start, end, []
    tickers = sorted(dec["ticker"].unique().tolist())
    if frames is None:
        from ..agent.train import load_frames

        frames = load_frames(cfg, offline=not refresh, refresh=refresh, tickers=tickers)   # the morning run pulls yesterday's close
    fees = FeeBook.from_config(cfg)
    max_position = float(cfg.get_path("execution.max_position", 0.10))
    allow_short = bool(env_settings(cfg).get("allow_short", False))
    rows: list[dict] = []
    for t in tickers:
        df = frames.get(t)
        if df is None or len(df) == 0:
            continue
        closes = df["close"].astype(float)
        closes = closes[[d.date() < today for d in closes.index]]          # never a partial bar of today
        unit = closes[[start <= d.date() <= end for d in closes.index]]
        if len(unit) == 0:
            continue
        sub = dec[dec["ticker"] == t].sort_values("_d")
        by_day = {r["_d"]: r for _, r in sub.iterrows()}
        if max(by_day) > unit.index[-1].date():
            continue                                                       # the unit has not settled yet
        days = [d.date() for d in unit.index]
        first = by_day.get(days[0])
        prev_close = closes.shift(1).reindex(unit.index).iloc[0]
        p_prev = float(first["price"]) if first is not None and float(first.get("price") or 0.0) > 0 else float(prev_close)
        if not np.isfinite(p_prev) or p_prev <= 0:
            continue
        vals = unit.to_numpy(float)
        r = np.concatenate([[vals[0] / p_prev - 1.0], vals[1:] / vals[:-1] - 1.0])
        first_rec = sub.iloc[0]
        obs0 = first_rec["obs"]
        start_level = float(np.clip(np.asarray(obs0, float)[-6], 0.0, 1.0)) if isinstance(obs0, list) and len(obs0) >= 6 else 0.0
        sched = fees.for_ticker(t)
        slice_notional = max(float(first_rec.get("equity") or 0.0) * max_position, 1.0)
        fee_frac = float(sched.cost(slice_notional / p_prev, p_prev, "buy")) / slice_notional if sched is not None else 0.0
        best, path = path_oracle(r, fee_frac, levels, start_level=start_level)
        day_str = [d.isoformat() for d in days]
        for k, d in enumerate(days):
            rec = by_day.get(d)
            if rec is None:
                continue
            price = float(rec.get("price") or 0.0) or float(vals[k])
            chosen = rec.get("chosen") if "chosen" in rec else None
            rows.append({"ticker": t, "date": str(rec["date"]), "action": str(rec.get("decision") or ""), "k": k, "days": day_str,
                         "level_best": float(path[k]), "level_ours": float(np.clip(float(rec.get("target_exposure") or 0.0), 0.0, 1.0)),
                         "r_day": float(r[k]), "r_to_end": float(vals[-1] / price - 1.0) if price > 0 else 0.0, "path": [float(x) for x in path],
                         "obs": rec["obs"], "equity": float(rec.get("equity") or 0.0), "chosen": None if chosen is None or chosen != chosen else bool(chosen),
                         "action_raw": float(rec.get("action") or 0.0), "unit_best_pnl": float(best), "allow_short": allow_short})
    return start, end, rows


VERDICTS = ("right", "missed", "wrong side", "under-sized", "over-sized")


def verdict_of(ours: float, best: float) -> str:
    if ours <= 0.05 and best >= 0.25:
        return "missed"
    if ours >= 0.25 and best <= 0.05:
        return "wrong side"
    if best > ours + 0.25:
        return "under-sized"
    if ours > best + 0.25:
        return "over-sized"
    return "right"


def better_move(row: dict) -> str:
    """What the fee-aware best path did with this name from the decision day on, in words."""
    path, k, days = row["path"], int(row["k"]), row["days"]
    best = path[k]
    if best <= 0.05:
        nxt = next((j for j in range(k + 1, len(path)) if path[j] > 0.05), None)
        return "stay out" + (f" until {days[nxt]}" if nxt is not None else " for the whole unit")
    txt = f"hold {best:.0%} of a slot"
    exit_j = next((j for j in range(k + 1, len(path)) if path[j] <= 0.05), None)
    if exit_j is not None:
        txt += f", sell on {days[exit_j]}"
    return txt


def unit_verdicts(rows: list[dict], max_position: float = 0.10) -> tuple[list[dict], dict]:
    """The decision ledger of a unit and its summary: every decision with its verdict, its regret (fraction of equity)
    and the better move; counts per verdict, hit rate, total regret, the worst decisions first."""
    ledger = []
    for row in rows:
        v = verdict_of(row["level_ours"], row["level_best"])
        regret = max(0.0, (row["level_best"] - row["level_ours"]) * row["r_day"]) * max_position
        ledger.append({"ticker": row["ticker"], "date": row["date"], "action": row["action"], "level_ours": row["level_ours"],
                       "level_best": row["level_best"], "verdict": v, "better": better_move(row), "r_day": row["r_day"],
                       "r_to_end": row["r_to_end"], "regret": regret})
    ledger.sort(key=lambda d: -d["regret"])
    n = len(ledger)
    counts = {v: sum(1 for d in ledger if d["verdict"] == v) for v in VERDICTS}
    summary = {"n": n, "counts": counts, "hit_rate": counts["right"] / n if n else None, "regret": float(sum(d["regret"] for d in ledger)),
               "worst": ledger[:5]}
    return ledger, summary


def build_samples(cfg: Config, obs_dim: int, period: str = "day", end: date | None = None, frames: dict[str, pd.DataFrame] | None = None,
                  refresh: bool = False, today: date | None = None, levels: tuple[float, ...] = LEVELS) -> dict:
    """Hindsight-labelled samples for one unit: ``X`` observations, ``y`` target conviction in -1..1, ``w`` weights, plus
    per-sample metadata.  Only decisions whose observation matches the current layout are used, and only when the
    unit's bars have all settled (every account's records count)."""
    recs = load_all_records(cfg)
    start, end, rows = unit_paths(cfg, period, end, recs, frames=frames, today=today, refresh=refresh, levels=levels)
    empty = {"X": np.zeros((0, obs_dim), np.float32), "y": np.zeros(0, np.float32), "w": np.zeros(0, np.float32), "meta": [],
             "period": period, "start": start.isoformat() if start else None, "end": end.isoformat() if end else None,
             "decisions_in_unit": len(rows)}
    rows = [r for r in rows if isinstance(r["obs"], list) and len(r["obs"]) == obs_dim]   # only the current layout
    if not rows:
        return empty
    X, y, w, meta = [], [], [], []
    for row in rows:
        level, ours = row["level_best"], row["level_ours"]
        target = level if row["allow_short"] else 2.0 * level - 1.0
        focus = 1.0 if row["chosen"] is None or row["chosen"] else 0.25     # names the rank layer did not pick teach the policy less
        X.append(np.asarray(row["obs"], np.float32))
        y.append(target)
        w.append(max(0.1, abs(level - ours)) * focus)
        meta.append({"ticker": row["ticker"], "date": row["date"], "level_best": level, "level_ours": ours, "target": target,
                     "r_day": row["r_day"], "unit_best_pnl": row["unit_best_pnl"], "action": row["action_raw"],
                     "verdict": verdict_of(ours, level)})
    return {"X": np.asarray(X, np.float32), "y": np.asarray(y, np.float32), "w": np.asarray(w, np.float32), "meta": meta,
            "period": period, "start": start.isoformat(), "end": end.isoformat(), "decisions_in_unit": empty["decisions_in_unit"]}


# ---------------------------------------------------------------------- fine-tune
def policy_mean(model, X: np.ndarray) -> np.ndarray:
    import torch

    policy = model.policy
    with torch.no_grad():
        obs = torch.as_tensor(np.asarray(X, np.float32), device=policy.device)
        return policy.get_distribution(obs).distribution.mean.detach().cpu().numpy().reshape(len(X), -1)[:, 0]


def reference_states(model, dataset, env_cfg: dict, n_steps: int, seed: int = 0, obs_dim: int | None = None) -> np.ndarray:
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
            o = np.asarray(obs, np.float32)
            if obs_dim and o.shape[-1] > obs_dim:
                o = o[:obs_dim]
            out.append(o)
            action, _ = model.predict(o.reshape(1, -1), deterministic=True)
            obs, _, terminated, truncated, _ = env.step(np.asarray(action).reshape(-1))
            done = terminated or truncated
    return np.asarray(out, np.float32)


def finetune(model, X: np.ndarray, y: np.ndarray, w: np.ndarray, X_ref: np.ndarray, epochs: int = 8, lr: float = 5e-5,
             batch_size: int = 256, anchor_weight: float = 1.0, seed: int = 0, deadline: float | None = None) -> dict:
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
    done_epochs = 0
    for _ in range(int(epochs)):
        if deadline is not None and done_epochs >= 1 and time.time() > deadline:
            break                                                          # out of time: keep what was learned so far
        done_epochs += 1
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
    return {"loss_before": loss0, "loss_after": loss1, "anchor_drift": drift, "samples": int(n), "epochs": int(done_epochs)}


# ---------------------------------------------------------------------- orchestration
def _score(model, ds_test, env_cfg: dict, tickers: list[str], max_bars: int, cash_levels: list[float] | None = None) -> float:
    from ..agent.evaluate import aggregate, evaluate

    summary, _ = evaluate(model, ds_test, env_cfg, tickers, max_bars=max_bars, cash_levels=cash_levels)
    agg = aggregate(summary)
    return float(agg.get("mean_sharpe", -np.inf) + agg.get("median_excess_return", 0.0))


def _state(ckpt: Path) -> dict:
    f = ckpt / "hindsight.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def learn(cfg: Config, period: str = "day", end: date | None = None, force: bool = False, frames: dict[str, pd.DataFrame] | None = None,
          dataset=None, refresh: bool = False, budget_minutes: float | None = None, today: date | None = None) -> dict:
    """Fine-tune every member of the current policy on one unit's hindsight labels and keep the members that
    pass the out-of-sample guard.  Idempotent per unit (``models/policy/hindsight.json`` remembers what was
    learned).  ``budget_minutes`` bounds the wall-clock: epochs stop early and later members are skipped
    rather than overrunning (the morning run must end before the pre-open warm-up); a generous budget
    switches the guard to the trainer's full yardstick and more epochs."""
    from ..agent.policy import EnsemblePolicy, PolicyBundle, load_model
    from ..agent.train import cached_dataset

    t_start = time.time()
    t_end = t_start + 60.0 * float(budget_minutes) if budget_minutes else None
    s = settings(cfg)
    ckpt = cfg.path("train.checkpoint_dir", "models/policy")
    report: dict = {"period": period, "ran_at": datetime.now(timezone.utc).isoformat(), "checkpoint": str(ckpt), "budget_minutes": budget_minutes}
    if not s.get("enabled", True) or not PolicyBundle.exists(ckpt):
        report["skipped"] = "disabled" if not s.get("enabled", True) else f"no policy in {ckpt}"
        return report
    bundle = PolicyBundle.load(ckpt)
    from ..signals.layout import PORTFOLIO_FEATURES

    obs_dim_now = bundle.layout.signal_dim + len(PORTFOLIO_FEATURES)   # what the runner records today, whatever the policy was trained on
    samples = build_samples(cfg, obs_dim_now, period=period, end=end, frames=frames, refresh=refresh, today=today)
    n = len(samples["meta"])
    expected = bundle.model_obs_dim()
    if n and expected and samples["X"].shape[1] > expected:      # the policy predates the newest portfolio feature
        samples["X"] = samples["X"][:, :expected]
    report.update({"samples": n, "start": samples["start"], "end": samples["end"], "signature": bundle.layout.signature()})
    prev = _state(ckpt)
    learned = prev.get("learned", {})
    if samples["end"] is None:
        report["skipped"] = f"no finished {period} unit to learn from yet"
        log.info("hindsight: %s", report["skipped"])
        return report
    if not force and learned.get(period) == samples["end"] and prev.get("signature") == bundle.layout.signature():
        report["skipped"] = f"the {period} ending {samples['end']} was already learned"
        log.info("hindsight: %s", report["skipped"])
        return report
    if n < int(s["min_samples"]):
        other = samples.get("decisions_in_unit", n) - n
        report["skipped"] = f"only {n} labelled decisions in the {period} ending {samples['end']} (need {s['min_samples']})" +             (f"; {other} more were recorded under an older observation layout" if other > 0 else "")
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
    X_ref = reference_states(members[0], ds_train, env_cfg, int(s["ref_steps"]), obs_dim=expected) if ds_train is not None \
        else np.zeros((0, expected or bundle.layout.obs_dim), np.float32)
    generous = budget_minutes is None or float(budget_minutes) >= float(s["big_budget_minutes"])
    n_eval, eval_bars, epochs = (int(s["eval_tickers_full"]), int(s["eval_bars_full"]), int(s["epochs_full"])) if generous \
        else (int(s["eval_tickers"]), int(s["eval_bars"]), int(s["epochs"]))
    eval_tickers = ds_test.tickers[:n_eval] if ds_test is not None else []
    cash_levels = cfg.get_path("train.eval_cash")
    report["yardstick"] = {"tickers": len(eval_tickers), "bars": eval_bars, "epochs": epochs}
    per_member = {}
    accepted = 0
    member_seconds: float | None = None
    for model, path in zip(members, paths):
        rec: dict = {"path": path}
        name = Path(path).parent.name + "/" + Path(path).name
        if t_end is not None and member_seconds is not None and time.time() + 1.2 * member_seconds > t_end:
            rec.update({"accepted": False, "skipped": "out of time"})
            per_member[name] = rec
            log.info("hindsight: %s skipped - the time budget is used up", Path(path).name)
            continue
        t_member = time.time()
        try:
            before = _score(model, ds_test, env_cfg, eval_tickers, eval_bars, cash_levels) if guard else None
            fit = finetune(model, samples["X"], samples["y"], samples["w"], X_ref, epochs=epochs, lr=float(s["lr"]),
                           batch_size=int(s["batch_size"]), anchor_weight=float(s["anchor_weight"]),
                           deadline=None if t_end is None else t_end - (time.time() - t_member))   # leave time for the second eval
            rec.update(fit)
            after = _score(model, ds_test, env_cfg, eval_tickers, eval_bars, cash_levels) if guard else None
            rec.update({"score_before": before, "score_after": after})
            ok = fit["loss_after"] <= fit["loss_before"] and (not guard or after >= before - float(s["max_score_drop"]))
            if ok:
                tmp = Path(path).with_suffix(".tmp.zip")             # atomic: a kill mid-save never leaves a broken member
                model.save(str(tmp))
                os.replace(tmp, path)
                accepted += 1
                rec["accepted"] = True
                log.info("hindsight: %s updated on the %s ending %s (loss %.4f -> %.4f, score %s -> %s)", Path(path).name, period, samples["end"],
                         fit["loss_before"], fit["loss_after"], f"{before:.3f}" if before is not None else "-", f"{after:.3f}" if after is not None else "-")
            else:
                rec["accepted"] = False
                reloaded = load_model(path, bundle.algo)
                model.policy.load_state_dict(reloaded.policy.state_dict())
                log.info("hindsight: %s rejected (loss %.4f -> %.4f, score %s -> %s) - kept the previous weights", Path(path).name,
                         fit["loss_before"], fit["loss_after"], f"{before:.3f}" if before is not None else "-", f"{after:.3f}" if after is not None else "-")
        except Exception as e:  # noqa: BLE001 - one member failing must not lose the others
            rec.update({"accepted": False, "error": str(e)[:200]})
            log.warning("hindsight: %s failed: %s", path, e)
        member_seconds = time.time() - t_member
        rec["seconds"] = round(member_seconds, 1)
        per_member[name] = rec
    report["seconds"] = round(time.time() - t_start, 1)
    report.update({"members": per_member, "accepted": accepted, "mean_target": float(np.mean(samples["y"])),
                   "mean_level_best": float(np.mean([m["level_best"] for m in samples["meta"]])),
                   "mean_level_ours": float(np.mean([m["level_ours"] for m in samples["meta"]]))})
    if period != "day" or not learned.get("day") or learned["day"] < samples["end"]:
        learned[period] = samples["end"]
    history = prev.get("history", [])[-60:]
    history.append({"ran_at": report["ran_at"], "period": period, "end": samples["end"], "samples": n, "accepted": accepted})
    (ckpt / "hindsight.json").write_text(json.dumps({"learned": learned, "signature": bundle.layout.signature(), "last": report, "history": history},
                                                    indent=1, default=str), encoding="utf-8")
    return report


def learn_due(cfg: Config, today: date | None = None, force: bool = False, frames: dict[str, pd.DataFrame] | None = None, dataset=None,
              refresh: bool = False, budget_minutes: float | None = None) -> dict[str, dict]:
    """Every finished unit not learned yet: yesterday's day, the last week, the previous month, the previous year."""
    today = today or datetime.now(timezone.utc).date()
    ckpt = cfg.path("train.checkpoint_dir", "models/policy")
    learned = _state(ckpt).get("learned", {})
    recs = load_all_records(cfg)
    last_dec = None
    if recs is not None and len(recs):
        dec = recs[recs["type"] == "decision"]
        if len(dec):
            last_dec = pd.to_datetime(dec["date"]).dt.date.max()
    out: dict[str, dict] = {}
    t0 = time.time()
    units: list[tuple[str, date]] = []
    if last_dec is not None:                                           # every session day not learned yet (a missed morning catches up)
        learned_day = date.fromisoformat(learned["day"]) if learned.get("day") and not force else None
        days = sorted({d for d in pd.to_datetime(dec["date"]).dt.date.unique() if d < today and (learned_day is None or d > learned_day)})
        units += [("day", d) for d in days[-5:]]
    for period in ("week", "month", "year"):
        end = default_unit_end(period, today, last_dec)
        if end is not None and (force or not learned.get(period) or date.fromisoformat(learned[period]) < end):
            units.append((period, end))
    for period, end in units:
        key = period if period != "day" else f"day {end.isoformat()}"
        left = None if budget_minutes is None else max(0.0, float(budget_minutes) - (time.time() - t0) / 60.0)
        if left is not None and left < 1.0:
            out[key] = {"period": period, "end": end.isoformat(), "skipped": "out of time"}
            continue
        try:
            out[key] = learn(cfg, period=period, end=end, force=force, frames=frames, dataset=dataset, refresh=refresh,
                             budget_minutes=left, today=today)
        except Exception as e:  # noqa: BLE001
            log.warning("hindsight %s failed: %s", key, e)
            out[key] = {"period": period, "end": end.isoformat(), "error": str(e)[:200]}
    return out


def _samples_key(samples: dict) -> str:   # kept for callers that want a fingerprint of a unit's labels
    return hashlib.sha1(f"{samples.get('period')}|{samples.get('end')}|{len(samples['meta'])}".encode()).hexdigest()[:12]
