"""Training pipeline: data -> signals -> dataset -> parallel simulators -> PPO/SAC -> checkpoints."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from ..config import Config, env_settings
from ..data.loader import load_universe, synthetic_universe
from ..env.dataset import MarketDataset
from ..env.vec import make_vec_env
from ..logging_utils import get_logger
from ..signals.layout import ObservationLayout
from ..signals.registry import build_context, build_layout, build_providers
from .evaluate import aggregate, evaluate
from .policy import PolicyBundle, build_model, load_model

log = get_logger(__name__)


# ---------------------------------------------------------------------- data / dataset
def load_frames(cfg: Config, offline: bool = False, refresh: bool = False, synthetic: bool = False,
                tickers: list[str] | None = None):
    tickers = tickers or list(cfg.get("universe", []))
    if synthetic:
        log.info("using SYNTHETIC data for %d tickers", len(tickers))
        return synthetic_universe(tickers, n=int(cfg.get_path("data.synthetic_bars", 2500)))
    d = cfg.section("data")
    return load_universe(tickers, d.get("start", "2008-01-01"), d.get("end"), d.get("interval", "1d"),
                         cfg.path("data.cache_dir", "data/cache"), float(d.get("refresh_days", 1)),
                         offline=offline, refresh=refresh)


def prepare_dataset(cfg: Config, offline: bool = False, refresh: bool = False, synthetic: bool = False,
                    fit: bool = True, save: bool = True, with_llm: bool = True) -> tuple[MarketDataset, list, object]:
    frames = load_frames(cfg, offline=offline, refresh=refresh, synthetic=synthetic)
    ctx = build_context(cfg, with_llm=with_llm, with_news=True)
    providers = build_providers(cfg, ctx)
    layout = build_layout(providers)
    train_end = cfg.get_path("data.train_end")
    t0 = time.time()
    ds = MarketDataset.build(frames, providers, layout, ctx, fit=fit, train_end=train_end)
    log.info("dataset built: %d tickers, obs_dim=%d, layout=%s (%.1fs)", len(ds), layout.obs_dim, layout.signature(), time.time() - t0)
    if save:
        ds.save(cfg.path("models_dir", "models") / "dataset")
    return ds, providers, ctx


def _archive_old_layout(ckpt: Path, signature: str) -> None:
    """A policy trained on a different signal layout is moved to ``archive/<signature>`` instead of being overwritten."""
    layout_file = ckpt / "layout.json"
    if not layout_file.exists():
        return
    try:
        old_sig = ObservationLayout.load(layout_file).signature()
    except Exception:  # noqa: BLE001
        return
    if old_sig == signature:
        return
    import shutil

    dest = ckpt / "archive" / old_sig
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("latest.zip", "best.zip", "best.json", "layout.json", "meta.json", "train_log.json", "ensemble.json", "ensemble"):
        f = ckpt / name
        if f.exists():
            shutil.move(str(f), str(dest / name))
    log.info("archived the previous policy (layout %s) to %s", old_sig, dest)


def cached_dataset(cfg: Config, max_age_days: float = 3.0) -> MarketDataset | None:
    """Reuse ``models/dataset`` when it was built recently for the same universe, split and signal layout."""
    folder = cfg.path("models_dir", "models") / "dataset"
    meta_file = folder / "meta.json"
    if not (folder / "layout.json").exists() or not meta_file.exists():
        return None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        ctx = build_context(cfg, with_llm=False, with_news=False)
        layout = build_layout(build_providers(cfg, ctx))
        same = (meta.get("signature") == layout.signature() and set(meta.get("tickers", [])) == set(cfg.get("universe", []))
                and str(meta.get("train_end")) == str(cfg.get_path("data.train_end")))
        fresh = (time.time() - meta_file.stat().st_mtime) < max_age_days * 86400
        if same and fresh:
            ds = MarketDataset.load(folder)
            log.info("reusing cached dataset %s (%d tickers, layout %s)", folder, len(ds), layout.signature())
            return ds
    except Exception as e:  # noqa: BLE001
        log.debug("cached dataset not reusable: %s", e)
    return None


# ---------------------------------------------------------------------- callbacks
class TimeBudgetCallback(BaseCallback):
    """Stop ``learn()`` cleanly after ``max_seconds`` of wall-clock time (unattended runs with a deadline)."""

    def __init__(self, max_seconds: float, verbose: int = 0):
        super().__init__(verbose)
        self.max_seconds = float(max_seconds)
        self.t0 = time.time()
        self.stopped = False

    def _on_training_start(self) -> None:
        self.t0 = time.time()

    def _on_step(self) -> bool:
        if time.time() - self.t0 >= self.max_seconds:
            if not self.stopped:
                log.info("time budget of %.1f minutes reached at %d timesteps - stopping training", self.max_seconds / 60.0, self.num_timesteps)
            self.stopped = True
            return False
        return True


class OOSEvalCallback(BaseCallback):
    """Periodically evaluate on held-out data and keep the best checkpoint."""

    def __init__(self, ds_test: MarketDataset, env_cfg: dict, eval_freq: int, n_tickers: int, save_dir: Path,
                 layout, meta: dict, max_bars: int = 504, verbose: int = 1, keep_best: bool = False):
        super().__init__(verbose)
        self.ds_test = ds_test
        self.env_cfg = env_cfg
        self.eval_freq = int(eval_freq)
        self.tickers = ds_test.tickers[: max(1, n_tickers)]
        self.save_dir = Path(save_dir)
        self.layout = layout
        self.meta = meta
        self.max_bars = max_bars
        self.best = -np.inf
        # scores are only comparable when the evaluation set and the environment economics are the same
        self.eval_key = hashlib.sha1(json.dumps({
            "tickers": self.tickers, "max_bars": max_bars, "signature": layout.signature(), "train_end": str(meta.get("train_end")),
            "env": {k: env_cfg.get(k) for k in ("allow_short", "vol_target", "benchmark_mix", "turnover_penalty",
                                              "short_penalty", "deadband", "reward", "commission", "slippage")},
        }, sort_keys=True, default=str).encode()).hexdigest()[:12]
        best_file = self.save_dir / "best.json"
        if best_file.exists():
            try:
                prev = json.loads(best_file.read_text(encoding="utf-8"))
                if keep_best and prev.get("eval_key") == self.eval_key:
                    self.best = float(prev.get("score", -np.inf))  # continue improving on the same yardstick
                else:  # different yardstick: keep the old files aside, start a fresh comparison
                    for name in ("best.zip", "best.json"):
                        f = self.save_dir / name
                        if f.exists():
                            f.replace(self.save_dir / name.replace("best", "best_prev"))
            except Exception:  # noqa: BLE001
                pass
        self.last_eval = 0
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps
        summary, _ = evaluate(self.model, self.ds_test, self.env_cfg, self.tickers, max_bars=self.max_bars)
        agg = aggregate(summary)
        # median excess return: robust to one runaway buy-and-hold ticker dominating the mean
        score = agg.get("mean_sharpe", -np.inf) + agg.get("median_excess_return", 0.0)
        agg["timesteps"] = int(self.num_timesteps)
        agg["score"] = float(score)
        self.history.append(agg)
        log.info("eval @%d: sharpe=%.2f (b&h %.2f) median excess=%.1f%% win_vs_bh=%.0f%% exposure=%.2f short=%.0f%% score=%.2f",
                 self.num_timesteps, agg.get("mean_sharpe", 0), agg.get("mean_bh_sharpe", 0),
                 100 * agg.get("median_excess_return", 0), 100 * agg.get("win_rate_vs_bh", 0),
                 agg.get("mean_avg_exposure", 0), 100 * agg.get("mean_short_share", 0), score)
        if score > self.best:
            self.best = score
            PolicyBundle(self.model, self.layout, {**self.meta, "best_eval": agg}).save(self.save_dir, "best")
            (self.save_dir / "best.json").write_text(json.dumps({**agg, "signature": self.layout.signature(),
                                                                 "eval_key": self.eval_key}, indent=1), encoding="utf-8")
        return True


# ---------------------------------------------------------------------- training
def train(cfg: Config, total_timesteps: int | None = None, resume: str | None = None, dataset: MarketDataset | None = None,
          offline: bool = False, refresh: bool = False, synthetic: bool = False, n_envs: int | None = None,
          seeds: int | None = None, max_minutes: float | None = None) -> Path:
    """``max_minutes`` (optional) stops the PPO updates after that much wall-clock time; whatever was
    learned is saved as usual and ``best.zip`` still only changes when the out-of-sample score improves."""
    tr = cfg.section("train")
    env_cfg = env_settings(cfg)
    ckpt = cfg.path("train.checkpoint_dir", "models/policy")
    ckpt.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        dataset = cached_dataset(cfg) if not (refresh or synthetic) else None
    if dataset is None:
        dataset, _providers, _ctx = prepare_dataset(cfg, offline=offline, refresh=refresh, synthetic=synthetic, fit=True)
    n_seeds = int(seeds or tr.get("seeds", 1) or 1)
    if n_seeds > 1:
        return train_ensemble(cfg, dataset, n_seeds, total_timesteps, resume, n_envs, max_minutes=max_minutes)
    if resume:  # a checkpoint from another signal layout cannot be continued
        rp = Path(resume)
        rp = rp if rp.suffix == ".zip" else (rp / "latest.zip")
        if not rp.exists() and (rp.parent / "ensemble.json").exists():  # resuming an ensemble: continue from seed0
            rp = rp.parent / "ensemble" / "seed0" / "latest.zip"
        layout_file = rp.parent / "layout.json"
        if not rp.exists():
            log.warning("checkpoint %s not found - starting a fresh policy", rp)
            resume = None
        elif layout_file.exists() and ObservationLayout.load(layout_file).signature() != dataset.layout.signature():
            log.warning("checkpoint %s was trained on a different signal layout - starting a fresh policy", rp)
            resume = None
    _archive_old_layout(ckpt, dataset.layout.signature())
    train_end = cfg.get_path("data.train_end")
    ds_train, ds_test = dataset.split(train_end) if train_end else (dataset, dataset)
    if len(ds_train) == 0:
        raise RuntimeError("training split is empty - check data.train_end")
    if len(ds_test) == 0:
        log.warning("test split is empty - evaluating on the training data")
        ds_test = ds_train
    log.info("train tickers: %d  test tickers: %d  obs_dim=%d", len(ds_train), len(ds_test), dataset.layout.obs_dim)

    n_envs = int(n_envs or tr.get("n_envs", 8))
    seed = int(tr.get("seed", 0))
    vec = make_vec_env(ds_train, env_cfg, n_envs=n_envs, seed=seed, vec=tr.get("vec_env", "auto"),
                       recent_bias=float(tr.get("recent_bias", 0.0)), recent_days=int(tr.get("recent_days", 400)))
    algo = str(tr.get("algo", "ppo")).lower()
    if resume:
        model = load_model(rp, algo, env=vec, device=tr.get("device", "cpu"))
        log.info("resumed from %s", rp)
    else:
        model = build_model(algo, vec, dict(tr), seed=seed)

    meta = {
        "algo": algo, "trained_at": datetime.now(timezone.utc).isoformat(), "train_end": train_end,
        "tickers": dataset.tickers, "n_envs": n_envs, "env_cfg": env_cfg, "train_cfg": dict(tr),
        "synthetic": synthetic, "signature": dataset.layout.signature(),
    }
    total = int(total_timesteps or tr.get("total_timesteps", 300_000))
    eval_freq = max(int(tr.get("eval_freq", 20_000)), n_envs)
    callbacks = [
        CheckpointCallback(save_freq=max(eval_freq // n_envs, 1), save_path=str(ckpt / "checkpoints"), name_prefix="step"),
        # best.zip only ever improves for a given signal layout, whether the run is fresh or resumed
        OOSEvalCallback(ds_test, env_cfg, eval_freq, int(tr.get("eval_tickers", 5)), ckpt, dataset.layout, meta,
                        max_bars=int(tr.get("eval_bars", 750)), keep_best=True),
    ]
    if max_minutes is not None and max_minutes > 0:
        callbacks.append(TimeBudgetCallback(max_minutes * 60.0))
        meta["max_minutes"] = float(max_minutes)
    log.info("training %s for %d timesteps on %d parallel simulators%s ...", algo.upper(), total, n_envs,
             f" (at most {max_minutes:.0f} minutes)" if max_minutes else "")
    t0 = time.time()
    model.learn(total_timesteps=total, callback=callbacks, reset_num_timesteps=not resume, progress_bar=False)
    meta["train_seconds"] = round(time.time() - t0, 1)
    meta["timesteps"] = int(model.num_timesteps)
    meta["eval_history"] = callbacks[1].history
    bundle = PolicyBundle(model, dataset.layout, meta)
    path = bundle.save(ckpt, "latest")
    (ckpt / "train_log.json").write_text(json.dumps(callbacks[1].history, indent=1), encoding="utf-8")
    vec.close()
    log.info("saved policy to %s (%.0fs)", path, meta["train_seconds"])
    return path


def select_members(scores: dict[str, float], min_score: float) -> list[str]:
    """Keep members whose best out-of-sample score reaches ``min_score``; the best member is always kept."""
    if not scores:
        return []
    keep = [m for m, s in scores.items() if s >= min_score]
    return keep or [max(scores, key=scores.get)]


def train_ensemble(cfg: Config, dataset: MarketDataset, n_seeds: int, total_timesteps: int | None, resume: str | None,
                   n_envs: int | None, max_minutes: float | None = None) -> Path:
    """Train ``n_seeds`` independent policies (different seeds) and register their best checkpoints as one ensemble.

    A ``max_minutes`` budget is shared: each member gets an equal share of whatever time is left."""
    import copy
    import shutil

    ckpt = cfg.path("train.checkpoint_dir", "models/policy")
    _archive_old_layout(ckpt, dataset.layout.signature())
    base_seed = int(cfg.get_path("train.seed", 0) or 0)
    min_score = float(cfg.get_path("train.ensemble_min_score", 0.0) or 0.0)
    scores: dict[str, float] = {}
    deadline = time.time() + max_minutes * 60.0 if max_minutes else None
    for k in range(n_seeds):
        sub = copy.deepcopy(cfg)
        member_dir = ckpt / "ensemble" / f"seed{k}"
        sub.set_path("train.checkpoint_dir", str(member_dir))
        sub.set_path("train.seed", base_seed + k)
        sub.set_path("train.seeds", 1)
        member_resume = str(member_dir / "latest.zip") if resume and (member_dir / "latest.zip").exists() else None
        member_minutes = None
        if deadline is not None:
            member_minutes = max(0.5, (deadline - time.time()) / 60.0 / (n_seeds - k))
        log.info("=== ensemble member %d/%d (seed %d)%s%s ===", k + 1, n_seeds, base_seed + k, " resumed" if member_resume else "",
                 f" {member_minutes:.0f} min" if member_minutes else "")
        train(sub, total_timesteps=total_timesteps, resume=member_resume, dataset=dataset, n_envs=n_envs, seeds=1, max_minutes=member_minutes)
        best = member_dir / "best.zip"
        member = str((best if best.exists() else member_dir / "latest.zip").relative_to(ckpt))
        score = -float("inf")
        if (member_dir / "best.json").exists():
            try:
                score = float(json.loads((member_dir / "best.json").read_text(encoding="utf-8")).get("score", -np.inf))
            except Exception:  # noqa: BLE001
                pass
        scores[member] = score
    members = select_members(scores, min_score)
    dropped = [m for m in scores if m not in members]
    if dropped:
        log.info("ensemble: dropped %s (score below %.2f)", dropped, min_score)
    best_member = max(scores, key=scores.get)
    for name in ("layout.json", "meta.json", "train_log.json", "best.json"):
        src = ckpt / Path(best_member).parent / name
        if src.exists():
            shutil.copy(src, ckpt / name)
    (ckpt / "ensemble.json").write_text(json.dumps({"members": members, "scores": scores, "signature": dataset.layout.signature(),
                                                    "seeds": n_seeds}, indent=1), encoding="utf-8")
    log.info("ensemble of %d/%d policies registered in %s (scores %s)", len(members), n_seeds, ckpt / "ensemble.json",
             {m: round(s, 2) for m, s in scores.items()})
    return ckpt / "ensemble.json"


def register_ensemble(cfg: Config, min_score: float | None = None) -> Path:
    """Register every trained ``<checkpoint_dir>/ensemble/seed*`` member (e.g. trained in parallel) as one ensemble."""
    import shutil

    ckpt = cfg.path("train.checkpoint_dir", "models/policy")
    root = ckpt / "ensemble"
    min_score = float(cfg.get_path("train.ensemble_min_score", 0.0) or 0.0) if min_score is None else float(min_score)
    scores: dict[str, float] = {}
    signature = None
    for member_dir in sorted(p for p in root.glob("seed*") if p.is_dir()):
        best = member_dir / "best.zip"
        target = best if best.exists() else member_dir / "latest.zip"
        if not target.exists() or not (member_dir / "layout.json").exists():
            continue
        sig = ObservationLayout.load(member_dir / "layout.json").signature()
        if signature is None:
            signature = sig
        elif sig != signature:
            log.warning("skipping %s: different signal layout", member_dir.name)
            continue
        score = -float("inf")
        if (member_dir / "best.json").exists():
            try:
                score = float(json.loads((member_dir / "best.json").read_text(encoding="utf-8")).get("score", -np.inf))
            except Exception:  # noqa: BLE001
                pass
        scores[str(target.relative_to(ckpt))] = score
    if not scores:
        raise FileNotFoundError(f"no trained members under {root}")
    members = select_members(scores, min_score)
    best_member = max(scores, key=scores.get)
    for name in ("layout.json", "meta.json", "train_log.json", "best.json"):
        src = ckpt / Path(best_member).parent / name
        if src.exists():
            shutil.copy(src, ckpt / name)
    (ckpt / "ensemble.json").write_text(json.dumps({"members": members, "scores": scores, "signature": signature,
                                                    "seeds": len(scores)}, indent=1), encoding="utf-8")
    log.info("ensemble of %d/%d members registered in %s (scores %s)", len(members), len(scores), ckpt / "ensemble.json",
             {m: round(s, 2) for m, s in scores.items()})
    return ckpt / "ensemble.json"


def retrain(cfg: Config, total_timesteps: int | None = None, n_envs: int | None = None, offline: bool = False,
            synthetic: bool = False, from_scratch: bool = False, seeds: int | None = None, max_minutes: float | None = None,
            reuse_dataset_days: float = 0.0) -> Path:
    """Feedback loop: refresh data, refit every sub-model, continue training with recent windows weighted up.

    ``reuse_dataset_days > 0`` skips the (slow) refit when ``models/dataset`` was built within that many
    days for the same universe / layout / split - the daily session uses this so the PPO updates get
    the time, and the weekly retrain (``0``) rebuilds everything."""
    ds = cached_dataset(cfg, max_age_days=reuse_dataset_days) if reuse_dataset_days > 0 and not synthetic else None
    if ds is None:
        ds, _, _ = prepare_dataset(cfg, refresh=not offline, offline=offline, synthetic=synthetic, fit=True)
    if float(cfg.get_path("train.recent_bias", 0.0) or 0.0) == 0.0:
        cfg.set_path("train.recent_bias", 0.5)
    ckpt = cfg.path("train.checkpoint_dir", "models/policy")
    has_policy = (ckpt / "latest.zip").exists() or (ckpt / "ensemble.json").exists()
    resume = str(ckpt / "latest.zip") if has_policy and not from_scratch else None
    return train(cfg, total_timesteps=total_timesteps, resume=resume, dataset=ds, n_envs=n_envs, seeds=seeds, max_minutes=max_minutes)


from ..config import rolling_train_end  # noqa: E402,F401 - re-exported: `data.train_end: rolling:N` is resolved by load_config
