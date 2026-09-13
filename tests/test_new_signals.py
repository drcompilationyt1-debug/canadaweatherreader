"""The seven third-party inputs added 2026-09-13: indicator / allocation / alpha blocks run for real,
the foundation models and FinBERT are exercised with fakes (their caching and alignment logic)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockbot.signals.registry import build_context, build_providers
from stockbot.signals.thirdparty.chronos_signal import ChronosSignal
from stockbot.signals.thirdparty.finbert_signal import FinBERTSignal
from stockbot.signals.thirdparty.kronos_signal import KronosSignal
from stockbot.signals.thirdparty.timesfm_signal import TimesFMSignal


def _provider(cfg, frames, name):
    cfg.set_path(f"signals.{name}.enabled", True)          # the shared fixture keeps the heavy models off
    ctx = build_context(cfg, with_llm=False, with_news=False)
    ctx.extra["frames"] = frames
    return next(p for p in build_providers(cfg, ctx) if p.name == name), ctx


def test_pandas_ta_block(cfg, frames):
    pytest.importorskip("pandas_ta_classic")
    p, _ = _provider(cfg, frames, "pandas_ta")
    assert p.availability()[0]
    arr = p.safe_history("AAA", frames["AAA"])
    assert arr is not None and arr.shape == (len(frames["AAA"]), 12)
    valid = ~np.isnan(arr).any(axis=1)
    assert valid[-100:].all()
    assert set(np.unique(arr[valid, 0])) <= {-1.0, 1.0}                   # supertrend direction
    assert np.abs(arr[valid, 5]).max() <= 1.0 + 1e-6                     # mfi scaled to -1..1


def test_pypfopt_block(cfg, frames):
    pytest.importorskip("pypfopt")
    cfg.set_path("signals.pypfopt.min_assets", 3)
    p, _ = _provider(cfg, frames, "pypfopt")
    out = p.compute_history_all(frames)
    assert set(out) == set(frames)
    a = out["AAA"]
    assert a.shape == (len(frames["AAA"]), 4)
    valid = ~np.isnan(a).any(axis=1)
    assert valid.sum() > 500 and valid[-1]
    assert 0 <= a[valid, 0].min() and a[valid, 0].max() <= 5 and np.abs(a[valid, 2]).max() <= 1.0
    # HRP weights of the three names sum to ~1 (scaled by N=3)
    last = np.array([out[t][-1, 0] for t in frames])
    assert abs(last.sum() - 3.0) < 1e-3
    lat = p.safe_latest("AAA", frames["AAA"])
    assert lat is not None and lat.shape == (4,)


def test_alpha101_block(cfg, frames):
    from stockbot.signals.thirdparty.alpha101_signal import SOURCE

    if not SOURCE.exists():
        pytest.skip("WorldQuant_alpha101_code submodule not cloned")
    p, _ = _provider(cfg, frames, "alpha101")
    assert p.availability()[0]
    out = p.compute_history_all(frames)
    a = out["BBB"]
    assert a.shape == (len(frames["BBB"]), 13)
    valid = ~np.isnan(a).any(axis=1)
    assert valid.sum() > 1000
    assert a[valid].min() >= -1.0 - 1e-6 and a[valid].max() <= 1.0 + 1e-6
    # cross-sectional: on any date the three ranks of one alpha are a permutation of {-1, 0, 1}
    row = np.array([out[t][-1, 3] for t in frames])                      # alpha012 ranks
    assert sorted(np.round(row, 6).tolist()) == [-1.0, 0.0, 1.0]
    assert p.safe_latest("AAA", frames["AAA"]) is not None


class _Fake:
    """Deterministic stand-in for the model call: feature k = k + 0.01 * window position."""

    def __init__(self):
        self.calls = 0

    def chronos(self, windows):
        self.calls += len(windows)
        return np.tile(np.arange(4, dtype=np.float32), (len(windows), 1)) + 0.01 * windows[:, -1:].astype(np.float32) * 0

    def kronos(self, df, rows):
        self.calls += len(rows)
        return np.tile(np.array([0.5, -0.5, 1.0], dtype=np.float32), (len(rows), 1))

    def timesfm(self, windows):
        self.calls += len(windows)
        return np.tile(np.array([0.2, 0.4, 0.6], dtype=np.float32), (len(windows), 1))


def test_chronos_cache_and_alignment(cfg, frames, monkeypatch):
    cfg.set_path("signals.chronos.context", 64)
    cfg.set_path("signals.chronos.history_years", 1)
    p, _ = _provider(cfg, frames, "chronos")
    fake = _Fake()
    monkeypatch.setattr(p, "_forecast", fake.chronos)
    monkeypatch.setattr(p, "availability", lambda: (True, "fake"))
    df = frames["AAA"]
    a = p.safe_history("AAA", df)
    assert a.shape == (len(df), 4)
    n_first = fake.calls
    assert 200 < n_first < 300                                           # ~1 year of bars, every bar
    assert np.isnan(a[: len(df) - n_first]).all() and not np.isnan(a[-n_first:]).any()
    a2 = p.safe_history("AAA", df)                                       # second pass: everything cached
    assert fake.calls == n_first and np.array_equal(np.nan_to_num(a), np.nan_to_num(a2))
    assert p.cache.path("AAA").exists()
    lat = p.safe_latest("AAA", df)
    assert lat is not None and fake.calls == n_first                    # newest bar served from the cache
    # a brand-new bar is forecast once and then cached
    longer = pd.concat([df, df.iloc[[-1]].set_index(pd.DatetimeIndex([df.index[-1] + pd.Timedelta(days=1)]))])
    assert p.safe_latest("AAA", longer) is not None and fake.calls == n_first + 1
    assert p.safe_latest("AAA", longer) is not None and fake.calls == n_first + 1


def test_kronos_and_timesfm_stride(cfg, frames, monkeypatch):
    cfg.set_path("signals.kronos.context", 64)
    cfg.set_path("signals.kronos.history_years", 1)
    cfg.set_path("signals.kronos.stride", 5)
    cfg.set_path("signals.timesfm.context", 64)
    cfg.set_path("signals.timesfm.history_years", 1)
    cfg.set_path("signals.timesfm.stride", 10)
    df = frames["CCC"]
    for name, attr in (("kronos", "kronos"), ("timesfm", "timesfm")):
        p, _ = _provider(cfg, frames, name)
        fake = _Fake()
        monkeypatch.setattr(p, "_forecast", getattr(fake, attr))
        monkeypatch.setattr(p, "availability", lambda: (True, "fake"))
        a = p.safe_history("CCC", df)
        assert a.shape == (len(df), 3)
        stride = p.stride
        assert fake.calls < 300 / stride + 5                             # sampled history
        valid = ~np.isnan(a).any(axis=1)
        assert valid[-1] and valid[-stride:].all()                       # forward-filled between samples
        assert p.safe_latest("CCC", df) is not None


class _FakePipe:
    def __init__(self):
        self.n = 0

    def __call__(self, texts):
        self.n += len(texts)
        out = []
        for t in texts:
            pos = 0.9 if "beat" in t.lower() else 0.1
            out.append([{"label": "positive", "score": pos}, {"label": "negative", "score": 1 - pos - 0.05}, {"label": "neutral", "score": 0.05}])
        return out


def test_finbert_history_from_csv(cfg, frames, monkeypatch, tmp_path):
    cfg.set_path("signals.finbert.max_texts", 50)
    p, _ = _provider(cfg, frames, "finbert")
    fake = _FakePipe()
    monkeypatch.setattr(p, "_pipeline", lambda: fake)
    monkeypatch.setattr(p, "availability", lambda: (True, "fake"))
    df = frames["AAA"]
    days = df.index[-30:]
    rows = []
    for i, d in enumerate(days):
        rows.append({"ticker": "AAA", "date": d.strftime("%Y-%m-%d"),
                     "title": (f"AAA beats estimates again, day {i}" if i % 2 == 0 else f"AAA misses badly, day {i}"), "summary": "x", "url": "u"})
    p.csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p.csv_dir / "google_AAA.csv", index=False)
    a = p.safe_history("AAA", df)
    assert a is not None and a.shape == (len(df), 4)
    assert fake.n == 30 and (p.folder / "AAA_scores.parquet").exists()
    valid = ~np.isnan(a).any(axis=1)
    assert valid[-30:].all() and not valid[:-40].any()
    assert -2.0 <= a[-1, 0] <= 2.0 and a[-1, 3] > 0
    a2 = p.safe_history("AAA", df)
    assert fake.n == 30 and np.array_equal(np.nan_to_num(a), np.nan_to_num(a2))   # scored once, cached
    assert p.safe_latest("AAA", df) is not None
