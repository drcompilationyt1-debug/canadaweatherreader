import json

import numpy as np
import pandas as pd

from stockbot.signals.base import SignalContext, SignalProvider
from stockbot.signals.layout import ObservationLayout
from stockbot.signals.news_llm import LLMNewsSignal, result_to_vector
from stockbot.signals.registry import build_context, build_layout, build_providers
from stockbot.signals.sentiment import _analyzer, score_texts
from stockbot.signals.thirdparty.trendet_signal import causal_runs, trendet_labels_native
from stockbot.env.dataset import MarketDataset


class Broken(SignalProvider):
    name = "broken"
    feature_names = ["x", "y"]

    def compute_history(self, ticker, df):
        raise RuntimeError("boom")


class Missing(SignalProvider):
    name = "missing"
    feature_names = ["z"]

    def availability(self):
        return False, "not installed"

    def compute_history(self, ticker, df):
        return np.zeros((len(df), 1))


def test_layout_masks_missing_blocks():
    layout = ObservationLayout([("a", ["a1", "a2"]), ("b", ["b1"])])
    arr_a = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
    out = layout.assemble(3, {"a": arr_a, "b": None})
    assert out.shape == (3, layout.signal_dim)
    assert out[:, 0].tolist() == [1.0, 0.0, 1.0]          # availability flag of a
    assert out[1, 1:3].tolist() == [0.0, 0.0]              # NaN row zeroed
    assert (out[:, 3] == 0).all() and (out[:, 4] == 0).all()  # block b fully masked
    assert layout.availability_of(out[0]) == {"a": True, "b": False}
    assert len(layout.column_names()) == layout.obs_dim
    assert ObservationLayout.from_dict(layout.to_dict()).signature() == layout.signature()


def test_broken_and_unavailable_providers_do_not_crash(cfg, frames, tmp_path):
    ctx = SignalContext(cfg=cfg, models_dir=tmp_path)
    b, m = Broken(cfg, ctx), Missing(cfg, ctx)
    df = frames["AAA"]
    assert b.safe_history("AAA", df) is None
    assert m.safe_history("AAA", df) is None
    assert m.safe_latest("AAA", df) is None
    assert b.describe()["available"] is True and m.describe()["reason"] == "not installed"


def test_registry_and_dataset_build(cfg, frames):
    ctx = build_context(cfg, with_llm=True, with_news=True)
    providers = build_providers(cfg, ctx)
    names = [p.name for p in providers]
    for required in ("technical", "candles", "trend", "alpha_factors", "market_regime", "trendet", "sentiment", "news_llm"):
        assert required in names
    layout = build_layout(providers)
    ds = MarketDataset.build(frames, providers, layout, ctx, fit=True, train_end="2018-12-31")
    assert set(ds.tickers) == set(frames)
    td = ds.data["AAA"]
    assert td.signals.shape == (len(frames["AAA"]), layout.signal_dim)
    assert np.isfinite(td.signals).all()
    last = td.signals[-1]
    avail = layout.availability_of(last)
    assert avail["technical"] and avail["candles"] and avail["trend"] and avail["trendet"]
    assert avail["alpha_factors"] and avail["market_regime"]
    assert not avail["news_llm"]  # no LLM key / cache in tests -> masked
    assert td.min_start > 0
    # persistence roundtrip
    ds.save(cfg.path("models_dir") / "dataset")
    loaded = MarketDataset.load(cfg.path("models_dir") / "dataset")
    assert loaded.layout.signature() == layout.signature()
    assert np.allclose(loaded.data["AAA"].signals, td.signals)
    # fitted state is reusable without refitting
    ctx2 = build_context(cfg, with_llm=False, with_news=False)
    p2 = build_providers(cfg, ctx2)
    alpha = next(p for p in p2 if p.name == "alpha_factors")
    assert alpha.load_state() and alpha.availability()[0]
    train, test = ds.split("2018-12-31")
    assert len(train) == 3 and len(test) == 3
    assert pd.Timestamp(test.data["AAA"].dates[0]) > pd.Timestamp("2018-12-31")


def test_trendet_causal_runs():
    close = np.array([10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9], float)
    up = causal_runs(-close)
    down = causal_runs(close)
    assert up[5] == 6            # six rising bars = one run
    assert up[6] == 7            # a single pullback above the running mean keeps the run alive (trendet rule)
    assert up[8] == 0            # falling below the running mean ends it
    assert down[-1] >= 5
    assert (down[:5] <= 1).all()  # no down-run while prices rise (a broken run restarts at length 1)
    # trendet records a trend only once it breaks -> add a rebound bar so the down-run closes
    ups, downs = trendet_labels_native(np.append(close, 12.0), window_size=3)
    assert ups == [(0, 5)] and downs == [(6, 11)]  # the bar that breaks a run does not start the next one


def test_vader_and_llm_vector():
    sia = _analyzer()
    assert sia is not None
    v = score_texts(["Company beats earnings expectations, stock soars", "Regulators fine company over fraud"], sia)
    assert v.shape == (4,) and -1 <= v[0] <= 1
    vec = result_to_vector({"sentiment": 0.6, "impact": 0.9, "confidence": 0.7, "horizon": "months", "direction": "bullish"}, 12)
    assert vec.tolist() == [0.6000000238418579, 0.8999999761581421, 0.699999988079071, 1.0, 1.0, 1.2000000476837158]
    assert result_to_vector({"sentiment": "bad"}, 0)[0] == 0.0


def test_news_llm_uses_cache_without_backend(cfg, frames, tmp_path):
    ctx = build_context(cfg, with_llm=True, with_news=True)
    sig = LLMNewsSignal(cfg, ctx)
    df = frames["AAA"]
    day = pd.Timestamp(df.index[-3]).strftime("%Y-%m-%d")
    f = sig.cache_dir / "AAA" / f"{day}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"hash": "x", "n_items": 4, "result": {"sentiment": -0.5, "impact": 0.4, "confidence": 0.8,
                                                                       "horizon": "days", "direction": "bearish"}}))
    hist = sig.compute_history("AAA", df)
    assert hist.shape == (len(df), sig.size)
    assert np.isnan(hist[0]).all()
    assert hist[-3][0] == np.float32(-0.5) and hist[-1][0] == np.float32(-0.5)  # forward filled within lookback
    assert sig.availability()[0]
