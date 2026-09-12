import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from stockbot.data.loader import resample_ohlcv, synthetic_ohlcv
from stockbot.execution.autopilot import Autopilot
from stockbot.feedback.experience import ExperienceStore
from stockbot.llm.base import LLMQuotaError
from stockbot.llm.openai_backend import GeminiBackend, GroqBackend, OpenAIBackend, OpenRouterBackend
from stockbot.llm.router import BACKENDS, LLMRouter
from stockbot.news import gdelt
from stockbot.report import build_dashboard


# ---------------------------------------------------------------------- free LLM presets
def test_presets_need_their_key(monkeypatch):
    for env in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    assert OpenRouterBackend({}).is_configured()[0] is False
    assert GroqBackend({}).is_configured()[0] is False
    assert GeminiBackend({}).is_configured()[0] is False
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    b = OpenRouterBackend({"model": None})  # None must not override the preset default
    ok, why = b.is_configured()
    assert ok and "openrouter.ai" in why and b.model.endswith(":free")
    kw = b.request_kwargs([{"role": "user", "content": "hi"}])
    assert kw["extra_body"]["models"][0] == b.model and len(kw["extra_body"]["models"]) == 3
    assert "extra_body" not in GroqBackend({}).request_kwargs([])
    # a custom OpenAI-compatible server without a key is fine (local LM Studio etc.)
    assert OpenAIBackend({"base_url": "http://localhost:1234/v1"}).is_configured()[0]
    assert {"openrouter", "groq", "gemini"} <= set(BACKENDS)


def test_router_uses_per_backend_cooldown(tmp_path):
    from tests.test_llm import Fake

    a = Fake("a", LLMQuotaError("busy"))
    a.cfg["cooldown_seconds"] = 5
    b = Fake("b", {"ok": 1})
    r = LLMRouter([a, b], cooldown_seconds=1000, state_file=tmp_path / "s.json")
    assert r.complete_json("s", "u", {}) == {"ok": 1}
    assert 0 < r.status()[0]["cooldown_s"] <= 5


# ---------------------------------------------------------------------- GDELT
def test_gdelt_helpers(monkeypatch):
    assert gdelt.clean_company_name("NVIDIA Corporation") == "NVIDIA"
    assert gdelt.company_name("NVDA", {"NVDA": "Nvidia"}) == "Nvidia"
    assert gdelt.gdelt_query("Nvidia").startswith("Nvidia stock")
    assert gdelt.gdelt_query("Meta Platforms").startswith('"Meta Platforms" stock')
    chunks = list(gdelt.month_chunks(datetime(2024, 1, 15), datetime(2024, 3, 10)))
    assert len(chunks) == 3 and chunks[0][0] == datetime(2024, 1, 15)
    quarters = list(gdelt.period_chunks(datetime(2023, 11, 1), datetime(2024, 6, 30), months=3))
    assert len(quarters) == 3 and quarters[1][0] == datetime(2024, 2, 1) and quarters[-1][1] == datetime(2024, 6, 30)

    class R:
        status_code = 200
        ok = True
        text = "{}"

        def json(self):
            return {"articles": [{"title": "Nvidia beats", "seendate": "20240115T120000Z", "domain": "x.com", "url": "u"},
                                 {"title": "Nvidia beats", "seendate": "20240116T120000Z", "domain": "y.com", "url": "v"}]}

    monkeypatch.setattr(gdelt.requests, "get", lambda *a, **k: R())
    arts = gdelt.fetch_gdelt("q", datetime(2024, 1, 1), datetime(2024, 1, 31))
    df = gdelt.articles_to_frame("NVDA", arts)
    assert list(df.columns) == ["ticker", "date", "title", "summary", "url"]
    assert len(df) == 1 and df.iloc[0]["date"] == "2024-01-15"


def test_google_history_parser_and_splitting(monkeypatch):
    from stockbot.news import google_history as gh

    xml = ('<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>'
           '<item><title>Nvidia beats - Reuters</title><link>http://x</link><pubDate>Mon, 15 Jan 2024 12:00:00 GMT</pubDate>'
           '<source url="http://r">Reuters</source></item>'
           '<item><title>Nvidia beats - Reuters</title><link>http://y</link><pubDate>Tue, 16 Jan 2024 12:00:00 GMT</pubDate>'
           '<source url="http://r">Reuters</source></item></channel></rss>')

    class R:
        status_code = 200
        ok = True
        text = xml

    monkeypatch.setattr(gh.requests, "get", lambda *a, **k: R())
    df = gh.fetch_google_history("NVDA", "Nvidia", datetime(2024, 1, 1), datetime(2024, 1, 31), sleep=0)
    assert len(df) == 1 and df.iloc[0]["title"] == "Nvidia beats" and df.iloc[0]["summary"] == "Reuters"
    assert gh.google_query("Nvidia", datetime(2024, 1, 1), datetime(2024, 2, 1)) == "Nvidia stock after:2024-01-01 before:2024-02-01"

    # a full window (100 items) is split until it is shorter than min_days
    calls = []

    def fake_window(name, a, b, **kw):
        calls.append((a, b))
        n = 100 if (b - a).days > 3 else 2
        return [{"date": a.strftime("%Y-%m-%d"), "title": f"{a}-{b}-{i}", "summary": "", "url": ""} for i in range(n)]

    monkeypatch.setattr(gh, "fetch_window", fake_window)
    df = gh.fetch_google_history("NVDA", "Nvidia", datetime(2024, 1, 1), datetime(2024, 1, 13), sleep=0, window_days=12)
    assert len(calls) == 7 and len(df) == 8  # 12 -> 6+6 -> 3+3+3+3 leaves, two headlines each


# ---------------------------------------------------------------------- resampling / TA-Lib
def test_resample_weekly():
    df = synthetic_ohlcv(300, seed=2)
    w = resample_ohlcv(df, "W")
    assert 55 <= len(w) <= 62
    assert (w["high"] >= w["low"]).all() and (w["high"] >= w["close"]).all()
    assert len(resample_ohlcv(df, "D")) == len(df)


def test_talib_block_if_available(cfg, frames, tmp_path):
    talib = pytest.importorskip("talib")
    from stockbot.signals.base import SignalContext
    from stockbot.signals.talib_candles import TALIB_PATTERNS, TalibCandleSignal, talib_patterns

    assert set(TALIB_PATTERNS) == set(talib.get_function_groups()["Pattern Recognition"])
    pat = talib_patterns(frames["AAA"])
    assert pat.shape == (len(frames["AAA"]), 61) and set(np.unique(pat.to_numpy())) <= {-1, 0, 1}
    sig = TalibCandleSignal(cfg, SignalContext(cfg=cfg, models_dir=tmp_path))
    sig.fit(frames, "2018-12-31")
    sig.save_state()
    arr = sig.safe_history("AAA", frames["AAA"])
    assert arr.shape == (len(frames["AAA"]), sig.size) and np.isfinite(arr[50:]).all()
    assert sig.load_state() and sig.stats.n_bars > 0


# ---------------------------------------------------------------------- sizing / ensemble
def test_vol_targeted_sizing():
    from stockbot.agent.sizing import realized_vol, realized_vol_series, size_exposure

    close = np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.02, 300))) * 100
    v = realized_vol(close, 20)
    assert 0.15 < v < 0.6
    series = realized_vol_series(close, 20)
    assert np.isnan(series[:20]).all() and abs(series[-1] - v) < 1e-9
    assert size_exposure(1.0, 0.50, 0.25) == 0.5          # twice too volatile -> half size
    assert size_exposure(1.0, 0.10, 0.25) == 1.0          # calm stock capped by max_leverage
    assert size_exposure(1.0, 0.10, 0.25, max_leverage=2.0) == 1.5  # ... and by max_scale
    assert size_exposure(-0.5, 0.50, 0.25) == -0.25
    assert size_exposure(0.7, float("nan"), 0.25) == 0.7 and size_exposure(0.7, 0.3, 0.0) == 0.7
    from stockbot.agent.sizing import conviction_to_exposure

    assert conviction_to_exposure(-1.0, False) == 0.0 and conviction_to_exposure(0.0, False) == 0.5
    assert conviction_to_exposure(1.0, False) == 1.0 and conviction_to_exposure(-0.4, True) == -0.4


def test_env_uses_vol_targeting(cfg, frames):
    from stockbot.env.dataset import MarketDataset
    from stockbot.env.trading_env import TradingEnv
    from stockbot.signals.registry import build_context, build_layout, build_providers

    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical",)]
    ds = MarketDataset.build(frames, providers, build_layout(providers), ctx, fit=False, train_end="2018-12-31")
    env_cfg = dict(cfg.section("env"))
    env_cfg.update(vol_target=0.05, signal_dropout=0.0, deadband=0.0)  # tiny target -> heavily scaled down
    env = TradingEnv(ds, env_cfg, tickers=["AAA"], seed=0)
    env.reset(options={"ticker": "AAA", "start": 300, "length": 20})
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    assert 0.0 < info["exposure"] < 0.5


def test_select_members():
    from stockbot.agent.train import select_members

    assert select_members({"a": 1.0, "b": -0.5, "c": 0.2}, 0.0) == ["a", "c"]
    assert select_members({"a": -1.0, "b": -0.5}, 0.0) == ["b"]  # best one always kept
    assert select_members({}, 0.0) == []


def test_ensemble_bundle(cfg, frames, tmp_path):
    from stockbot.agent.policy import EnsemblePolicy, PolicyBundle
    from stockbot.agent.train import train
    from stockbot.env.dataset import MarketDataset
    from stockbot.signals.registry import build_context, build_layout, build_providers

    cfg.set_path("train.ensemble_min_score", -999)  # tiny runs score badly; keep every member here
    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    ds = MarketDataset.build(frames, providers, build_layout(providers), ctx, fit=True, train_end="2018-12-31")
    path = train(cfg, total_timesteps=128, dataset=ds, n_envs=2, seeds=2)
    assert path.name == "ensemble.json"
    ckpt = cfg.path("train.checkpoint_dir")
    bundle = PolicyBundle.load(ckpt)
    assert isinstance(bundle.model, EnsemblePolicy) and len(bundle.model.members) == 2
    a = bundle.predict(np.zeros(bundle.layout.obs_dim))
    assert -1.0 <= a <= 1.0
    assert PolicyBundle.exists(ckpt)
    # a second training round resumes every member
    path2 = train(cfg, total_timesteps=128, dataset=ds, n_envs=2, seeds=2, resume=str(ckpt / "latest.zip"))
    assert path2.exists() and (ckpt / "ensemble" / "seed1" / "latest.zip").exists()
    # members trained separately are registered with `stockbot ensemble`
    from stockbot.agent.train import register_ensemble

    (ckpt / "ensemble.json").unlink()
    reg = json.loads(register_ensemble(cfg, min_score=-999).read_text())
    assert len(reg["members"]) == 2 and reg["signature"] == bundle.layout.signature()
    assert isinstance(PolicyBundle.load(ckpt).model, EnsemblePolicy)


# ---------------------------------------------------------------------- reference-repo ports
def test_strategy_zoo_values(frames):
    from stockbot.features.strategies import STRATEGY_FEATURES, compute_strategies

    s = compute_strategies(frames["AAA"])
    assert list(s.columns) == STRATEGY_FEATURES and s.shape[0] == len(frames["AAA"])
    body = s.iloc[300:]
    assert not body.isna().any().any()
    for col in STRATEGY_FEATURES[:-1]:
        assert set(np.unique(body[col].to_numpy())) <= {-1.0, 0.0, 1.0}, col
    assert body["vote"].abs().max() <= 1.0
    assert body["lean_ema_cross"].min() == 0.0  # long / flat only, like the Lean algorithm


def test_es_agent_fit_and_features(cfg, frames, tmp_path):
    from stockbot.signals.base import SignalContext
    from stockbot.signals.es_agent import ESAgentSignal, make_states

    cfg.set_path("signals.es_agent.iterations", 5)
    cfg.set_path("signals.es_agent.train_bars", 300)
    cfg.set_path("signals.es_agent.layer_size", 16)
    cfg.set_path("signals.es_agent.holdout_years", 1)  # the synthetic history is short
    st = make_states(np.array([1.0, 1.1, 1.0, 1.2, 1.3]), 3)
    assert st.shape == (5, 3) and abs(st[4, 2] - (1.3 - 1.2) / 1.3 * 100) < 1e-9
    sig = ESAgentSignal(cfg, SignalContext(cfg=cfg, models_dir=tmp_path))
    sig.fit(frames, "2018-12-31")
    sig.save_state()
    arr = sig.safe_history("AAA", frames["AAA"])
    assert arr.shape == (len(frames["AAA"]), 3) and np.isfinite(arr[-50:]).all()
    assert (arr[-50:, 0] >= 0).all() and (arr[-50:, 0] <= 1).all()
    fresh = ESAgentSignal(cfg, SignalContext(cfg=cfg, models_dir=tmp_path))
    assert fresh.availability()[0] and set(fresh.models) == set(frames)
    # the agent's own fit window (up to train_end - holdout_years) is masked for the policy
    fit_end = pd.Timestamp(fresh.fit_end["AAA"])
    assert fit_end <= pd.Timestamp("2018-12-31") - pd.DateOffset(years=1)
    arr2 = fresh.safe_history("AAA", frames["AAA"])
    assert np.isnan(arr2[frames["AAA"].index <= fit_end]).all() and np.isfinite(arr2[-1]).all()


def test_dl_forecast_walk_forward(cfg, frames, tmp_path):
    pytest.importorskip("torch")
    from stockbot.signals.base import SignalContext
    from stockbot.signals.dl_forecast import DLForecastSignal

    for k, v in {"window": 10, "hidden": 8, "epochs": 1, "min_train_years": 1, "refit_every": 1, "max_train_rows": 3000}.items():
        cfg.set_path(f"signals.dl_forecast.{k}", v)
    sig = DLForecastSignal(cfg, SignalContext(cfg=cfg, models_dir=tmp_path))
    sig.fit(frames, "2018-12-31")
    assert len(sig.preds) > 0 and sig.state is not None
    sig.save_state()
    out = sig.compute_history_all(frames)
    arr = out["AAA"]
    assert arr.shape == (len(frames["AAA"]), 2) and np.isfinite(arr[-1]).all()
    fresh = DLForecastSignal(cfg, SignalContext(cfg=cfg, models_dir=tmp_path))
    assert fresh.availability()[0]


def test_resume_with_changed_layout_starts_fresh(cfg, frames, tmp_path):
    from stockbot.agent.train import train
    from stockbot.env.dataset import MarketDataset
    from stockbot.signals.registry import build_context, build_layout, build_providers

    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    layout = build_layout(providers)
    ds = MarketDataset.build(frames, providers, layout, ctx, fit=True, train_end="2018-12-31")
    path = train(cfg, total_timesteps=128, dataset=ds, n_envs=2)
    providers2 = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend", "strategy_zoo")]
    ds2 = MarketDataset.build(frames, providers2, build_layout(providers2), ctx, fit=True, train_end="2018-12-31")
    path2 = train(cfg, total_timesteps=128, dataset=ds2, n_envs=2, resume=str(path))  # must not crash
    assert path2.exists()


def test_dqn_agent_fit_and_features(cfg, frames, tmp_path):
    pytest.importorskip("torch")
    from stockbot.signals.base import SignalContext
    from stockbot.signals.dqn_agent import DQNAgentSignal, SingleStockEnv, price_windows

    env = SingleStockEnv(np.array([10.0, 11.0, 12.0, 11.0]), price_windows(np.array([10.0, 11.0, 12.0, 11.0]), 3), 100.0)
    s, r, done = env.step(2)  # buy 10 shares at 10
    assert env.shares == 10 and env.cash == 0 and abs(r - 10.0) < 1e-9 and not done
    s, r, done = env.step(0)  # sell at 11
    assert env.shares == 0 and abs(env.cash - 110.0) < 1e-9
    cfg.set_path("signals.dqn_agent.episodes", 1)
    cfg.set_path("signals.dqn_agent.train_bars", 150)
    cfg.set_path("signals.dqn_agent.holdout_years", 1)
    sig = DQNAgentSignal(cfg, SignalContext(cfg=cfg, models_dir=tmp_path))
    sig.fit(frames, "2018-12-31")
    sig.save_state()
    arr = sig.safe_history("AAA", frames["AAA"])
    assert arr.shape == (len(frames["AAA"]), 3) and np.isfinite(arr[-50:]).all()
    assert set(np.unique(arr[-50:, 2])) <= {-1.0, 0.0, 1.0}
    fit_end = pd.Timestamp(sig.models["AAA"]["fit_end"])
    assert np.isnan(arr[frames["AAA"].index <= fit_end]).all()  # own fit window masked
    assert DQNAgentSignal(cfg, SignalContext(cfg=cfg, models_dir=tmp_path)).availability()[0]


def test_llm_trader_with_fake_router(cfg, frames, tmp_path):
    from stockbot.signals.base import SignalContext
    from stockbot.signals.llm_trader import LLMTraderSignal, decision_to_vector, market_briefing

    v = decision_to_vector({"action": "OPEN_NEW", "direction": "short", "position_size_pct": 40, "confidence": 80})
    assert v.tolist() == [-1.0, 0.800000011920929, 0.4000000059604645, 1.0, 0.0]
    assert decision_to_vector({"action": "FULL_CLOSE"})[0] == -1.0 and decision_to_vector({"action": "nonsense"})[0] == 0.0
    text = market_briefing("AAA", frames["AAA"], {"exposure": 0.5, "pnl_pct": 3.2, "peak_pnl_pct": 4.0, "days": 7})
    assert "Symbol: AAA" in text and "RSI14" in text and "peak PnL +4.00%" in text

    class FakeRouter:
        def __init__(self):
            self.calls = 0

        def usable(self):
            return [type("B", (), {"name": "fake"})()]

        def complete_json(self, system, user, schema, max_tokens=None):
            self.calls += 1
            return {"action": "ADD_POSITION", "direction": "long", "position_size_pct": 25, "confidence": 70, "reasoning": "trend"}

    ctx = SignalContext(cfg=cfg, models_dir=tmp_path, llm=FakeRouter())
    ctx.extra["positions"] = {"AAA": {"exposure": 0.3, "pnl_pct": 1.0, "peak_pnl_pct": 2.0, "days": 3}}
    sig = LLMTraderSignal(cfg, ctx)
    assert sig.availability()[0]
    vec = sig.safe_latest("AAA", frames["AAA"])
    assert vec is not None and vec[0] == 0.5 and vec[3] == 1.0
    assert sig.safe_latest("AAA", frames["AAA"])[0] == 0.5 and ctx.llm.calls == 1  # cached per day
    assert sig.safe_history("AAA", frames["AAA"]) is None  # live only


def test_news_llm_rag_grading(cfg, frames, tmp_path):
    from stockbot.news.fetcher import NewsItem
    from stockbot.signals.base import SignalContext
    from stockbot.signals.news_llm import LLMNewsSignal

    prompts = []

    class FakeRouter:
        def usable(self):
            return [type("B", (), {"name": "fake"})()]

        @property
        def available(self):
            return True

        def complete_json(self, system, user, schema, max_tokens=None):
            prompts.append(user)
            if "relevant" in schema["properties"]:
                return {"relevant": [1]}
            return {"sentiment": 0.3, "impact": 0.5, "confidence": 0.6, "horizon": "weeks", "direction": "bullish", "key_points": []}

    cfg.set_path("signals.news_llm.rag_grading", True)
    ctx = SignalContext(cfg=cfg, models_dir=tmp_path, llm=FakeRouter())
    sig = LLMNewsSignal(cfg, ctx)
    items = [NewsItem("AAA", f"headline {i}", "2026-01-05", "src") for i in range(5)]
    res = sig.score_items("AAA", items, "2026-01-05")
    assert res["direction"] == "bullish" and len(prompts) == 2
    assert "headline 1" in prompts[1] and "headline 0" not in prompts[1]  # only the graded-relevant headline is scored


def test_ta_keras_alignment(cfg, frames, tmp_path, monkeypatch):
    from stockbot.signals.base import SignalContext
    from stockbot.signals.thirdparty import ta_keras_signal as tk

    sig = tk.TAKerasSignal(cfg, SignalContext(cfg=cfg, models_dir=tmp_path))
    df = frames["AAA"]
    dates = [d.strftime("%Y-%m-%d") for d in df.index[100:]]
    fake = {"dates": dates, "action": ["Buy"] * len(dates), "p_buy": [0.7] * len(dates), "p_sell": [0.1] * len(dates),
            "price_pred": (df["close"].iloc[100:] * 1.01).round(4).tolist(), "close": df["close"].iloc[100:].round(4).tolist()}
    monkeypatch.setattr(sig, "_run", lambda ticker, d: fake)
    arr = sig.compute_history("AAA", df)
    assert arr.shape == (len(df), 4) and np.isnan(arr[50]).all() and arr[-1, 0] == 1.0 and abs(arr[-1, 3] - 1.0) < 1e-3
    latest = sig.compute_latest("AAA", df)
    assert latest is not None and latest[1] == np.float32(0.7)


# ---------------------------------------------------------------------- moomoo broker (fake OpenD gateway)
def test_moomoo_broker_with_fake_gateway(monkeypatch):
    import pandas as pd

    from stockbot.execution.base import Order
    from stockbot.execution.moomoo import MoomooBroker

    class FakeTrd:
        def __init__(self):
            self.orders = []
            self.unlocked = False

        def get_acc_list(self):
            return 0, pd.DataFrame([{"acc_id": 111, "trd_env": "SIMULATE"}, {"acc_id": 222, "trd_env": "REAL"}])

        def accinfo_query(self, trd_env, acc_id, currency):
            return 0, pd.DataFrame([{"total_assets": 50_000.0, "cash": 20_000.0}])

        def position_list_query(self, trd_env, acc_id, currency):
            return 0, pd.DataFrame([{"code": "US.AAPL", "qty": 10.0, "cost_price": 150.0, "position_side": "LONG"},
                                    {"code": "US.BRK.B", "qty": 3.0, "cost_price": 400.0, "position_side": "LONG"}])

        def place_order(self, **kw):
            self.orders.append(kw)
            return 0, pd.DataFrame([{"order_id": "42"}])

        def unlock_trade(self, password=None, **kw):
            self.unlocked = True
            return 0, "ok"

    class FakeQuote:
        def get_market_snapshot(self, codes):
            return 0, pd.DataFrame([{"code": codes[0], "last_price": 200.0}])

    trd, quote = FakeTrd(), FakeQuote()
    b = MoomooBroker(env="simulate", trd_ctx=trd, quote_ctx=quote)
    assert b.equity() == 50_000.0 and b.cash() == 20_000.0
    pos = b.positions()
    assert pos["AAPL"].shares == 10 and pos["BRK-B"].shares == 3 and pos["BRK-B"].avg_price == 400.0
    assert b.price("BRK-B") == 200.0
    fill = b.submit(Order("AAPL", "buy", 5.7))
    assert fill.qty == 5 and trd.orders[-1]["code"] == "US.AAPL" and trd.orders[-1]["trd_side"] == "BUY"
    assert b.submit(Order("AAPL", "sell", 25)).qty == 10        # long-only: never sells more than it holds
    assert b.submit(Order("MSFT", "sell", 3)) is None            # nothing to sell
    assert not trd.unlocked                                       # paper account never needs the trade password
    real = MoomooBroker(env="real", trd_ctx=trd, quote_ctx=quote)
    monkeypatch.delenv("MOOMOO_TRADE_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="MOOMOO_TRADE_PASSWORD"):
        real.submit(Order("AAPL", "buy", 1))
    monkeypatch.setenv("MOOMOO_TRADE_PASSWORD", "secret")
    assert real.submit(Order("AAPL", "buy", 1)).qty == 1 and trd.unlocked


# ---------------------------------------------------------------------- agent frameworks (subprocess runner)
def test_agent_runner(tmp_path):
    import sys

    from stockbot.signals.thirdparty.agent_runner import resolve_python, run_agent

    py, why = resolve_python(str(tmp_path / "missing" / "python.exe"), ".venv-does-not-exist", "no_such_module_xyz")
    assert py is None and "not found" in why
    py, why = resolve_python(None, ".venv-does-not-exist", "no_such_module_xyz")
    assert py is None and "setup_agent_envs" in why
    py, why = resolve_python(None, ".venv-does-not-exist", "json")  # importable here -> current interpreter
    assert py == __import__("pathlib").Path(sys.executable)

    script = tmp_path / "fake_agent.py"
    script.write_text("import json,sys\nprint('progress noise')\nprint(json.dumps({'decision': 'BUY', 'ticker': sys.argv[1]}))\n")
    out = run_agent(py, script, ["NVDA"], timeout=60)
    assert out == {"decision": "BUY", "ticker": "NVDA"}
    bad = tmp_path / "bad_agent.py"
    bad.write_text("import json\nprint(json.dumps({'error': 'no key'}))\n")
    with pytest.raises(RuntimeError, match="no key"):
        run_agent(py, bad, [], timeout=60)


def test_trading_agents_adapter_unavailable_without_env(cfg, tmp_path, monkeypatch):
    from stockbot.signals.base import SignalContext
    from stockbot.signals.thirdparty.trading_agents_signal import TradingAgentsSignal, decision_code

    cfg.set_path("signals.trading_agents.enabled", True)
    cfg.set_path("signals.trading_agents.python", str(tmp_path / "nope.exe"))
    sig = TradingAgentsSignal(cfg, SignalContext(cfg=cfg, models_dir=tmp_path))
    ok, why = sig.availability()
    assert not ok and "not found" in why
    assert decision_code("FINAL TRANSACTION PROPOSAL: **BUY**") == 0.7 and decision_code("hold") == 0.0
    # OpenRouter key is passed through as OPENAI_API_KEY when backend_url points at OpenRouter
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    cfg.set_path("signals.trading_agents.backend_url", "https://openrouter.ai/api/v1")
    sig = TradingAgentsSignal(cfg, SignalContext(cfg=cfg, models_dir=tmp_path))
    missing, extra = sig._key_env()
    assert missing is None and extra["OPENAI_API_KEY"] == "sk-or-test"


# ---------------------------------------------------------------------- autopilot / dashboard
def test_autopilot_schedule(cfg, tmp_path):
    cfg.set_path("autopilot.state_file", (tmp_path / "ap.json").as_posix())
    ap = Autopilot(cfg, mode="paper")
    fri_late = datetime(2026, 9, 11, 17, 0, tzinfo=ap.tz)   # Friday after the close
    nxt = ap.next_trade_time(fri_late)
    assert nxt.weekday() == 0 and nxt.hour == 16 and nxt.minute == 30
    assert ap.next_trade_time(datetime(2026, 9, 9, 9, 0, tzinfo=ap.tz)).day == 9
    assert ap.retrain_due()
    ap.state["last_retrain"] = datetime.now().isoformat()
    ap._save()
    assert not Autopilot(cfg, mode="paper").retrain_due()


def test_dashboard_from_state(cfg, tmp_path):
    state = {"cash": 50_000.0, "initial_cash": 100_000.0, "created": "2026-01-01",
             "positions": {"AAA": {"shares": 100.0, "avg_price": 400.0}}, "last_prices": {"AAA": 520.0},
             "fills": [], "peak_equity": 102_000.0,
             "equity_history": [{"ts": "2026-01-02", "equity": 100_000.0}, {"ts": "2026-01-03", "equity": 102_000.0}]}
    sf = cfg.path("execution.state_file")
    sf.parent.mkdir(parents=True, exist_ok=True)
    sf.write_text(json.dumps(state))
    st = ExperienceStore(cfg.path("feedback.experience_file"))
    st.record(mode="paper", ticker="AAA", date="2026-01-02", obs=np.zeros(2), action=0.4, target_exposure=0.4, weight=0.1,
              decision="BUY", price=500.0, equity=1e5, availability={"technical": True, "news_llm": False})
    st.settle("AAA", "2026-01-03", 520.0)
    out = build_dashboard(cfg, mode="paper", out=tmp_path / "dash.html")
    html = out.read_text(encoding="utf-8")
    assert "StockBot dashboard" in html and "AAA" in html and "<svg" in html
    assert "signals on: technical" in html and "off: news_llm" in html
