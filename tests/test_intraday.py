"""The broker's record (Alpaca history: equity per day, fills, intraday bars), the intraday exit model, and the
session selling a held name when the model flags it."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from stockbot.execution.alpaca_history import AlpacaHistory
from stockbot.feedback.intraday import ExitModel, build_dataset, features_at, label_at, paths_from_bars, what_if_exit

DAY = date(2026, 9, 11)
BARS_UTC = [datetime(2026, 9, 11, 13, 30, tzinfo=timezone.utc) + timedelta(minutes=15 * i) for i in range(26)]   # 09:30 .. 15:45 New York


class FakeResp:
    def __init__(self, data, status=200):
        self.data, self.status_code = data, status
        self.text = json.dumps(data)

    def json(self):
        return self.data


class FakeAlpaca:
    """Answers the three feeds the way Alpaca does (pagination included)."""

    def __init__(self):
        self.calls = []
        closes = 100.0 + np.cumsum(np.r_[0.0, np.full(9, 0.12), np.full(16, -0.09)])          # up 1% by 11:45, then gives it back
        self.bars = [{"t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": float(c) - 0.05, "h": float(c) + 0.1, "l": float(c) - 0.1, "c": float(c), "v": 1000}
                     for ts, c in zip(BARS_UTC, closes)]
        self.fills = [{"id": f"f{i}", "transaction_time": f"2026-09-1{d}T13:35:0{i}Z", "symbol": "AAPL", "side": "buy" if i < 2 else "sell",
                       "qty": "10", "price": str(100 + i), "order_id": f"o{i}", "type": "fill", "cum_qty": "10", "leaves_qty": "0"}
                      for i, d in ((0, 0), (1, 1), (2, 1))]

    def get(self, url, params=None, headers=None, timeout=None):
        params = dict(params or {})
        self.calls.append((url, params))
        assert headers["APCA-API-KEY-ID"] == "k" and headers["APCA-API-SECRET-KEY"] == "s"
        if "portfolio/history" in url:
            return FakeResp({"timestamp": [1757462400, 1757548800, 1757635200], "equity": [100000.0, 100500.0, 99819.53],
                             "profit_loss": [0.0, 500.0, -680.47], "profit_loss_pct": [0.0, 0.005, -0.0068], "base_value": 100000.0})
        if "activities/FILL" in url:
            after = params.get("after", "")
            rows = [f for f in self.fills if f["transaction_time"] > after] if after else list(self.fills)
            token = params.get("page_token")
            if token:
                rows = [f for f in rows if f["id"] > token]
            return FakeResp(rows[: int(params["page_size"])])
        if "/v2/stocks/bars" in url:
            assert params["symbols"] == "AAPL" and params["timeframe"] == "15Min" and params["feed"] == "iex"
            if not params.get("page_token"):
                return FakeResp({"bars": {"AAPL": self.bars[:20]}, "next_page_token": "p2"})
            return FakeResp({"bars": {"AAPL": self.bars[20:]}, "next_page_token": None})
        if url.endswith("/v2/positions"):
            return FakeResp([{"symbol": "AAPL", "qty": "10", "avg_entry_price": "100", "current_price": "101", "market_value": "1010",
                              "unrealized_pl": "10", "unrealized_plpc": "0.01", "unrealized_intraday_pl": "3"}])
        return FakeResp({"message": "not found"}, 404)


def test_alpaca_history_feeds(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    fake = FakeAlpaca()
    h = AlpacaHistory(paper=True, cache_dir=tmp_path / "alpaca", session=fake)
    assert AlpacaHistory.available() and h.base.startswith("https://paper-api")
    daily = h.daily("1M")
    assert list(daily["direction"]) == ["flat", "up", "down"] and daily["equity"].iloc[-1] == pytest.approx(99819.53)
    assert (tmp_path / "alpaca" / "history_1M_1D.json").exists()
    fills = h.fills(page_size=2)                                               # two pages
    assert len(fills) == 3 and list(fills["side"]) == ["buy", "buy", "sell"] and str(fills["ts"].dt.tz) == "America/New_York"
    assert fills["notional"].iloc[0] == pytest.approx(1000.0)
    cached = h.sync_fills()
    assert len(cached) == 3 and (tmp_path / "alpaca" / "fills.jsonl").exists()
    n_calls = len(fake.calls)
    cached2 = h.sync_fills()                                                  # incremental: asks only for newer fills, nothing doubled
    assert len(cached2) == 3 and fake.calls[n_calls][1].get("after", "") > "2026-09-11T13:35"
    pos = h.positions()
    assert pos.iloc[0]["ticker"] == "AAPL" and pos.iloc[0]["today_pl"] == 3.0
    bars = h.bars_cached(["AAPL"], DAY, DAY)
    assert len(bars["AAPL"]) == 26 and (tmp_path / "alpaca" / "bars_15Min" / "AAPL.parquet").exists()
    paths = h.day_paths(["AAPL"], DAY)
    s = paths["AAPL"]
    assert len(s) == 26 and s.index[0].strftime("%H:%M") == "09:30" and s.index[-1].strftime("%H:%M") == "15:45"
    n_calls = len(fake.calls)
    h.day_paths(["AAPL"], DAY)                                                 # served from the parquet cache
    assert len(fake.calls) == n_calls
    # a later day forces a fetch that is merged with the cached bars: the index must stay a proper NY DatetimeIndex
    cached_df = pd.read_parquet(tmp_path / "alpaca" / "bars_15Min" / "AAPL.parquet")
    cached_df.index = cached_df.index.tz_convert("UTC")                       # a different tz object, as another pandas / pyarrow may give
    cached_df.to_parquet(tmp_path / "alpaca" / "bars_15Min" / "AAPL.parquet")
    merged = h.bars_cached(["AAPL"], DAY, DAY + timedelta(days=1))
    assert isinstance(merged["AAPL"].index, pd.DatetimeIndex) and str(merged["AAPL"].index.tz) == "America/New_York"
    assert len(merged["AAPL"]) == 26 and len(fake.calls) == n_calls + 2       # fetched (two pages), nothing duplicated


def _synthetic_paths(n_tickers=12, n_days=40, seed=0):
    """Intraday paths with a learnable habit: a name up 0.4% by mid-morning tends to give it back."""
    rng = np.random.default_rng(seed)
    paths = {}
    for i in range(n_tickers):
        for d in range(n_days):
            day = date(2026, 7, 1) + timedelta(days=d)
            steps = rng.normal(0.0, 0.002, 26)
            c = 100.0 * np.cumprod(1.0 + steps)
            if c[9] / 100.0 - 1.0 >= 0.004:                                        # about a quarter of the name-days
                c[10:] = c[9] * np.cumprod(1.0 + rng.normal(-0.0015, 0.002, 16))
            paths[(f"T{i}", day)] = {"closes": c, "open": 100.0, "prev_close": 99.5}
    return paths


def test_exit_model_learns_when_to_sell(tmp_path):
    c = np.array([100.5, 101.0, 101.2, 100.8, 100.2, 99.9])
    f = features_at(c, 2, 100.0, 99.0)
    assert f["ret_open"] == pytest.approx(0.012) and f["ret_max"] == pytest.approx(0.012) and f["gap"] == pytest.approx(100 / 99 - 1)
    assert label_at(c, 2, 0.001) == 1 and label_at(c, 5, 0.001) == 0
    w = what_if_exit(c, 100.0, 0.001)
    assert w["best_exit_k"] == 2 and w["gain_vs_hold"] == pytest.approx(0.011988, abs=1e-6)   # 101.2 net of 0.1% vs the 99.9 close
    paths = _synthetic_paths()
    df = build_dataset(paths, fee_rt=0.001)
    assert len(df) == len(paths) * 24 and set(df["label"].unique()) == {0, 1}
    model = ExitModel()
    m = model.fit(df, holdout_days=5)
    assert m["n_holdout"] > 0 and m["auc"] > 0.62
    pop = 100.0 * np.cumprod(1.0 + np.full(10, 0.0012))                        # +1.2% by mid-morning: the habit says sell
    adv = model.advice(pop, 100.0, 99.5, fee_rt=0.001, min_prob=0.6, min_gain=0.005)
    assert adv["exit"] is True and adv["prob"] > 0.6 and adv["reason"] == "take profit"
    flat = np.full(10, 100.05)
    assert model.advice(flat, 100.0, 99.5, fee_rt=0.001, min_prob=0.6, min_gain=0.005)["exit"] is False   # nothing to take
    rp = model.replay(np.r_[pop, pop[-1] * np.cumprod(1.0 + np.full(16, -0.001))], 100.0, 99.5, 0.001)
    assert rp["exit_k"] is not None and rp["exit_return"] > rp["hold_return"]
    model.save(tmp_path / "exit")
    loaded = ExitModel.load(tmp_path / "exit")
    assert loaded is not None and abs(loaded.predict(features_at(pop, 9, 100.0, 99.5))[0] - adv["prob"]) < 1e-6
    assert ExitModel.load(tmp_path / "nothing") is None
    # the same model file with Windows line endings (git autocrlf on a Linux-written file) still loads
    f = tmp_path / "exit" / "model.txt"
    if f.exists():                                                            # lightgbm build; the logistic fallback has no model.txt
        f.write_bytes(f.read_bytes().replace(b"\n", b"\r\n"))
        crlf = ExitModel.load(tmp_path / "exit")
        assert crlf is not None and abs(crlf.predict(features_at(pop, 9, 100.0, 99.5))[0] - adv["prob"]) < 1e-6
        f.write_text("tree\nversion=v4\n", encoding="utf-8")
        assert ExitModel.load(tmp_path / "exit") is None                       # a truncated file is ignored, not fatal
    bars = {"AAPL": pd.DataFrame({"open": [100.0] * 26, "high": 0, "low": 0, "close": np.linspace(100, 101, 26), "volume": 1},
                                 index=pd.DatetimeIndex([b.astimezone(timezone.utc) for b in BARS_UTC]).tz_convert("America/New_York"))}
    p = paths_from_bars(bars)
    assert ("AAPL", DAY) in p and len(p[("AAPL", DAY)]["closes"]) == 26 and p[("AAPL", DAY)]["times"][0] == "09:30"


def test_session_sells_when_the_exit_model_flags_a_held_name(cfg, frames, monkeypatch):
    from stockbot.agent.policy import PolicyBundle
    from stockbot.execution.base import Order
    from stockbot.execution.market_hours import NY, MarketClock
    from stockbot.execution.runner import TradingRunner
    from stockbot.execution.session import TradingSession
    from stockbot.signals.registry import build_context, build_layout, build_providers

    class Clock(MarketClock):
        def __init__(self, start):
            self.t = start
            super().__init__(source="builtin", now_fn=lambda: self.t)

        def sleep(self, s):
            self.t += timedelta(seconds=s)

    class Hold:                                                               # the daily policy: keep what we have
        num_timesteps = 0

        def predict(self, obs, deterministic=True):
            return np.array([0.0], dtype=np.float32), None

    class Stub:                                                               # the exit model: sell once a name is up 0.5%
        meta = {"metrics": {"auc": 0.7}}

        def advice(self, closes, open_px, prev_close, fee_rt, **kw):
            r = closes[-1] / open_px - 1.0
            return {"exit": bool(r >= 0.005), "prob": 0.9, "ret_open": r, "ret_max": r, "reason": "take profit"}

    ctx = build_context(cfg, with_llm=False, with_news=False)
    providers = [p for p in build_providers(cfg, ctx) if p.name in ("technical", "trend")]
    bundle = PolicyBundle(Hold(), build_layout(providers), {"algo": "ppo"})
    start = datetime(2026, 9, 14, 9, 31, tzinfo=NY)
    clock = Clock(start)
    runner = TradingRunner(cfg, mode="paper", bundle=bundle, frames_loader=lambda refresh: frames, with_llm=False, clock=clock)
    base = {t: float(frames[t]["close"].iloc[-1]) for t in frames}
    runner.broker.submit(Order("AAA", "buy", 10))                              # a position from before
    monkeypatch.setattr(runner, "price", lambda t: base[t] * (1.01 if t == "AAA" and clock.t > start + timedelta(minutes=20) else 1.0))
    session = TradingSession(cfg, mode="paper", hours=1, train=False, clock=clock, sleep=clock.sleep, runner=runner, snapshot_minutes=15,
                             after_open_minutes=0, review_after=False, learn_before_open=False)
    session.exit_model = Stub()
    assert session._exit_model_has_edge({"auc": 0.62, "gain_when_exit": 0.002, "gain_all": 0.0005})
    assert not session._exit_model_has_edge({"auc": 0.51, "gain_when_exit": 0.0009, "gain_all": 0.0009})   # what 20 real days gave: stays OFF
    summary = session.run()
    assert summary["exits"] == 1 and runner.broker.position("AAA").shares == 0
    recs = [json.loads(line) for line in (cfg.path("session.log_dir") / "session_2026-09-14.jsonl").read_text().splitlines() if line.strip()]
    ex = [r for r in recs if r.get("type") == "exit"]
    assert len(ex) == 1 and ex[0]["ticker"] == "AAA" and ex[0]["qty"] > 10 and ex[0]["ret_open"] == pytest.approx(0.01)   # the open cycle had added to the position; all of it was sold
