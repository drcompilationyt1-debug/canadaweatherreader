"""Cross-check the trained policy inside backtrader (independent accounting / order engine)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..agent.policy import PolicyBundle
from ..agent.sizing import conviction_to_exposure, realized_vol_series, size_exposure
from ..env.dataset import TickerData
from ..logging_utils import get_logger
from ..paths import import_optional

log = get_logger(__name__)


def run_backtrader(bundle: PolicyBundle, td: TickerData, env_cfg: dict, start: int | None = None,
                   end: int | None = None) -> dict:
    bt = import_optional("backtrader", "backtrader")
    if bt is None:
        raise ImportError("pip install backtrader (or clone the backtrader submodule)")
    start = td.min_start if start is None else max(start, td.min_start)
    end = len(td) if end is None else min(end, len(td))
    df = pd.DataFrame({"open": td.open[start:end], "high": td.high[start:end], "low": td.low[start:end],
                       "close": td.close[start:end], "volume": td.volume[start:end]},
                      index=pd.DatetimeIndex(td.dates[start:end]))
    signals = td.signals[start:end]
    vol_target = float(env_cfg.get("vol_target", 0.0) or 0.0)
    vols = realized_vol_series(td.close, int(env_cfg.get("vol_window", 20)))[start:end]
    max_scale = float(env_cfg.get("vol_max_scale", 1.5))
    initial = float(env_cfg.get("initial_cash", 100_000))
    deadband = float(env_cfg.get("deadband", 0.05))
    allow_short = bool(env_cfg.get("allow_short", True))
    fee = float(env_cfg.get("commission", 0.0005)) + float(env_cfg.get("slippage", 0.0005))

    class PolicyStrategy(bt.Strategy):
        def __init__(self):
            self.peak = initial
            self.pos_age = 0
            self.exposures = []

        def next(self):
            i = len(self) - 1
            price = float(self.data.close[0])
            value = float(self.broker.getvalue())
            self.peak = max(self.peak, value)
            exposure = self.position.size * price / value if value > 0 else 0.0
            port = np.array([exposure, np.clip(value / initial - 1.0, -1.0, 3.0) * 2.0, (1.0 - value / self.peak) * 5.0,
                             min(self.pos_age / 252.0, 2.0), float(exposure > 0.01), float(exposure < -0.01)], dtype=np.float32)
            obs = np.concatenate([signals[i], port])
            target = conviction_to_exposure(bundle.predict(obs), allow_short)
            if vol_target > 0:
                target = size_exposure(target, float(vols[i]), vol_target, 1.0, max_scale)
            self.exposures.append(exposure)
            if abs(target - exposure) < deadband:
                return
            self.order_target_percent(target=target)
            same = abs(exposure) > 0.01 and np.sign(target) == np.sign(exposure)
            self.pos_age = self.pos_age + 1 if same else (1 if abs(target) > 0.01 else 0)

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(PolicyStrategy)
    cerebro.broker.setcash(initial)
    cerebro.broker.setcommission(commission=fee)
    cerebro.broker.set_coc(True)  # execute at the current close, like the simulator
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, riskfreerate=0.0, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="ret", timeframe=bt.TimeFrame.NoTimeFrame)
    res = cerebro.run()[0]
    final = float(cerebro.broker.getvalue())
    sharpe = res.analyzers.sharpe.get_analysis().get("sharperatio")
    dd = res.analyzers.dd.get_analysis()
    bh = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0)
    return {
        "engine": "backtrader", "ticker": td.ticker, "bars": len(df), "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
        "final_equity": final, "total_return": final / initial - 1.0, "bh_return": bh,
        "sharpe": float(sharpe) if sharpe is not None else float("nan"),
        "max_drawdown": float(dd.get("max", {}).get("drawdown", 0.0)) / 100.0,
        "avg_exposure": float(np.mean(res.exposures)) if res.exposures else 0.0,
    }
