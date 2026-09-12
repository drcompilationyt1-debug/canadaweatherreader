"""Run akurgat/automating-technical-analysis' trained Keras models on one OHLCV parquet file.

Executed inside ``.venv-tf`` (``python scripts/setup_agent_envs.py --only tf``).  Reproduces the
repo's pipeline (Technical_Calculations -> Indications -> Preprocessing 60-day windows) on our
data instead of its own downloader, then applies ``action_prediction_model.h5`` (Buy/Hold/Sell)
and ``price_prediction_model.h5`` (next close) to every window.

    python scripts/agents/run_ta_keras.py --parquet bars.parquet --out result.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

FEATURES = ['High', 'Low', 'Open', 'Volume', 'Adj Close', 'P', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3',
            'OBV', 'MACD', 'MACDS', 'MACDH', 'SMA', 'LMA', 'SEMA', 'LEMA', 'RSI', 'SR_K', 'SR_D',
            'SR_RSI_K', 'SR_RSI_D', 'ATR', 'HL_PCT', 'PCT_CHG']
PRICE_FEATURES = [f for f in FEATURES if f != 'Adj Close']
STEPS = ("pivot_point", "on_balance_volume", "moving_average_convergence_divergence", "moving_averages",
         "relative_strength_index", "slow_stochastic", "stochastic_rsi", "average_true_range", "price_analysis",
         "engulfing_analysis", "support_resistance", "moving_average_analysis", "macd_analysis",
         "stochastic_analysis", "rsi_divergence_convergence", "price_action")
CATS = ["Buy", "Hold", "Sell"]


def load_keras_model(path: Path):
    try:
        from tensorflow.keras.models import load_model

        return load_model(str(path), compile=False)
    except Exception as first:  # noqa: BLE001 - legacy h5 -> try the Keras 2 shim
        try:
            os.environ["TF_USE_LEGACY_KERAS"] = "1"
            from tf_keras.models import load_model as legacy_load

            return legacy_load(str(path), compile=False)
        except Exception as second:  # noqa: BLE001
            raise RuntimeError(f"cannot load {path.name}: {first} / {second}") from second


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2] / "third_party" / "automating-technical-analysis"))
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    args.parquet = str(Path(args.parquet).resolve())  # resolve before chdir into the repo
    args.out = str(Path(args.out).resolve())
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.modules.setdefault("yfinance", types.ModuleType("yfinance"))  # their downloader is bypassed

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    from app.model import Prediction

    df = pd.read_parquet(args.parquet)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Adj Close", "volume": "Volume"})
    df = df[["High", "Low", "Open", "Volume", "Adj Close"]].astype(float)

    obj = Prediction.__new__(Prediction)
    obj.engulfing_period, obj.sma, obj.lma = -5, -15, -20
    obj.fast_length, obj.slow_length, obj.signal_smoothing = 12, 26, 9
    obj.short_run, obj.long_run, obj.rsi_period = 20, 50, 14
    obj.df = df.copy()

    def price_action_compat():
        # identical to the repo's price_action, minus the ewm(axis=1) keyword pandas 3 removed
        obj.indication_estimate = 3
        ind = obj.df.loc[:, "Engulfing_Indication":]
        obj.df["Indication"] = ind.T.ewm(com=obj.indication_estimate - 1, min_periods=obj.indication_estimate).mean().T.iloc[:, -1].round(3)
        obj.df.loc[(obj.df["Indication"] >= 1.25) & (obj.df["Adj Close"] <= obj.df["P"]), "Distinct_Action"] = "Buy"
        obj.df.loc[(obj.df["Indication"] <= 0.75) & (obj.df["Adj Close"] >= obj.df["P"]), "Distinct_Action"] = "Sell"
        obj.df["Distinct_Action"] = obj.df["Distinct_Action"].fillna("Hold")
        obj.df = obj.df.drop(["Indication"], axis=1).dropna()

    for step in STEPS:
        if step == "price_action":
            price_action_compat()
        else:
            getattr(obj, step)()
    if len(obj.df) < 61:
        json.dump({"error": f"only {len(obj.df)} usable rows"}, open(args.out, "w"))
        return 1

    df_action = obj.df[FEATURES + ["Distinct_Action"]].copy()
    df_price = obj.df[FEATURES].copy()
    scaler = StandardScaler()
    df_price["Adj Close_Scaled"] = scaler.fit_transform(df_price[["Adj Close"]].values).reshape(-1)
    X_action, _ = obj.scaling(df_action)
    X_price, _ = obj.scaling(df_price[PRICE_FEATURES + ["Adj Close_Scaled"]])
    X_action = np.asarray(X_action, dtype=np.float32)
    X_price = np.asarray(X_price, dtype=np.float32)

    action_model = load_keras_model(repo / "models" / "action_prediction_model.h5")
    price_model = load_keras_model(repo / "models" / "price_prediction_model.h5")
    pa = np.asarray(action_model.predict(X_action, verbose=0), dtype=float)
    pp = scaler.inverse_transform(np.asarray(price_model.predict(X_price, verbose=0), dtype=float).reshape(-1, 1)).flatten()
    idx = obj.df.index[59:59 + len(pa)]
    close = obj.df["Adj Close"].to_numpy(float)[59:59 + len(pa)]
    out = {
        "dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in idx],
        "action": [CATS[int(i)] for i in pa.argmax(axis=1)],
        "p_buy": pa[:, 0].round(4).tolist(),
        "p_sell": pa[:, 2].round(4).tolist(),
        "price_pred": pp.round(4).tolist(),
        "close": close.round(4).tolist(),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(json.dumps({"rows": len(pa), "last": out["dates"][-1], "last_action": out["action"][-1]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
