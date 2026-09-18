#!/usr/bin/env python
"""Train the standard Qlib workflow (Alpha158 -> LightGBM) and export predictions for StockBot.

Steps::

    pip install pyqlib lightgbm                        # or: pip install -e third_party/qlib
    python scripts/qlib_train.py --download            # one-off: fetch qlib's US daily bundle
    python scripts/qlib_train.py                       # train + write models/qlib/pred.parquet

The exported table (columns: datetime, instrument, score) is picked up automatically by the
``qlib`` signal provider (config ``signals.qlib.predictions``).

If the official data bundle is unavailable, build one from Yahoo with qlib's collector::

    python third_party/qlib/scripts/data_collector/yahoo/collector.py download_data --source_dir ~/.qlib/stock_data/source/us_data --region US --start 2008-01-01 --end 2026-01-01 --delay 1 --interval 1d
    python third_party/qlib/scripts/data_collector/yahoo/collector.py normalize_data --source_dir ~/.qlib/stock_data/source/us_data --normalize_dir ~/.qlib/stock_data/source/us_1d_nor --region US --interval 1d
    python third_party/qlib/scripts/dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/us_1d_nor --qlib_dir ~/.qlib/qlib_data/us_data --freq day --exclude_fields date,symbol
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider-uri", default="~/.qlib/qlib_data/us_data")
    ap.add_argument("--download", action="store_true", help="download qlib's US data bundle first")
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--train-end", default=None, help="defaults to config data.train_end")
    ap.add_argument("--end", default=None)
    ap.add_argument("--instruments", nargs="*", help="defaults to the config universe")
    ap.add_argument("--out", default=None, help="prediction table (default models/qlib/pred.parquet, pred_tra.parquet for --model tra)")
    ap.add_argument("--model", choices=["lgb", "tra"], default="lgb", help="lgb = LightGBM (daily); tra = qlib's Temporal Routing Adaptor (weekend)")
    ap.add_argument("--epochs", type=int, default=8, help="TRA: training epochs (early stop 3)")
    ap.add_argument("--seq-len", type=int, default=30, help="TRA: sequence length in bars")
    ap.add_argument("--hidden", type=int, default=64, help="TRA: LSTM hidden size")
    args = ap.parse_args()
    if args.out is None:
        args.out = "models/qlib/pred_tra.parquet" if args.model == "tra" else "models/qlib/pred.parquet"

    from stockbot.config import load_config
    from stockbot.paths import add_submodule_to_syspath, resolve

    cfg = load_config()
    try:
        import qlib  # noqa: F401
    except ImportError:
        add_submodule_to_syspath("qlib")
        try:
            import qlib  # noqa: F401
        except ImportError:
            print("qlib is not installed: pip install pyqlib   (or pip install -e third_party/qlib)")
            return 1

    import os

    import pandas as pd

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # qlib's recorder uses mlflow's ./mlruns file store
    import qlib
    from qlib.constant import REG_US

    provider = Path(args.provider_uri).expanduser()
    if args.download:
        from qlib.tests.data import GetData

        GetData().qlib_data(target_dir=str(provider), region="us", exists_skip=True)
    if not provider.exists():
        print(f"no qlib data at {provider}; run with --download or build it with the yahoo collector (see --help)")
        return 1
    qlib.init(provider_uri=str(provider), region=REG_US)

    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH

    # the bundle decides the usable range: the official snapshot ends 2020-11-10, a yahoo-collector
    # bundle ends whenever you built it.  Splits are derived from its calendar.
    cal = pd.read_csv(provider / "calendars" / "day.txt", header=None)[0]
    last = pd.Timestamp(cal.iloc[-1])
    first = max(pd.Timestamp(cal.iloc[0]), pd.Timestamp(args.start))
    end = min(pd.Timestamp(args.end), last) if args.end else last
    train_end = pd.Timestamp(args.train_end or cfg.get_path("data.train_end", "2022-12-31"))
    if train_end >= end - pd.DateOffset(years=1):
        train_end = end - pd.DateOffset(years=1)  # keep one year of test data inside the bundle
    valid_start = train_end - pd.DateOffset(years=2)
    fit_end = valid_start - pd.Timedelta(days=1)
    test_start = train_end + pd.Timedelta(days=1)
    fmt = "%Y-%m-%d"
    print(f"bundle {cal.iloc[0]} -> {last.date()}; train {first.date()}..{fit_end.date()}, "
          f"valid {valid_start.date()}..{train_end.date()}, test {test_start.date()}..{end.date()}")

    instruments = [t.upper() for t in (args.instruments or cfg.get("universe", []))]
    segments = {"train": (first.strftime(fmt), fit_end.strftime(fmt)), "valid": (valid_start.strftime(fmt), train_end.strftime(fmt)),
                "test": (test_start.strftime(fmt), end.strftime(fmt))}
    if args.model == "tra":
        # qlib's TRA (Lin et al. 2021): Alpha158 with the benchmark's processors, sequences of seq_len bars, a small LSTM with
        # attention and 3 routed predictors - the official config shrunk to what a CPU runner trains in a weekend hour
        from qlib.contrib.data.dataset import MTSDatasetH
        from qlib.contrib.model.pytorch_tra import TRAModel

        handler = Alpha158(instruments=instruments, start_time=first.strftime(fmt), end_time=end.strftime(fmt),
                           fit_start_time=first.strftime(fmt), fit_end_time=fit_end.strftime(fmt),
                           infer_processors=[{"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                                             {"class": "Fillna", "kwargs": {"fields_group": "feature"}}],
                           learn_processors=[{"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}],
                           label=["Ref($close, -2) / Ref($close, -1) - 1"])
        dataset = MTSDatasetH(handler, segments=segments, seq_len=int(args.seq_len), num_states=3, memory_mode="sample", batch_size=1024,
                              drop_last=True, input_size=158)
        out = resolve(args.out)
        model = TRAModel(model_config={"input_size": 158, "hidden_size": int(args.hidden), "num_layers": 1, "rnn_arch": "LSTM", "use_attn": True,
                                       "dropout": 0.1},
                         tra_config={"num_states": 3, "rnn_arch": "LSTM", "hidden_size": 16, "num_layers": 1, "dropout": 0.0, "tau": 1.0,
                                     "src_info": "LR_TPE"},
                         model_type="RNN", lr=1e-3, n_epochs=int(args.epochs), early_stop=3, lamb=1.0, rho=0.99, alpha=0.5, seed=0,
                         transport_method="router", memory_mode="sample", eval_train=False, eval_test=False, pretrain=True,
                         logdir=str(out.parent / "tra_logs"))
        model.fit(dataset)
        preds = []
        for s in ("valid", "test"):
            p = model.predict(dataset, segment=s)
            if hasattr(p, "columns"):
                p = p["score"] if "score" in p.columns else p.iloc[:, 0]
            preds.append(p)
        pred = pd.concat(preds).sort_index()
    else:
        handler = Alpha158(instruments=instruments, start_time=first.strftime(fmt), end_time=end.strftime(fmt),
                           fit_start_time=first.strftime(fmt), fit_end_time=fit_end.strftime(fmt))
        dataset = DatasetH(handler, segments=segments)
        model = LGBModel(loss="mse", colsample_bytree=0.8879, learning_rate=0.0421, subsample=0.8789, lambda_l1=205.6999,
                         lambda_l2=580.9768, max_depth=8, num_leaves=210, num_threads=8)
        model.fit(dataset)
        preds = [model.predict(dataset, segment=s) for s in ("valid", "test")]
        pred = pd.concat(preds).sort_index()
    df = pred.to_frame("score").reset_index()
    df.columns = ["datetime", "instrument", "score"]
    df["instrument"] = df["instrument"].str.upper()
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"wrote {len(df)} predictions for {df['instrument'].nunique()} instruments to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
