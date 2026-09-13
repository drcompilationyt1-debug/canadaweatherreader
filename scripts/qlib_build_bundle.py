#!/usr/bin/env python
"""Build a qlib data bundle for the configured universe from StockBot's own Yahoo cache.

qlib's public US bundle (``GetData().qlib_data(region="us")``) stops on 2020-11-10, so a model
trained on it can never score today's bars.  This script writes a fresh bundle in the same
layout (``features/<symbol>/{open,high,low,close,volume,factor,change}.day.bin`` plus calendars
and instruments) from ``data/cache/*.parquet`` - the auto-adjusted daily bars every other block
already uses - with qlib's own ``scripts/dump_bin.py`` from the ``third_party/qlib`` submodule.

    python scripts/qlib_build_bundle.py                    # universe from config, cache as is
    python scripts/qlib_build_bundle.py --refresh          # re-download the bars first
    python scripts/qlib_train.py                           # then train / export predictions

Takes about a minute for 30 tickers.  The previous bundle at ``--provider-uri`` is replaced.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DUMP_BIN = ROOT / "third_party" / "qlib" / "scripts" / "dump_bin.py"


def write_csvs(frames: dict, csv_dir: Path) -> int:
    import numpy as np

    n = 0
    for ticker, df in frames.items():
        d = df[["open", "high", "low", "close", "volume"]].astype(float).copy()
        d = d[d["close"] > 0]
        if len(d) < 300:
            continue
        d["factor"] = 1.0                                   # bars are already dividend/split adjusted
        d["change"] = d["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        d.index.name = "date"
        d.reset_index().to_csv(csv_dir / f"{ticker.upper()}.csv", index=False, date_format="%Y-%m-%d")
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider-uri", default="~/.qlib/qlib_data/us_data")
    ap.add_argument("--refresh", action="store_true", help="re-download the daily bars before building")
    ap.add_argument("--tickers", nargs="*", help="defaults to the config universe")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if not DUMP_BIN.exists():
        print(f"{DUMP_BIN} missing: git submodule update --init --depth 1 third_party/qlib")
        return 2
    from stockbot.agent.train import load_frames
    from stockbot.config import load_config

    cfg = load_config()
    frames = load_frames(cfg, refresh=args.refresh, tickers=args.tickers)
    provider = Path(args.provider_uri).expanduser()
    with tempfile.TemporaryDirectory() as tmp:
        csv_dir = Path(tmp) / "csv"
        csv_dir.mkdir()
        n = write_csvs(frames, csv_dir)
        if n == 0:
            print("no bars to dump")
            return 1
        if provider.exists():
            shutil.rmtree(provider)
        provider.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(DUMP_BIN), "dump_all", "--data_path", str(csv_dir), "--qlib_dir", str(provider), "--freq", "day",
               "--exclude_fields", "date", "--date_field_name", "date", "--max_workers", str(args.workers)]
        print("  $", " ".join(cmd[1:]))
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"dump_bin failed ({r.returncode})")
            return 1
    cal = (provider / "calendars" / "day.txt").read_text().split()
    inst = (provider / "instruments" / "all.txt").read_text().strip().splitlines()
    print(f"qlib bundle at {provider}: {len(inst)} instruments, {len(cal)} days, {cal[0]} -> {cal[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
