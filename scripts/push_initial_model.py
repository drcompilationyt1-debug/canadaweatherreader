#!/usr/bin/env python
"""Push a locally trained policy to the ``state`` branch as the model GitHub continues from.

The state branch is one force-pushed snapshot of models + paper state + experience.  The runners keep
the live parts (paper accounts, experience, sessions, caches) up to date, so a local push must not
overwrite those with stale local copies: this script refreshes everything from the state branch
EXCEPT ``models/policy`` (the locally trained ensemble), then saves the combined tree back.

    python scripts/push_initial_model.py            # refresh live state, keep local models/policy, push
    python scripts/push_initial_model.py --dry-run  # show what would be pushed
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ci_state  # noqa: E402

KEEP_LOCAL = ["models/policy"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--branch", default="state")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    policy = ROOT / "models" / "policy"
    if not (policy / "ensemble.json").exists() and not (policy / "latest.zip").exists():
        print(f"no trained policy in {policy}")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        keep = Path(tmp) / "keep"
        for rel in KEEP_LOCAL:
            src = ROOT / rel
            if src.exists():
                shutil.copytree(src, keep / rel, ignore=shutil.ignore_patterns("checkpoints", "archive"))
        print("refreshing the live state from", f"{args.remote}/{args.branch}")
        ci_state.restore(args.remote, args.branch)
        for rel in KEEP_LOCAL:
            src = keep / rel
            if src.exists():
                dst = ROOT / rel
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print("kept the local", rel)
    if args.dry_run:
        r = subprocess.run(["git", "status", "--short", "--", "models/policy"], cwd=ROOT, capture_output=True, text=True)
        print(r.stdout or "(models/policy unchanged vs the working tree)")
        print("dry run - not pushed")
        return 0
    return ci_state.save(args.remote, args.branch)


if __name__ == "__main__":
    raise SystemExit(main())
