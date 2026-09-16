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
    ap.add_argument("--keep", nargs="*", default=[], help="extra local paths (dirs or files) to keep over the state branch, e.g. models/dataset models/signals")
    ap.add_argument("--no-policy", action="store_true", help="leave the state branch's policy alone (push only the --keep paths)")
    args = ap.parse_args()
    keep_local = ([] if args.no_policy else KEEP_LOCAL) + [k for k in args.keep if k not in KEEP_LOCAL]
    policy = ROOT / "models" / "policy"
    if not args.no_policy and not (policy / "ensemble.json").exists() and not (policy / "latest.zip").exists():
        print(f"no trained policy in {policy}")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        keep = Path(tmp) / "keep"
        for rel in keep_local:
            src = ROOT / rel
            if src.is_dir():
                shutil.copytree(src, keep / rel, ignore=shutil.ignore_patterns("checkpoints", "archive"))
            elif src.is_file():
                (keep / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, keep / rel)
        print("refreshing the live state from", f"{args.remote}/{args.branch}")
        ci_state.restore(args.remote, args.branch)
        for rel in keep_local:
            src = keep / rel
            dst = ROOT / rel
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print("kept the local", rel)
            elif src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print("kept the local", rel)
    if args.dry_run:
        r = subprocess.run(["git", "status", "--short", "--"] + keep_local, cwd=ROOT, capture_output=True, text=True)
        print(r.stdout or "(kept paths unchanged vs the working tree)")
        print("dry run - not pushed")
        return 0
    return ci_state.save(args.remote, args.branch)


if __name__ == "__main__":
    raise SystemExit(main())
