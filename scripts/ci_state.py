#!/usr/bin/env python
"""Persist the bot's state (models, paper account, experience, caches) in a ``state`` git branch.

GitHub Actions runners start empty, so every run restores the state branch first and saves it
back at the end.  The branch always holds exactly one commit (force-pushed), so the repository
does not grow with history.

    python scripts/ci_state.py restore      # git fetch origin state && check the files out
    python scripts/ci_state.py save         # commit the state paths to a fresh tree and push -f origin state
    python scripts/ci_state.py save --remote origin --branch state
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATHS = ["models", "data/paper", "data/experience", "data/news_cache", "data/autopilot.json", "data/llm_state.json",
               "reports/dashboard.html"]
EXCLUDE = ["models/policy/checkpoints", "models/policy/ensemble/*/checkpoints", "models/policy/archive", "models/cache/*.parquet",
           "models/cache/ta_keras/*.parquet"]


def _excluded(path: str) -> bool:
    from fnmatch import fnmatch

    p = path.replace("\\", "/")
    return any(fnmatch(p, pat) or fnmatch(p, pat.rstrip("/") + "/*") for pat in EXCLUDE)


def git(*args: str, check: bool = True, env: dict | None = None, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, check=check, text=True, capture_output=capture, env=env)


def restore(remote: str, branch: str) -> int:
    r = git("fetch", remote, branch, "--depth", "1", check=False)
    if r.returncode != 0:
        print(f"no '{branch}' branch on {remote} yet - nothing to restore")
        return 0
    files = git("ls-tree", "-r", "--name-only", f"{remote}/{branch}").stdout.split()
    if not files:
        print("state branch is empty")
        return 0
    git("checkout", f"{remote}/{branch}", "--", *files)
    git("reset", "-q", "--", *files, check=False)  # keep the files, do not stage them on the working branch
    print(f"restored {len(files)} files from {remote}/{branch}")
    return 0


def save(remote: str, branch: str) -> int:
    present = [p for p in STATE_PATHS if (ROOT / p).exists()]
    if not present:
        print("nothing to save")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        git("read-tree", "--empty", env=env)
        git("add", "-f", "--", *present, env=env)
        listed = git("ls-files", "--cached", env=env).stdout.split("\n")
        drop = [f for f in listed if f and _excluded(f)]
        for i in range(0, len(drop), 200):
            git("rm", "-q", "--cached", "--", *drop[i:i + 200], env=env)
        kept = len(listed) - len(drop)
        print(f"state: {kept} files kept, {len(drop)} excluded (checkpoints / archives / caches)")
        tree = git("write-tree", env=env).stdout.strip()
        msg = f"state {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        cenv = {**env, "GIT_AUTHOR_NAME": "stockbot", "GIT_AUTHOR_EMAIL": "stockbot@localhost",
                "GIT_COMMITTER_NAME": "stockbot", "GIT_COMMITTER_EMAIL": "stockbot@localhost"}
        commit = git("commit-tree", tree, "-m", msg, env=cenv).stdout.strip()
        size = git("count-objects", "-vH").stdout.strip().splitlines()[-1]
        git("push", "-f", remote, f"{commit}:refs/heads/{branch}", capture=False)
        print(f"saved state as {commit[:10]} on {remote}/{branch} ({len(present)} paths; {size})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["restore", "save"])
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--branch", default="state")
    args = ap.parse_args()
    return restore(args.remote, args.branch) if args.action == "restore" else save(args.remote, args.branch)


if __name__ == "__main__":
    sys.exit(main())
