#!/usr/bin/env python
"""Persist the bot's state (models, paper account, experience, caches) in a ``state`` git branch.

GitHub Actions runners start empty, so every run restores the state branch first and saves it
back at the end.  The branch always holds exactly one commit (force-pushed), so the repository
does not grow with history.

    python scripts/ci_state.py restore      # git fetch origin state && check the files out
    python scripts/ci_state.py save         # commit the state paths to a fresh tree and push -f origin state
    python scripts/ci_state.py save --remote origin --branch state
    python scripts/ci_state.py save --fresh # the very first push, when there is no state to restore

A save only goes through after a restore in the same checkout (a marker under .git records it), and never when it would
drop more than half of the restored files: a job that failed or was cancelled before its restore step must not push a
near-empty tree over the real state (the branch keeps a single commit, so that would lose it).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATHS = ["models", "data/paper", "data/experience", "data/news_cache", "data/news", "data/autopilot.json", "data/llm_state.json",
               "reports"]   # data/news = the headline archive FinBERT / the LLM news history score (grows weekly)
EXCLUDE = ["models/policy/checkpoints", "models/policy/ensemble/*/checkpoints", "models/policy/archive", "models/cache/*.parquet",
           "models/cache/ta_keras/*.parquet"]


def _excluded(path: str) -> bool:
    from fnmatch import fnmatch

    p = path.replace("\\", "/")
    return any(fnmatch(p, pat) or fnmatch(p, pat.rstrip("/") + "/*") for pat in EXCLUDE)


def git(*args: str, check: bool = True, env: dict | None = None, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, check=check, text=True, capture_output=capture, env=env)


def _marker() -> Path:
    r = git("rev-parse", "--git-path", "stockbot-state-restored", check=False)
    return ROOT / (r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else ".state-restored")


def _mark(files: int) -> None:
    m = _marker()
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(json.dumps({"files": files}), encoding="utf-8")


def restore(remote: str, branch: str) -> int:
    r = git("fetch", remote, branch, "--depth", "1", check=False)
    if r.returncode != 0:
        print(f"no '{branch}' branch on {remote} yet - nothing to restore")
        _mark(0)
        return 0
    files = git("ls-tree", "-r", "--name-only", f"{remote}/{branch}").stdout.split()
    if not files:
        print("state branch is empty")
        _mark(0)
        return 0
    # directory pathspecs, not one argument per file: hundreds of file names overflow the Windows command line
    specs = [p for p in STATE_PATHS if any(f == p or f.startswith(p.rstrip("/") + "/") for f in files)]
    extra = sorted({f for f in files if not any(f == p or f.startswith(p.rstrip("/") + "/") for p in STATE_PATHS)})
    for chunk in [specs] + [extra[i:i + 100] for i in range(0, len(extra), 100)]:
        if chunk:
            git("checkout", f"{remote}/{branch}", "--", *chunk)
            git("reset", "-q", "--", *chunk, check=False)  # keep the files, do not stage them on the working branch
    print(f"restored {len(files)} files from {remote}/{branch}")
    _mark(len(files))
    return 0


MAX_FILE_MB = 95            # GitHub rejects a file over 100 MB and with it the whole push: such a file is left out, loudly


def _oversized(path: Path, limit_mb: float = MAX_FILE_MB) -> bool:
    try:
        return path.is_file() and path.stat().st_size > limit_mb * 1024 * 1024
    except OSError:
        return False


def save(remote: str, branch: str, fresh: bool = False) -> int:
    restored = None
    if not fresh:
        try:
            restored = int(json.loads(_marker().read_text(encoding="utf-8"))["files"])
        except (OSError, ValueError, KeyError):
            print(f"ERROR: the state was not restored in this checkout - refusing to save over {remote}/{branch} "
                  "(run restore first; --fresh only for the very first push)")
            return 1
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
        big = [f for f in listed if f and f not in drop and _oversized(ROOT / f)]
        for f in big:
            print(f"WARNING: {f} is {(ROOT / f).stat().st_size / 2**20:.0f} MB, over GitHub's limit - left out of the state (the day's work is not lost with it)")
        drop += big
        for i in range(0, len(drop), 200):
            git("rm", "-q", "--cached", "--", *drop[i:i + 200], env=env)
        kept = len(listed) - len(drop)
        print(f"state: {kept} files kept, {len(drop)} excluded (checkpoints / archives / caches)")
        if restored and kept < restored / 2:
            print(f"ERROR: only {kept} files against the {restored} restored - refusing to save a state this much smaller")
            return 1
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
    ap.add_argument("--fresh", action="store_true", help="save without a restore first (only the very first push)")
    args = ap.parse_args()
    return restore(args.remote, args.branch) if args.action == "restore" else save(args.remote, args.branch, args.fresh)


if __name__ == "__main__":
    sys.exit(main())
