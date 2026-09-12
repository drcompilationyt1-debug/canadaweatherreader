#!/usr/bin/env python
"""One-shot GitHub setup: private repo, push, encrypted Actions secrets, first dry-run workflow.

    set GITHUB_TOKEN=ghp_...            (classic token with `repo` + `workflow` scopes, or a fine-grained
                                         token with Administration, Contents, Secrets, Actions: read/write)
    python scripts/github_setup.py --name quiet-orchard

What it does
1. POST /user/repos            - creates a private repository (no README, no description)
2. git remote add origin       - and pushes `main`
3. python scripts/ci_state.py save   - pushes the trained state to the `state` branch
4. PUT  /repos/.../actions/secrets/*  - ALPACA_API_KEY, ALPACA_SECRET_KEY and any LLM keys found in the environment
5. POST /repos/.../actions/workflows/trade.yml/dispatches  - a dry run of the daily trade job

Nothing about the project's purpose is written to GitHub except the code itself.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"
SECRET_NAMES = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
                "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]


def gh(method: str, path: str, token: str, **kw) -> requests.Response:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    r = requests.request(method, API + path, headers=headers, timeout=60, **kw)
    return r


def encrypt(public_key_b64: str, value: str) -> str:
    from nacl import encoding, public

    pk = public.PublicKey(public_key_b64.encode(), encoding.Base64Encoder())
    return base64.b64encode(public.SealedBox(pk).encrypt(value.encode())).decode()


def run(cmd: list[str]) -> None:
    print("  $", " ".join(c if not c.startswith("https://x-access-token") else "https://<token>@github.com/..." for c in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="repository name (pick something unrelated to trading)")
    ap.add_argument("--public", action="store_true", help="public repo = unlimited Actions minutes, visible code")
    ap.add_argument("--no-dispatch", action="store_true", help="do not trigger the dry-run workflow")
    args = ap.parse_args()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("set GITHUB_TOKEN first")
        return 2

    me = gh("GET", "/user", token)
    if not me.ok:
        print("token rejected:", me.status_code, me.text[:200])
        return 2
    owner = me.json()["login"]
    print(f"authenticated as {owner}")

    r = gh("POST", "/user/repos", token, json={"name": args.name, "private": not args.public, "auto_init": False,
                                                "has_issues": False, "has_projects": False, "has_wiki": False})
    if r.status_code == 422 and "already exists" in r.text:
        print(f"repository {owner}/{args.name} already exists - reusing it")
    elif not r.ok:
        print("repo creation failed:", r.status_code, r.text[:300])
        return 1
    else:
        print(f"created {'public' if args.public else 'private'} repository {owner}/{args.name}")
    full = f"{owner}/{args.name}"
    url = f"https://x-access-token:{token}@github.com/{full}.git"

    remotes = subprocess.run(["git", "remote"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    if "origin" in remotes:
        run(["git", "remote", "set-url", "origin", url])
    else:
        run(["git", "remote", "add", "origin", url])
    run(["git", "push", "-u", "origin", "HEAD:main"])
    run([sys.executable, "scripts/ci_state.py", "save"])
    # never leave the token in .git/config
    run(["git", "remote", "set-url", "origin", f"https://github.com/{full}.git"])

    key = gh("GET", f"/repos/{full}/actions/secrets/public-key", token).json()
    n = 0
    for name in SECRET_NAMES:
        val = os.environ.get(name)
        if not val:
            continue
        rr = gh("PUT", f"/repos/{full}/actions/secrets/{name}", token,
                json={"encrypted_value": encrypt(key["key"], val), "key_id": key["key_id"]})
        print(f"  secret {name}: {'ok' if rr.ok else rr.text[:120]}")
        n += rr.ok
    print(f"{n} secrets set")

    if not args.no_dispatch:
        time.sleep(3)
        rr = gh("POST", f"/repos/{full}/actions/workflows/trade.yml/dispatches", token,
                json={"ref": "main", "inputs": {"mode": "alpaca", "dry_run": True}})
        print("dry-run trade workflow dispatched" if rr.status_code == 204 else f"dispatch failed: {rr.status_code} {rr.text[:200]}")
    print(f"\nrepository: https://github.com/{full}    actions: https://github.com/{full}/actions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
