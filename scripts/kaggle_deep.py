#!/usr/bin/env python
"""Weekly GPU training on Kaggle for the heavy ranking heads (the neural ensemble and TabPFN).

GitHub's runners have no GPU; Kaggle gives a free weekly GPU quota and an API.  This script pushes a private Kaggle script
kernel that clones the code (main) and the saved dataset (the ``state`` branch), runs ``stockbot gpu-train`` on the GPU and
leaves the heads' state files as its output; it then waits for the kernel and downloads them.  The weekly workflow copies
them into ``models/signals`` and saves the state, and from then on every runner reuses them (``signals.<head>.pretrained``).

    KAGGLE_API_TOKEN=... python scripts/kaggle_deep.py run --out kaggle_out      # push, wait, download
    python scripts/kaggle_deep.py status
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = "https://github.com/drcompilationyt1-debug/canadaweatherreader"
SLUG = "cwr-trainer"
HEAD_FILES = ["xs_nn.joblib", "xs_nn_preds.parquet", "xs_tabpfn.joblib", "xs_tabpfn_preds.parquet"]

KERNEL = r'''
import os, shutil, subprocess, sys, json, time
t0 = time.time()
REPO = "__REPO__"
os.environ["TABPFN_TOKEN"] = "__TABPFN_TOKEN__"
os.environ["TABPFN_NO_BROWSER"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
def run(cmd, cwd=None):
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, shell=isinstance(cmd, str))
run(["git", "clone", "--depth", "1", REPO, "/kaggle/temp/code"])
run(["git", "clone", "--depth", "1", "--branch", "state", "--filter=blob:none", "--sparse", REPO, "/kaggle/temp/state"])
run(["git", "sparse-checkout", "set", "models/dataset", "models/signals"], cwd="/kaggle/temp/state")
shutil.copytree("/kaggle/temp/state/models", "/kaggle/temp/code/models", dirs_exist_ok=True)
run([sys.executable, "-m", "pip", "install", "-q", "-e", "/kaggle/temp/code"])
run([sys.executable, "-m", "pip", "install", "-q", "tabpfn", "chronos-forecasting", "einops", "safetensors", "huggingface_hub"])
subprocess.run(["git", "submodule", "update", "--init", "--depth", "1", "third_party/Kronos"], cwd="/kaggle/temp/code")
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", flush=True)
r = subprocess.run(__COMMAND__, cwd="/kaggle/temp/code", capture_output=True, text=True, shell=True)
print(r.stdout[-6000:], r.stderr[-6000:], flush=True)
out = "/kaggle/working/out"
os.makedirs(out, exist_ok=True)
copied = []
import glob
for pattern in __COLLECT__:
    for src in glob.glob(os.path.join("/kaggle/temp/code", pattern), recursive=True):
        if os.path.isfile(src):
            rel = os.path.relpath(src, "/kaggle/temp/code")
            dst = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
json.dump({"returncode": r.returncode, "copied": copied, "minutes": round((time.time() - t0) / 60, 1),
           "cuda": torch.cuda.is_available(), "report": r.stdout[-4000:]}, open(os.path.join(out, "report.json"), "w"), indent=1)
print("done", copied, flush=True)
'''


def kaggle(*args: str, check: bool = True) -> str:
    exe = Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle")
    cmd = [str(exe) if exe.exists() else "kaggle", *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed: {r.stderr or r.stdout}")
    return (r.stdout or "") + (r.stderr or "")


def username() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    m = re.search(r"username:\s*(\S+)", kaggle("config", "view"))
    if not m or m.group(1) == "None":
        raise SystemExit("could not tell the Kaggle username - set KAGGLE_USERNAME")
    return m.group(1)


HEADS_COMMAND = "python -m stockbot gpu-train --ic-start 2019-01-01"
HEADS_COLLECT = [f"models/signals/{f}" for f in HEAD_FILES]


def push(user: str, slug: str = SLUG, command: str = HEADS_COMMAND, collect: list[str] | None = None) -> str:
    kid = f"{user}/{slug}"
    with tempfile.TemporaryDirectory() as d:
        src = (KERNEL.replace("__REPO__", REPO).replace("__TABPFN_TOKEN__", os.environ.get("TABPFN_TOKEN", ""))
               .replace("__COMMAND__", json.dumps(command)).replace("__COLLECT__", json.dumps(collect or HEADS_COLLECT)))
        (Path(d) / "run.py").write_text(src, encoding="utf-8")
        meta = {"id": kid, "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
        (Path(d) / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        for attempt in range(60):                                      # Kaggle runs at most 2 GPU sessions at a time
            out = kaggle("kernels", "push", "-p", d, check=False).strip()
            print(out, flush=True)
            if "session count" in out.lower():
                print("all GPU slots busy - retrying in 5 minutes", flush=True)
                time.sleep(300)
                continue
            if "error" in out.lower() and "successfully" not in out.lower():
                raise SystemExit(f"kernel push failed: {out}")
            break
        else:
            raise SystemExit("no GPU slot became free")
    return kid


def status(kid: str) -> str:
    out = kaggle("kernels", "status", kid, check=False)
    m = re.search(r'status "?([A-Za-z_.]+)"?', out)
    return (m.group(1) if m else out.strip()).split(".")[-1].upper()


def wait(kid: str, max_minutes: float) -> str:
    t0 = time.time()
    last = ""
    while time.time() - t0 < max_minutes * 60:
        st = status(kid)
        if st != last:
            print(f"{time.strftime('%H:%M:%S')} {kid}: {st}", flush=True)
            last = st
        if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELREQUESTED", "CANCELLED"):
            return st
        time.sleep(60)
    return "TIMEOUT"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "push", "status", "download"])
    ap.add_argument("--out", default="kaggle_out")
    ap.add_argument("--max-minutes", type=float, default=300)
    ap.add_argument("--slug", default=SLUG, help="the Kaggle kernel's name (research runs use their own)")
    ap.add_argument("--cmd", default=HEADS_COMMAND, help="what to run in the repo on the GPU")
    ap.add_argument("--collect", default=None, help="comma-separated globs (relative to the repo) returned as output")
    args = ap.parse_args()
    collect = [c.strip() for c in args.collect.split(",")] if args.collect else HEADS_COLLECT
    kid = f"{username()}/{args.slug}"
    if args.action == "status":
        print(kid, status(kid))
        return 0
    if args.action in ("run", "push"):
        kid = push(username(), args.slug, args.cmd, collect)
        if args.action == "push":
            return 0
        time.sleep(30)
        st = wait(kid, args.max_minutes)
        print("final status:", st)
        if st != "COMPLETE":
            print(kaggle("kernels", "output", kid, "-p", args.out, check=False)[-3000:])
            return 1
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", kid, "-p", args.out, check=False)[-2000:])
    rep = next(Path(args.out).rglob("report.json"), None)
    if rep is None:
        print("no report.json in the kernel output")
        return 1
    r = json.loads(rep.read_text(encoding="utf-8"))
    print(json.dumps({k: v for k, v in r.items() if k != "report"}, indent=1))
    print(r.get("report", ""))
    return 0 if r.get("returncode") == 0 and r.get("copied") else 1


if __name__ == "__main__":
    raise SystemExit(main())
