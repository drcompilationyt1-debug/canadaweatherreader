#!/usr/bin/env python
"""Create isolated virtualenvs for the LLM agent frameworks.

TradingAgents and ai-hedge-fund pin dependency versions that conflict with each other and with
StockBot (e.g. ai-hedge-fund downgrades the anthropic / openai SDKs), so each gets its own venv and
the StockBot adapters run them as subprocesses (``stockbot/signals/thirdparty/agent_runner.py``).

    python scripts/setup_agent_envs.py                 # both
    python scripts/setup_agent_envs.py --only tradingagents
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ENVS = {
    "tradingagents": ("third_party/TradingAgents", ".venv-tradingagents", "tradingagents"),
    "aihf": ("third_party/ai-hedge-fund", ".venv-aihf", "hedge_fund"),
    # TensorFlow for the Keras models of automating-technical-analysis (and any other .h5 you add)
    "tf": (None, ".venv-tf", "tensorflow"),
}
TF_PACKAGES = ["tensorflow-cpu", "tf_keras", "pandas", "numpy", "scikit-learn", "h5py", "pyarrow"]


def env_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def setup(name: str) -> bool:
    repo, venv, module = ENVS[name]
    env_dir = ROOT / venv
    if repo is not None:
        repo_dir = ROOT / repo
        if not (repo_dir / "pyproject.toml").exists() and not (repo_dir / "setup.py").exists():
            print(f"[{name}] repo not cloned: run python scripts/setup_submodules.py --only {repo.split('/')[-1]}")
            return False
    py = env_python(env_dir)
    if not py.exists():
        print(f"[{name}] creating {env_dir}")
        subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "-q", "-U", "pip"], check=False)
    if repo is None:
        print(f"[{name}] installing {' '.join(TF_PACKAGES)} (a few minutes)")
        r = subprocess.run([str(py), "-m", "pip", "install", "-q", *TF_PACKAGES], cwd=ROOT)
    else:
        print(f"[{name}] installing {repo} (this pulls langchain & co, a few minutes)")
        r = subprocess.run([str(py), "-m", "pip", "install", "-q", "-e", str(ROOT / repo)], cwd=ROOT)
    if r.returncode != 0:
        print(f"[{name}] pip failed ({r.returncode})")
        return False
    chk = subprocess.run([str(py), "-c", f"import {module}; print('ok')"], capture_output=True, text=True, cwd=ROOT)
    ok = chk.returncode == 0 and "ok" in chk.stdout
    print(f"[{name}] import {module}: {'OK' if ok else 'FAILED: ' + chk.stderr.strip()[-300:]}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(ENVS))
    args = ap.parse_args()
    names = [args.only] if args.only else list(ENVS)
    results = {n: setup(n) for n in names}
    print("\nresult:", results)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
