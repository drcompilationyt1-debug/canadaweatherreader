#!/usr/bin/env python
"""Register / update the third-party model repositories as git submodules.

Every external project StockBot can learn from lives under ``third_party/<name>``
as a *shallow* git submodule, so updating them later is just::

    python scripts/setup_submodules.py --update      # pull latest of every repo
    git submodule update --remote --merge            # (plain git equivalent)

Usage::

    python scripts/setup_submodules.py                 # core group (the ones we wrap)
    python scripts/setup_submodules.py --all           # everything, incl. reference-only repos
    python scripts/setup_submodules.py --only qlib,FinRL
    python scripts/setup_submodules.py --update        # fetch newest commit of registered repos
    python scripts/setup_submodules.py --list

Groups
------
core       repos StockBot has (or attempts) a live adapter for
reference  repos kept for strategy / code reuse only (no python adapter yet, or
           not a python project at all)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"

# name -> (url, group, note)
REPOS: dict[str, tuple[str, str, str]] = {
    # ---- core: wrapped by stockbot/signals/thirdparty/* --------------------------------
    "qlib": ("https://github.com/microsoft/qlib", "core",
             "Microsoft Qlib - Alpha158 factors + LightGBM/NN forecasters (most important)"),
    "FinRL": ("https://github.com/AI4Finance-Foundation/FinRL", "core",
              "FinRL - DRL trading environments / agents"),
    "backtrader": ("https://github.com/mementum/backtrader", "core",
                   "backtrader - event-driven backtesting engine (used to cross-check the policy)"),
    "trendet": ("https://github.com/alvarobartt/trendet", "core",
                "trendet - trend detection on OHLC series"),
    "stocksight": ("https://github.com/shirosaidev/stocksight", "core",
                   "stocksight - news/twitter sentiment approach (VADER + TextBlob)"),
    "freqtrade": ("https://github.com/freqtrade/freqtrade", "core",
                  "freqtrade - strategy framework, qtpylib indicators"),
    "TradingAgents": ("https://github.com/TauricResearch/TradingAgents", "core",
                      "TradingAgents - LLM multi-agent analyst/trader/risk debate"),
    "ai-hedge-fund": ("https://github.com/virattt/ai-hedge-fund", "core",
                      "ai-hedge-fund - LLM investor personas (Buffett, Munger, ...)"),
    "chronos-forecasting": ("https://github.com/amazon-science/chronos-forecasting", "core",
                            "Chronos-Bolt - zero-shot time-series foundation model (pip install chronos-forecasting)"),
    "Kronos": ("https://github.com/shiyu-coder/Kronos", "core",
               "Kronos - foundation model pre-trained on candlesticks (imported from the submodule)"),
    "timesfm": ("https://github.com/google-research/timesfm", "core",
                "TimesFM - Google's time-series foundation model (pip install timesfm[torch])"),
    "finBERT": ("https://github.com/ProsusAI/finBERT", "core",
                "FinBERT - financial headline sentiment (weights from Hugging Face, pip install transformers)"),
    "pandas-ta-classic": ("https://github.com/xgboosted/pandas-ta-classic", "core",
                          "pandas-ta - 150+ indicators (pip install pandas-ta-classic)"),
    "PyPortfolioOpt": ("https://github.com/robertmartin8/PyPortfolioOpt", "core",
                       "PyPortfolioOpt - hierarchical risk parity / risk-based weights (pip install PyPortfolioOpt)"),
    "WorldQuant_alpha101_code": ("https://github.com/yli188/WorldQuant_alpha101_code", "core",
                                 "WorldQuant 101 formulaic alphas (imported from the submodule)"),
    # ---- reference: included for code / strategy reuse ------------------------------
    "automating-technical-analysis": ("https://github.com/akurgat/automating-technical-analysis", "reference",
                                      "TA signal models + streamlit dashboard"),
    "AI-Stock-Trader": ("https://github.com/henryboisdequin/AI-Stock-Trader", "reference",
                        "Simple RL/ML stock trader"),
    "stockBot": ("https://github.com/romaingrx/stockBot", "reference",
                 "RL stock bot experiments"),
    "Stock-Prediction-Models": ("https://github.com/huseinzol05/Stock-Prediction-Models", "reference",
                                "Large collection of forecasting models + RL agents (notebooks)"),
    "nofx": ("https://github.com/NoFxAiOS/nofx", "reference",
             "NoFx - multi-LLM trading OS (Go backend, not importable from python)"),
    "stocks-insights-ai-agent": ("https://github.com/vinay-gatech/stocks-insights-ai-agent", "reference",
                                 "LangGraph stock insights agent"),
    "Stock-Market-AI-GUI": ("https://github.com/crypto-code/Stock-Market-AI-GUI", "reference",
                            "LSTM forecaster + evolution-strategy agent with GUI"),
    "Lean": ("https://github.com/QuantConnect/Lean", "reference",
             "QuantConnect LEAN engine (C#) - optional external backtester, large repo"),
}

LARGE = {"Lean", "Stock-Prediction-Models", "freqtrade"}


def run(cmd: list[str], check: bool = True, **kw) -> subprocess.CompletedProcess:
    print("  $", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, **kw)


def registered() -> set[str]:
    gm = ROOT / ".gitmodules"
    if not gm.exists():
        return set()
    out = subprocess.run(["git", "config", "-f", str(gm), "--get-regexp", r"submodule\..*\.path"],
                         cwd=ROOT, capture_output=True, text=True)
    names = set()
    for line in out.stdout.splitlines():
        path = line.split()[-1].replace("\\", "/")
        names.add(path.split("/")[-1])
    return names


def ensure_git_repo() -> None:
    if not (ROOT / ".git").exists():
        run(["git", "init", "-q"])
    # long paths (Lean) + never prompt for credentials on a public clone
    run(["git", "config", "core.longpaths", "true"], check=False)


def add(name: str) -> bool:
    url, _group, _note = REPOS[name]
    dest = THIRD_PARTY / name
    rel = f"third_party/{name}"
    if name in registered() and (dest / ".git").exists():
        print(f"[skip] {name}: already registered")
        return True
    if dest.exists() and any(dest.iterdir()) and not (dest / ".git").exists():
        print(f"[warn] {name}: {dest} exists and is not a submodule - leaving it alone")
        return False
    print(f"[add ] {name} <- {url}")
    full_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    r = run(["git", "submodule", "add", "--depth", "1", "--force", url, rel], check=False, env=full_env)
    if r.returncode != 0:
        print(f"[fail] {name}: git exited {r.returncode}")
        return False
    run(["git", "config", "-f", ".gitmodules", f"submodule.{rel}.shallow", "true"], check=False)
    return True


def update(names: list[str]) -> None:
    ensure_git_repo()
    run(["git", "submodule", "sync", "--recursive"], check=False)
    for name in names:
        rel = f"third_party/{name}"
        if not (THIRD_PARTY / name).exists():
            continue
        print(f"[upd ] {name}")
        run(["git", "submodule", "update", "--init", "--depth", "1", "--remote", "--", rel], check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="register every repo (core + reference)")
    ap.add_argument("--only", help="comma separated repo names")
    ap.add_argument("--skip-large", action="store_true", help=f"skip {sorted(LARGE)}")
    ap.add_argument("--update", action="store_true", help="pull newest commit of registered repos")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n, (url, group, note) in REPOS.items():
            flag = "*" if n in registered() else " "
            print(f"{flag} {n:32s} {group:9s} {note}")
        print("\n* = registered as submodule")
        return 0

    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in names if n not in REPOS]
        if unknown:
            print("unknown repos:", unknown, "\nknown:", list(REPOS))
            return 2
    elif args.all:
        names = list(REPOS)
    else:
        names = [n for n, v in REPOS.items() if v[1] == "core"]
    if args.skip_large:
        names = [n for n in names if n not in LARGE]

    if args.update:
        update(names)
        return 0

    ensure_git_repo()
    THIRD_PARTY.mkdir(exist_ok=True)
    (THIRD_PARTY / ".gitkeep").touch()
    ok, bad = [], []
    for n in names:
        (ok if add(n) else bad).append(n)
    print("\nregistered:", ok)
    if bad:
        print("FAILED    :", bad)
        print("re-run later with: python scripts/setup_submodules.py --only", ",".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
