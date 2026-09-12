"""Run ai-hedge-fund's staffed alpha models for one ticker/date and print a JSON line (inside .venv-aihf).

    python scripts/agents/run_ai_hedge_fund.py NVDA 2026-09-11 --mandate third_party/ai-hedge-fund/hedge_fund/fund/example.yaml
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("date")
    ap.add_argument("--mandate", required=True)
    args = ap.parse_args()

    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    signals, errors = [], []
    try:
        from hedge_fund.data import CachedDataClient, FDClient
        from hedge_fund.fund import Fund, load_spec

        fund = Fund(load_spec(args.mandate))
        with FDClient() as raw:
            fd = CachedDataClient(raw)
            for _strategy, staff in fund.strategies:
                for model in staff:
                    try:
                        sig = model.predict(args.ticker, args.date, fd)
                        signals.append({"model": getattr(sig, "model_name", type(model).__name__), "value": float(sig.value),
                                        "reasoning": (getattr(sig, "reasoning", None) or "")[:400]})
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"{type(model).__name__}: {type(e).__name__}: {e}")
        result = {"ticker": args.ticker, "date": args.date, "signals": signals, "errors": errors}
    except Exception as e:  # noqa: BLE001
        result = {"ticker": args.ticker, "date": args.date, "signals": [], "error": f"{type(e).__name__}: {e}"}
    finally:
        sys.stdout = real_stdout
    print(json.dumps(result))
    return 0 if signals else 1


if __name__ == "__main__":
    sys.exit(main())
