"""Run TradingAgents for one ticker/date and print a JSON line (executed inside .venv-tradingagents).

    python scripts/agents/run_trading_agents.py NVDA 2026-09-11 --config-json '{"llm_provider": "openai"}'
"""
from __future__ import annotations

import argparse
import copy
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("date")
    ap.add_argument("--config-json", default="{}")
    args = ap.parse_args()
    overrides = json.loads(args.config_json or "{}")

    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # the framework prints progress; keep stdout for the JSON result only
    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        config = copy.deepcopy(DEFAULT_CONFIG)
        config.update({k: v for k, v in overrides.items() if v is not None})
        config.setdefault("online_tools", True)
        graph = TradingAgentsGraph(debug=False, config=config)
        _state, decision = graph.propagate(args.ticker, args.date)
        result = {"ticker": args.ticker, "date": args.date, "decision": str(decision)}
    except Exception as e:  # noqa: BLE001
        result = {"ticker": args.ticker, "date": args.date, "error": f"{type(e).__name__}: {e}"}
    finally:
        sys.stdout = real_stdout
    print(json.dumps(result))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
