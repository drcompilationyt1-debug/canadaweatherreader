"""Run an LLM agent framework in its own virtualenv as a subprocess and read back one JSON line."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ...logging_utils import get_logger
from ...paths import ROOT

log = get_logger(__name__)


def env_python(env_dir: str | Path) -> Path | None:
    d = Path(env_dir)
    d = d if d.is_absolute() else ROOT / d
    py = d / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    return py if py.exists() else None


def resolve_python(configured: str | None, default_env: str, module: str) -> tuple[Path | None, str]:
    """Interpreter that can import ``module``: configured path, the default isolated venv, or this one."""
    if configured:
        p = Path(configured)
        p = p if p.is_absolute() else ROOT / p
        return (p, f"python={p}") if p.exists() else (None, f"configured python not found: {p}")
    py = env_python(default_env)
    if py is not None:
        return py, f"isolated env {default_env}"
    try:
        __import__(module)
        return Path(sys.executable), "importable in the current environment"
    except Exception:  # noqa: BLE001
        return None, f"run: python scripts/setup_agent_envs.py  (creates {default_env})"


def run_agent(python: Path, script: Path, args: list[str], timeout: float = 1800.0, env_extra: dict | None = None) -> dict:
    # PYTHONUTF8: the frameworks open text files without an encoding; on a non-UTF-8 Windows locale that breaks
    env = {**os.environ, **(env_extra or {}), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    cmd = [str(python), str(script), *args]
    log.info("running agent: %s (timeout %ds)", " ".join(cmd[1:3]), int(timeout))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(ROOT), encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"agent timed out after {timeout:.0f}s") from e
    for line in reversed((proc.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                out = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in out and not out.get("signals") and not out.get("decision"):
                raise RuntimeError(f"agent error: {out['error']}")
            return out
    tail = (proc.stderr or "").strip()[-600:]
    raise RuntimeError(f"agent produced no JSON (exit {proc.returncode}): {tail}")
