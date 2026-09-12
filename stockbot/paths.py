"""Project paths and helpers for importing code out of ``third_party/`` submodules."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
CONFIG_DIR = ROOT / "config"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root unless it is absolute."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def third_party_dir(name: str) -> Path | None:
    """Return ``third_party/<name>`` if the submodule has been cloned (non-empty)."""
    d = THIRD_PARTY / name
    if d.is_dir() and any(d.iterdir()):
        return d
    return None


def add_submodule_to_syspath(name: str, subdir: str | None = None) -> bool:
    """Put a cloned submodule (or a sub-directory of it) on ``sys.path``.

    Returns False when the submodule is missing so callers can degrade gracefully.
    """
    d = third_party_dir(name)
    if d is None:
        return False
    if subdir:
        d = d / subdir
        if not d.is_dir():
            return False
    s = str(d)
    if s not in sys.path:
        sys.path.insert(0, s)
    return True


def import_optional(module: str, submodule_dir: str | None = None, subdir: str | None = None) -> ModuleType | None:
    """Import ``module``; if that fails and a submodule dir is given, try again from the submodule."""
    try:
        return importlib.import_module(module)
    except Exception:  # noqa: BLE001 - any import-time failure means "unavailable"
        pass
    if submodule_dir and add_submodule_to_syspath(submodule_dir, subdir):
        try:
            return importlib.import_module(module)
        except Exception:  # noqa: BLE001
            return None
    return None


def load_module_from_file(alias: str, file: Path) -> ModuleType | None:
    """Load a single python file as a module without importing its parent package."""
    if not file.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location(alias, file)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:  # noqa: BLE001
        sys.modules.pop(alias, None)
        return None
