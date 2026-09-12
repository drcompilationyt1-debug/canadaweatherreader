#!/usr/bin/env python
"""Zip the project (without the virtualenv, data, models and the big submodules) for upload to Colab.

    python scripts/make_colab_zip.py            -> StockBot_colab.zip next to the project
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "data", "models",
             "reports", "third_party", "stockbot.egg-info"}


def main() -> int:
    out = ROOT.parent / "StockBot_colab.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob("*"):
            rel = p.relative_to(ROOT)
            if any(part in SKIP_DIRS for part in rel.parts) or p.is_dir() or p.suffix in (".pyc", ".zip"):
                continue
            z.write(p, Path("StockBot") / rel)
            n += 1
        z.writestr("StockBot/third_party/.gitkeep", "")
    print(f"wrote {out} ({n} files, {out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
