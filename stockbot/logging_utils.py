from __future__ import annotations

import logging
import os

_CONFIGURED = False


def get_logger(name: str = "stockbot") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        level = os.environ.get("STOCKBOT_LOG", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        # third-party chatter
        for noisy in ("yfinance", "peewee", "urllib3", "httpx", "httpx2", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _CONFIGURED = True
    return logging.getLogger(name)
