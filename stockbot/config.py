"""Configuration: ``config/default.yaml`` merged with an optional user YAML and ``--set`` overrides."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from .paths import DEFAULT_CONFIG, resolve


class Config(dict):
    """A dict with attribute access and dotted-path helpers.

    >>> cfg = Config({"env": {"allow_short": True}})
    >>> cfg.env.allow_short
    True
    >>> cfg.get_path("env.allow_short")
    True
    """

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as e:  # pragma: no cover - trivial
            raise AttributeError(key) from e
        return Config(val) if isinstance(val, dict) and not isinstance(val, Config) else val

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) and not isinstance(node, Config) else node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: dict = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def section(self, dotted: str) -> "Config":
        val = self.get_path(dotted, {})
        return val if isinstance(val, Config) else Config(val or {})

    def path(self, dotted: str, default: str | None = None) -> Path:
        """Filesystem path option resolved relative to the project root."""
        return resolve(self.get_path(dotted, default))

    def to_yaml(self) -> str:
        return yaml.safe_dump(json.loads(json.dumps(self)), sort_keys=False)


def rolling_train_end(months: int, today=None) -> str:
    """First day of the month ``months`` months ago: the newest data the policy may train on.

    Snapping to a month start keeps the split (and therefore the cached dataset) stable for a month
    while the paper-trading days still flow into the training set once they are old enough."""
    from datetime import date

    today = today or date.today()
    m = today.month - 1 - int(months)
    return f"{today.year + m // 12:04d}-{m % 12 + 1:02d}-01"


def resolve_train_end(cfg: Config) -> None:
    """``data.train_end: rolling:12`` -> the concrete month-start date (in place)."""
    from datetime import date, datetime

    val = cfg.get_path("data.train_end")
    if isinstance(val, str) and val.lower().startswith("rolling:"):
        cfg.set_path("data.train_end", rolling_train_end(int(val.split(":", 1)[1] or 12)))
    elif isinstance(val, (date, datetime)):  # `--set data.train_end=2022-12-31` is YAML-coerced to a date
        cfg.set_path("data.train_end", val.strftime("%Y-%m-%d"))


def env_settings(cfg: Config) -> dict:
    """The simulator's settings: the ``env`` section plus the fee schedule the brokers use, so the
    policy trains, is evaluated and trades with the same costs."""
    out = dict(cfg.section("env"))
    out.setdefault("fees", str(cfg.get_path("fees.preset", "moomoo") or "moomoo"))
    out.setdefault("fees_by_market", {str(m): str(p) for m, p in (cfg.get_path("fees.by_market", {}) or {}).items()})
    out.setdefault("fee_scale", float(cfg.get_path("execution.max_position", 1.0) or 1.0))
    return out


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def account_names(cfg) -> list[str]:
    """Extra accounts traded in the same session (``accounts:`` in the config), enabled ones only."""
    accs = cfg.get("accounts") or {}
    return [n for n, a in accs.items() if isinstance(a, dict) and a.get("enabled", True)]


def account_config(cfg, name: str | None):
    """The config of one extra account: the base config with the account's overrides (execution, env, feedback,
    session, report ...) deep-merged in and ``account`` set to its name.  ``None`` / ``main`` = the base config."""
    if not name or name == "main":
        return cfg
    accs = cfg.get("accounts") or {}
    if name not in accs:
        raise KeyError(f"unknown account {name!r}; configured: {list(accs)}")
    over = {k: v for k, v in dict(accs[name]).items() if k != "enabled"}
    out = Config(deep_merge(dict(cfg), over))
    out["account"] = name
    return out


def _coerce(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def load_config(path: str | Path | None = None, overrides: Iterable[str] | None = None) -> Config:
    """Load defaults, merge a user config (YAML) and ``key.path=value`` overrides."""
    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if path:
        p = resolve(path)
        with open(p, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        data = deep_merge(data, user)
    cfg = Config(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, val = item.split("=", 1)
        cfg.set_path(key.strip(), _coerce(val.strip()))
    resolve_train_end(cfg)
    return cfg
