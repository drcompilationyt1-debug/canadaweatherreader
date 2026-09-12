"""API key pools: several keys per provider, used round-robin, rested individually on rate limits.

Keys are read from the environment, all of these forms combined and de-duplicated:

    OPENROUTER_API_KEY            one key
    OPENROUTER_API_KEY_2 .. _9    more keys
    OPENROUTER_API_KEYS           a comma / semicolon / newline separated list

The same scheme works for GROQ_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY, ... (``KeyPool("GROQ_API_KEY")``).
Free tiers are capped per account, so spreading calls over several accounts multiplies the daily
budget; a key that hits its cap rests for the provider's cooldown while the others keep working.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

_SPLIT = re.compile(r"[,;\s]+")


def load_keys(env_name: str) -> list[str]:
    keys: list[str] = []
    for name in [env_name] + [f"{env_name}_{i}" for i in range(2, 10)] + [f"{env_name}S"]:
        for k in _SPLIT.split(os.environ.get(name, "") or ""):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    return keys


@dataclass
class KeyPool:
    env_name: str
    keys: list[str] = field(default_factory=list)
    cooldown_until: dict[int, float] = field(default_factory=dict)
    bad: dict[int, str] = field(default_factory=dict)
    calls: int = 0

    def __post_init__(self) -> None:
        if not self.keys:
            self.keys = load_keys(self.env_name)

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def configured(self) -> bool:
        return bool(self.keys)

    def usable(self) -> list[int]:
        """Indices of the keys that can be used now, starting after the last one used (round-robin)."""
        now = time.time()
        n = len(self.keys)
        if n == 0:
            return []
        start = self.calls % n
        order = [(start + i) % n for i in range(n)]
        return [i for i in order if i not in self.bad and self.cooldown_until.get(i, 0.0) <= now]

    def pick(self) -> str | None:
        """Next usable key (advances the round-robin pointer); None when every key is resting / bad."""
        idx = self.usable()
        if not idx:
            return None
        self.calls += 1
        return self.keys[idx[0]]

    def index_of(self, key: str) -> int:
        return self.keys.index(key)

    def rest(self, key: str, seconds: float) -> None:
        self.cooldown_until[self.index_of(key)] = time.time() + max(1.0, float(seconds))

    def disable(self, key: str, why: str) -> None:
        self.bad[self.index_of(key)] = why

    def seconds_until_available(self) -> float:
        """0 when a key is usable now, else the shortest wait; inf when every key is disabled."""
        if self.usable():
            return 0.0
        now = time.time()
        waits = [t - now for i, t in self.cooldown_until.items() if i not in self.bad]
        return max(0.0, min(waits)) if waits else float("inf")

    def describe(self) -> str:
        n = len(self.keys)
        if n == 0:
            return f"set {self.env_name}"
        resting = sum(1 for i, t in self.cooldown_until.items() if t > time.time() and i not in self.bad)
        parts = [f"{n} key{'s' if n != 1 else ''}"]
        if resting:
            parts.append(f"{resting} resting")
        if self.bad:
            parts.append(f"{len(self.bad)} rejected")
        return ", ".join(parts)


_POOLS: dict[str, KeyPool] = {}


def pool(env_name: str) -> KeyPool:
    """Process-wide pool per environment variable name (so every caller shares the rotation)."""
    p = _POOLS.get(env_name)
    if p is None or set(p.keys) != set(load_keys(env_name)):
        p = KeyPool(env_name)
        _POOLS[env_name] = p
    return p
