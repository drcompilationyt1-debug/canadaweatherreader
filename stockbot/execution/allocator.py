"""Turn per-ticker target exposures (each relative to its capital slice) into portfolio weights."""
from __future__ import annotations


def allocate(targets: dict[str, float], max_position: float = 0.25, max_gross_exposure: float = 1.0,
             allow_short: bool = True) -> dict[str, float]:
    """weights[t] = target[t] * max_position, scaled down so that sum(|w|) <= max_gross_exposure."""
    weights = {}
    for t, e in targets.items():
        e = float(max(-1.0, min(1.0, e)))
        if not allow_short:
            e = max(0.0, e)
        weights[t] = e * max_position
    gross = sum(abs(w) for w in weights.values())
    if gross > max_gross_exposure > 0:
        scale = max_gross_exposure / gross
        weights = {t: w * scale for t, w in weights.items()}
    return weights
