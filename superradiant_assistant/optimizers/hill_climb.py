"""Tiny hill-climb / random-restart optimizer for Phase 0."""
from __future__ import annotations
import random
from typing import Dict, List

from superradiant_assistant.params import ParamSpec
from superradiant_assistant.signals import ShotSignal


class HillClimbOptimizer:
    def __init__(self, params: List[ParamSpec], target_metric: str,
                 step_frac: float = 0.1, seed: int = 0):
        self.params = params
        self.target_metric = target_metric
        self.step_frac = step_frac
        self.rng = random.Random(seed)

    def suggest(self, history: List[ShotSignal]) -> Dict[str, float]:
        if not history:
            return {p.name: p.init for p in self.params}

        scored = [(getattr(s, self.target_metric, None), s) for s in history]
        scored = [(v, s) for v, s in scored if v is not None]
        if not scored:
            return {p.name: p.init for p in self.params}
        scored.sort(key=lambda t: t[0], reverse=True)
        _, best_shot = scored[0]

        # Robustly extract a scalar starting point per param
        base: Dict[str, float] = {}
        for p in self.params:
            base[p.name] = _coerce_scalar(
                best_shot.atom_loading_globals.get(p.name), fallback=p.init
            )

        proposal: Dict[str, float] = {}
        for p in self.params:
            span = p.hi - p.lo
            delta = self.rng.uniform(-self.step_frac, self.step_frac) * span
            proposal[p.name] = p.clip(base[p.name] + delta)
        return proposal


def _coerce_scalar(v, fallback: float) -> float:
    """Turn lists / arrays / weird globals into a single float, else fallback."""
    if v is None:
        return fallback
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)):
        try:
            return float(v[0])
        except Exception:
            return fallback
    try:
        return float(v)
    except (TypeError, ValueError):
        return fallback