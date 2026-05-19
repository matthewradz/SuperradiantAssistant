"""PlanStep type + deterministic fallback planner."""
from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel

from superradiant_assistant.goal import Stage
from superradiant_assistant.signals import ShotSignal
from superradiant_assistant.params import ParamSpec

PlanKind = Literal[
    "propose_shot",
    "ask_user",
    "end",
    "stop_target_met",
    "stop_max_iters",
    "stop_timeout",
    "stop_error",
]


class PlanStep(BaseModel):
    kind: PlanKind
    rationale: str = ""
    prompt_for_coder: str = ""
    signal_description: str = ""
    user_question: str = ""
    suggested_params: Dict[str, Any] = {}

    model_config = {"populate_by_name": True}

    @classmethod
    def model_validate(cls, obj, **kwargs):
        # coerce None string fields to empty string
        if isinstance(obj, dict):
            for f in ("rationale", "prompt_for_coder",
                      "signal_description", "user_question"):
                if obj.get(f) is None:
                    obj[f] = ""
        return super().model_validate(obj, **kwargs)


def _best_metric(history: List[ShotSignal], metric: str) -> Optional[float]:
    vals = [getattr(s, metric, None) for s in history]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


class DeterministicPlanner:
    """Original hill-climb planner — kept as offline fallback."""
    def __init__(self, stage: Stage, params: List[ParamSpec], optimizer):
        self.stage = stage
        self.params = params
        self.optimizer = optimizer

    def plan_next(self, iteration: int, history: List[ShotSignal]) -> PlanStep:
        if iteration >= self.stage.max_iterations:
            return PlanStep(kind="stop_max_iters")
        best = _best_metric(history, self.stage.target_metric)
        if best is not None and self.stage.threshold_met(best):
            return PlanStep(kind="stop_target_met",
                            rationale=f"best={best:.2f} meets threshold")
        suggested = self.optimizer.suggest(history)
        return PlanStep(
            kind="propose_shot",
            rationale=f"iter {iteration+1}",
            suggested_params=suggested,
        )