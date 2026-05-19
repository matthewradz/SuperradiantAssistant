"""Structured goal and stage types."""
from __future__ import annotations
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel

ThresholdOp = Literal[">", ">=", "<", "<=", "=="]


class Stage(BaseModel):
    name: str
    description: str = ""
    target_metric: str = "Neta_2"
    threshold: Optional[float] = None
    threshold_op: ThresholdOp = ">"
    sequence_file: str = ""
    params_file: str = "examples/params.txt"
    max_iterations: int = 50
    status: Literal["pending", "active", "complete", "failed"] = "pending"
    result: Optional[Dict[str, Any]] = None
    notes: str = ""
    # sweep support
    stage_kind: Literal["optimize", "sweep"] = "optimize"
    sweep_param: Optional[str] = None   # global being swept (e.g. "yellow_doublepass_freq_list")
    plot_ratio: Optional[str] = None    # e.g. "Neta_5/Neta_4" — plotted on y-axis after sweep

    def threshold_met(self, value: float) -> bool:
        if self.threshold is None:
            return False
        ops = {
            ">":  lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<":  lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
        }
        return ops[self.threshold_op](value, self.threshold)


class Goal(BaseModel):
    stages: List[Stage] = []
    max_dollars: float = 100.0
    max_tokens: int = 1_000_000
    timeout_seconds: int = 3600
    raw_prompt: str = ""

    @property
    def current_stage(self) -> Optional[Stage]:
        for s in self.stages:
            if s.status in ("pending", "active"):
                return s
        return None

    @property
    def all_complete(self) -> bool:
        return bool(self.stages) and all(s.status == "complete" for s in self.stages)

    # ---- backward-compat shims for single-stage code ----
    @property
    def target_metric(self) -> str:
        return self.stages[0].target_metric if self.stages else "Neta_2"

    @property
    def threshold(self) -> Optional[float]:
        return self.stages[0].threshold if self.stages else None

    @property
    def threshold_op(self) -> ThresholdOp:
        return self.stages[0].threshold_op if self.stages else ">"

    @property
    def sequence_file(self) -> str:
        return self.stages[0].sequence_file if self.stages else ""

    @property
    def params_file(self) -> str:
        return self.stages[0].params_file if self.stages else "examples/params.txt"

    @property
    def max_iterations(self) -> int:
        return self.stages[0].max_iterations if self.stages else 50

    def threshold_met(self, value: float) -> bool:
        s = self.current_stage
        return s.threshold_met(value) if s else False