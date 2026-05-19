"""Developer: turns a PlanStep into a concrete Action.

In HAL the developer writes Python. Here we just produce a structured Action,
which the executor consumes. Later we can let an LLM author Action subclasses
or generate analysis code.
"""
from __future__ import annotations
from typing import Dict, Any
from pydantic import BaseModel


class ShotRequest(BaseModel):
    sequence_file: str
    globals_to_set: Dict[str, Any]
    required_metric: str = "Neta_2"
    notes: str = ""


class Developer:
    def __init__(self, sequence_file: str):
        self.sequence_file = sequence_file

    def make_shot_request(self, suggested_params: Dict[str, Any]) -> ShotRequest:
        return ShotRequest(
            sequence_file=self.sequence_file,
            globals_to_set=dict(suggested_params),
        )