"""Deterministic sweep coder — no LLM needed when range and step are explicit.

Reads the current value of sweep_param from runmanager, then steps through
evenly-spaced values covering the requested range.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional

from superradiant_assistant.goal import Stage
from superradiant_assistant.signals import ShotSignal
from superradiant_assistant.orchestrator.developer import ShotRequest


class DeterministicSweepCoder:
    def __init__(
        self,
        runmanager=None,   # RunmanagerInterface or None (uses center_override)
        center_override: Optional[float] = None,
    ):
        self.rm = runmanager
        self.center_override = center_override
        self._values: List[float] = []
        self._initialized = False

    def _init_values(self, stage: Stage) -> None:
        if self._initialized:
            return

        # Get center value
        center = self.center_override
        if center is None and self.rm is not None:
            try:
                globals_dict = self.rm.get_globals()
                raw = globals_dict.get(stage.sweep_param)
                if raw is not None:
                    center = float(raw)
            except Exception as e:
                print(f"  [sweep] could not read current value from runmanager: {e}")

        if center is None:
            print(f"  [sweep] WARNING: could not determine center for {stage.sweep_param}, defaulting to 0.0")
            center = 0.0

        step = stage.sweep_step_mhz or 0.00025
        total = stage.sweep_range_mhz or (step * max(stage.max_iterations, 1))
        half = total / 2.0

        # Generate evenly spaced values across the range
        n = max(int(round(total / step)), 1)
        self._values = [
            round(center - half + i * step, 9)
            for i in range(n + 1)
        ]
        print(f"  [sweep] {stage.sweep_param}: {n+1} points from "
              f"{self._values[0]:.6f} to {self._values[-1]:.6f} MHz "
              f"(center={center:.6f}, step={step*1000:.4f}kHz)")
        self._initialized = True

    def make_shot_request(
        self,
        plan,            # PlanStep (unused — value determined deterministically)
        stage: Stage,
        history: List[ShotSignal],
        state: Dict[str, Any],
    ) -> ShotRequest:
        self._init_values(stage)

        idx = len(history)
        if idx < len(self._values):
            value = self._values[idx]
        else:
            # Exhausted planned values — clamp to last
            value = self._values[-1]

        print(f"  [sweep] {stage.sweep_param} = {value:.6f} MHz (point {idx+1}/{len(self._values)})")

        return ShotRequest(
            sequence_file=stage.sequence_file,
            globals_to_set={stage.sweep_param: value},
            required_metric=stage.target_metric,
        )
