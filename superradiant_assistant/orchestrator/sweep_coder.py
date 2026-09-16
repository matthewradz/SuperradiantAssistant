"""Deterministic sweep coder — queues the full sweep as a single runmanager engage.

Instead of queueing one shot per iteration, generates a numpy linspace expression
for the sweep parameter and calls engage() once. Runmanager then queues all N shots
automatically (one per element of the list), which is the standard labscript workflow.

The loop runs for max_iterations=1, then exits.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional

from superradiant_assistant.goal import Stage
from superradiant_assistant.signals import ShotSignal
from superradiant_assistant.orchestrator.developer import ShotRequest


class DeterministicSweepCoder:
    def __init__(
        self,
        runmanager=None,           # RunmanagerInterface or None
        center_override: Optional[float] = None,
    ):
        self.rm = runmanager
        self.center_override = center_override
        self._done = False

    def _get_center(self, stage: Stage) -> float:
        """Get the center value of the sweep from runmanager or override."""
        if self.center_override is not None:
            return self.center_override
        if self.rm is not None:
            try:
                g = self.rm.get_globals()
                raw = g.get(stage.sweep_param)
                # If the value is a list/array (e.g. already a sweep expression),
                # try the scalar base param (strip _list suffix)
                if isinstance(raw, list):
                    raw = None
                if raw is None and stage.sweep_param.endswith("_list"):
                    base = stage.sweep_param[:-5]  # strip "_list"
                    raw = g.get(base)
                if raw is not None:
                    return float(raw)
            except Exception as e:
                print(f"  [sweep] could not read center from runmanager: {e}")
        print(f"  [sweep] WARNING: could not determine center, defaulting to 0.0")
        return 0.0

    def make_shot_request(
        self,
        plan,
        stage: Stage,
        history: List[ShotSignal],
        state: Dict[str, Any],
    ) -> ShotRequest:
        if self._done:
            # Already queued — return a dummy that signals completion
            return ShotRequest(
                sequence_file=stage.sequence_file,
                globals_to_set={},
                notes="sweep_complete",
            )

        center = None
        # Explicit start/end takes priority (e.g. calibration_precession_time in ms)
        if stage.sweep_start is not None and stage.sweep_end is not None:
            start = stage.sweep_start
            end   = stage.sweep_end
            n     = stage.max_iterations
        else:
            # Frequency sweep: center ± range/2 in MHz
            center = self._get_center(stage)
            step   = stage.sweep_step_mhz  or 0.00025
            total  = stage.sweep_range_mhz or (step * max(stage.max_iterations - 1, 1))
            half   = total / 2.0
            n      = max(int(round(total / step)) + 1, 2)
            start  = round(center - half, 9)
            end    = round(center + half, 9)

        # Generate numpy linspace expression — runmanager evaluates this and queues n shots
        linspace_expr = f"np.linspace({start}, {end}, {n})"

        # Increment delta_duration by 0.1 so this sweep group is uniquely identifiable
        globals_to_set: dict = {stage.sweep_param: linspace_expr}
        new_delta = None
        if self.rm is not None:
            try:
                g = self.rm.get_globals()
                current_delta = float(g.get("delta_duration", 0))
                new_delta = round(current_delta + 0.1, 2)
                globals_to_set["delta_duration"] = new_delta
            except Exception:
                pass

        print(f"  [sweep] {stage.sweep_param}: np.linspace({start:.6f}, {end:.6f}, {n})")
        if center is not None:
            print(f"          = {n} shots, center={center:.6f} MHz, step={step*1000:.4f} kHz")
        if new_delta is not None:
            print(f"  [sweep] delta_duration -> {new_delta} (incremented for this sweep group)")
        print(f"  [sweep] Setting full list in runmanager — all {n} shots will be queued at once")

        self._done = True  # only engage once

        return ShotRequest(
            sequence_file=stage.sequence_file,
            globals_to_set=globals_to_set,
            required_metric="Neta_2",
            notes=f"sweep {stage.sweep_param} from {start} to {end} in {n} steps",
        )
