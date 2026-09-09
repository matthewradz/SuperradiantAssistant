"""Deterministic sweep coder — queues the full sweep as a single runmanager engage.

Instead of queueing one shot per iteration, generates a numpy linspace expression
for the sweep parameter and calls engage() once. Runmanager then queues all N shots
automatically (one per element of the list), which is the standard labscript workflow.

The loop runs for max_iterations=1, then exits.
"""
from __future__ import annotations
from superradiant_assistant import splash as _S
from typing import List, Dict, Any, Optional

from superradiant_assistant.goal import Stage
from superradiant_assistant.signals import ShotSignal
from superradiant_assistant.orchestrator.developer import ShotRequest


def _collapse_to_scalar(value: Any) -> Optional[float]:
    """First element of a multi-valued global, or None if it is already scalar.

    Used to undo a previous sweep. Runmanager takes the *product* of every
    multi-valued global, so a leftover 5-point list turns the next 5-point
    sweep into 25 shots — silently, and with the leftover parameter varying
    when the operator asked for one variable to move.
    """
    if isinstance(value, (list, tuple)):
        if len(value) <= 1:
            return None
        try:
            return float(value[0])
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        from superradiant_assistant.safety import parse_linspace
        parsed = parse_linspace(value)
        if parsed is not None and parsed[2] > 1:
            return float(parsed[0])
    return None


class DeterministicSweepCoder:
    def __init__(
        self,
        runmanager=None,           # RunmanagerInterface or None
        center_override: Optional[float] = None,
        collapse_other_sweeps: bool = True,
    ):
        self.rm = runmanager
        self.center_override = center_override
        # Set False only for a deliberate multi-dimensional scan.
        self.collapse_other_sweeps = collapse_other_sweeps
        self._done = False

    def _stale_sweep_resets(self, sweep_param: str, g: Dict[str, Any]) -> Dict[str, float]:
        """Scalars that collapse every *other* multi-valued writable global."""
        from superradiant_assistant.safety import load_global_specs
        writable = load_global_specs()
        resets: Dict[str, float] = {}
        for name, value in (g or {}).items():
            if name == sweep_param or name not in writable:
                continue
            scalar = _collapse_to_scalar(value)
            if scalar is not None:
                resets[name] = scalar
        return resets

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
                print(f"  {_S.tag('sweep')} could not read center from runmanager: {e}")
        print(f"  {_S.tag('sweep')} WARNING: could not determine center, defaulting to 0.0")
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
                # Undo any previous sweep before setting this one, in the same
                # write — otherwise runmanager multiplies them together.
                if self.collapse_other_sweeps:
                    resets = self._stale_sweep_resets(stage.sweep_param, g)
                    for rname, rval in resets.items():
                        print(f"  {_S.tag('sweep')} collapsing leftover sweep: {rname} -> {rval} "
                              f"(was multi-valued; would have multiplied the shot count)")
                    globals_to_set.update(resets)
                # Only tag the group when the experiment actually defines this
                # global. `.get(..., 0)` used to mask its absence and then write
                # it back, which runmanager refuses ("not found in any active
                # group") — taking the whole sweep down over a cosmetic label.
                if "delta_duration" in g:
                    current_delta = float(g["delta_duration"] or 0)
                    new_delta = round(current_delta + 0.1, 2)
                    globals_to_set["delta_duration"] = new_delta
            except Exception:
                pass

        print(f"  {_S.tag('sweep')} {stage.sweep_param}: np.linspace({start:.6f}, {end:.6f}, {n})")
        if center is not None:
            print(f"          = {n} shots, center={center:.6f} MHz, step={step*1000:.4f} kHz")
        if new_delta is not None:
            print(f"  {_S.tag('sweep')} delta_duration -> {new_delta} (incremented for this sweep group)")
        print(f"  {_S.tag('sweep')} Setting full list in runmanager — all {n} shots will be queued at once")

        self._done = True  # only engage once

        return ShotRequest(
            sequence_file=stage.sequence_file,
            globals_to_set=globals_to_set,
            required_metric="Neta_2",
            notes=f"sweep {stage.sweep_param} from {start} to {end} in {n} steps",
        )
