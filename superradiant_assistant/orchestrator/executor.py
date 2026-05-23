"""Executor: runs a ShotRequest and returns a ShotSignal.

Phase 0 (offline): finds the historical shot whose Atom Loading globals are
closest to the requested ones and returns its result. This lets us 'replay'
the experiment.

Phase 2 (live): set globals via runmanager, queue in BLACS, wait, read HDF5.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import numpy as np

from superradiant_assistant.signals import ShotSignal
from superradiant_assistant.interfaces.hdf5_reader import list_shots, read_shot
from superradiant_assistant.orchestrator.developer import ShotRequest


class OfflineReplayExecutor:
    """Executor that pulls from a historical HDF5 corpus."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self._cache: List[ShotSignal] = []
        self._used_ids: set[str] = set()

    def _ensure_loaded(self) -> None:
        if self._cache:
            return
        for p in list_shots(self.data_root):
            try:
                self._cache.append(read_shot(p))
            except Exception:
                pass

    def execute(self, req: ShotRequest) -> ShotSignal:
        self._ensure_loaded()
        # For sweeps: need Neta_4 and Neta_5 for ratio. For optimize: need Neta_2.
        # Use all_globals for distance so sweep params (e.g. clock frequency) are matched.
        candidates = [s for s in self._cache
                      if s.shot_id not in self._used_ids
                      and (s.Neta_4 is not None and s.Neta_5 is not None
                           or s.Neta_2 is not None)]
        if not candidates:
            raise RuntimeError("No more historical shots available to replay.")

        best = min(candidates,
                   key=lambda s: _distance(req.globals_to_set,
                                           {**s.atom_loading_globals, **s.all_globals}))
        self._used_ids.add(best.shot_id)
        # Tag the signal with what was requested so sweep plots use the right x-values
        best = best.model_copy(update={"requested_globals": dict(req.globals_to_set)})
        return best


def _distance(target: dict, actual: dict) -> float:
    """Normalized L2 distance over keys present in both."""
    keys = [k for k in target if k in actual]
    if not keys:
        return float("inf")
    diffs = []
    for k in keys:
        try:
            t = float(target[k])
            a = float(actual[k])
            scale = max(abs(t), 1e-9)
            diffs.append(((t - a) / scale) ** 2)
        except (TypeError, ValueError):
            continue
    if not diffs:
        return float("inf")
    return float(np.sqrt(np.mean(diffs)))