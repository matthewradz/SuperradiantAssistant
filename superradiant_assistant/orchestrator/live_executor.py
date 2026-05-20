"""Live executor: queues shots via runmanager/BLACS, returns results from offline replay.

Phase 2 hybrid: the shot is visibly queued in BLACS (so the user can see it and confirm
hardware safety), but results come from offline replay since the hardware may not be live.
Phase 3 will replace the replay fallback with real shot polling.
"""
from __future__ import annotations
from pathlib import Path
from typing import List

from superradiant_assistant.signals import ShotSignal
from superradiant_assistant.orchestrator.executor import OfflineReplayExecutor
from superradiant_assistant.orchestrator.developer import ShotRequest
from superradiant_assistant.interfaces.runmanager_iface import RunmanagerInterface


class LiveExecutor:
    def __init__(
        self,
        data_root: Path,
        runmanager: RunmanagerInterface,
    ):
        self.runmanager = runmanager
        self.replay = OfflineReplayExecutor(data_root)

    def execute(self, req: ShotRequest) -> ShotSignal:
        # 1. Set globals and queue in BLACS
        try:
            self.runmanager.set_globals_and_engage(req.globals_to_set)
            print(f"  [live] shot queued in BLACS ✓")
        except Exception as e:
            print(f"  [live] warning: could not queue shot: {e}")

        # 2. Get result from offline replay (hardware not live yet)
        return self.replay.execute(req)
