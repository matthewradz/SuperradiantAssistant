"""Live executor: queues a shot via runmanager/BLACS and reads back what it measured.

It used to queue the shot and then return a *historical* shot matched by nearest
parameter value, on the grounds that the hardware might not be live. That made
every optimize iteration read five-day-old data: on 2026-08-17 a 15-iteration run
was driven entirely by files from 2026-08-12, the values printed per iteration
belonged to angles up to 8.6 deg away from the one being set, and the reported
best (0.5366) was a number from the older session while the shots actually fired
that day peaked at 0.7675.

So there is no replay path here any more. Queue the shot, wait for it, read it,
or fail -- a shot that cannot be read has not measured anything, and saying
otherwise is worse than saying nothing. Offline replay still exists for offline
runs; it is chosen by not using this executor at all.
"""
from __future__ import annotations
import time
from superradiant_assistant import splash as _S
from pathlib import Path
from typing import List, Optional

from superradiant_assistant.signals import ShotSignal
from superradiant_assistant.orchestrator.developer import ShotRequest
from superradiant_assistant.interfaces.runmanager_iface import RunmanagerInterface


def _expected_shots(globals_to_set: dict) -> Optional[int]:
    """How many shots this request alone implies, or None if undeterminable.

    Only counts the values being written now; anything already multi-valued in
    runmanager and left untouched is exactly what this is meant to catch.
    """
    from superradiant_assistant.safety import parse_linspace

    total = 1
    for value in (globals_to_set or {}).values():
        if isinstance(value, (list, tuple)):
            total *= max(len(value), 1)
        elif isinstance(value, str):
            parsed = parse_linspace(value)
            if parsed is None:
                return None
            total *= max(parsed[2], 1)
    return total


class LiveExecutor:
    def __init__(
        self,
        data_root: Path,
        runmanager: RunmanagerInterface,
        shot_timeout: float = 300.0,
    ):
        self.runmanager = runmanager
        self.data_root = Path(data_root)
        self.shot_timeout = float(shot_timeout)

    def execute(self, req: ShotRequest) -> ShotSignal:
        # Stamped before the engage, so the shot this returns is provably one
        # that ran after the request rather than whatever is newest on disk.
        since_ts = time.time()

        # 1. Set globals and queue in BLACS.
        # A failure here must be fatal: swallowing it used to leave the caller
        # looking at a replayed signal that reads exactly like a successful
        # queue, so a dead runmanager was indistinguishable from a live one.
        # Runmanager multiplies every multi-valued global together, so a leftover
        # list from an earlier sweep silently inflates the shot count. Check the
        # shape the request implies *before* engaging, not after.
        expected = _expected_shots(req.globals_to_set)
        self.runmanager.set_globals(req.globals_to_set)
        n = self.runmanager.n_shots()
        if n < 1:
            raise RuntimeError(
                "current globals expand to 0 shots — nothing to queue"
            )
        if expected is not None and n > expected:
            raise RuntimeError(
                f"refusing to engage: this request asks for {expected} shot(s) but "
                f"runmanager would queue {n}. Another global is still multi-valued "
                f"from a previous sweep — collapse it to a scalar first."
            )
        self.runmanager.engage()
        print(f"  {_S.tag('live')} queued in BLACS [OK] | {n} shot(s) from current globals")

        # 2. Wait for that shot to run and for lyse to analyse it, then read it.
        from superradiant_assistant.orchestrator.loop import wait_for_shot_signal
        sig, diagnosis = wait_for_shot_signal(
            self.data_root, since_ts=since_ts, timeout=self.shot_timeout,
        )
        if sig is not None:
            return sig

        asked = ", ".join(f"{k}={v}" for k, v in (req.globals_to_set or {}).items())
        if diagnosis == "no_shot_files":
            raise RuntimeError(
                f"the shot was queued ({asked}) but no shot file appeared in "
                f"{self.data_root} within {self.shot_timeout:.0f}s. Either the "
                f"sequence failed to compile or BLACS is not running the queue — "
                f"check the BLACS window. Nothing was measured.")
        raise RuntimeError(
            f"the shot ran ({asked}) but lyse saved no results for it within "
            f"{self.shot_timeout:.0f}s, so there is no number to act on. Check "
            f"that the singleshot routine is loaded and enabled in lyse. Nothing "
            f"was measured.")
