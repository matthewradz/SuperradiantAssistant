"""An optimize iteration must read the shot it fired, not one from history.

Reproduces the 2026-08-17 run: fifteen iterations whose printed values all came
from 2026-08-12 files, and a reported best that belonged to the older session.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


import h5py

from superradiant_assistant.orchestrator.live_executor import LiveExecutor
from superradiant_assistant.orchestrator.loop import wait_for_shot_signal
from superradiant_assistant.orchestrator.developer import ShotRequest

TMP = Path(os.environ.get("TEMP", ".")) / "test_live_feedback"


def write_shot(root: Path, name: str, angle: float, mean: float, results=True):
    """A shot file shaped the way a real one is.

    Globals live in a *subgroup* of /globals and are stored as strings — that is
    what runmanager writes and what the reader walks. Putting them on /globals
    itself parses to nothing.
    """
    root.mkdir(parents=True, exist_ok=True)
    p = root / name
    with h5py.File(p, "w") as f:
        g = f.create_group("globals/AWG")
        g.attrs["waveplate_angle"] = repr(float(angle))
        f.create_group("shot_properties")
        if results:
            r = f.create_group("results/waveplate_transmission")
            r.attrs["CHAN1_mean"] = float(mean)
    return p


class FakeRunmanager:
    """Records what was queued, and writes the shot file BLACS would have."""

    def __init__(self, root, produce=True, delay=0.0, with_results=True):
        self.root, self.produce = Path(root), produce
        self.delay, self.with_results = delay, with_results
        self.queued = []
        self._n = 0

    def set_globals(self, g):
        self.last = dict(g)

    def n_shots(self):
        return 1

    def get_globals(self):
        return dict(getattr(self, "last", {}))

    def engage(self):
        angle = float(self.last["waveplate_angle"])
        self.queued.append(angle)
        if not self.produce:
            return
        self._n += 1
        # The value a real detector would give at this angle: a Malus curve with
        # the fast axis at 9.5 deg, so a wrong angle gives a visibly wrong value.
        mean = 0.81 * np.cos(np.deg2rad(2 * (angle - 9.5))) ** 2
        time.sleep(self.delay)
        write_shot(self.root, f"2026-08-17_{self._n:04d}_new_0.h5", angle, mean,
                   results=self.with_results)


print("\n=== 1. the value returned belongs to the angle just set ===")
root = TMP / "case1"
if root.exists():
    for p in root.glob("*.h5"):
        p.unlink()
# History that the old code would have matched against: a 10 deg grid, five days
# old, with values from a different alignment.
for i, a in enumerate(range(150, 220, 10)):
    write_shot(root, f"2026-08-12_0005_old_{i:02d}.h5", a,
               0.45 + 0.01 * i)
old = sorted(root.glob("2026-08-12_*.h5"))
past = time.time() - 3600
for p in old:
    os.utime(p, (past, past))
check(len(old) == 7, f"{len(old)} historical shots on disk to be tempted by")

rm = FakeRunmanager(root)
ex = LiveExecutor(root, rm, shot_timeout=20.0)
sig = ex.execute(ShotRequest(sequence_file="waveplate_rotate.py",
                             globals_to_set={"waveplate_angle": 198.57},
                             required_metric="CHAN1_mean"))
got_angle = float(sig.all_globals.get("waveplate_angle"))
got_val = sig.metric_value("CHAN1_mean")
expect = 0.81 * np.cos(np.deg2rad(2 * (198.57 - 9.5))) ** 2
print(f"    set 198.57 -> shot {sig.shot_id}, angle={got_angle}, "
      f"CHAN1_mean={got_val:.4f} (expected {expect:.4f})")
check(abs(got_angle - 198.57) < 1e-6,
      "the signal carries the angle that was set, not a nearest neighbour")
check(abs(got_val - expect) < 1e-9, "and the value measured at that angle")
check("2026-08-12" not in sig.shot_id,
      f"the shot is not one from the old session ({sig.shot_id})")

print("\n=== 2. a shot that never appears is an error, not a replay ===")
root2 = TMP / "case2"
if root2.exists():
    for p in root2.glob("*.h5"):
        p.unlink()
write_shot(root2, "2026-08-12_0005_old_00.h5", 190.0, 0.5366)
os.utime(root2 / "2026-08-12_0005_old_00.h5", (past, past))
rm2 = FakeRunmanager(root2, produce=False)
ex2 = LiveExecutor(root2, rm2, shot_timeout=3.0)
try:
    ex2.execute(ShotRequest(sequence_file="s.py",
                            globals_to_set={"waveplate_angle": 190.0},
                            required_metric="CHAN1_mean"))
    check(False, "a shot that never ran should raise")
except RuntimeError as e:
    print(f"    {str(e)[:110]}...")
    check("no shot file appeared" in str(e), "it says no file appeared")
    check("Nothing was measured" in str(e), "and that nothing was measured")
    check("0.5366" not in str(e), "and it did not hand back the historical value")

print("\n=== 3. a shot with no lyse results is also an error ===")
root3 = TMP / "case3"
if root3.exists():
    for p in root3.glob("*.h5"):
        p.unlink()
rm3 = FakeRunmanager(root3, with_results=False)
ex3 = LiveExecutor(root3, rm3, shot_timeout=3.0)
try:
    ex3.execute(ShotRequest(sequence_file="s.py",
                            globals_to_set={"waveplate_angle": 10.0},
                            required_metric="CHAN1_mean"))
    check(False, "an unanalysed shot should raise")
except RuntimeError as e:
    print(f"    {str(e)[:110]}...")
    check("no results" in str(e), "it says lyse saved no results")
    check("singleshot routine" in str(e), "and names what to check")

print("\n=== 4. the waiter ignores anything older than the stamp ===")
root4 = TMP / "case4"
if root4.exists():
    for p in root4.glob("*.h5"):
        p.unlink()
write_shot(root4, "2026-08-12_0005_old_00.h5", 190.0, 0.5366)
os.utime(root4 / "2026-08-12_0005_old_00.h5", (past, past))
sig4, diag = wait_for_shot_signal(root4, since_ts=time.time(), timeout=2.0,
                                  poll=0.3, first_file_grace=1.0)
check(sig4 is None, "an old file does not satisfy the wait")
check(diag == "no_shot_files", f"and the diagnosis says so ({diag})")

print("\n=== 5. it gives up early rather than burning the timeout ===")
t0 = time.time()
wait_for_shot_signal(root4, since_ts=time.time(), timeout=60.0, poll=0.3,
                     first_file_grace=1.0)
took = time.time() - t0
print(f"    gave up after {took:.1f}s of a 60s timeout")
check(took < 10, "a failed compile is reported in seconds, not minutes")

print("\n=== 6. there is no path back to the replay executor ===")
import superradiant_assistant.orchestrator.live_executor as le
src = Path(le.__file__).read_text(encoding="utf-8")
check("OfflineReplayExecutor" not in src,
      "live_executor no longer imports the replay executor")
check(not hasattr(ex, "replay"), "and a LiveExecutor has no .replay to fall back to")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
