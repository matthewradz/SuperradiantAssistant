"""The sweep progress bar must survive a running spinner.

The bug: `_heartbeat` redraws `* coder Thinking...` on the same line the bar
uses, eight times a second, so the bar was drawn and immediately overwritten.
This replays what a terminal would actually show.
"""
import io
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


class Tty(io.StringIO):
    def isatty(self):
        return True


def replay(raw: str):
    """Every frame a terminal would have *displayed*, in order.

    Measuring raw bytes is not measuring the screen, but neither is keeping only
    what survives to the end: with in-place redraw every frame is eventually
    erased, so the last-frame-only view shows nothing at all. What matters is
    the sequence that appeared, so each frame is snapshotted as it is wiped.
    """
    frames, cur = [], ""
    i = 0
    while i < len(raw):
        if raw.startswith("\x1b[2K", i) or raw[i] in "\r\n":
            if cur.strip():
                frames.append(cur)
            cur = ""
            i += 4 if raw[i] == "\x1b" else 1
        else:
            cur += raw[i]
            i += 1
    if cur.strip():
        frames.append(cur)
    return frames


print("\n=== 1. the bar is drawn under a live spinner ===")

from superradiant_assistant.orchestrator import loop as L
from superradiant_assistant.orchestrator import agents as A

# A sweep that never completes: four shots expected, none appear. Previously
# this window showed nothing but spinner frames.
L.list_shots = lambda root: []
L.read_shot = lambda p: None
import superradiant_assistant.interfaces.hdf5_reader as H
H.list_shots = lambda root: []

real_stdout = sys.stdout
buf = Tty()
sys.stdout = buf
try:
    with A._heartbeat("coder"):
        time.sleep(0.4)                      # let the spinner get going
        res = L.collect_sweep_results(
            Path("."), since_ts=0, expected=4, sweep_param="phase",
            timeout=2.0, poll=0.25, first_file_grace=99, stall_grace=99)
        time.sleep(0.3)
finally:
    sys.stdout = real_stdout

raw = buf.getvalue()
screen = replay(raw)
bar_frames = [l for l in screen if "░" in l or "█" in l]
spin_frames = [l for l in screen if "Thinking" in l or "coder" in l and "░" not in l]

print(f"    {len(screen)} frames on screen: {len(bar_frames)} bar, {len(spin_frames)} spinner")
check(len(bar_frames) >= 3, f"the bar was drawn repeatedly ({len(bar_frames)} frames)")

# The real test: no spinner frame lands between the last bar frame and the end
# of the wait. That is what "the bar is never seen" looked like.
last_bar = max((i for i, l in enumerate(screen) if "░" in l or "█" in l), default=-1)
after = [l for l in screen[last_bar + 1:] if "Thinking" in l]
check(last_bar >= 0, "a bar frame exists at all")

print("\n=== 2. the spinner is silent for the whole wait ===")
first_bar = min((i for i, l in enumerate(screen) if "░" in l), default=10 ** 6)
interleaved = [l for l in screen[first_bar:last_bar + 1]
               if "Thinking" in l or "esc to" in l]
check(not interleaved,
      f"no spinner frame between the first and last bar frame "
      f"({len(interleaved)} found)")

print("\n=== 3. the spinner comes back afterwards ===")
check(any("Thinking" in l or "coder" in l for l in screen[last_bar + 1:]),
      "the heartbeat resumes once the wait is over")

print("\n=== 4. the bar reports what it should ===")
sample = next(l for l in screen if "░" in l)
plain = re.sub(r"\x1b\[[0-9;]*m", "", sample)
print(f"    {plain.strip()}")
check("0/4 analysed" in plain, "counts analysed shots against the expected total")
check("shot file(s)" in plain, "reports shot files separately from results")
check(re.search(r"\d+s\s*$", plain.rstrip()), "shows elapsed seconds")
check("sweep" in plain, "carries the [sweep] tag")

print("\n=== 5. the summary line survives too ===")
check(any("shots analysed" in re.sub(r"\x1b\[[0-9;]*m", "", l) for l in screen),
      "'0/4 shots analysed' was printed, not overwritten")
check(res["diagnosis"] == "no_shot_files", f"diagnosis: {res['diagnosis']}")

print("\n=== 6. suppression is released even if the loop raises ===")
H.list_shots = lambda root: (_ for _ in ()).throw(RuntimeError("boom"))
L.list_shots = H.list_shots
sys.stdout = Tty()
try:
    try:
        L.collect_sweep_results(Path("."), 0, 4, "phase", timeout=1.0, poll=0.2)
    except RuntimeError:
        pass
finally:
    sys.stdout = real_stdout
check(not A._QUIET.is_set(), "_QUIET cleared after an exception inside the wait")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
