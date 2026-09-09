"""A report embeds the FINAL figure of each kind, not the partial ones.

lyse re-runs every multishot routine after every shot, so a routine that names
its output after the newest shot leaves one file per shot. Two different routines
did that on 2026-08-17; both reports came out carrying six half-drawn copies of
one plot and none of the finished one.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from superradiant_assistant.creative.tools import (
    _shot_figures, _figure_section, _MAX_SHOT_FIGURES,
)

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


TMP = Path(os.environ.get("TEMP", ".")) / "test_report_one_figure"


class Evidence:
    def __init__(self, shots):
        self.measurements = [{"shot_path": str(s)} for s in shots]


def scenario(name, n_shots, per_shot_kinds, anchored_kinds=()):
    """Lay out shot files and the figures a routine would have left beside them.

    `per_shot_kinds` are saved beside EVERY shot (the bug); `anchored_kinds` only
    beside the first (the fix). Later shots get later mtimes, so "newest" is
    well defined.
    """
    root = TMP / name
    if root.exists():
        for f in root.iterdir():
            f.unlink()
    root.mkdir(parents=True, exist_ok=True)
    shots = []
    for i in range(n_shots):
        h5 = root / ("2026-08-17_%04d_waveplate_rotate_0.h5" % (48 + i))
        h5.write_bytes(b"")
        shots.append(h5)
        for kind in per_shot_kinds:
            f = root / ("%s__%s.png" % (h5.stem, kind))
            f.write_bytes(b"x")
            os.utime(f, (time.time() + i, time.time() + i))
    for kind in anchored_kinds:
        f = root / ("%s__%s.png" % (shots[0].stem, kind))
        f.write_bytes(b"x")
        os.utime(f, (time.time() + n_shots, time.time() + n_shots))
    return shots


print("\n=== 1. the bug: one figure per shot ===")
shots = scenario("pershot", 20, ["optimizer_approach"])
got = _shot_figures(Evidence(shots))
print("    20 shots, 20 files on disk -> %d embedded" % len(got))
check(len(got) == 1, "exactly one figure is embedded, not six")
check(got[0].stem.startswith("2026-08-17_0067"),
      "and it is the NEWEST one, drawn after the last shot (%s)" % got[0].stem)

print("\n=== 2. the fix in the routine: one file, anchored ===")
shots = scenario("anchored", 20, [], ["optimizer_approach"])
got = _shot_figures(Evidence(shots))
check(len(got) == 1, "one figure")
check(got[0].stem.startswith("2026-08-17_0048"),
      "beside the first shot, overwritten each time (%s)" % got[0].stem)

print("\n=== 3. different kinds are all kept, one each ===")
shots = scenario("kinds", 12, ["optimizer_approach", "Waveplate_Malus_Drift"])
got = _shot_figures(Evidence(shots))
kinds = sorted(f.stem.partition("__")[2] for f in got)
print("    kinds: %s" % kinds)
check(len(got) == 2, "two figures for two kinds")
check(kinds == ["Waveplate_Malus_Drift", "optimizer_approach"],
      "one of each, and the order does not depend on shot walk order")
check(all(f.stem.startswith("2026-08-17_0059") for f in got),
      "both are the newest of their kind")

print("\n=== 4. a per-shot figure is still per-shot when it should be ===")
# A singleshot routine legitimately draws one figure per shot -- a Lissajous
# curve is a property of that shot alone. Those have distinct kinds only if the
# routine names them so; with one shared kind the newest wins, which is the
# documented behaviour and what the report wants.
shots = scenario("single", 3, [])
for i, s in enumerate(shots):
    f = s.parent / ("%s__Lissajous_Figure.png" % s.stem)
    f.write_bytes(b"x")
    os.utime(f, (time.time() + i, time.time() + i))
got = _shot_figures(Evidence(shots))
check(len(got) == 1, "one Lissajous figure, the newest shot's")

print("\n=== 5. the cap still holds ===")
shots = scenario("many", 4, ["k%d" % i for i in range(_MAX_SHOT_FIGURES + 4)])
got = _shot_figures(Evidence(shots))
check(len(got) == _MAX_SHOT_FIGURES,
      "%d kinds present, %d embedded" % (_MAX_SHOT_FIGURES + 4, len(got)))

print("\n=== 6. no shots, no figures, no crash ===")
check(_shot_figures(Evidence([])) == [], "empty evidence yields nothing")
check(_shot_figures(Evidence([TMP / "gone" / "nope.h5"])) == [],
      "a shot whose directory does not exist is skipped")
missing = Evidence([])
missing.measurements = [{"shot_path": ""}, {}]
check(_shot_figures(missing) == [], "blank and absent shot_path are skipped")

print("\n=== 7. the caption names the figure, not the shot order ===")
shots = scenario("caption", 5, ["optimizer_approach"])
section = _figure_section(None, _shot_figures(Evidence(shots)))
check(section.count("![") == 1, "one image in the section")
check("optimizer approach" in section, "captioned by kind")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
