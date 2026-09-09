"""A full turn around a periodic parameter must sweep, not be refused."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from superradiant_assistant.safety import (
    load_global_specs, wrap_sweep, validate_sweep, SafetyViolation,
)

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


print("\n=== 1. the period is declared ===")
specs = load_global_specs()
ph = specs["phase"]
print(f"    phase: {ph.describe_range()}")
check(ph.is_periodic and ph.period == 360.0, "phase declares period 360")
check(not specs["amplitude"].is_periodic, "amplitude is not periodic")
check("wraps at 360" in ph.describe_range(), "the range description says so")

print("\n=== 2. the case that failed twice tonight ===")
s, e, n, note = wrap_sweep("phase", 0.0, 360.0, 13)
print(f"    0..360 x13  ->  {s:g}..{e:g} x{n}")
print("    " + note.replace("\n", "\n    "))
check((s, e, n) == (0.0, 330.0, 12), f"became 0..330 in 12 points, got {s},{e},{n}")
check(note, "explains the change")
try:
    validate_sweep("phase", s, e, n)
    check(True, "the adjusted sweep passes validation")
except SafetyViolation as ex:
    check(False, f"adjusted sweep still refused: {ex}")

print("\n=== 3. spacing is preserved, coverage is one full turn ===")
step = (e - s) / (n - 1)
check(abs(step - 30.0) < 1e-9, f"step is 30 deg, got {step:g}")
check(abs((e + step) - 360.0) < 1e-9, "the next point after the last is 360 = 0")

print("\n=== 4. it does not touch anything it should not ===")
cases = [
    ("phase", 0.0, 180.0, 7, "half turn, already in range"),
    ("phase", 0.0, 359.9, 13, "already the declared max"),
    ("phase", 90.0, 270.0, 5, "partial span"),
    ("sine_frequency", 100.0, 25000.0, 26, "non-periodic parameter"),
    ("amplitude", 0.0, 1.0, 5, "non-periodic, endpoint at max"),
]
for name, a, b, k, why in cases:
    s2, e2, n2, note2 = wrap_sweep(name, a, b, k)
    same = (s2, e2, n2) == (a, b, k) and not note2
    check(same, f"untouched: {name} {a:g}..{b:g} x{k}  ({why})")

print("\n=== 5. a genuinely out-of-range sweep is still refused ===")
for a, b, k, why in [(0.0, 400.0, 5, "beyond one turn"),
                     (-10.0, 350.0, 5, "negative start")]:
    s3, e3, n3, _ = wrap_sweep("phase", a, b, k)
    try:
        validate_sweep("phase", s3, e3, n3)
        check(False, f"WRONGLY ALLOWED {a:g}..{b:g} ({why})")
    except SafetyViolation:
        check(True, f"still refused: {a:g}..{b:g}  ({why})")

print("\n=== 6. run_sweep now accepts the original request ===")
from superradiant_assistant.tools import build_registry, configure_session
from superradiant_assistant.hooks import default_gate
configure_session(live=False)
reg = build_registry(gate=default_gate(auto_approve=True), creative=True)
# Whatever this install actually whitelists, not a path from the machine the
# test was written on: run_sweep validates against the whitelist, so a foreign
# absolute path is refused for the wrong reason and the range check under test
# never runs.
from superradiant_assistant.safety import allowed_sequence_files
_allowed = allowed_sequence_files()
sequence_file = _allowed[0] if _allowed else "sequences/none_configured.py"
out = reg.dispatch("coder", "run_sweep", {
    "sweep_param": "phase", "mode": "explicit",
    "start": 0, "end": 360, "n_points": 13,
    "sequence_file": sequence_file,
    "signal_spec": "CHAN2 vs CHAN1",
    "reason": "rotating Lissajous",
})
first = out.split("\n")[0]
print(f"    {first[:150]}")
check("outside the safe range" not in out, "no longer refused for being out of range")
check("wraps at 360" in out or "repeats its first value" in out,
      "the adjustment is reported to the model")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
