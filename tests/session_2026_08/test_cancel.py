"""A cancelled call must hand back an instruction, not a shrug."""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from superradiant_assistant import splash as S
from superradiant_assistant.hooks import (
    ToolGate, ChoiceOption, ChoicePrompt, CANCELLED, DECLINED,
)

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


class Tty(io.StringIO):
    def isatty(self):
        return True


class TtyIn:
    def isatty(self):
        return True


opts = [
    ChoiceOption("1", "new", "dated file", apply=lambda a: a),
    ChoiceOption("2", "replace", "same name", apply=lambda a: a),
    ChoiceOption("3", "reuse", "use an existing one", apply=None,
                 deny_reason="reuse an existing script"),
]
prompt = ChoicePrompt(question="q", options=opts, context="")


def drive(keys):
    it = iter(keys)
    S.read_key = lambda: next(it)
    gate = ToolGate(audit_path=Path("nul"), session_id="t",
                    always_confirm=frozenset({"write_analysis"}))
    buf, ro, ri = Tty(), sys.stdout, sys.stdin
    sys.stdout, sys.stdin = buf, TtyIn()
    try:
        return gate.before_tool_call("write_analysis", {"filename": "x"},
                                     preview="p", choices=prompt)
    finally:
        sys.stdout, sys.stdin = ro, ri


print("\n=== 1. cancel returns an instruction ===")
dec = drive(["down", "down", "down", "enter"])     # -> cancel
check(dec.denied, "cancel refuses the write")
r = dec.reason
print("    reason given to the model:")
for line in r.split("\n"):
    print(f"      {line}")
for want, why in (
    ("Do NOT retry", "forbids retrying"),
    ("different existing file", "forbids substituting another file"),
    ("lyse routines", "forbids reconfiguring lyse as a workaround"),
    ("Do NOT mark the step done", "forbids claiming completion"),
    ("ask the operator", "tells it to ask"),
):
    check(want in r, why)

print("\n=== 2. 'n' and esc land on cancel too ===")
for key in ("n", "esc", "q"):
    d = drive([key])
    check(d.denied and d.reason == CANCELLED, f"{key!r} -> the same instruction")

print("\n=== 3. the plain y/N decline says the same thing ===")
import builtins
gate = ToolGate(audit_path=Path("nul"), session_id="t",
                always_confirm=frozenset({"run_sweep"}))
real_in, real_out, real_input = sys.stdin, sys.stdout, builtins.input
sys.stdin, sys.stdout = TtyIn(), Tty()
builtins.input = lambda *a: "n"
try:
    d = gate.before_tool_call("run_sweep", {"sweep_param": "f"}, preview="p")
finally:
    sys.stdin, sys.stdout, builtins.input = real_in, real_out, real_input
check(d.denied and d.reason == DECLINED, f"declined -> {d.reason[:48]}...")
check("route around" in d.reason, "forbids routing around a declined sweep")

print("\n=== 4. reuse keeps its own, different instruction ===")
dec = drive(["down", "down", "enter"])
check(dec.reason == "reuse an existing script",
      "reuse is not overwritten by the cancel text")

print("\n=== 5. replacing lyse routines names what it dropped ===")
import superradiant_assistant.interfaces.lyse_iface as L
real_live, real_call = L.live_routines, None
state = {"n": 0}


def fake_live(kind="singleshot"):
    state["n"] += 1
    if state["n"] == 1:
        return [r"C:\x\fourier_spectrum_20260809.py"]   # what the operator loaded
    return [r"C:\x\scope_test.py"]


import superradiant_assistant.interfaces.runmanager_iface as R
L.live_routines = fake_live
L.Path.is_file = lambda self: True
real_rcall = R._call
R._call = lambda *a, **k: "ok"
try:
    out = L.set_live_routines([r"C:\x\scope_test.py"], replace=True,
                              kind="singleshot")
finally:
    L.live_routines = real_live
    R._call = real_rcall
    del L.Path.is_file

print("   ", out.replace("\n", "\n    "))
check("REPLACED and no longer running" in out, "the displaced routine is named")
check("fourier_spectrum_20260809.py" in out, "by filename")
check("operator may have loaded those deliberately" in out,
      "and the model is told to report it")

print("\n" + "=" * 58)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
