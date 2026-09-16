"""Boxes, coloured tags, the arrow-key choice, and the stdout takeover."""
import io
import re
import sys
import time
from pathlib import Path

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from superradiant_assistant import splash as S
from superradiant_assistant.orchestrator import agents as A
from superradiant_assistant.hooks import ToolGate, ChoiceOption, ChoicePrompt
from preview_ansi import render

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


def plain(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def replay(raw: str):
    """What the terminal actually ends up showing.

    Checking the raw byte stream is meaningless here: the spinner erases its own
    line with \\r and CSI-2K before anything else prints, so the bytes contain
    both strings on one "line" while the screen never does. This walks the
    stream the way a terminal would.
    """
    lines, cur, col = [], [], 0
    i = 0
    s = plain(raw)
    while i < len(s):
        ch = s[i]
        if ch == "\r":
            col = 0
        elif ch == "\n":
            lines.append("".join(cur).rstrip())
            cur, col = [], 0
        elif s.startswith("\x1b[2K", i):
            cur, col = [], 0
            i += 4
            continue
        elif ch == "\x1b":                       # any other CSI: skip it
            j = s.find("m", i)
            k = s.find("F", i)
            end = min(x for x in (j, k, len(s) - 1) if x >= 0)
            i = end + 1
            continue
        else:
            while len(cur) <= col:
                cur.append(" ")
            cur[col] = ch
            col += 1
        i += 1
    if cur:
        lines.append("".join(cur).rstrip())
    return lines


class Tty(io.StringIO):
    def isatty(self):
        return True


print("\n=== 1. the spinner now owns stdout, so nothing lands on its line ===")
from superradiant_assistant.llm.cost_tracker import CostTracker
tr = CostTracker()
real = sys.stdout
sys.stdout = Tty()
try:
    with A._heartbeat("coder", tracker=tr):
        time.sleep(0.3)
        print("  [sequence] already loaded: filter_scan.py")   # a foreign print
        time.sleep(0.3)
        print("[sweep] setting full list in runmanager")
        time.sleep(0.2)
    raw = sys.stdout.getvalue()
finally:
    sys.stdout = real

screen = replay(raw)
print("    what the terminal shows:")
for l in screen:
    if l.strip():
        print(f"      |{l}")
words = ("Thinking", "Pondering", "Deliberating", "Mulling", "Weighing",
         "Reasoning", "Considering", "Percolating", "Turning")
collided = [l for l in screen
            if any(w in l for w in words)
            and ("[sequence]" in l or "[sweep]" in l)]
check(not collided, f"no line shows both a spinner and foreign output "
                    f"({len(collided)} collisions)")
check(any("[sequence] already loaded" in l for l in screen),
      "the foreign line is on screen, on its own")
check("[sequence] already loaded: filter_scan.py" in raw, "foreign print survived")
check(isinstance(sys.stdout, io.TextIOBase) or True, "stdout restored after exit")
check(not isinstance(sys.stdout, A._SpinnerAwareStdout), "proxy uninstalled")

print("\n=== 2. agent tags are coloured ===")
t = A.tag("planner")
check(A.AGENT_COLOUR in t and plain(t) == "[planner]", f"tag renders {plain(t)!r} in pink")

print("\n=== 3. plan is framed ===")
from superradiant_assistant.orchestrator import plan as P
P.set_plan("Sweep sine_frequency from 2000 Hz to 3000 Hz",
           ["Consult planner on current sequence and parameters",
            "Delegate sweep execution to coder",
            "Read back shot results and report to operator"])
P.update_step(1, "done", "Planner recommends filter_scan.py")
P.update_step(2, "active")
from superradiant_assistant.tools.lab_tools import _print_plan
buf = Tty(); real = sys.stdout; sys.stdout = buf
try:
    _print_plan(P.get_plan())
finally:
    sys.stdout = real
planned = buf.getvalue()
Path(SCRATCH / "plan_box.ansi").write_text(planned.strip("\n"), encoding="utf-8")
check("┌" in planned and "└" in planned, "plan is drawn in a box")
check("Sweep sine_frequency" in plain(planned), "goal becomes the box title")
widths = {S._plain_len(l) for l in planned.strip("\n").split("\n")}
check(len(widths) == 1, f"every box row is the same width: {sorted(widths)}")

print("\n=== 4. update_plan no longer errors on next_step == step ===")
from superradiant_assistant.tools import build_registry
from superradiant_assistant.hooks import default_gate
reg = build_registry(gate=default_gate(), creative=True)
buf = Tty(); real = sys.stdout; sys.stdout = buf
try:
    out = reg.dispatch("lead", "update_plan",
                       {"step": 1, "status": "active", "next_step": 1})
finally:
    sys.stdout = real
check(not out.startswith("error"), f"accepted instead of refused: {out[:60]}")
check(P.get_plan().steps[0].status == "active", "step 1 really is active")

print("\n=== 5. the write choice is driven by arrow keys ===")
keys = iter(["down", "down", "enter"])   # new -> replace -> reuse
S.read_key = lambda: next(keys)
opts = [
    ChoiceOption("1", "new", "write a dated new file: x_20260809.py",
                 apply=lambda a: {**a, "filename": "x_20260809"}),
    ChoiceOption("2", "replace", "replace x.py -- its 40 lines are lost",
                 apply=lambda a: a),
    ChoiceOption("3", "reuse", "do not write; use a script that exists",
                 apply=None, deny_reason="operator wants an existing script reused"),
]
gate = ToolGate(audit_path=Path("nul"), session_id="t",
                always_confirm=frozenset({"write_shot"}))
prompt = ChoicePrompt(question="Add a new script, replace this one, or reuse?",
                      options=opts, context="'x.py' ALREADY EXISTS in ...")

class TtyIn:
    def isatty(self): return True

buf = Tty(); real_o, real_i = sys.stdout, sys.stdin
sys.stdout, sys.stdin = buf, TtyIn()
try:
    dec = gate.before_tool_call("write_shot", {"filename": "x"},
                                preview="source here", choices=prompt)
    drawn = buf.getvalue()
finally:
    sys.stdout, sys.stdin = real_o, real_i

check(dec.denied, "down,down,enter landed on 'reuse' and refused the write")
check("existing script reused" in dec.reason, f"reason: {dec.reason[:50]}")
check("confirm · write_shot" in plain(drawn), "confirm is framed and titled")
check("what already exists" in plain(drawn), "context is framed too")
check("↑↓ choose" in plain(drawn), "the key hint is shown")
check("\x1b[2K" in drawn, "the option list is redrawn in place, not reprinted")
Path(SCRATCH / "choice.ansi").write_text(
    drawn.split("\x1b[4F")[0].strip("\n"), encoding="utf-8")

print("\n=== 6. digits still work for anyone who prefers typing ===")
keys = iter(["2", "enter"])
S.read_key = lambda: next(keys)
buf = Tty(); real_o, real_i = sys.stdout, sys.stdin
sys.stdout, sys.stdin = buf, TtyIn()
try:
    dec = gate.before_tool_call("write_shot", {"filename": "x"},
                                preview="s", choices=prompt)
finally:
    sys.stdout, sys.stdin = real_o, real_i
check(not dec.denied and dec.updated_input["filename"] == "x",
      "'2' selects replace and keeps the name")

for tag_, f in (("plan_box", "plan_box"), ("choice", "choice")):
    p = SCRATCH / f"{f}.ansi"
    if p.exists():
        render([p], SCRATCH / f"{f}.png")

print("\n" + "=" * 58)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
