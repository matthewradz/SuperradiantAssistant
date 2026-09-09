"""The choice block must redraw in place: no drift, no leftovers, at any width."""
import io
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from superradiant_assistant import splash as S
from superradiant_assistant import script_inventory as inv
from superradiant_assistant.hooks import ToolGate, ChoiceOption, ChoicePrompt

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


def plain(s):
    """Visible text only. Must strip EVERY CSI sequence, not just colours:
    counting `\\x1b[2K` (4 chars) as content inflates each row by 4 and makes a
    fitting line look like it wraps."""
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s)


class Tty(io.StringIO):
    def isatty(self):
        return True


class TtyIn:
    def isatty(self):
        return True


LONG = "fourier_spectrum_20260809_20260809.py"
opts = [
    ChoiceOption("1", "new", f"write a dated new file: {LONG}",
                 apply=lambda a: {**a, "filename": "x"}),
    ChoiceOption("2", "replace",
                 f"write {LONG} under the name the agent chose", apply=lambda a: a),
    ChoiceOption("3", "reuse", "do not write; use a script that already exists",
                 apply=None, deny_reason="reuse"),
]
prompt = ChoicePrompt(question="Add a new script, replace this one, or reuse "
                               "what exists?", options=opts, context="")

print("\n=== 1. no drawn row ever exceeds the terminal width ===")
real_size = shutil.get_terminal_size
for cols in (60, 80, 100, 140):
    shutil.get_terminal_size = lambda fb=(80, 24), c=cols: type(
        "T", (), {"columns": c, "lines": 30})()
    keys = iter(["down", "down", "enter"])
    S.read_key = lambda: next(keys)
    gate = ToolGate(audit_path=Path("nul"), session_id="t",
                    always_confirm=frozenset({"w"}))
    buf, ro, ri = Tty(), sys.stdout, sys.stdin
    sys.stdout, sys.stdin = buf, TtyIn()
    try:
        gate.before_tool_call("w", {"filename": "x"}, preview="p", choices=prompt)
        raw = buf.getvalue()
    finally:
        sys.stdout, sys.stdin = ro, ri
    rows = [plain(r) for r in raw.split("\n")]
    over = [len(r) for r in rows if len(r) > cols - 1]
    check(not over, f"{cols:>4} cols: longest row {max((len(r) for r in rows), default=0)}"
                    f" -> {'fits' if not over else f'WRAPS {over}'}")
shutil.get_terminal_size = real_size

print("\n=== 2. the redraw moves up exactly as many lines as it drew ===")
shutil.get_terminal_size = lambda fb=(80, 24): type(
    "T", (), {"columns": 80, "lines": 30})()
keys = iter(["down", "down", "enter"])
S.read_key = lambda: next(keys)
gate = ToolGate(audit_path=Path("nul"), session_id="t",
                always_confirm=frozenset({"w"}))
buf, ro, ri = Tty(), sys.stdout, sys.stdin
sys.stdout, sys.stdin = buf, TtyIn()
try:
    gate.before_tool_call("w", {"filename": "x"}, preview="p", choices=prompt)
    raw = buf.getvalue()
finally:
    sys.stdout, sys.stdin = ro, ri
shutil.get_terminal_size = real_size

frames = raw.split("\x1b[")
moves = [int(m.group(1)) for m in re.finditer(r"\x1b\[(\d+)F", raw)]
blocks = [b for b in re.split(r"\x1b\[\d+F", raw)]
drawn_counts = [b.count("\x1b[2K") for b in blocks if "\x1b[2K" in b]
print(f"    drew {drawn_counts} rows per frame; moved up {moves} lines")
check(len(set(drawn_counts)) == 1,
      f"every frame draws the same number of rows: {set(drawn_counts)}")
check(all(m == drawn_counts[0] for m in moves),
      "each move-up equals the rows drawn -> the block stays put")

print("\n=== 3. no leftover tail from a previous frame ===")
lines, cur, col = [], [], 0
i, s = 0, plain(raw)
while i < len(s):
    ch = s[i]
    if ch == "\r":
        col = 0
    elif ch == "\n":
        lines.append("".join(cur).rstrip()); cur, col = [], 0
    elif s.startswith("\x1b[2K", i):
        cur, col = [], 0; i += 4; continue
    elif ch == "\x1b":
        i = s.find("F", i) + 1 if "F" in s[i:i + 8] else i + 1
        continue
    else:
        while len(cur) <= col:
            cur.append(" ")
        cur[col] = ch; col += 1
    i += 1
if cur:
    lines.append("".join(cur).rstrip())
tail = [l for l in lines if l.strip().startswith("Add a new script")
        and l.rstrip().endswith(".py")]
check(not tail, f"the question line carries no wrapped remnant ({tail})")

print("\n=== 4. dated_stem does not double-date ===")
from datetime import datetime
today = datetime.now().strftime("%Y%m%d")
a = inv.dated_stem("fourier_spectrum", "singleshot")
b = inv.dated_stem(f"fourier_spectrum_{today}", "singleshot")
print(f"    'fourier_spectrum'          -> {a}")
print(f"    'fourier_spectrum_{today}' -> {b}")
check(a == b, "an already-dated stem is not dated twice")
check(a.count(today) == 1, f"the date appears once: {a}")

print("\n" + "=" * 58)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for x in bad:
    print("    - " + x)
sys.exit(1 if bad else 0)
