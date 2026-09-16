"""Verify the add/replace/reuse patch without touching hardware or the LLM."""
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from superradiant_assistant import script_inventory as inv
from superradiant_assistant.creative import tools as ctools
from superradiant_assistant.hooks import ToolGate
from superradiant_assistant.tools.registry import ToolRegistry, ToolSpec

ok = lambda m: print(f"  PASS  {m}")
bad = []


def check(cond, msg):
    if cond:
        ok(msg)
    else:
        print(f"  FAIL  {msg}")
        bad.append(msg)


print("\n=== 1. inventory sees the real folders ===")
shots = inv.scripts_of_kind("shot")
singles = inv.scripts_of_kind("singleshot")
multis = inv.scripts_of_kind("multishot")
print(f"  shot={len(shots)}  singleshot={len(singles)}  multishot={len(multis)}")
check(len(shots) > 0, "found shot sequences")
check(any(s.name == "Lissajous.py" for s in shots), "Lissajous.py listed")
# Not a fixed filename: the agent renames and replaces its own routines, and
# `lissajous_singleshot.py` is now `plot_lissajous.py`. What matters is that the
# inventory finds this apparatus's lissajous analysis, whatever it is called.
check(any("lissajous" in s.name.lower() for s in singles),
      f"a lissajous singleshot routine is listed: "
      f"{[s.name for s in singles if 'lissajous' in s.name.lower()]}")
check(len(singles) < 40, f"singleshot list filtered to this apparatus ({len(singles)}, "
                          f"not the ~600 in the folder)")
gen = [s.name for s in shots if s.generated]
check("filter_scan_ch2.py" in gen, f"creative-written shots tagged [gen]: {gen}")
check(not any(s.name == "Lissajous.py" and s.generated for s in shots),
      "hand-written Lissajous.py NOT tagged [gen]")

print("\n=== 2. dated_stem ===")
d1 = inv.dated_stem("filter_scan", "shot")
check(d1.startswith("filter_scan_2") and len(d1) == len("filter_scan_20260808"),
      f"dated name is {d1}")
check(not (inv.SHOT_DIR / f"{d1}.py").exists(), "dated name does not collide")
# collision path: make the dated file, ask again
probe = inv.SHOT_DIR / f"{d1}.py"
probe.write_text("# temp\n", encoding="utf-8")
try:
    d2 = inv.dated_stem("filter_scan", "shot")
    check(d2 == f"{d1}_2", f"collision falls through to {d2}")
finally:
    probe.unlink()

print("\n=== 3. the three-way prompt is built correctly ===")
p_exist = ctools._script_choices("shot", {"filename": "Lissajous"})
check(p_exist is not None, "prompt built for an existing script")
check([o.key for o in p_exist.options] == ["1", "2", "3"], "keys are 1/2/3")
check("ALREADY EXISTS" in p_exist.context, "context warns the file exists")
check("hand-written" in p_exist.context, "context flags hand-written work")
check("Lissajous.py" in p_exist.options[0].detail or
      p_exist.options[0].detail.startswith("write a dated"),
      f"option 1 offers a dated file: {p_exist.options[0].detail}")
check(p_exist.options[2].apply is None, "option 3 (reuse) refuses the write")
check("list_scripts" in p_exist.options[2].deny_reason,
      "reuse tells the model what to do instead")

p_new = ctools._script_choices("shot", {"filename": "brand_new_thing"})
check("No 'brand_new_thing.py' exists yet" in p_new.context,
      "context says nothing exists at that name")

p_multi = ctools._script_choices("multishot", {"filename": "filter_ch2_gap_multishot"})
check("multishot_routines" in p_multi.context, "multishot prompt uses the multishot dir")

print("\n=== 4. the gate applies the answer (scripted stdin) ===")


class FakeStdin:
    def __init__(self, answers):
        self.answers = list(answers)

    def isatty(self):
        return True


class TtyBuf(io.StringIO):
    """A capture buffer that claims to be a terminal.

    _ask_choice refuses to draw a cursor UI into something that is not a tty,
    so a plain StringIO makes it decline before a key is ever read.
    """
    def isatty(self):
        return True


def run_gate(answer, args, kind="shot"):
    """Drive one confirmation with a canned answer; return the decision."""
    import builtins
    gate = ToolGate(audit_path=Path("nul"), session_id="test",
                    always_confirm=frozenset({"write_shot"}))
    real_stdin, real_input = sys.stdin, builtins.input
    sys.stdin = FakeStdin([answer])
    builtins.input = lambda *a: answer
    from superradiant_assistant import splash as _S
    _keys = iter([answer, "enter"] if answer else ["", "enter"])
    _S.read_key = lambda: next(_keys, "esc")
    buf = TtyBuf()
    real_stdout = sys.stdout
    sys.stdout = buf
    try:
        return gate.before_tool_call(
            "write_shot", args, preview="(preview)",
            choices=ctools._script_choices(kind, args)), buf.getvalue()
    finally:
        sys.stdin, builtins.input, sys.stdout = real_stdin, real_input, real_stdout


args = {"filename": "Lissajous", "source": "x", "description": "d", "reason": "r"}

dec, out = run_gate("1", dict(args))
check(not dec.denied, "'1' allows the call")
check(dec.updated_input["filename"].startswith("Lissajous_2"),
      f"'1' rewrote filename to {dec.updated_input['filename']}")

dec, out = run_gate("2", dict(args))
check(not dec.denied, "'2' allows the call")
check(dec.updated_input["filename"] == "Lissajous",
      "'2' keeps the original filename")

dec, out = run_gate("3", dict(args))
check(dec.denied, "'3' refuses the write")
check("reused" in dec.reason or "existing script reused" in dec.reason,
      "'3' explains itself to the model")

dec, out = run_gate("", dict(args))
check(dec.denied and "cancel" in dec.reason, "empty line cancels")

dec, out = run_gate("n", dict(args))
check(dec.denied, "'n' still means no, not 'new'")

# With a visible cursor, an unrecognised key is ignored rather than treated as
# an answer: the highlighted row is what enter commits, and that row is on
# screen. Silently reinterpreting a stray keypress as a choice would be worse.
from datetime import datetime as _dt
_today = f"Lissajous_{_dt.now():%Y%m%d}"      # not hardcoded: this rolls over
dec, out = run_gate("zzz", dict(args))
check(not dec.denied and dec.updated_input["filename"] == _today,
      f"garbage is ignored; enter still commits the highlighted option ({_today})")
check("> new" in out.replace("\x1b[1m", "").replace("\x1b[22m", "")
      or "new" in out, "the highlighted option was visible on screen")

print("\n=== 5. registry honours updated_input ===")
seen = {}


def fake_handler(**kw):
    seen.update(kw)
    return "wrote it"


spec = ToolSpec(name="write_shot", description="", parameters={"type": "object"},
                handler=fake_handler, allowed_agents=frozenset({"lead"}),
                choices=lambda a: ctools._script_choices("shot", a))
reg = ToolRegistry([spec], gate=ToolGate(audit_path=Path("nul"), session_id="t",
                                          always_confirm=frozenset({"write_shot"})))
import builtins
real_stdin, real_input, real_stdout = sys.stdin, builtins.input, sys.stdout
sys.stdin = FakeStdin(["1"]); builtins.input = lambda *a: "1"
from superradiant_assistant import splash as _S2
_k = iter(["1", "enter"]); _S2.read_key = lambda: next(_k, "esc")
sys.stdout = TtyBuf()
try:
    res = reg.dispatch("lead", "write_shot", dict(args))
finally:
    sys.stdin, builtins.input, sys.stdout = real_stdin, real_input, real_stdout
check(seen.get("filename", "").startswith("Lissajous_2"),
      f"handler received the rewritten filename: {seen.get('filename')}")

print("\n=== 6. multishot preview no longer shows the singleshot path ===")
prev = ctools._preview_write_analysis({
    "filename": "somecurve", "kind": "multishot", "reason": "r",
    "source": "from lyse import data\nimport matplotlib.pyplot as plt\n"
              "df = data()\nplt.figure('x')\nplt.plot([1,2],[3,4])\n",
})
check("multishot_routines" in prev, "preview points at multishot_routines")
check("kind   : multishot" in prev, "preview states the kind")
check("guard  : passed" in prev, f"guard judged with multishot rules\n{prev[:400]}")

print("\n=== 7. list_scripts tool renders ===")
from superradiant_assistant.tools.lab_tools import list_scripts
text = list_scripts("shot")
check("SHOT SEQUENCES" in text, "list_scripts('shot') renders")
check("error" in list_scripts("nonsense"), "bad kind is refused")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print(f"    - {b}")
sys.exit(1 if bad else 0)



