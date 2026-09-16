"""`remember` must actually write, survive compaction, and be confirmed."""
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


from superradiant_assistant.memory.store import MemoryStore, NOTE_TARGETS, day_key

RULE = ("start frequency sweeps at 3 kHz, not 100 Hz — below ~2.1 kHz the "
        "600 us scope window holds under one cycle and the amplitude fit fails")

print("\n=== 1. it writes, to the right file and section ===")
tmp = Path(tempfile.mkdtemp(prefix="mem_"))
M = MemoryStore(root=tmp)
out = M.remember(RULE, "instruction")
print(f"    {out}")
check("INSTRUCTIONS.md" in out,
      f"an instruction goes to the apparatus's INSTRUCTIONS.md: {out}")
user = M.read_instructions()
check("## Standing instructions" in user, "under a fixed heading")
check(RULE in user, "the fact is there verbatim")
check(day_key() in user, f"and dated {day_key()}")

out2 = M.remember("the DG1022Z stops at 25 MHz", "apparatus")
check("MEMORY.md" in out2, f"an apparatus fact goes to MEMORY.md: {out2}")
check("## Operator notes" in M.read_memory(), "under its own heading")

print("\n=== 2. it does not write the same thing twice ===")
again = M.remember(RULE, "instruction")
print(f"    {again}")
check("already remembered" in again, "a repeat is reported, not appended")
check(M.read_instructions().count("3 kHz") == 1, "still one copy")
# wording differs, meaning does not
loose = M.remember("Start frequency sweeps at 3 kHz, not 100 Hz: below ~2.1 kHz "
                   "the 600 us scope window holds under one cycle and the "
                   "amplitude fit fails.", "instruction")
check("already remembered" in loose,
      f"punctuation and case do not make it a new note: {loose[:60]}")

print("\n=== 3. a second, different instruction appends ===")
M.remember("never drive amplitude above 1.0 V on this bench", "instruction")
user = M.read_instructions()
check(user.count("\n- ") >= 2, f"two bullets now ({user.count(chr(10) + '- ')})")
check(RULE in user, "the first one survived")
sec = user.split("## Standing instructions")[1]
check("1.0 V" in sec.split("## ")[0], "the new one is inside the section")

print("\n=== 4. it appends into an existing file without eating it ===")
q = Path(tempfile.mkdtemp(prefix="mem2_"))
M2 = MemoryStore(root=q)
M2._write(M2.instructions_file,
          "# Operator\n\n## How he works\n\n- prefers English replies\n\n"
          "## Something else\n\n- keep this too\n")
M2.remember("always start sweeps at 3 kHz", "instruction")
t = M2.read_instructions()
print("    " + t.replace("\n", "\n    "))
check("prefers English replies" in t and "keep this too" in t,
      "existing sections untouched")
check("## Standing instructions" in t, "the new section was created")
check(t.index("## Something else") < t.index("## Standing instructions"),
      "and appended at the end, not spliced into the middle")

print("\n=== 5. inserting into a section that is not last ===")
M2._write(M2.instructions_file,
          "## Standing instructions\n\n- first rule\n\n## Later\n\n- unrelated\n")
M2.remember("second rule", "instruction")
t = M2.read_instructions()
first = t.split("## Later")[0]
check("first rule" in first and "second rule" in first,
      f"both rules are inside the section:\n      {first.strip()}")
check("unrelated" in t, "the following section is intact")
check(t.index("second rule") < t.index("## Later"),
      "the new bullet went before the next heading")

print("\n=== 6. the tool refuses a bad kind, and empty input ===")
from superradiant_assistant.tools.lab_tools import remember as tool_remember
import superradiant_assistant.memory as mem_pkg
mem_pkg.MEMORY = M
r = tool_remember("something", kind="whatever")
check(r.startswith("error"), f"bad kind refused: {r[:70]}")
check("instruction" in r and "apparatus" in r, "and the valid kinds are named")
check("nothing to remember" in M.remember("   ", "instruction"),
      "an empty fact is refused")

print("\n=== 7. the confirmation preview shows what will be written ===")
from superradiant_assistant.tools.lab_tools import _preview_remember
p = _preview_remember({"fact": RULE, "kind": "instruction", "reason": "operator asked"})
print("    " + p.replace("\n", "\n    "))
check("INSTRUCTIONS.md" in p, "names the file")
check("Standing instructions" in p, "names the section")
check(RULE in p, "shows the exact line")
check("operator asked" in p, "shows the reason")

print("\n=== 8. it is registered, confirmed and audited ===")
from superradiant_assistant.tools import build_registry
from superradiant_assistant.hooks import default_gate
gate = default_gate()
reg = build_registry(gate=gate, creative=False)
names = {d.name for d in reg.declarations_for("lead")}
check("remember" in names, "available to the lead outside creative mode")
check("remember" in {d.name for d in reg.declarations_for("coder")},
      "and to the coder")
check("remember" in gate.always_confirm,
      "always_confirm, so --yes cannot wave it through")
check("remember" in gate.audit, "and audited")
decl = [d for d in reg.declarations_for("lead") if d.name == "remember"][0]
props = set(decl.parameters.properties or {})
check(props == {"fact", "kind", "reason"}, f"schema: {sorted(props)}")

print("\n=== 9. the curator is told to preserve it ===")
from superradiant_assistant.memory.store import _COMPACT_SYSTEM
check("## Operator notes" in _COMPACT_SYSTEM, "MEMORY.md section named")
# Standing instructions are safer than VERBATIM now: INSTRUCTIONS.md is never
# handed to the curator and never written by it, so it cannot be paraphrased away.
import inspect
csrc = inspect.getsource(MemoryStore.compact)
check("instructions_file" not in csrc,
      "compaction never writes INSTRUCTIONS.md")
check("read_instructions" not in csrc,
      "and never even shows it to the curator")
check("updated_labscript" in _COMPACT_SYSTEM,
      "the shared labscript tier has its own block")
check("Do not move a fact between these three files" in _COMPACT_SYSTEM,
      "and the curator is forbidden from moving facts between tiers")
check(_COMPACT_SYSTEM.count("VERBATIM") >= 1,
      "the operator-notes section is still marked verbatim")

print("\n=== 10. the agent is told the tool exists ===")
from superradiant_assistant.orchestrator.agents import _SHARED_RULES
# Normalised: the rules are hard-wrapped, so a phrase can straddle a newline.
rules = " ".join(_SHARED_RULES.split())
check("`remember`" in rules, "named in the shared rules")
check("records nothing" in rules,
      "and warned that saying so is not doing so")
check("the tool wins" in rules, "stale memory loses to a tool result")
check("actionable next session" in rules, "and told how to word it")

shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(q, ignore_errors=True)
print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
