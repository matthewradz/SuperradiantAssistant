"""Switching lab must switch everything derived from it, not just the label.

The bug this locks down: `MEMORY` is a module-level singleton whose paths are
computed at construction from the data root. `select_apparatus` rebuilt CONFIG
and left the store alone, so `/memory` showed `lab ybclock` above the testbench's
files, and a session started on ybclock would have been injected the AWG's facts.
A mislabelled memory is worse than no switch.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


from superradiant_assistant import config as C
from superradiant_assistant.memory import MEMORY
from superradiant_assistant.safety import load_global_specs

START = C.APPARATUS

print("\n=== 1. the labs are discovered from config files ===")
labs = C.available_apparatus()
print(f"    {labs}")
check(C.DEFAULT_APPARATUS in labs, f"the default is listed as {C.DEFAULT_APPARATUS!r}")
check(len(labs) >= 2, "more than one lab exists to switch between")
check(C.apparatus_config_path(C.DEFAULT_APPARATUS).name == "config.json",
      "the default maps to config.json")
check(C.apparatus_config_path("ybclock").name == "config.ybclock.json",
      "a named lab maps to config.<name>.json")

print("\n=== 2. everything derived follows the switch ===")
seen = {}
for lab in labs:
    C.select_apparatus(lab)
    check(C.APPARATUS == lab, f"APPARATUS is {lab!r}")
    seen[lab] = {
        "data": C.CONFIG.historical_data_root,
        "memory": MEMORY.root,
        "globals": tuple(load_global_specs()),
        "injected": len(MEMORY.build_context_block()),
    }
    print(f"    {lab:<10} data={str(seen[lab]['data'])[-34:]}")
    print(f"    {'':<10} mem ={str(seen[lab]['memory'])[-34:]}")

a, b = labs[0], labs[1]
check(seen[a]["data"] != seen[b]["data"], "the two labs read different data roots")
check(seen[a]["memory"] != seen[b]["memory"],
      f"and different memory directories:\n"
      f"           {seen[a]['memory']}\n           {seen[b]['memory']}")
check(seen[a]["globals"] != seen[b]["globals"],
      f"and different globals: {seen[a]['globals'][:2]} vs {seen[b]['globals'][:2]}")
check(seen[a]["injected"] != seen[b]["injected"],
      f"so a different memory is injected: {seen[a]['injected']:,} vs "
      f"{seen[b]['injected']:,} chars")

print("\n=== 3. the memory root is inside the data root's parent ===")
for lab, v in seen.items():
    check(v["memory"].parent == v["data"].parent,
          f"{lab}: memory sits beside its data, not beside the code")

print("\n=== 4. the shared tier does NOT follow the switch ===")
shared = {}
for lab in labs:
    C.select_apparatus(lab)
    shared[lab] = (MEMORY.shared_root, MEMORY.labscript_file, MEMORY.operator_file)
check(len({s[0] for s in shared.values()}) == 1,
      f"one shared directory for every lab: {shared[labs[0]][0]}")
check(all(s[1].exists() for s in shared.values()),
      "LABSCRIPT.md is readable from every lab")
check(all(s[2].exists() for s in shared.values()),
      "OPERATOR.md is readable from every lab")

print("\n=== 5. the injected block names the lab it belongs to ===")
for lab in labs:
    C.select_apparatus(lab)
    block = MEMORY.build_context_block()
    # An empty layer is left out entirely, so the label is only expected when
    # that lab actually has instructions. A fresh branch has none.
    has = MEMORY.instructions_file.exists() and MEMORY.read_instructions()
    labelled = f"Standing instructions for {lab}" in block
    check(labelled == bool(has),
          f"{lab}: instructions layer labelled iff it has any "
          f"(has={bool(has)}, labelled={labelled})")
    others = [o for o in labs if o != lab]
    check(not any(f"apparatus: {o} " in block for o in others),
          f"{lab}: no other lab's apparatus memory leaked in")
    check(f"This apparatus: {lab}" in block or not MEMORY.memory_file.exists(),
          f"{lab}: the apparatus layer names this lab")

print("\n=== 6. an unknown lab is refused, not silently accepted ===")
C.select_apparatus(labs[0])
before = C.APPARATUS
got = C.select_apparatus("no_such_lab")
check(got == before and C.APPARATUS == before,
      f"stayed on {before!r} rather than switching to a lab with no config")

print("\n=== 7. the boot screen and lab screen agree ===")
from superradiant_assistant import splash as S
C.select_apparatus(labs[0])
facts = dict(S._lab_facts(labs[1]))
check(C.APPARATUS == labs[0],
      "reading another lab's facts does not switch the session to it")
check("globals" in facts and "last opened" in facts,
      f"the lab screen reports the fields it promises: {sorted(facts)}")

C.select_apparatus(START)
print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b_ in bad:
    print("    - " + b_)
sys.exit(1 if bad else 0)
