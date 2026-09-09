"""The coder can read many shots in one call instead of one call per shot.

On 2026-08-17 the lead delegated an optimize run to the coder and asked it to
report the numbers. The coder had `inspect_shot` but not `read_shot_results`, so
it read fifteen shots one at a time -- twelve model round trips, ~123k tokens of
fixed overhead, and it still tabulated only ten of them and left out the best.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from superradiant_assistant.tools import build_registry
from superradiant_assistant.hooks import default_gate
from superradiant_assistant.tools.lab_tools import read_shot_results

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


reg = build_registry(gate=default_gate(), creative=True)

print("\n=== 1. every agent that reports numbers can read them in bulk ===")
for role in ("lead", "coder", "planner"):
    names = reg.names_for(role)
    print(f"    {role:8} read_shot_results={'read_shot_results' in names}  "
          f"inspect_shot={'inspect_shot' in names}")
    check("read_shot_results" in names, f"{role} has the bulk reader")

print("\n=== 2. the two tools stay distinct ===")
spec = {s.name: s for s in reg.specs_for("coder")}
check("inspect_shot" in spec, "the coder keeps inspect_shot")
check("LAYOUT" in spec["read_shot_results"].description
      or "layout" in spec["read_shot_results"].description,
      "the bulk reader's description says what inspect_shot is for instead")
check("one call" in spec["read_shot_results"].description,
      "and that one call covers many shots")
check("laid out" in spec["inspect_shot"].description,
      "inspect_shot still describes itself as the layout tool")

print("\n=== 3. the coder is actually allowed to call it ===")
out = reg.dispatch("coder", "read_shot_results", {"limit": 4,
                                                   "metrics": ["CHAN1_mean"]})
print(f"    {out.splitlines()[0][:90]}")
check(not out.startswith("error:") and not out.startswith("refused:"),
      "dispatch as the coder is permitted")
check("read 4 shots" in out or "read 0 shots" in out,
      "and it returns the bulk summary line")

print("\n=== 4. one call carries what a dozen used to ===")
# Compared per shot, not in absolute bytes: which lab is open decides how many
# metrics a shot has, so the absolute sizes move with the dataset. What does not
# move is that one call covers N shots and the other covers one.
N = 12
bulk = read_shot_results(limit=N, metrics=["CHAN1_mean"])
single = reg.dispatch("coder", "inspect_shot", {"shot": "latest"})
n_read = int(re.search(r"read (\d+) shots", bulk).group(1))
print(f"    read_shot_results({N}): {len(bulk):>6} chars, 1 call, {n_read} shots"
      f"  -> {len(bulk) / max(n_read, 1):.0f} chars/shot")
print(f"    inspect_shot         : {len(single):>6} chars, 1 call, 1 shot"
      f"  -> {len(single)} chars/shot")
check(n_read > 1, f"the bulk call really read several shots ({n_read})")
check(len(bulk) / max(n_read, 1) < len(single) / 2,
      "bulk costs less than half as many characters per shot")
check(len(bulk) < len(single) * n_read,
      f"and one call is smaller than {n_read} inspect_shot calls put together")

print("\n=== 5. the coder's rules point at the right tool ===")
from superradiant_assistant.orchestrator.agents import _CREATIVE_RULES
check("read_shot_results" in _CREATIVE_RULES,
      "the creative rules name the bulk reader")
check("round trip per shot" in _CREATIVE_RULES,
      "and say what the per-shot alternative costs")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
