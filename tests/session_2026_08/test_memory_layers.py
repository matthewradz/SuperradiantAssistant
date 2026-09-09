"""Each role carries only the memory it uses.

The whole block is 18.7k characters, it is 68% of a system prompt that is resent
on every tool call, and all four agents used to carry all of it -- while the four
role prompts differed from one another by 2%.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["AGENT_APPARATUS"] = "cesium"

from superradiant_assistant.memory import MEMORY
from superradiant_assistant.memory.store import CONTEXT_LAYERS
from superradiant_assistant.orchestrator.agents import build_team, _MEMORY_LAYERS
from superradiant_assistant.tools import build_registry
from superradiant_assistant.hooks import default_gate
from superradiant_assistant.llm.cost_tracker import CostTracker

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


print("\n=== 1. layers can be selected, and default to all ===")
full = MEMORY.build_context_block()
same = MEMORY.build_context_block(CONTEXT_LAYERS)
check(full == same, "no argument means every layer")
check(MEMORY.build_context_block(()) == "", "an empty selection yields nothing")
one = MEMORY.build_context_block(("instructions",))
check(0 < len(one) < len(full), f"one layer is a fraction of all ({len(one)} of {len(full)})")
check("Standing instructions" in one, "and it is the layer that was asked for")
check("Operator profile" not in one, "with nothing else along for the ride")

print("\n=== 2. an unknown layer name is simply not emitted ===")
check(MEMORY.build_context_block(("nonsense",)) == "",
      "a name that is not a layer yields nothing rather than everything")

print("\n=== 3. what each role actually carries ===")
reg = build_registry(gate=default_gate(), creative=True)
team = build_team(reg, model="claude-opus-5", cost_tracker=CostTracker())
sizes = {}
for role, agent in team.items():
    s = agent.build_system_instruction()
    sizes[role] = len(s)
    layers = [name for name, marker in (
        ("objective", "OBJECTIVE.md"),
        ("labscript", "How labscript, BLACS and lyse behave"),
        ("memory", "This apparatus:"),
        ("operator", "Operator profile"),
        ("instructions", "Standing instructions"),
        ("notebook", "Today's page from the lab notebook"),
    ) if marker in s]
    print(f"    {role:8} {len(s):6d} chars   {layers}")
    sizes[role + "_layers"] = layers

# Every layer that has anything in it -- not `CONTEXT_LAYERS` itself. An empty
# layer emits nothing, so before the first session of the day there is no notebook
# page and the literal comparison failed at midnight for reasons having nothing to
# do with what this test is about.
present = [k for k in CONTEXT_LAYERS if MEMORY.build_context_block((k,))]
check(sizes["lead_layers"] == present,
      f"the lead keeps everything that exists ({present})")
check("labscript" in sizes["coder_layers"],
      "the coder keeps LABSCRIPT.md -- savefig, /data/traces, the analysis traps")
check("operator" not in sizes["coder_layers"], "but not the operator profile")
check("notebook" not in sizes["coder_layers"], "and not the day's narrative")
check(sizes["planner_layers"] == ["instructions"],
      "the planner keeps only the standing instructions")
check("answer" not in team,
      "there is no answer agent: nothing could ever dispatch to it")

print("\n=== 4. the saving, in characters resent on every tool call ===")
for role in ("coder", "planner"):
    saved = sizes["lead"] - sizes[role]
    print(f"    {role:8} {saved:6d} chars lighter than the lead"
          f"  (~{saved // 4} tokens per call)")
    check(saved > 8000, f"{role} sheds more than 8k characters")
check(set(team) == {"lead", "planner", "coder", "advisor"},
      f"the team is the four reachable roles ({sorted(team)})")

print("\n=== 5. safety-relevant layers are never dropped ===")
# The standing instructions are rules the operator set. Every role keeps them.
for role, agent in team.items():
    check("Standing instructions" in agent.build_system_instruction(),
          f"{role} still carries the operator's standing instructions")

print("\n=== 6. a role not in the map gets everything ===")
check("nobody" not in _MEMORY_LAYERS, "the map only names roles that are trimmed")
check(_MEMORY_LAYERS.get("lead") is None,
      "the lead is absent from it, so it falls through to all layers")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
