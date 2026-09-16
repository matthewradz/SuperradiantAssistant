"""An explicit, visible plan for the current request.

Without one the agent worked move-to-move: it would search, guess, write a probe
script, fire a shot, and by the time that came back it had lost the thread and
started over. Nothing recorded what it was trying to achieve or how far it had
got, and the operator could not see whether it was making progress or circling.

The plan is deliberately a plain ordered list with one active step. Its value is
not bookkeeping -- it is that writing the steps down first forces the design
decisions to happen before the hardware does anything.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

STATUS_MARK = {
    "pending": "[ ]",
    "active": "[>]",
    "done": "[x]",
    "failed": "[!]",
    "skipped": "[-]",
}


@dataclass
class Step:
    text: str
    status: str = "pending"
    note: str = ""


@dataclass
class Plan:
    goal: str = ""
    steps: List[Step] = field(default_factory=list)

    def render(self) -> str:
        if not self.steps:
            return "(no plan set)"
        lines = [f"  PLAN: {self.goal}"] if self.goal else ["  PLAN:"]
        for i, s in enumerate(self.steps, 1):
            mark = STATUS_MARK.get(s.status, "[ ]")
            lines.append(f"    {mark} {i}. {s.text}"
                         + (f"   -- {s.note}" if s.note else ""))
        done = sum(1 for s in self.steps if s.status in ("done", "skipped"))
        lines.append(f"    ({done}/{len(self.steps)} complete)")
        return "\n".join(lines)

    @property
    def active(self) -> Optional[Step]:
        return next((s for s in self.steps if s.status == "active"), None)

    def summary_for_model(self) -> str:
        if not self.steps:
            return "No plan has been set. Call set_plan before doing anything else."
        return self.render().replace("  PLAN", "PLAN")


# One plan per agent.
#
# A single shared plan meant a consulted teammate's `set_plan` silently replaced
# the lead's: the lead then found its own step numbers gone ("step 6 does not
# exist (plan has 3 steps)") and lost track of where it was.
_PLANS: Dict[str, Plan] = {}
_CURRENT_AGENT = "lead"


def set_current_agent(name: str) -> None:
    """Whose plan the plan tools operate on. Set by the agent loop."""
    global _CURRENT_AGENT
    _CURRENT_AGENT = name or "lead"


def current_agent() -> str:
    """Who is dispatching right now.

    `ToolSpec.handler(**args)` never receives the caller: `registry.dispatch`
    checks the whitelist and then drops the name. This is already set before
    every dispatch, so a tool that must know who called it -- the mailbox, which
    stamps a note's sender -- reads it here instead of the handler contract
    changing for all thirty tools.
    """
    return _CURRENT_AGENT


def get_plan(agent: Optional[str] = None) -> Plan:
    return _PLANS.setdefault(agent or _CURRENT_AGENT, Plan())


def set_plan(goal: str, steps: List[str], agent: Optional[str] = None) -> Plan:
    who = agent or _CURRENT_AGENT
    _PLANS[who] = Plan(goal=goal, steps=[Step(text=s) for s in steps])
    return _PLANS[who]


def update_step(number: int, status: str, note: str = "",
                agent: Optional[str] = None) -> Optional[str]:
    """Set one step's status. Returns an error string, or None on success."""
    plan = get_plan(agent)
    if not plan.steps:
        return "no plan has been set yet — call set_plan first"
    if not (1 <= number <= len(plan.steps)):
        return f"step {number} does not exist (plan has {len(plan.steps)} steps)"
    if status not in STATUS_MARK:
        return f"status must be one of {sorted(STATUS_MARK)}"
    step = plan.steps[number - 1]
    step.status = status
    if note:
        step.note = note
    # Only one step is active at a time; starting a new one closes the previous.
    if status == "active":
        for i, s in enumerate(plan.steps, 1):
            if i != number and s.status == "active":
                s.status = "pending"
    return None
