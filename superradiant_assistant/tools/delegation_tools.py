"""Consulting a teammate, as registry tools.

These were `google.genai.types.FunctionDeclaration` objects hung on
`Agent.extra_tools`, and `make_session` hands that list to the Gemini branch
only. A lead on Claude therefore had no `ask_planner`, no `ask_coder` and no
`ask_advisor` — the whole team structure was inert, silently, from the day the
second provider landed.

As `ToolSpec`s the whitelist, the description and the schema are one definition,
and `specs_for` (Claude) and `declarations_for` (Gemini) are two views of it. The
bug cannot recur without deleting a tool outright.

The handlers bind late, via `orchestrator.delegate`, because the team is built
after the registry.

Not gated: a consult changes nothing on the apparatus, and whatever the teammate
goes on to do is gated at its own dispatch. A confirmation here would prompt the
operator twice for one action.
"""
from __future__ import annotations
from typing import List

from superradiant_assistant.orchestrator import delegate
from superradiant_assistant.tools.registry import ToolSpec

#: Only the lead. The plan is how the operator follows the work and the lead is
#: the only agent they talk to, so a chain of consults would produce reasoning
#: nobody sees and a bill nobody predicted.
DELEGATORS = frozenset({"lead"})


def _one_arg(name: str, description: str) -> dict:
    return {
        "type": "object",
        "properties": {name: {"type": "string", "description": description}},
        "required": [name],
    }


def build_delegation_tool_specs() -> List[ToolSpec]:
    return [
        ToolSpec(
            name="ask_planner",
            description=(
                "Consult the planner on what to do next and why. It reads results "
                "and parameters but cannot run anything."
            ),
            parameters=_one_arg(
                "question",
                "The decision to make, with the context needed to make it."),
            handler=lambda question: delegate.consult("planner", question),
            allowed_agents=DELEGATORS,
        ),
        ToolSpec(
            name="ask_coder",
            description=(
                "Hand a decided step to the coder to execute. It owns "
                "run_optimization and run_sweep."
            ),
            parameters=_one_arg(
                "task",
                "What to run, including the target, bounds and sequence."),
            handler=lambda task: delegate.consult("coder", task),
            allowed_agents=DELEGATORS,
        ),
        ToolSpec(
            name="ask_advisor",
            description=(
                "Consult the advisor: a physicist who diagnoses results. Use it "
                "when a measurement is surprising, disagrees with what you "
                "predicted, contradicts an earlier run, or looks like it might be "
                "an instrument artifact rather than physics — it will rank the "
                "explanations and name the cheapest measurement that tells them "
                "apart. It reads shots, scripts, the notebook, past reports, the "
                "lab knowledge base and the web, and cannot change anything. It "
                "is the strongest model and it is expensive: do not call it to "
                "look up a number, and do not call it to confirm something you "
                "already know. For a question you can answer from the data in "
                "front of you, answer it."
            ),
            parameters=_one_arg(
                "question",
                "The result to explain, with the numbers you have, what you "
                "expected, and how it was measured. State your own reading if you "
                "have one — it will be judged, not assumed."),
            handler=lambda question: delegate.consult("advisor", question),
            allowed_agents=DELEGATORS,
        ),
    ]
