"""Where the team is, so a registry tool can reach it.

`ask_planner`, `ask_coder` and `ask_advisor` need the teammate *objects*, and
those are built by `build_team()` -- which runs after `build_registry()`. That
ordering is why the delegation tools were originally hung on `Agent.extra_tools`
as Gemini `FunctionDeclaration`s: at registry-build time there was nothing to
close over.

The cost of that shortcut was a silent, total failure. `make_session` passes
`tool_declarations` (which carries `extra_tools`) to the Gemini branch and
`tool_specs` (which does not) to Claude -- so from the day the second provider
landed, a lead on Claude had no teammates at all. `planner`, `coder` and
`advisor` were constructed, held sessions, showed up in `/team`, and could not be
reached. Nothing errored; the lead simply reported it had no such tool, which was
true.

So the tools become ordinary `ToolSpec`s and look the team up here at call time.
Late binding rather than a second wiring path: `plan.py` already holds the
current agent this way, and `tools/__init__.py` the session context.
"""
from __future__ import annotations
from typing import Dict

#: Filled by `build_team`. Empty until then, which is a real state -- a tool can
#: be declared before the team exists.
_TEAM: Dict[str, object] = {}

#: Names currently mid-turn. A teammate cannot delegate today, so this cannot
#: trigger; it is here because the failure it prevents is a blown stack rather
#: than a wrong answer, and the guard is three lines.
_IN_FLIGHT: set = set()


def register_team(team: Dict[str, object]) -> None:
    """Publish the team so the delegation tools can find it."""
    _TEAM.clear()
    _TEAM.update(team)
    _IN_FLIGHT.clear()


def names() -> list:
    return sorted(_TEAM)


def consult(name: str, message: str) -> str:
    """Hand `message` to teammate `name` and return its reply.

    Synchronous on purpose. Planning must finish before coding starts, and
    concurrency around hardware control would buy nothing but race conditions --
    the asynchronous channel is the mailbox, not this.
    """
    agent = _TEAM.get(name)
    if agent is None:
        return (f"error: no teammate named '{name}'"
                + (f" — the team is {', '.join(names())}" if _TEAM
                   else " — the team has not been built yet"))
    if name in _IN_FLIGHT:
        return (f"error: {name} is already working on something further up this "
                f"call — it cannot be consulted re-entrantly")

    from superradiant_assistant.orchestrator.agents import _consult
    _IN_FLIGHT.add(name)
    try:
        return _consult(agent, message)
    finally:
        _IN_FLIGHT.discard(name)
