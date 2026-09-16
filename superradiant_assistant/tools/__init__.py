"""Tool layer: session context, registry construction.

Tools need access to per-run objects (the LLM client, the knowledge base, the
executor, the goal-mode switch) that they can't take as arguments — those are
decided by the operator, not the model. The session context holds them.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional

_KNOWLEDGE: Optional[List[Any]] = None
_LAB_API = None


@dataclass
class SessionContext:
    llm_client: Any = None
    knowledge: Any = None
    executor: Any = None
    goal_mode: bool = True
    live: bool = False
    # Creative mode: the model may write shots, analyses and new parameters,
    # each behind an operator confirmation it cannot skip.
    creative: bool = False


_SESSION = SessionContext()


def get_session_context() -> SessionContext:
    return _SESSION


def configure_session(**kwargs) -> SessionContext:
    for k, v in kwargs.items():
        if not hasattr(_SESSION, k):
            raise AttributeError(f"SessionContext has no field '{k}'")
        setattr(_SESSION, k, v)
    return _SESSION


def get_knowledge():
    """Knowledge base, loaded once on first use."""
    global _KNOWLEDGE
    if _SESSION.knowledge is not None:
        return _SESSION.knowledge
    if _KNOWLEDGE is None:
        from superradiant_assistant.knowledge.loader import load_knowledge_base
        _KNOWLEDGE = load_knowledge_base()
    return _KNOWLEDGE


def get_lab_api():
    global _LAB_API
    if _LAB_API is None:
        from superradiant_assistant.labscript_api import LabscriptAPI
        _LAB_API = LabscriptAPI()
    return _LAB_API


def build_registry(gate=None, creative: bool = False):
    """The tool set. Creative mode adds the tools that write files.

    Those tools are absent rather than merely refused when creative mode is off,
    so the model cannot name them at all — the same reason parameter enums are
    built from config.json instead of being validated after the fact.
    """
    from superradiant_assistant.tools.advisor_tools import build_advisor_tool_specs
    from superradiant_assistant.tools.delegation_tools import (
        build_delegation_tool_specs,
    )
    from superradiant_assistant.tools.lab_tools import build_tool_specs
    from superradiant_assistant.tools.registry import ToolRegistry
    from superradiant_assistant.tools.team_tools import build_team_tool_specs

    # The advisor's own tools are read-only and always present: creative mode
    # gates writes, and reading the notebook is not one. The mailbox likewise --
    # a note changes nothing on the apparatus. Delegation is here rather than on
    # `Agent.extra_tools` so that both providers get it from one definition; the
    # bypass it replaced reached Gemini only.
    specs = (build_tool_specs() + build_advisor_tool_specs()
             + build_team_tool_specs() + build_delegation_tool_specs())
    if creative:
        from superradiant_assistant.creative.tools import build_creative_tool_specs
        specs = specs + build_creative_tool_specs()
    return ToolRegistry(specs, gate=gate)
