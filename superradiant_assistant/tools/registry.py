"""Tool registry: declaration, per-agent whitelisting, and gated dispatch.

Tools are declared with plain dicts so this module (and the tools themselves) can
be imported and unit-tested without the Gemini SDK present. Conversion to
google.genai FunctionDeclaration happens only when a chat session is built.

Dispatch always runs through the ToolGate, so confirmation and audit apply no
matter which agent made the call.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from superradiant_assistant.hooks import ChoicePrompt, ToolGate, default_gate


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]              # JSON-schema-style object schema
    handler: Callable[..., str]
    allowed_agents: frozenset               # which agent roles may call this
    preview: Optional[Callable[[Dict[str, Any]], str]] = None  # confirmation text
    #: Turns the y/N confirmation into a multi-way question, and may rewrite the
    #: arguments with the operator's answer. Return None to keep the plain y/N.
    choices: Optional[Callable[[Dict[str, Any]], Optional[ChoicePrompt]]] = None


class ToolRegistry:
    def __init__(self, specs: List[ToolSpec], gate: Optional[ToolGate] = None):
        self._specs = {s.name: s for s in specs}
        self.gate = gate or default_gate()

    def names_for(self, agent: str) -> List[str]:
        return [n for n, s in self._specs.items() if agent in s.allowed_agents]

    def specs_for(self, agent: str) -> List[ToolSpec]:
        """The tools this agent may call, provider-neutral.

        `ToolSpec.parameters` is already JSON Schema, which is what Anthropic's
        `input_schema` wants verbatim and what `_to_genai_schema` converts for
        Gemini. So a second provider needs no second definition of any tool: the
        whitelist, the descriptions and the schemas are shared, and only the
        wire format differs.
        """
        return [self._specs[n] for n in self.names_for(agent)]

    def declarations_for(self, agent: str) -> List[Any]:
        """google.genai FunctionDeclaration objects for the tools this agent may use."""
        from google.genai import types
        decls = []
        for name in self.names_for(agent):
            spec = self._specs[name]
            decls.append(types.FunctionDeclaration(
                name=spec.name,
                description=spec.description,
                parameters=_to_genai_schema(spec.parameters),
            ))
        return decls

    def dispatch(self, agent: str, name: str, args: Dict[str, Any]) -> str:
        """Run a tool on behalf of `agent`, enforcing the whitelist and the gate."""
        spec = self._specs.get(name)
        if spec is None:
            return f"error: unknown tool '{name}'"
        if agent not in spec.allowed_agents:
            # A hard boundary, not a suggestion: the answer agent can never reach hardware.
            return f"error: agent '{agent}' is not permitted to call '{name}'"

        preview = spec.preview(args) if spec.preview else ""
        try:
            choices = spec.choices(args) if spec.choices else None
        except Exception as e:
            # A broken choice builder must not become a silent plain y/N: the
            # whole point of the question is that "yes" was ambiguous.
            return (f"error: could not build the confirmation choices for "
                    f"{name}: {type(e).__name__}: {e}")

        decision = self.gate.before_tool_call(name, args, preview=preview,
                                              choices=choices)
        if decision.denied:
            self.gate.after_tool_call(name, args, result=None, error=decision.reason)
            return f"refused: {decision.reason}"

        # The operator's answer can rewrite the arguments -- choosing "new file"
        # changes the filename the handler writes to. Audit what actually ran.
        args = decision.updated_input if decision.updated_input is not None else args

        try:
            result = spec.handler(**args)
        except Exception as e:
            self.gate.after_tool_call(name, args, result=None, error=str(e))
            return f"error: {type(e).__name__}: {e}"

        self.gate.after_tool_call(name, args, result=result)
        return result

    def dispatcher_for(self, agent: str) -> Callable[[str, Dict[str, Any]], str]:
        """A dispatch callback bound to one agent, for GeminiAgentSession."""
        def _dispatch(name: str, args: Dict[str, Any]) -> str:
            return self.dispatch(agent, name, args)
        return _dispatch


_TYPE_MAP = {
    "string": "STRING", "number": "NUMBER", "integer": "INTEGER",
    "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT",
}


def _to_genai_schema(node: Dict[str, Any]):
    from google.genai import types

    kwargs: Dict[str, Any] = {"type": _TYPE_MAP[node["type"]]}
    if "description" in node:
        kwargs["description"] = node["description"]
    if "enum" in node:
        kwargs["enum"] = [str(v) for v in node["enum"]]
    if node["type"] == "object":
        kwargs["properties"] = {
            k: _to_genai_schema(v) for k, v in node.get("properties", {}).items()
        }
        if node.get("required"):
            kwargs["required"] = list(node["required"])
    if node["type"] == "array" and "items" in node:
        kwargs["items"] = _to_genai_schema(node["items"])
    return types.Schema(**kwargs)
