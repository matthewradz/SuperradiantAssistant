"""The mailbox, as tools.

Read-only in the sense that matters: a note changes no parameter and fires no
shot. So neither tool is gated -- a confirmation prompt between the lead and the
advisor would make the channel unusable for the thing it exists for.

These are plain `ToolSpec`s on purpose. The delegation tools (`ask_planner`,
`ask_coder`, `ask_advisor`) are Gemini `FunctionDeclaration` objects hung on
`Agent.extra_tools`, and `make_session` passes that list to the Gemini branch
only -- so on Claude the lead has had no teammates at all since the day the
second provider landed. A tool in the registry cannot have that bug: the
whitelist, the description and the schema are one definition and both providers
read it.
"""
from __future__ import annotations
from typing import List, Tuple

from superradiant_assistant.orchestrator import mailbox
from superradiant_assistant.tools.registry import ToolSpec

#: The two agents that talk without the other being in the room. The lead is
#: reached by the operator; the advisor is reached by `/advisor`, which bypasses
#: the lead entirely -- which is exactly why they need somewhere to leave things.
MAILBOX_AGENTS = frozenset(mailbox.BOXES)

#: `send_note` fires mid-turn, while the model is still composing the reply that
#: the operator is about to see framed in its own panel. Printing the envelope
#: right there put the delivery notice ABOVE the panel it was announcing.
#: Queuing it here and flushing after the panel is printed (see
#: `flush_mail_animations`) keeps the turn's own answer on top and the delivery
#: notice below it, matching the order the operator actually experienced them.
_PENDING_MAIL: List[Tuple[str, int]] = []


def send_note(to: str, content: str) -> str:
    from superradiant_assistant.orchestrator.plan import current_agent
    receipt = mailbox.send(to=to, frm=current_agent(), content=content)
    if not receipt.startswith("error"):
        _PENDING_MAIL.append((to, len(content or "")))
    return receipt


def flush_mail_animations() -> None:
    """Show the envelope for every note sent this turn, below the turn's panel."""
    if not _PENDING_MAIL:
        return
    from superradiant_assistant import splash as S
    pending = list(_PENDING_MAIL)
    _PENDING_MAIL.clear()
    for to, chars in pending:
        S.envelope_sent(to, chars=chars)


def read_notes() -> str:
    from superradiant_assistant.orchestrator.plan import current_agent
    return mailbox.render(mailbox.read(current_agent()))


def build_team_tool_specs() -> List[ToolSpec]:
    others = ", ".join(sorted(mailbox.BOXES))
    return [
        ToolSpec(
            name="send_note",
            description=(
                "Leave a note for a teammate who is not in the room. The note "
                "waits in their inbox and is read on their next turn — this is "
                "how the advisor's diagnosis reaches the lead, and how the lead "
                "reports back what it measured. Write it for someone who did not "
                "see your session: they have none of your tool results, so state "
                "the numbers, the angles, the filenames in full rather than "
                "referring to 'the four shots' or 'the value above'. "
                f"Mailboxes exist for: {others}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "enum": sorted(mailbox.BOXES),
                            "description": "Which teammate's inbox."},
                    "content": {
                        "type": "string",
                        "description": "The note, self-contained. Include every "
                                       "number and identifier the recipient "
                                       "needs — they cannot see your session.",
                    },
                },
                "required": ["to", "content"],
            },
            handler=send_note,
            allowed_agents=MAILBOX_AGENTS,
        ),
        ToolSpec(
            name="read_notes",
            description=(
                "Read the notes waiting in your inbox. Reading empties it, so "
                "each note is handled once — copy anything you still need into "
                "your reply or your plan. Call this when you are told notes are "
                "waiting, and before answering a request that refers to advice "
                "or numbers you have not seen yourself."
            ),
            parameters={"type": "object", "properties": {}},
            handler=read_notes,
            allowed_agents=MAILBOX_AGENTS,
        ),
    ]
