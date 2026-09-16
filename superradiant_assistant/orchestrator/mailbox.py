"""Notes the agents leave for each other.

The team's only channel used to be a synchronous function call: the lead asks a
teammate, blocks, gets one string back. That works when the lead starts the
conversation. It cannot express the case that actually came up on 2026-08-18 --
the operator summoned the advisor with `/advisor`, the advisor worked out which
four angles to measure, and the lead never saw them, because `/advisor` bypasses
the lead by design. The operator had to retype the recommendation, and the lead's
own guess at "the four shots" was four different angles.

So: a mailbox. One file per agent, send appends, read drains. Deliberately not a
queue, a broker, or a thread -- the send is a file append and the read is a tool
call, which is all the asynchrony this team needs. A note written now is read on
the recipient's next turn, whenever that is; the advisor's inbox may sit for an
hour between summons and that is fine.

Two properties worth stating because they were not free:

* **It survives a restart.** The files live beside the apparatus's memory, so a
  note written before an exit is still there afterwards. Sessions do not survive
  a restart; notes do.
* **It is provider-neutral.** The tools are ordinary `ToolSpec`s in the registry,
  not declarations bolted onto one agent. The delegation tools were bolted on
  (`Agent.extra_tools`), and `make_session` hands those to Gemini only -- so on
  Claude the lead has had no `ask_planner`, `ask_coder` or `ask_advisor` at all.
  Anything that lives outside the registry has to be re-plumbed per provider, and
  that is exactly the bug it produces.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

#: A note is a message, not a document. Long enough for a full diagnosis with a
#: table in it, short enough that a runaway loop cannot fill the disk.
MAX_CHARS = 8000

#: Who may hold an inbox. Not every agent needs one: the planner and the coder
#: are only ever reached synchronously by the lead, and giving them a mailbox
#: would add a tool to their prompt that nothing writes to.
BOXES = ("lead", "advisor")


def _root() -> Path:
    from superradiant_assistant.memory import MEMORY
    return Path(MEMORY.root) / "team"


def _inbox(agent: str) -> Path:
    return _root() / f"{agent}.jsonl"


def _history() -> Path:
    """Every note ever sent. The inbox drains; this does not.

    Draining is what stops a note being processed twice, but it also destroys the
    record -- and on this apparatus the record is the point. A note that changed
    what got measured should still be readable next week.
    """
    return _root() / "history.jsonl"


def send(to: str, frm: str, content: str) -> str:
    """Append one note to `to`'s inbox. Returns a one-line receipt."""
    to, frm = (to or "").strip(), (frm or "?").strip()
    if to not in BOXES:
        return (f"error: no mailbox for '{to}' — the agents with one are "
                f"{', '.join(BOXES)}")
    if to == frm:
        return "error: a note to yourself is a note nobody reads"
    text = str(content or "").strip()
    if not text:
        return "error: an empty note says nothing"
    truncated = len(text) > MAX_CHARS
    if truncated:
        text = text[:MAX_CHARS]

    note = {"to": to, "from": frm, "content": text,
            "at": datetime.now().isoformat(timespec="seconds")}
    line = json.dumps(note, ensure_ascii=False)
    try:
        _root().mkdir(parents=True, exist_ok=True)
        for path in (_inbox(to), _history()):
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError as e:
        return f"error: could not deliver the note: {e}"

    return (f"note delivered to {to} ({len(text):,} chars"
            + (f", truncated from {len(str(content))}" if truncated else "")
            + f") — it will be read on {to}'s next turn")


def read(agent: str) -> List[Dict[str, str]]:
    """Drain `agent`'s inbox. Reading clears it, so nothing is handled twice."""
    path = _inbox(agent)
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return []
    notes = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            notes.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if notes:
        try:
            path.write_text("", encoding="utf-8")
        except OSError:
            # Better to hand the notes over twice than to lose them.
            pass
    return notes


def pending(agent: str) -> int:
    """How many notes are waiting, without draining them."""
    try:
        raw = _inbox(agent).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return 0
    return sum(1 for ln in raw.splitlines() if ln.strip())


def render(notes: List[Dict[str, str]]) -> str:
    """The notes as a tool result."""
    if not notes:
        return "no notes waiting"
    out = [f"{len(notes)} note(s), oldest first — the inbox is now empty"]
    for n in notes:
        out.append(f"\n--- from {n.get('from', '?')} at {n.get('at', '?')} ---\n"
                   + n.get("content", ""))
    return "\n".join(out)


def directive(agent: str) -> str:
    """A per-turn nudge, appended to what the model sees but not remembered.

    Without it the mailbox is write-only in practice: a model does not call
    `read_notes` on the off chance that something arrived. Same mechanism as
    `language_directive` -- shown to the model for this turn, not recorded as
    part of the operator's request.
    """
    n = pending(agent)
    if not n:
        return ""
    return (f"\n\n[{n} unread note(s) in your inbox. Call `read_notes` before "
            f"answering — a teammate left something you have not seen.]")
