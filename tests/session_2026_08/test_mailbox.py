"""The lead and the advisor can leave each other notes.

The gap this closes, from 2026-08-18: the operator asked the advisor why the
optimiser stopped after two shots, the advisor worked out which four angles would
discriminate, and the lead never saw them -- `/advisor` bypasses the lead by
design. The operator retyped the recommendation, and the lead still proposed four
different angles because it was reconstructing rather than reading.

Three things are load-bearing:

  * the tools are ordinary registry specs, so BOTH providers get them — the same
    fix the delegation tools then got, after the bypass left them Gemini-only;
  * reading drains the inbox, so a note is never handled twice -- but a separate
    history file keeps the record, because on this apparatus the record is the
    point;
  * a per-turn directive announces the inbox, because a model will not call
    `read_notes` on the off chance something arrived.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["AGENT_APPARATUS"] = "cesium"

from superradiant_assistant.memory import MEMORY

# Before anything reads MEMORY.root: notes must not land in the real lab store.
MEMORY.root = Path(tempfile.mkdtemp(prefix="mailbox_test_"))

from superradiant_assistant.hooks import default_gate
from superradiant_assistant.orchestrator import mailbox, plan
from superradiant_assistant.tools import build_registry

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


reg = build_registry(gate=default_gate(), creative=True)


def as_agent(name, tool, args=None):
    plan.set_current_agent(name)
    return reg.dispatch(name, tool, args or {})


print("\n=== 1. only the two agents that talk without the other present ===")
for role, want in (("lead", True), ("advisor", True),
                   ("planner", False), ("coder", False)):
    has = {"send_note", "read_notes"} <= set(reg.names_for(role))
    check(has == want,
          f"{role}: mailbox {'present' if has else 'absent'}"
          + ("" if has == want else "  -- WRONG"))
check(set(mailbox.BOXES) == {"lead", "advisor"},
      f"BOXES names exactly those two ({mailbox.BOXES})")
# The planner and the coder are only ever reached synchronously by the lead, so a
# mailbox would be a tool in their prompt that nothing ever writes to.
out = as_agent("coder", "send_note", {"to": "lead", "content": "x"})
check(out.startswith("error: agent 'coder' is not permitted"),
      f"and the whitelist refuses them at dispatch: {out[:52]}")

print("\n=== 2. advisor -> lead, the case that failed on 2026-08-18 ===")
ANGLES = ("Measure waveplate_angle = 17.5, 62.5, 107.5, 230.55 deg, plus one "
          "shot with the beam blocked.")
receipt = as_agent("advisor", "send_note", {"to": "lead", "content": ANGLES})
check("delivered to lead" in receipt, f"delivered: {receipt[:60]}")
check(mailbox.pending("lead") == 1, "one note waiting for the lead")
check(mailbox.pending("advisor") == 0, "and none for the sender")

d = mailbox.directive("lead")
check("read_notes" in d and "1 unread" in d,
      "the lead's next turn is told to read it")
check(mailbox.directive("advisor") == "",
      "an empty inbox adds no directive at all")

got = as_agent("lead", "read_notes")
check(ANGLES in got, "the lead reads the angles verbatim -- no reconstruction")
check("from advisor" in got, "and can see who sent it")
check(mailbox.pending("lead") == 0, "reading drains the inbox")
check("no notes waiting" in as_agent("lead", "read_notes"),
      "a second read says so rather than repeating the note")

print("\n=== 3. the reverse direction: the lead reports back ===")
MEASURED = ("Ran them: 17.5->0.7412 V, 62.5->0.0104 V, 107.5->0.7331 V, "
            "230.55->0.0748 V; beam blocked -> -0.0213 V.")
as_agent("lead", "send_note", {"to": "advisor", "content": MEASURED})
check(mailbox.pending("advisor") == 1, "the advisor has mail")
check(MEASURED in as_agent("advisor", "read_notes"),
      "and reads the measured numbers on its next summon")

print("\n=== 4. reading drains, history does not ===")
hist = Path(MEMORY.root) / "team" / "history.jsonl"
lines = [ln for ln in hist.read_text(encoding="utf-8").splitlines() if ln.strip()]
check(len(lines) == 2, f"both notes are still on the record ({len(lines)})")
check(any(ANGLES in ln for ln in lines) and any(MEASURED in ln for ln in lines),
      "with their content, after both inboxes were drained")
check(mailbox.pending("lead") == 0 and mailbox.pending("advisor") == 0,
      "while the inboxes themselves are empty")

print("\n=== 5. guards ===")
for args, want in (
        ({"to": "advisor", "content": "x"}, "note to yourself"),
        ({"to": "coder", "content": "x"}, "no mailbox for 'coder'"),
        ({"to": "", "content": "x"}, "no mailbox"),
        ({"to": "lead", "content": "   "}, "empty note"),
):
    out = as_agent("advisor", "send_note", args)
    check(want in out, f"{args} -> {out[:56]}")

huge = "y" * (mailbox.MAX_CHARS + 5000)
out = as_agent("advisor", "send_note", {"to": "lead", "content": huge})
check("truncated" in out, f"an oversized note is capped, and says so: {out[:70]}")
body = as_agent("lead", "read_notes")
check(len(body) < mailbox.MAX_CHARS + 500,
      f"and the delivered note is actually capped ({len(body)} chars)")

print("\n=== 6. it survives a restart, unlike a session ===")
as_agent("advisor", "send_note", {"to": "lead", "content": "still here"})
import importlib
importlib.reload(mailbox)          # a fresh process would re-import it
check(mailbox.pending("lead") == 1,
      "a note written before the exit is still waiting after re-import")
check("still here" in as_agent("lead", "read_notes"), "and reads back intact")

print("\n=== 7. provider-neutral, unlike the bypass it replaced ===")
# The bug this shape avoids: `ask_*` used to be Gemini FunctionDeclaration objects
# on `Agent.extra_tools`, and `make_session` hands that list to the Gemini branch
# only -- so on Claude the lead had no teammates at all. The delegation tools are
# now registry specs too, so the same check covers both sets.
specs = {s.name for s in reg.specs_for("lead")}          # what Claude receives
decls = {getattr(d, "name", "") for d in reg.declarations_for("lead")}  # Gemini
for name in ("send_note", "read_notes",
             "ask_planner", "ask_coder", "ask_advisor"):
    check(name in specs and name in decls,
          f"{name} is in both the Claude specs and the Gemini declarations")
from superradiant_assistant.orchestrator.agents import Agent
check(not any(f in Agent.__dataclass_fields__
              for f in ("extra_tools", "extra_dispatch")),
      "and the bypass fields no longer exist on Agent, so it cannot regress")

print("\n=== 8. the envelope (static box, no animation) ===")
import io
import superradiant_assistant.splash as S

buf, real = io.StringIO(), sys.stdout
sys.stdout = buf
try:
    S.envelope_sent("lead", chars=4231)
finally:
    sys.stdout = real
out = buf.getvalue()
check("Mail has been sent to lead" in out, f"red status line names the recipient: {out!r}")
check("4,231 chars" in out, "and the char count")
import re
strip_ansi = lambda s: re.sub(r"\x1b\[[0-9;]*m", "", s)
lines = [strip_ansi(ln) for ln in out.rstrip("\n").split("\n")]
check("email" in lines[0], "titled 'email' on the top border")
check(lines[0].startswith("┌") and lines[0].endswith("┐"),
      "the top border is a single outer frame, opened once")
check(lines[-1].startswith("└") and lines[-1].endswith("┘"),
      "closed once at the bottom -- the glyph and the text share this one frame")
check(all("Mail has been sent" not in ln or (ln.startswith("│") and ln.endswith("│"))
          for ln in lines),
      "and the status line sits inside the frame's side borders, not outside them")
check("\x1b[8A" not in out and "\x1b[2K" not in out,
      "with no cursor-up or erase sequences -- it is static, not an animation")

print("\n=== 9. the envelope prints after the panel, not during the tool call ===")
from superradiant_assistant.tools import team_tools

buf, real = io.StringIO(), sys.stdout
sys.stdout = buf
try:
    receipt = as_agent("advisor", "send_note", {"to": "lead", "content": "z" * 20})
finally:
    sys.stdout = real
check(buf.getvalue() == "", f"send_note itself prints nothing: {buf.getvalue()!r}")
check("delivered to lead" in receipt, "but the note was actually sent")

buf, real = io.StringIO(), sys.stdout
sys.stdout = buf
try:
    team_tools.flush_mail_animations()
finally:
    sys.stdout = real
check("Mail has been sent to lead" in buf.getvalue(),
      "flushing after the panel is what draws the envelope")
check(mailbox.pending("lead") == 1, "and the note itself is unaffected -- still waiting")

buf, real = io.StringIO(), sys.stdout
sys.stdout = buf
try:
    team_tools.flush_mail_animations()
finally:
    sys.stdout = real
check(buf.getvalue() == "", "a second flush with nothing queued draws nothing")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
