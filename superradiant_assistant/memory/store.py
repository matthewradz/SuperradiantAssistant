"""Cross-session memory.

Distinct from `state.STATE`, which holds progress within one run and is discarded
when the process exits. Three layers are injected into the system prompt each
turn, backed by a raw log that never enters context:

  MEMORY.md        core memory — durable facts: working parameter values,
                   calibration results, past incidents
  USER.md          operator profile — who runs the experiment and how they work
  YYYY-MM-DD.md    episodic memory — what happened today
  history.jsonl    raw append-only log of every turn (not injected)

The three layers are produced by compaction: when a session grows long, its older
turns are distilled into these files and dropped from context. Memory is re-read
from disk on every turn rather than cached at startup, so editing MEMORY.md by
hand takes effect on the next turn.
"""
from __future__ import annotations
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from superradiant_assistant.config import CONFIG

#: Compaction is triggered by how much context has accumulated, not by how many
#: times the operator typed. Counting operator messages is why memory was never
#: written at all: across 31 sessions and 468 logged turns the busiest single
#: agent reached 14 of the 18 required, so `compact()` never ran once and
#: MEMORY.md was never created. One request can be 20 tool calls and 60k
#: characters of context while still counting as a single "turn".
#: The layers `build_context_block` can emit, in the order they are written.
#: `objective` leads: an agent asked to judge a result should know what the
#: apparatus is FOR before it reads a single number.
CONTEXT_LAYERS = ("objective", "labscript", "memory", "operator", "instructions",
                  "notebook")

COMPACT_AFTER_CHARS = 60_000
COMPACT_AFTER_TURNS = 18          # secondary trigger, for long slow chat sessions
KEEP_RECENT_TURNS = 10

#: MEMORY.md's own size budget -- distinct from COMPACT_AFTER_CHARS above, which
#: triggers folding a SESSION into memory, not the size of memory itself. Picked
#: relative to the file's actual size on this apparatus (~7.5k chars): room to
#: roughly double before a shrink pass is forced, while keeping the file from
#: becoming the dominant cost in the ~12-19k-char context block it is one layer
#: of. `remember()` never checks this -- it is a fast, synchronous file append
#: with no LLM call, and staying that way was the point of adding it. Only
#: `compact()` enforces the budget, because that is the one path that already
#: calls an LLM to rewrite the whole file, and rewriting is what shrinking needs.
MEMORY_FILE_MAX_CHARS = 12_000

_COMPACT_SYSTEM = """You are the memory curator for a physics lab assistant.

You are given older turns from a session that are about to be dropped from the
model's context. Distil what is worth keeping and return exactly three blocks:

<episode>
What happened, as a short bullet list — one entry in the day's log. Keep
concrete numbers: measured values, parameter changes, fit results. Say what the
operator was trying to find out, not only what was run; a log line that records
a sweep but not its question is unreadable a week later.

Do not repeat the date, the entry is timestamped already. No LaTeX — write
`fc`, `~14.6 kHz`, `-3 dB`, never `$f_c$` or `\\approx`. Drop conversational
filler.
</episode>

<updated_memory>
The full revised contents of MEMORY.md — facts about THIS APPARATUS ONLY. Its
hardware, its wiring, its globals, how its drivers misbehave, and the CURRENT
understanding of what has been measured on it. Start from the existing memory
and fold in anything durable: working parameter values, calibration results,
things that went wrong and why. Do not let it grow without bound — replace
superseded facts rather than appending to them. Omit anything true only of
this session.

This is not a day-by-day log. `<episode>` above already exists for "what
happened, when" — that is where a dated, per-run entry belongs, not here.
A fact in MEMORY.md describes the apparatus's state NOW, not the history of
how that state was discovered:

- Write ONE current line per fact, not one line per time it was measured.
  "theta0 ~17.5 deg, period 90 deg (half-wave plate) — stable across
  sessions" is a memory fact. "2026-08-17 seq 0069: theta0=17.68; 2026-08-18
  seq 0003: theta0=17.51" is two notebook entries wearing a memory-file
  disguise — put that in `<episode>` instead, or leave it where it already
  is if today's turns did not touch it.
- Do not stamp an entry with a specific date unless the date changes what the
  fact MEANS (a firmware update, a rewiring, a component swap — "as of the
  2026-08-10 driver update" is load-bearing). A date that only records WHEN a
  number was measured is provenance, not the fact itself, and provenance
  belongs in the notebook and in reports, which can be searched or read in
  full when it is actually needed — not carried in every system prompt.
- Do not carry shot IDs, sequence numbers, or a per-run value into MEMORY.md.
  If a quantity is still unsettled or drifts between sessions, say that in
  ONE line — what is unstable, and what to do about it (e.g. "read the
  multishot fit's own A_start/A_end fresh each time; do not trust a
  remembered absolute value") — not the value from every attempt that found
  it unstable.

Reproduce the `## Operator notes` section VERBATIM if it exists. Those lines were
written because the operator asked for them; you did not see that request and
cannot judge whether they still matter.
</updated_memory>

<updated_labscript>
The full revised contents of LABSCRIPT.md, or UNCHANGED.

This file holds what is true of labscript, BLACS, lyse and runmanager themselves —
so it is read in every lab, on every apparatus. Add to it only a fact that would
still be true with completely different hardware: that a shot file exists as soon
as runmanager compiles it, that lyse's `data()` is a two-level MultiIndex, that its
dataframe is cumulative across sweeps, that files under labscriptlib must be pure
ASCII. These are the lessons that cost hours and would otherwise be relearned by
repeating the mistake in the next lab.

Never put a device name, a channel, a measured value or a frequency in here.
</updated_labscript>

<updated_user>
The full revised contents of OPERATOR.md, or UNCHANGED. Who the operator is and
how they work — shared across every apparatus, because it is the same person.
How they like to be answered, what they have been burned by, what they always
state explicitly.

Nothing apparatus-specific goes here. A rule like "start filter sweeps at 3 kHz"
is about one bench and belongs in that apparatus's standing instructions, not in
the profile that follows the operator into every lab.
</updated_user>

Do not move a fact between these three files. Deciding that a measured cutoff is
"really" a labscript fact, or that a driver quirk belongs in the operator profile,
corrupts the boundary that makes switching apparatus safe. If a fact is in the
wrong file, leave it there and say so in the episode.

An instruction the operator gave about how to run the experiment — a range to
start from, a routine to prefer, something never to do again — is not a "parameter
value" or a "calibration result". Without being named, it fell through every
category and was lost, and the operator had to give it three times.
"""


_SHRINK_SYSTEM = """You are the memory curator for a physics lab assistant, on a
second pass. MEMORY.md is over its size budget and has to come down, not grow.

Return ONLY the full revised contents of MEMORY.md, under the character limit
given in the prompt. To get there:

- Do not drop a distinct fact. Every hardware quirk, calibration result and
  working parameter value that is still true must survive somewhere.
- A running log of past mistakes (values later found wrong, artefacts later
  explained) does not need every entry kept separately once the CURRENT
  correct value is established. Fold it into one line: the current value, and
  one clause on why earlier readings were wrong -- not the full history of
  each wrong reading.
- Tighten prose into denser bullets. Say the number once, not once per
  sentence that mentions it.
- Cut anything that duplicates a fact stated elsewhere in the file.

If the file truly cannot fit the limit without losing safety- or
correctness-relevant information, get as close as you can and end with one
line starting exactly with `STILL OVER BUDGET:` naming what could not be cut
and why -- do not silently return something still over budget without saying so.
"""


def _shrink_memory(text: str, llm_client, limit: int, max_passes: int = 2) -> str:
    """Ask the curator to cut MEMORY.md back under `limit`, retrying if needed.

    A separate pass from `compact()`'s own prompt on purpose: that prompt's job
    is "fold in these new turns", and a second, competing instruction to also
    shrink the file tends to lose out to the more concrete ask. Giving the
    shrink instruction its own call, with nothing else to think about, is what
    actually gets it followed.
    """
    for _ in range(max_passes):
        if len(text) <= limit:
            return text
        prompt = f"### Current MEMORY.md ({len(text):,} chars, budget is {limit:,})\n\n{text}"
        try:
            resp = llm_client.generate(prompt, system=_SHRINK_SYSTEM, temperature=0.2)
        except Exception as e:
            print(f"  [memory] shrink pass failed, keeping the over-budget version: {e}")
            return text
        shrunk = (resp.text or "").strip()
        if not shrunk:
            return text
        text = shrunk
    return text


_DAY_SUMMARY_SYSTEM = """You are keeping a lab notebook for a physics apparatus.

You are given one day's log of what the assistant did, and the durable memory
for context. Write that day's entry. Return exactly two blocks:

<summary>
What was done today and what it showed, for a physicist reading it next week
who was not there. Lead with the result, not the procedure. Keep every number
that matters -- measured values, fitted parameters, the sweep that produced
them. Say what was inconclusive as plainly as what worked; a day that ended
without an answer is a real day and should read like one.

Attach the conditions to each measured value: what was being measured, when,
and with which sequence. A cutoff frequency with no date and no device is the
one thing that will mislead the reader later.

3-8 sentences, or a short table if the day produced a set of numbers.
</summary>

<next_steps>
What to do next, as 2-5 concrete bullets. Each one must be actionable tomorrow
morning: name the parameter, the range, the file, or the thing to check. "Look
into the discrepancy" is not a next step; "re-run the 10-20 kHz sweep with the
scope timebase at 5 ms/div to get >2 cycles at 1 kHz" is.

Include anything left unfinished or unverified today, and any question the
day raised. If today genuinely closed the topic out, say so and suggest what
the apparatus is now ready for.
</next_steps>

Plain Markdown. No LaTeX -- write `fc`, `~14.6 kHz`, `-3 dB`, never `$f_c$` or
`\\approx`. This is read in a terminal."""

_WEEK_SUMMARY_SYSTEM = """You are writing the weekly summary of a lab notebook.

You are given several daily pages, oldest first. Write the week, not a list of
the days. Return plain Markdown with these sections:

## What was measured
The results, in a table where there is a set of them. One row per measurement
with its date and conditions.

## How it developed
The through-line: what was believed at the start of the week, what changed it,
what is now settled. Name the day a value changed and what changed it. If a
number was revised, give both values and the reason -- that is the most useful
thing a weekly summary can contain.

## Still open
Questions the week raised and did not close, and anything measured once that
has not been reproduced.

## Next week
3-5 concrete actions, each naming a parameter, file or check.

Keep every number. No LaTeX -- write `fc`, `~14.6 kHz`, `-3 dB`. This is read
in a terminal."""

#: Sessions that start before this hour belong to the previous day's page.
#:
#: 0 means plain calendar dates, which is what this should be. It was briefly 5,
#: on the evidence of 02:06 and 03:47 entries that looked like a habit of working
#: past midnight. They were not: the machine was set to UTC+8 while the operator
#: was in US Eastern, so afternoon work was stamped as the small hours of the
#: next day. Fixing the clock removed the symptom, and a 5 a.m. rollover would
#: now misfile genuine late-evening work into the previous day.
#:
#: Raise it only for someone who really does work through midnight, and only
#: after checking the machine's timezone against where they are.
DAY_ROLLOVER_HOUR = 0

_DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

SUMMARY_HEADING = "## Summary"
NEXT_HEADING = "## Next steps"
LOG_HEADING = "## Log"
_PLACEHOLDER = "_(not written yet)_"


def _guard_ids(*texts: str, evidence: List[str]) -> tuple:
    """Strip run identifiers that do not appear in `evidence`.

    Reuses the reply guard so a notebook page cannot claim a shot that was never
    reported. Import is local because id_guard lives in the orchestrator and
    memory is imported by it.
    """
    try:
        from superradiant_assistant.orchestrator.id_guard import scrub
    except Exception:
        return texts
    out, dropped = [], []
    for t in texts:
        clean, removed = scrub(t, evidence)
        out.append(clean)
        dropped += removed
    if dropped:
        print(f"  [memory] dropped {len(dropped)} unverified identifier(s) from "
              f"the summary: {', '.join(dropped[:5])}")
    return tuple(out)


def day_key(when: Optional[datetime] = None) -> str:
    """Which notebook page a moment belongs to."""
    when = when or datetime.now()
    if when.hour < DAY_ROLLOVER_HOUR:
        when = when - timedelta(days=1)
    return when.strftime("%Y-%m-%d")


def parse_episode(text: str) -> tuple:
    """Split a page into (summary, next_steps, log).

    Tolerates a page written before this format existed: with no headings the
    whole file is the log, which is what those pages actually contain.
    """
    if not text.strip():
        return "", "", ""
    marks = []
    for heading in (SUMMARY_HEADING, NEXT_HEADING, LOG_HEADING):
        m = re.search(rf"^{re.escape(heading)}\s*$", text, re.MULTILINE)
        if m:
            marks.append((m.start(), m.end(), heading))
    if not marks:
        body = re.sub(r"^#\s+\d{4}-\d{2}-\d{2}\s*$", "", text, count=1,
                      flags=re.MULTILINE)
        return "", "", body.strip()

    marks.sort()
    out = {}
    for i, (start, end, heading) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        section = text[end:stop].strip()
        out[heading] = "" if section == _PLACEHOLDER else section
    return (out.get(SUMMARY_HEADING, ""), out.get(NEXT_HEADING, ""),
            out.get(LOG_HEADING, ""))


def render_episode(day: str, summary: str, next_steps: str, log: str) -> str:
    """One page, always in the same shape so a week of them reads as a series."""
    return (f"# {day}\n\n"
            f"{SUMMARY_HEADING}\n{summary.strip() or _PLACEHOLDER}\n\n"
            f"{NEXT_HEADING}\n{next_steps.strip() or _PLACEHOLDER}\n\n"
            f"{LOG_HEADING}\n{log.strip()}\n")


#: Where a deliberately remembered fact goes, and under which heading. Fixed
#: headings so the curator can be told to leave them alone: a note the operator
#: asked for must not be paraphrased away by the next compaction.
NOTE_TARGETS = {
    "instruction": ("instructions", "## Standing instructions"),
    "apparatus": ("core", "## Operator notes"),
    "labscript": ("labscript", "## Operator notes"),
}

#: Which memory is about THIS apparatus and which is about the software stack.
#:
#: Splitting these is not tidiness. About a third of what was learned on the test
#: bench is not about the bench at all -- that a shot file exists as soon as
#: runmanager compiles it, that lyse's `data()` is a two-level MultiIndex, that
#: its dataframe is cumulative, that labscriptlib must be pure ASCII. Each of
#: those cost hours, each is true in any lab on this stack, and each would have to
#: be relearned by repeating the mistake if memory were partitioned per apparatus.
#:
#: The reverse is worse: one shared file would have a real experiment reading
#: "No atoms, no cavity -- this is the function-generator test bench" as fact
#: about itself.
SHARED_DIRNAME = "agent_memory_shared"


def CONFIG_APPARATUS() -> str:
    """The current apparatus name, read late so a switch is picked up."""
    from superradiant_assistant import config as _c
    return getattr(_c, "APPARATUS", "?")


def _extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _same_note(a: str, b: str) -> bool:
    """Whether two notes say the same thing, ignoring formatting and dates."""
    def norm(s):
        s = re.sub(r"\(\d{4}-\d{2}-\d{2}\)", "", s)
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return norm(a) == norm(b)


class MemoryStore:
    def reconfigure(self) -> None:
        """Recompute every path from the current CONFIG, in place.

        `MEMORY` is a module-level singleton that other modules have already
        bound, and its paths are derived from the data root at construction. So
        switching apparatus rebuilt CONFIG and left the store pointing at the
        previous lab: the memory screen showed the new lab's name above the old
        lab's files, and starting a session would have injected one apparatus's
        facts while claiming to be another. Worse than no switch at all.
        """
        self.__init__(self._explicit_root, self._explicit_shared)

    def __init__(self, root: Optional[Path] = None,
                 shared_root: Optional[Path] = None):
        # Remembered so `reconfigure` can rebuild without losing an override.
        self._explicit_root = root
        self._explicit_shared = shared_root
        # Per-apparatus memory lives beside that apparatus's data, so switching
        # apparatus switches memory with no further wiring -- and so lab notes
        # never get committed with the code.
        self.root = Path(root or Path(CONFIG.historical_data_root).parent / "agent_memory")
        self.memory_file = self.root / "MEMORY.md"          # this apparatus
        self.instructions_file = self.root / "INSTRUCTIONS.md"   # for this apparatus
        self.history_file = self.root / "history.jsonl"
        # What this apparatus is ultimately for. Operator-owned: nothing in the
        # code writes it, and `_note_path` deliberately does not name it. An agent
        # that can rewrite the goal it is judged against is worse than one with no
        # goal, and the goal is a statement of intent rather than a fact an agent
        # discovers. Absent is a legitimate state -- the layer then emits nothing
        # and the advisor says it is inferring the goal from the notebook.
        self.objective_file = self.root / "OBJECTIVE.md"

        # Shared memory travels with the code, because it is about the code:
        # labscript/lyse/BLACS mechanics, and who the operator is.
        #
        # An explicit `root` isolates the shared tier too. Without this a store
        # pointed at a temporary directory still wrote LABSCRIPT.md and
        # OPERATOR.md into the repo -- which is exactly what happened, and a test
        # replaced the real operator profile with a one-line stub.
        from superradiant_assistant.config import REPO_ROOT
        if shared_root is not None:
            self.shared_root = Path(shared_root)
        elif root is not None:
            self.shared_root = self.root / "shared"
        else:
            self.shared_root = REPO_ROOT / SHARED_DIRNAME
        self.labscript_file = self.shared_root / "LABSCRIPT.md"
        self.operator_file = self.shared_root / "OPERATOR.md"

        #: Kept as an alias so anything still asking for USER.md gets the
        #: operator profile rather than a missing file.
        self.user_file = self.operator_file
        # Weekly pages live in their own folder so browsing days stays a clean
        # list of dates.
        self.weekly_dir = self.root / "weekly"

    def _read(self, path: Path, default: str = "") -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            return default

    def _write(self, path: Path, text: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path.write_text(text.strip() + "\n", encoding="utf-8")

    def read_memory(self) -> str:
        return self._read(self.memory_file)

    def read_user(self) -> str:
        return self._read(self.operator_file)

    def read_labscript(self) -> str:
        return self._read(self.labscript_file)

    def read_instructions(self) -> str:
        return self._read(self.instructions_file)

    def read_objective(self) -> str:
        return self._read(self.objective_file)

    def _note_path(self, target: str) -> Path:
        return {
            "core": self.memory_file,
            "instructions": self.instructions_file,
            "labscript": self.labscript_file,
            "user": self.operator_file,
        }.get(target, self.instructions_file)

    def episode_path(self, day: Optional[str] = None) -> Path:
        return self.root / f"{day or day_key()}.md"

    def read_today_episode(self) -> str:
        return self._read(self.episode_path())

    # -------------------------------------------------------------- notes

    def remember(self, fact: str, kind: str = "instruction") -> str:
        """Write one fact into memory now, rather than hoping compaction keeps it.

        The assistant had no way to do this at all. Asked to remember that sweeps
        should start at 3 kHz, it answered "I've noted this rule" and wrote
        nothing -- there was no tool, and the compaction curator is told to keep
        "parameter values, calibration results, things that went wrong", none of
        which covers an instruction about how to run the experiment. So the same
        instruction had to be given three times.

        Returns a human-readable status; the caller reports it verbatim.
        """
        fact = " ".join(str(fact).split())
        if not fact:
            return "nothing to remember: the fact was empty"
        target, heading = NOTE_TARGETS.get(kind, NOTE_TARGETS["instruction"])
        path = self._note_path(target)
        text = self._read(path)

        bullets = re.findall(r"^- .*$", text, re.MULTILINE)
        for b in bullets:
            if _same_note(b, "- " + fact):
                return f"already remembered in {path.name}: {b.strip()}"

        entry = f"- {fact}  ({day_key()})"
        m = re.search(rf"^{re.escape(heading)}\s*$", text, re.MULTILINE)
        if m:
            # Append at the end of that section, before the next heading.
            rest = text[m.end():]
            nxt = re.search(r"^## ", rest, re.MULTILINE)
            cut = m.end() + (nxt.start() if nxt else len(rest))
            text = (text[:cut].rstrip() + "\n" + entry + "\n\n"
                    + text[cut:].lstrip("\n"))
        else:
            text = text.rstrip() + f"\n\n{heading}\n\n{entry}\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.strip() + "\n", encoding="utf-8")
        scope = "shared across all apparatus" if target == "labscript" \
            else f"this apparatus ({CONFIG_APPARATUS()})"
        return (f"remembered in {path.name} under {heading!r} — {scope}: {fact}")

    # ---------------------------------------------------------------- days

    def episode_days(self) -> List[str]:
        """Every day that has a notebook page, oldest first."""
        days = []
        for p in self.root.glob("*.md"):
            if _DAY_RE.fullmatch(p.stem):
                days.append(p.stem)
        return sorted(days)

    def read_episode(self, day: str) -> str:
        return self._read(self.episode_path(day))

    def ensure_episode(self, day: Optional[str] = None) -> Path:
        """Create today's page if it does not exist yet.

        Called at start-up so the notebook has a page for every day the agent
        was opened, even one where nothing was worth distilling. A missing page
        and an uneventful page mean different things to whoever reads the week.
        """
        day = day or day_key()
        path = self.episode_path(day)
        if not path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            self._write(path, render_episode(day, "", "", ""))
        return path

    def append_episode(self, text: str, day: Optional[str] = None) -> None:
        """Add one timestamped entry to the day's log, leaving the summary alone.

        The summary and next steps are rewritten in place by `summarise_day`;
        only the log grows. Appending a summary per exit gave a day several,
        each claiming to be the day's, and the last one written was whichever
        session happened to end last -- often a two-minute one.
        """
        day = day or day_key()
        path = self.episode_path(day)
        summary, nxt, log = parse_episode(self._read(path))
        stamp = datetime.now().strftime("%H:%M")
        entry = f"### {stamp}\n{text.strip()}\n"
        log = (log.rstrip() + "\n\n" + entry) if log.strip() else entry
        self.root.mkdir(parents=True, exist_ok=True)
        self._write(path, render_episode(day, summary, nxt, log))

    def write_day_summary(self, summary: str, next_steps: str,
                          day: Optional[str] = None) -> None:
        """Replace the day's summary and next steps, keeping the log verbatim."""
        day = day or day_key()
        path = self.episode_path(day)
        _, _, log = parse_episode(self._read(path))
        self.root.mkdir(parents=True, exist_ok=True)
        self._write(path, render_episode(day, summary, next_steps, log))

    def summarise_day(self, llm_client, day: Optional[str] = None) -> bool:
        """Rewrite the day's summary and next steps from the WHOLE day's log.

        Rewritten rather than appended, so exiting six times in a day leaves the
        same file as exiting once. The input is the day's page, not this
        session's turns: the last session of a day is often a short one, and a
        summary drawn from it alone would describe ten minutes as the day.
        """
        day = day or day_key()
        _, _, log = parse_episode(self.read_episode(day))
        if not log.strip():
            return False
        prompt = (f"### Date\n{day}\n\n"
                  f"### Today's log\n{log.strip()}\n\n"
                  f"### Durable memory for context\n"
                  f"{self.read_memory() or '(empty)'}")
        try:
            resp = llm_client.generate(prompt, system=_DAY_SUMMARY_SYSTEM,
                                       temperature=0.2)
        except Exception as e:
            print(f"  [memory] day summary failed: {e}")
            return False
        summary = _extract_tag(resp.text, "summary") or ""
        nxt = _extract_tag(resp.text, "next_steps") or ""
        if not (summary.strip() or nxt.strip()):
            return False
        # The same guard the replies get. A summary is written by a model from
        # the log, so it can invent a run ID exactly as a reply can -- and this
        # one is written to disk and read back weeks later, which is worse. The
        # log it was summarised from is the only admissible evidence.
        summary, nxt = _guard_ids(summary, nxt, evidence=[log])
        self.write_day_summary(summary, nxt, day)
        return True

    def weekly_summary(self, llm_client, days: Optional[List[str]] = None,
                       label: Optional[str] = None) -> Optional[Path]:
        """Write one page covering several days, from their pages.

        Built on the daily summaries rather than on raw history: the point of
        writing a summary every day is that the week does not have to be
        reconstructed from 600 turns.
        """
        days = days or self.episode_days()[-7:]
        pages = []
        for d in days:
            text = self.read_episode(d)
            if text.strip():
                pages.append(f"## {d}\n{text.strip()}")
        if not pages:
            return None
        label = label or (f"{days[0]}_to_{days[-1]}" if len(days) > 1 else days[0])
        try:
            resp = llm_client.generate("\n\n".join(pages),
                                       system=_WEEK_SUMMARY_SYSTEM,
                                       temperature=0.2)
        except Exception as e:
            print(f"  [memory] weekly summary failed: {e}")
            return None
        body, = _guard_ids(resp.text.strip(), evidence=pages)
        out = self.weekly_dir / f"{label}.md"
        self.weekly_dir.mkdir(parents=True, exist_ok=True)
        self._write(out, f"# Weekly summary · {label}\n\n{body}\n")
        return out

    def append_history(self, role: str, content: Any) -> None:
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "role": role,
            "content": content,
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.history_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            print(f"  [memory] could not append history: {e}")

    def read_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            lines = self.history_file.read_text(encoding="utf-8").strip().splitlines()
        except (FileNotFoundError, OSError):
            return []
        out = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def build_context_block(self, include=None) -> str:
        """The memory layers, injected into a system prompt when it is built.

        Read once per agent session, not per turn: the system instruction is built
        in `Agent._new_session()` and then held. A `remember` call mid-session is
        on disk immediately but is not re-read until a new session or a context
        compaction.

        Each layer is labelled with its scope, because the assistant has to know
        which facts travel: a cutoff frequency belongs to one apparatus, while how
        lyse's dataframe behaves is true wherever labscript runs.

        `include` selects layers. The whole block is 18.7k characters and every
        agent used to carry all of it -- which is 68% of a system prompt that is
        resent on every single tool call, and the four roles' prompts differed
        from each other by 2%. A teammate that never talks to the operator has no
        use for the operator profile, and one that only writes code has no use for
        the day's narrative.
        """
        apparatus = CONFIG_APPARATUS()
        layers = [
            ("objective",
             f"## What {apparatus} is for (OBJECTIVE.md, written by the operator)",
             self.read_objective()),
            ("labscript",
             "## How labscript, BLACS and lyse behave (shared by every apparatus)",
             self.read_labscript()),
            ("memory",
             f"## This apparatus: {apparatus} (MEMORY.md)", self.read_memory()),
            ("operator",
             "## Operator profile (shared by every apparatus)", self.read_user()),
            ("instructions",
             f"## Standing instructions for {apparatus}", self.read_instructions()),
            ("notebook",
             "## Today's page from the lab notebook", self.read_today_episode()),
        ]
        want = set(CONTEXT_LAYERS if include is None else include)
        parts = [f"{title}\n{body}" for key, title, body in layers
                 if body and key in want]
        return "\n\n".join(parts)

    def compact(self, older_turns: List[Dict[str, str]], llm_client) -> bool:
        """Fold older turns into MEMORY.md / OPERATOR.md / today's episode.

        Returns True if memory was updated. Failures are non-fatal: losing a
        compaction is better than losing the session.
        """
        if not older_turns:
            return False

        transcript = "\n".join(
            f"[{t.get('role', '?').upper()}] {t.get('content', '')}" for t in older_turns
        )
        prompt = (
            f"### Apparatus\n{CONFIG_APPARATUS()}\n\n"
            f"### Existing MEMORY.md — facts about THIS apparatus\n"
            f"{self.read_memory() or '(empty)'}\n\n"
            f"### Existing LABSCRIPT.md — shared by every apparatus\n"
            f"{self.read_labscript() or '(empty)'}\n\n"
            f"### Existing OPERATOR.md — shared by every apparatus\n"
            f"{self.read_user() or '(empty)'}\n\n"
            f"### Turns to distil\n{transcript}"
        )
        try:
            resp = llm_client.generate(prompt, system=_COMPACT_SYSTEM, temperature=0.2)
        except Exception as e:
            print(f"  [memory] compaction failed: {e}")
            return False

        episode = _extract_tag(resp.text, "episode")
        memory = _extract_tag(resp.text, "updated_memory")
        labscript = _extract_tag(resp.text, "updated_labscript")
        user = _extract_tag(resp.text, "updated_user")

        def unchanged(t):
            return not t or t.strip().upper() == "UNCHANGED"

        if episode:
            self.append_episode(episode)
        if memory:
            if len(memory) > MEMORY_FILE_MAX_CHARS:
                before = len(memory)
                memory = _shrink_memory(memory, llm_client, MEMORY_FILE_MAX_CHARS)
                print(f"  [memory] MEMORY.md was {before:,} chars over the "
                     f"{MEMORY_FILE_MAX_CHARS:,} budget, shrunk to {len(memory):,}")
            self._write(self.memory_file, memory)
        if not unchanged(labscript):
            self.shared_root.mkdir(parents=True, exist_ok=True)
            self.labscript_file.write_text(labscript.strip() + "\n", encoding="utf-8")
        if not unchanged(user):
            self.shared_root.mkdir(parents=True, exist_ok=True)
            self.operator_file.write_text(user.strip() + "\n", encoding="utf-8")
        return bool(episode or memory or not unchanged(labscript))


    def compact_on_exit(self, agents, llm_client, evidence=None) -> bool:
        """Distil what is left in each agent's session before the process dies.

        Without this, everything since the last compaction is lost -- and since
        compaction had never fired, that was everything, every time. Called from
        the exit path so a session that did real work leaves a note behind.

        Skipped when nothing happened: distilling two lines of chat costs an API
        call and writes a MEMORY.md that dilutes the real notes.
        """
        if evidence is not None and not evidence.did_substantive_work:
            return False

        turns: List[Dict[str, str]] = []
        for agent in agents:
            session = getattr(agent, "_session", None)
            if session is None:
                continue
            try:
                history = session.get_history()
            except Exception:
                continue
            for c in history:
                text = history_text(c)
                if text:
                    turns.append({"role": f"{agent.name}:{getattr(c, 'role', '?')}",
                                  "content": text})
        if not turns:
            return False
        return self.compact(turns, llm_client)


#: A tool result can be thousands of characters -- a file read, a loaded skill,
#: a shot inspection. Truncated so one enormous result cannot dominate the
#: transcript handed to the curator, but not dropped: the numbers a session
#: should remember arrive in tool results, not in the chat text around them.
_MAX_RESULT_CHARS = 1200


def _anthropic_block(block, key: str, default=None):
    """One field of a content block, dict or SDK object."""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _anthropic_text(message: Dict[str, Any], truncate: bool = True) -> str:
    """Plain text out of one Anthropic message, covering every block type."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    out = []
    for block in content or []:
        btype = _anthropic_block(block, "type", "")
        if btype == "text":
            out.append(_anthropic_block(block, "text") or "")
        elif btype == "thinking":
            continue
        elif btype == "tool_use":
            args = _anthropic_block(block, "input") or {}
            rendered = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
            out.append(f"[called {_anthropic_block(block, 'name')}({rendered})]")
        elif btype == "tool_result":
            payload = str(_anthropic_block(block, "content") or "")
            if truncate and len(payload) > _MAX_RESULT_CHARS:
                payload = payload[:_MAX_RESULT_CHARS] + " ...(truncated)"
            out.append(f"[-> {payload}]" if payload else "[result]")
    return " ".join(out).strip()


def history_text(content: Any) -> str:
    """Plain text out of a google.genai Content, covering every part type.

    Reading only `.text` scored every tool result as zero characters. That made
    the context-size trigger blind to exactly the parts that make context large,
    and it fed the memory curator the chat around the measurements without the
    measurements.

    Anthropic messages are dicts of blocks rather than Content objects, and go
    through `_anthropic_text`. Same reason: an agent on Claude whose history
    measured zero would never compact.
    """
    if isinstance(content, dict):
        return _anthropic_text(content)
    out = []
    for part in getattr(content, "parts", None) or []:
        text = getattr(part, "text", None)
        if text:
            out.append(text)
            continue
        call = getattr(part, "function_call", None)
        if call is not None and getattr(call, "name", None):
            args = getattr(call, "args", None)
            rendered = ", ".join(f"{k}={_short(v)}" for k, v in (args or {}).items())
            out.append(f"[called {call.name}({rendered})]")
            continue
        resp = getattr(part, "function_response", None)
        if resp is not None and getattr(resp, "name", None):
            body = getattr(resp, "response", None)
            payload = ""
            if isinstance(body, dict):
                payload = str(body.get("result", body))
            elif body is not None:
                payload = str(body)
            if len(payload) > _MAX_RESULT_CHARS:
                payload = payload[:_MAX_RESULT_CHARS] + " ...(truncated)"
            out.append(f"[{resp.name} -> {payload}]" if payload
                       else f"[result of {resp.name}]")
    return " ".join(out).strip()


def content_size(content: Any) -> int:
    """Untruncated size of one history entry, for the compaction trigger.

    Deliberately not `len(history_text(...))`: that truncates each tool result
    to keep the curator's prompt manageable, so a 10 kB file read would be
    measured as 1.2 kB and the size trigger would fire many times later than
    intended. What decides when to compact is what the model is actually
    carrying, not what the curator will be shown.
    """
    if isinstance(content, dict):
        return len(_anthropic_text(content, truncate=False))
    total = 0
    for part in getattr(content, "parts", None) or []:
        text = getattr(part, "text", None)
        if text:
            total += len(text)
        call = getattr(part, "function_call", None)
        if call is not None:
            total += len(str(getattr(call, "args", "") or "")) + 20
        resp = getattr(part, "function_response", None)
        if resp is not None:
            total += len(str(getattr(resp, "response", "") or ""))
    return total


def _short(v: Any, n: int = 60) -> str:
    s = str(v)
    return s if len(s) <= n else s[:n] + "..."


#: Kept for callers that used the private name.
_history_text = history_text


MEMORY = MemoryStore()
