"""Read the labscript suite's own logs so failures reach the agent verbatim.

When a shot does not produce data the agent could previously only infer why: the
runmanager remote API exposes no compile status, and BLACS reports nothing back
at all. The real cause was always sitting in a log file nobody was reading --
`VI_ERROR_INP_PROT_VIOL` aborting every shot, for instance, was plainly logged
while the agent guessed at "lyse may not be running".

Compile tracebacks are NOT here: runmanager pipes those to its GUI output pane
and never writes them to disk. Use the bridge's compile_check for those.
"""
from __future__ import annotations
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from superradiant_assistant.config import CONFIG

LOG_DIR = Path(CONFIG.labscript_suite_root) / "logs"

# "2026-08-07 04:33:50,094 INFO BLACS.queue_manager.thread: ..."
_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\s+(?P<level>[A-Z]+)\s+(?P<rest>.*)$"
)

INTERESTING = ("ERROR", "CRITICAL", "WARNING")


def _parse_ts(line: str) -> Optional[float]:
    m = _LINE.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S").timestamp()
    except ValueError:
        return None


def recent_errors(log_name: str, since_ts: float, max_blocks: int = 5,
                  max_lines_per_block: int = 25) -> List[str]:
    """Error blocks logged at or after `since_ts`.

    A block is an ERROR/CRITICAL line plus the untimestamped lines that follow
    it, which is how Python tracebacks appear in these logs.
    """
    path = LOG_DIR / log_name
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    blocks: List[List[str]] = []
    current: Optional[List[str]] = None
    # A whole second of slack: log timestamps have no sub-second resolution
    # here, so an error logged in the same second as the trigger would be lost.
    cutoff = since_ts - 1.0

    for line in text.splitlines():
        ts = _parse_ts(line)
        if ts is None:
            # Continuation (traceback body) belongs to the block above it.
            if current is not None:
                current.append(line)
            continue
        m = _LINE.match(line)
        level = m.group("level") if m else ""
        if ts >= cutoff and level in INTERESTING:
            current = [line]
            blocks.append(current)
        else:
            current = None

    trimmed = []
    for b in blocks[-max_blocks:]:
        if len(b) > max_lines_per_block:
            b = b[:max_lines_per_block] + ["    ... (truncated)"]
        trimmed.append("\n".join(b))
    return trimmed


def blacs_errors(since_ts: float, **kw) -> List[str]:
    """Device-level failures: the worker exceptions that make BLACS abort a shot."""
    return recent_errors("BLACS.log", since_ts, **kw)


def lyse_errors(since_ts: float, **kw) -> List[str]:
    """Analysis-routine failures."""
    return recent_errors("lyse.log", since_ts, **kw)


def describe_failures(since_ts: float) -> str:
    """One block of real log text for whatever went wrong, or '' if nothing did."""
    parts = []
    for label, errs in (("BLACS", blacs_errors(since_ts)),
                        ("lyse", lyse_errors(since_ts))):
        if errs:
            parts.append(f"--- {label} log, actual errors ---\n" + "\n".join(errs))
    return "\n".join(parts)
