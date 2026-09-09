"""Strip shot identifiers the model did not actually receive.

The model has a standing habit of inventing plausible run IDs: given one real
anchor (a status line naming the last shot) and a point count, it will emit a
contiguous range like `..._0035` through `..._0040` that looks entirely
legitimate and sends the operator to the wrong data. Prompt rules did not stop
it — it read both the shared "quote identifiers verbatim" rule and a tool result
saying no IDs existed, and produced a list anyway.

So this is enforced in code instead: every identifier in a reply must appear
somewhere in what the agent was actually given (tool results, or the operator's
own words). Anything else is replaced with a visible marker.
"""
from __future__ import annotations
import re
from typing import Iterable

# `2026-08-06_0048_Test_Speed_5`, and the bare `2026-08-06_0048` prefix form.
_RUN_ID = re.compile(r"\b\d{4}-\d{2}-\d{2}_\d{3,}(?:_[A-Za-z0-9]+)*\b")
# `shot_0000`, `shot_0050.h5`
_SHOT_ID = re.compile(r"\bshot_\d+\b", re.IGNORECASE)

PATTERNS = (_RUN_ID, _SHOT_ID)

REDACTION = "[unverified-id-removed]"


def _normalise(s: str) -> str:
    return s.replace("\\", "/").lower()


def find_ids(text: str) -> list[str]:
    """Every shot-identifier-shaped token in `text`, in order, deduplicated."""
    seen: list[str] = []
    for pat in PATTERNS:
        for m in pat.finditer(text or ""):
            tok = m.group(0)
            if tok not in seen:
                seen.append(tok)
    return seen


def scrub(reply: str, corpus: Iterable[str]) -> tuple[str, list[str]]:
    """Return (cleaned_reply, removed_ids).

    An identifier survives only if it occurs literally in `corpus` — the tool
    results and operator messages this agent actually saw. Substring matching is
    deliberate: a reply may cite `2026-08-06_0048_Test_Speed_5` when the corpus
    holds `.../2026-08-06_0048_Test_Speed_5.h5`, and that is a real citation.
    """
    if not reply:
        return reply, []

    haystack = "\n".join(_normalise(c) for c in corpus if c)
    removed: list[str] = []
    cleaned = reply

    for tok in find_ids(reply):
        if _normalise(tok) in haystack:
            continue
        removed.append(tok)
        # Word-boundary replace so `shot_1` does not eat `shot_12`.
        cleaned = re.sub(rf"`?{re.escape(tok)}`?(?!\w)", REDACTION, cleaned)

    return cleaned, removed
