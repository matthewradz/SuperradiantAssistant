"""What experiment scripts already exist, and where.

The model kept choosing badly between writing a new script, overwriting one, and
reusing one -- not out of recklessness but because it never had the list in front
of it. `list_lab_files` returns 256 paths across drivers, old analyses and dead
experiments; that is a haystack, not an inventory.

This module answers the narrower question the decision actually needs: which
sequences and analysis routines belong to THIS apparatus, when was each written,
which of them creative mode produced, and what each one is for. It is also what
the confirmation prompt shows the operator, so the choice is informed even when
the model did not bother to look first.
"""
from __future__ import annotations
import io
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from superradiant_assistant.config import CONFIG, REPO_ROOT

_SUITE = Path(CONFIG.labscript_suite_root)

SHOT_DIR = _SUITE / "userlib" / "labscriptlib" / "Cesium" / "Sequences"
ANALYSIS_DIR = _SUITE / "userlib" / "analysislib" / "Cesium" / "singleshot_routines"
MULTISHOT_DIR = _SUITE / "userlib" / "analysislib" / "Cesium" / "multishot_routines"

#: Fallback only. The real cutoff is derived from the shot folder -- see
#: `_apparatus_epoch`.
_FALLBACK_EPOCH = datetime(2026, 1, 1)


def _apparatus_epoch() -> datetime:
    """When work on THIS apparatus began, taken from the shot folder.

    The analysis folders are shared with the previous experiment on this suite
    (the MIT ybclock cavity work): several hundred routines that all carry the
    date the tree was copied here, not the date they were written. A fixed cutoff
    got that wrong in both directions, so derive it instead -- Sequences/ holds
    only this apparatus's shots, so its oldest file is when we started.
    """
    if not SHOT_DIR.is_dir():
        return _FALLBACK_EPOCH
    stamps = [datetime.fromtimestamp(p.stat().st_mtime)
              for p in SHOT_DIR.glob("*.py") if not p.name.startswith("_")]
    if not stamps:
        return _FALLBACK_EPOCH
    return min(stamps) - timedelta(days=1)


def _docstring_summary(path: Path) -> str:
    """First line of the module docstring, as a description.

    Analysis routines are not registered in config.json, so without this the
    inventory lists a column of bare filenames -- which is what sent the model
    to read six files one after another to find out what it already had.

    Scanned by hand rather than with `ast.parse`, which needs the whole file to
    be syntactically complete: reading a bounded head of a long file cuts it
    mid-statement, and every routine over ~4 KB silently lost its description.
    """
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            lines = [next(f, "") for _ in range(40)]
    except OSError:
        return ""

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        quote = next((q for q in ('"""', "'''") if line.startswith(q)), None)
        if quote is None:
            return ""                       # no module docstring
        first = line[len(quote):].strip()
        if first.endswith(quote):           # one-line docstring
            first = first[:-len(quote)].strip()
        if first:
            return first[:96]
        # Docstring opened on its own line; the summary is the next real line.
        for follow in lines[i + 1:]:
            text = follow.strip()
            if text and not text.startswith(quote):
                return text.rstrip(quote).strip()[:96]
        return ""
    return ""

KIND_DIRS = {
    "shot": SHOT_DIR,
    "singleshot": ANALYSIS_DIR,
    "multishot": MULTISHOT_DIR,
}


@dataclass
class ScriptInfo:
    path: Path
    kind: str
    modified: datetime
    n_lines: int
    generated: bool                 # written by creative mode, per config.json
    description: str = ""
    key_globals: Optional[List[str]] = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    def render(self, width: int = 34) -> str:
        tag = "[gen] " if self.generated else "      "
        line = (f"  {tag}{self.name:<{width}} {self.modified:%Y-%m-%d}"
                f" {self.n_lines:>4} lines")
        if self.description:
            line += f"\n         {self.description}"
        if self.key_globals:
            line += f"\n         globals: {', '.join(self.key_globals)}"
        return line


def _config_index() -> Dict[str, dict]:
    """config.json's `sequences`, keyed by lower-cased forward-slash path."""
    try:
        cfg = json.loads(io.open(REPO_ROOT / "config.json", encoding="utf-8").read())
    except (OSError, ValueError):
        return {}
    return {str(s.get("file", "")).replace("\\", "/").lower(): s
            for s in cfg.get("sequences", [])}


def _describe(path: Path, index: Dict[str, dict]) -> ScriptInfo:
    key = str(path).replace("\\", "/").lower()
    entry = index.get(key, {})
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        n_lines = len(text.splitlines())
    except OSError:
        n_lines = -1
    kind = next((k for k, d in KIND_DIRS.items() if d == path.parent), "shot")
    # config.json only registers sequences; analysis routines describe themselves.
    description = str(entry.get("description", "")) or _docstring_summary(path)
    return ScriptInfo(
        path=path,
        kind=kind,
        modified=datetime.fromtimestamp(path.stat().st_mtime),
        n_lines=n_lines,
        generated=entry.get("use_case") == "written in creative mode",
        description=description,
        key_globals=list(entry.get("key_globals") or []) or None,
    )


def scripts_of_kind(kind: str, recent_only: bool = True) -> List[ScriptInfo]:
    """Every script of one kind, newest first."""
    directory = KIND_DIRS.get(kind)
    if directory is None or not directory.is_dir():
        return []
    index = _config_index()
    epoch = _apparatus_epoch()
    out = []
    for p in directory.glob("*.py"):
        if p.name.startswith("_"):
            continue
        info = _describe(p, index)
        # The shot folder is small and entirely ours; the analysis folders are
        # shared with hundreds of files from the previous experiment.
        if recent_only and kind != "shot" and info.modified < epoch:
            continue
        out.append(info)
    return sorted(out, key=lambda i: i.modified, reverse=True)


def find(kind: str, stem: str) -> Optional[ScriptInfo]:
    """One script by bare stem, or None."""
    directory = KIND_DIRS.get(kind)
    if directory is None:
        return None
    path = directory / f"{stem}.py"
    if not path.is_file():
        return None
    return _describe(path, _config_index())


def dated_stem(stem: str, kind: str, when: Optional[datetime] = None) -> str:
    """`stem` with today's date appended, guaranteed not to collide.

    A new script has to be distinguishable from the one it was almost an
    overwrite of, and 'filter_scan2' does not say when or why it appeared.
    """
    when = when or datetime.now()
    directory = KIND_DIRS.get(kind, SHOT_DIR)
    # The model frequently dates its own filenames. Appending another gives
    # `fourier_spectrum_20260809_20260809.py` -- ugly, and long enough to wrap
    # the confirmation prompt.
    suffix = f"{when:%Y%m%d}"
    if stem.endswith(f"_{suffix}"):
        stem = stem[: -(len(suffix) + 1)]
    base = f"{stem}_{suffix}"
    if not (directory / f"{base}.py").exists():
        return base
    for n in range(2, 100):
        candidate = f"{base}_{n}"
        if not (directory / f"{candidate}.py").exists():
            return candidate
    return f"{base}_{when:%H%M%S}"


_HEADINGS = {
    "shot": "SHOT SEQUENCES        (runmanager compiles these)",
    "singleshot": "SINGLESHOT ROUTINES   (lyse runs one per shot)",
    "multishot": "MULTISHOT ROUTINES    (lyse runs one per sweep)",
}


def render_inventory(kinds: Optional[List[str]] = None,
                     limit_per_kind: int = 12) -> str:
    """The whole inventory as operator- and model-readable text."""
    blocks = []
    for kind in (kinds or ["shot", "singleshot", "multishot"]):
        items = scripts_of_kind(kind)
        head = f"{_HEADINGS.get(kind, kind.upper())}   [{KIND_DIRS[kind]}]"
        if not items:
            blocks.append(f"{head}\n  (none)")
            continue
        shown = items[:limit_per_kind]
        body = "\n".join(i.render() for i in shown)
        if len(items) > len(shown):
            body += f"\n  ... and {len(items) - len(shown)} older"
        blocks.append(f"{head}\n{body}")

    live = _live_lyse_routines()
    if live:
        blocks.append(live)
    blocks.append("[gen] = written by creative mode. Anything unmarked is "
                  "hand-written; overwriting it destroys someone's work.")
    return "\n\n".join(blocks)


def _live_lyse_routines() -> str:
    """Which routines lyse is actually running, if it can be asked."""
    try:
        from superradiant_assistant.interfaces.lyse_iface import live_routines
    except Exception:
        return ""
    parts = []
    for kind in ("singleshot", "multishot"):
        got = live_routines(kind)
        if got is None:
            parts.append(f"  {kind:<11} (lyse did not answer)")
        elif not got:
            parts.append(f"  {kind:<11} (none loaded)")
        else:
            parts.append(f"  {kind:<11} " + ", ".join(Path(p).name for p in got))
    return "CURRENTLY LOADED IN LYSE\n" + "\n".join(parts)


def compact_list(kind: str, limit: int = 8) -> List[str]:
    """Short 'name -- description' lines, for a confirmation prompt."""
    out = []
    for info in scripts_of_kind(kind)[:limit]:
        desc = info.description or "(no description on record)"
        if len(desc) > 62:
            desc = desc[:59] + "..."
        out.append(f"{info.name:<34} {desc}")
    return out
