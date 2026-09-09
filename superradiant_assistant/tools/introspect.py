"""Read-only introspection: let the agent look at the apparatus it works on.

The agent could write sequences and queue shots long before it could read a
file. Asked how the scope data was laid out, its only route was to guess a
format, write a probe analysis, fire a real shot, and wait -- minutes per guess,
against a working example sitting unread on disk.

Nothing here writes, executes, or touches an instrument. The restrictions that
matter for safety are on writing and on hardware; withholding the ability to
*look* bought nothing and cost the agent every chance to check its assumptions.

Paths are still confined to the experiment directories, for the same reason the
shot data root is fixed: the model should not be able to read arbitrary files
off this machine.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from superradiant_assistant.config import CONFIG

MAX_CHARS = 12000


def _roots() -> List[Path]:
    suite = Path(CONFIG.labscript_suite_root)
    return [
        suite / "userlib" / "labscriptlib",
        suite / "userlib" / "analysislib",
        suite / "labscript-devices" / "labscript_devices",
        Path(CONFIG.historical_data_root),
    ]


def _resolve(path_str: str) -> Optional[Path]:
    """Resolve `path_str` only if it lands inside an allowed root.

    A relative path is tried against the suite root and each allowed root
    before being rejected. `list_scripts` and the config files quote paths in
    several forms, and the model copies whichever it saw: a bare
    `userlib/labscriptlib/Cesium/Sequences/filter_scan.py` was refused as
    "outside the experiment directories" even though it names a file inside
    one, costing a round trip on the way to the same file.
    """
    raw = Path(str(path_str).strip().strip('"').strip("'"))
    candidates = [raw]
    if not raw.is_absolute():
        suite = Path(CONFIG.labscript_suite_root)
        candidates.append(suite / raw)
        candidates.extend(root / raw for root in _roots())
        # Also allow naming a file by the part after a root, e.g.
        # "Cesium/Sequences/filter_scan.py".
        candidates.extend(root / raw.name for root in _roots())

    roots = [r.resolve() for r in _roots()]
    for cand in candidates:
        try:
            p = cand.resolve()
        except (OSError, ValueError):
            continue
        for root in roots:
            try:
                p.relative_to(root)
            except (ValueError, OSError):
                continue
            if p.exists() or cand is raw:
                return p
    return None


# --------------------------------------------------------------------------
# read_lab_file
# --------------------------------------------------------------------------

def read_lab_file(path: str, max_chars: int = MAX_CHARS) -> str:
    """Read a sequence, analysis or device driver so it can be used as a model."""
    p = _resolve(path)
    if p is None:
        return (f"refused: '{path}' is outside the experiment directories. "
                f"Readable roots: {[str(r) for r in _roots()]}")
    if not p.is_file():
        return f"error: {p} is not a file"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"error: could not read {p}: {e}"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... (truncated at {max_chars} chars)"
    return f"{p}\n{'-' * 60}\n{text}"


# --------------------------------------------------------------------------
# list_lab_files
# --------------------------------------------------------------------------

#: Directories holding this apparatus's own code. Listing everything returned
#: 253 files -- almost all of them from an unrelated experiment -- and that
#: whole list then sat in the conversation for every subsequent turn.
_PRIMARY_HINTS = ("Cesium",)

MAX_LISTED = 40


def list_lab_files(subdir: str = "", pattern: str = "*.py",
                   include_all: bool = False) -> str:
    """List the sequences and analyses that already exist.

    Files for this apparatus come first and everything else is summarised by
    count, unless `include_all` is set.
    """
    primary: List[str] = []
    other = 0

    for root in _roots():
        if not root.exists() or root == Path(CONFIG.historical_data_root):
            continue
        base = root / subdir if subdir else root
        if not base.exists():
            continue
        for p in sorted(base.rglob(pattern)):
            if "__pycache__" in str(p):
                continue
            relevant = any(hint in str(p) for hint in _PRIMARY_HINTS)
            if not (relevant or include_all):
                other += 1
                continue
            try:
                n = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
            except OSError:
                n = -1
            primary.append(f"  {p}  ({n} lines)")

    if not primary and not other:
        return f"no files matching '{pattern}' under the experiment directories"

    truncated = len(primary) > MAX_LISTED
    shown = primary[:MAX_LISTED]
    head = f"{len(primary)} file(s) for this apparatus"
    tail = ""
    if truncated:
        tail += f"\n  ... and {len(primary) - MAX_LISTED} more; narrow with `subdir` or `pattern`"
    if other:
        tail += (f"\n  ({other} further file(s) belong to other apparatus and are "
                 f"hidden; pass include_all=true if you really need them)")
    return head + ":\n" + "\n".join(shown) + tail


# --------------------------------------------------------------------------
# inspect_shot
# --------------------------------------------------------------------------

def _describe(node, name: str, lines: List[str], depth: int = 0) -> None:
    pad = "  " * (depth + 1)
    if isinstance(node, h5py.Dataset):
        cols = node.dtype.names
        detail = f"columns={cols}" if cols else f"dtype={node.dtype}"
        lines.append(f"{pad}{name}  shape={node.shape}  {detail}")
    else:
        lines.append(f"{pad}{name}/")
        if node.attrs:
            for k, v in list(node.attrs.items())[:12]:
                lines.append(f"{pad}  .{k} = {_short(v)}")
        if depth < 2:
            for child in list(node.keys())[:20]:
                _describe(node[child], child, lines, depth + 1)


def _short(v: Any, n: int = 60) -> str:
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    s = str(v).replace("\n", " ")
    return s[:n] + "..." if len(s) > n else s


def inspect_shot(shot: str = "latest") -> str:
    """Show a shot file's real structure: groups, dataset columns, saved results.

    This is what makes the data layout knowable without firing a shot to find
    out. `shot` is a filename, a fragment of one, or "latest".
    """
    from superradiant_assistant.interfaces.hdf5_reader import list_shots

    paths = list_shots(Path(CONFIG.historical_data_root))
    if not paths:
        return f"no shot files under {CONFIG.historical_data_root}"

    if shot in ("latest", "", None):
        target = paths[-1]
    else:
        matches = [p for p in paths if shot.lower() in p.name.lower()]
        if not matches:
            return (f"no shot matching '{shot}'. Newest few: "
                    f"{[p.name for p in paths[-5:]]}")
        target = matches[-1]

    lines = [f"{target.name}", f"  (of {len(paths)} shots in {target.parent})"]
    try:
        with h5py.File(target, "r") as f:
            ran = "front_panel" in f
            lines.append(f"  executed by BLACS: {ran}"
                         + ("" if ran else "   <- compiled only, never run"))
            for key in ("globals", "data", "results", "devices"):
                if key in f:
                    _describe(f[key], key, lines)
            traces = f.get("data/traces")
            if traces is not None and len(traces):
                name = list(traces.keys())[0]
                arr = traces[name][:]
                lines.append(f"  first trace '{name}': {len(arr)} points")
                for col in (arr.dtype.names or ()):
                    if col == "t":
                        continue
                    v = arr[col].astype(float)
                    lines.append(f"    {col}: Vpp={np.ptp(v):.4g} mean={np.mean(v):.4g}")
    except Exception as e:
        return f"error reading {target.name}: {type(e).__name__}: {e}"
    return "\n".join(lines)
