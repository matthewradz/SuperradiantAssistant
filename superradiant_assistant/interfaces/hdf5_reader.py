"""Read labscript HDF5 shots and turn them into ShotSignal objects."""
from __future__ import annotations
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional
import h5py
import numpy as np

from superradiant_assistant.signals import ShotSignal


def list_shots(root: Path) -> List[Path]:
    return sorted(Path(root).rglob("*.h5"))


def _decode(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def _strip_comment(s: str) -> str:
    """Remove everything from the first '#' to end of line, handling quoted strings."""
    out = []
    in_str = False
    quote = ""
    for ch in s:
        if in_str:
            out.append(ch)
            if ch == quote:
                in_str = False
            continue
        if ch in ("'", '"'):
            in_str = True
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out).strip()


def _eval_global(raw: str) -> Any:
    """Try hard to turn a runmanager-style global expression into a Python value.

    Handles things like:
      '48.3'                           -> 48.3
      '48.3#np.linspace(-0.2,0.2,11)'  -> 48.3   (comment stripped)
      'np.linspace(0,1,5)'             -> [0, 0.25, 0.5, 0.75, 1.0]
      'True'                           -> True
      "'Magnification_RF'"             -> 'Magnification_RF'
    Returns the original string if all parsing fails.
    """
    s = _strip_comment(raw)
    if not s:
        return raw

    # 1) Pure Python literal?
    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    # 2) Try eval with numpy available
    try:
        return eval(s, {"__builtins__": {}}, {"np": np, "pi": np.pi})
    except Exception:
        return raw  # give up, return the original string


def _parse_attr(v: Any) -> Any:
    v = _decode(v)
    if isinstance(v, str):
        return _eval_global(v)
    return v


def _read_attrs(g: h5py.Group) -> Dict[str, Any]:
    return {k: _parse_attr(v) for k, v in g.attrs.items()}


def read_globals_group(f: h5py.File, group_name: str) -> Dict[str, Any]:
    path = f"globals/{group_name}"
    if path not in f:
        return {}
    return _read_attrs(f[path])


def read_results_attrs(f: h5py.File, results_subpath: str) -> Dict[str, Any]:
    full = f"results/{results_subpath}"
    if full not in f:
        return {}
    return _read_attrs(f[full])


def read_shot(path: Path) -> ShotSignal:
    path = Path(path)
    with h5py.File(path, "r") as f:
        cavity = read_results_attrs(f, "cavity_scan_analysis")

        atom_loading: Dict[str, Any] = {}
        all_globals: Dict[str, Any] = {}
        if "globals" in f:
            for group_name in f["globals"]:
                grp = _read_attrs(f[f"globals/{group_name}"])
                all_globals.update(grp)
                if group_name == "Atom Loading":
                    atom_loading = grp

        return ShotSignal(
            shot_id=path.stem,
            shot_path=str(path),
            sequence_name=path.stem,
            Neta_1=_safe_float(cavity.get("Neta_1")),
            Neta_2=_safe_float(cavity.get("Neta_2")),
            Neta_3=_safe_float(cavity.get("Neta_3")),
            Neta_4=_safe_float(cavity.get("Neta_4")),
            Neta_5=_safe_float(cavity.get("Neta_5")),
            chi_square_2=_safe_float(cavity.get("chi_square_2")),
            r_sq_2=_safe_float(cavity.get("r_sq_2")),
            atom_loading_globals=atom_loading,
            all_globals=all_globals,
        )


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None