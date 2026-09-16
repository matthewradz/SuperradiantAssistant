"""Read labscript HDF5 shots and turn them into ShotSignal objects."""
from __future__ import annotations
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional
import h5py
import numpy as np

from superradiant_assistant.signals import ShotSignal


def list_shots(root: Path) -> List[Path]:
    """Every shot under `root`, oldest first.

    Ordered by modification time rather than filename: a data folder mixes
    labscript's date-prefixed names with other conventions, and under filename
    sorting `2026-07-31_0003_...h5` lands before `shot_0000.h5`, so callers
    taking the last N would silently miss the newest shots.
    """
    return sorted(Path(root).rglob("*.h5"), key=lambda p: p.stat().st_mtime)


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


def read_all_results(f: h5py.File) -> Dict[str, float]:
    """Every numeric attr under /results, from every analysis routine.

    lyse writes each routine's `save_result` calls into `/results/<name>`, and
    the routine name is the analysis script's, not something this project picks.
    Reading only one hard-coded group meant an experiment whose analysis had a
    different name reported no metrics at all — which in turn made every
    threshold check unreachable.
    """
    out: Dict[str, float] = {}
    if "results" not in f:
        return out
    for group_name in f["results"]:
        node = f[f"results/{group_name}"]
        attrs = getattr(node, "attrs", {})
        for key, raw in attrs.items():
            value = _safe_float(_decode(raw))
            if value is not None:
                # Later groups win only if the earlier one had no usable value.
                out.setdefault(key, value)
    return out


def resolve_per_shot_globals(f: h5py.File, globals_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Replace swept globals with this shot's own value.

    labscript stores the *expression* in `globals/<group>` attrs, so a swept
    parameter reads back as the whole `np.linspace(...)` list on every shot of
    the sweep. The shot's position in the sweep lives in the root attrs as
    `run number` / `n_runs`, so index the expansion with it.

    Only applies when the list length matches `n_runs`, which is what makes it
    an expansion of this sweep rather than a parameter that is genuinely a list.
    """
    try:
        run_number = int(f.attrs["run number"])
        n_runs = int(f.attrs["n_runs"])
    except (KeyError, TypeError, ValueError):
        return globals_dict
    if n_runs <= 1:
        return globals_dict

    out = dict(globals_dict)
    for name, value in globals_dict.items():
        # np.linspace(...) evaluates to an ndarray, not a list — checking only
        # list/tuple silently left every swept value as the whole array.
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)) and len(value) == n_runs:
            if 0 <= run_number < n_runs:
                out[name] = value[run_number]
    return out


def read_shot(path: Path) -> ShotSignal:
    path = Path(path)
    with h5py.File(path, "r") as f:
        cavity = read_results_attrs(f, "cavity_scan_analysis")
        metrics = read_all_results(f)

        atom_loading: Dict[str, Any] = {}
        all_globals: Dict[str, Any] = {}
        if "globals" in f:
            for group_name in f["globals"]:
                grp = _read_attrs(f[f"globals/{group_name}"])
                all_globals.update(grp)
                if group_name == "Atom Loading":
                    atom_loading = grp
        all_globals = resolve_per_shot_globals(f, all_globals)
        atom_loading = resolve_per_shot_globals(f, atom_loading)

        return ShotSignal(
            shot_id=path.stem,
            shot_path=str(path),
            sequence_name=path.stem,
            Neta_1=_safe_float(cavity.get("Neta_1", metrics.get("Neta_1"))),
            Neta_2=_safe_float(cavity.get("Neta_2", metrics.get("Neta_2"))),
            Neta_3=_safe_float(cavity.get("Neta_3", metrics.get("Neta_3"))),
            Neta_4=_safe_float(cavity.get("Neta_4", metrics.get("Neta_4"))),
            Neta_5=_safe_float(cavity.get("Neta_5", metrics.get("Neta_5"))),
            chi_square_2=_safe_float(cavity.get("chi_square_2", metrics.get("chi_square_2"))),
            r_sq_2=_safe_float(cavity.get("r_sq_2", metrics.get("r_sq_2"))),
            metrics=metrics,
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