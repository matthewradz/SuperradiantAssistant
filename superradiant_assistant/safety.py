"""Parameter safety validation — the single source of truth for what may be written
to the experiment hardware.

Every write path (the LabscriptAPI wrapper, the raw RunmanagerInterface, the sweep
coder) validates through this module, so there is no code path that can push an
out-of-range value to runmanager. Rules come from `config.json` -> `globals`, never
from a prompt: a model cannot name a parameter that isn't on the list, because the
tool schemas are generated from the same list.

Validation rules (docs/tool-schema.md T8):
  1. name must be a configured global
  2. if the global declares min/max, the value must lie in the closed interval
  3. if min/max are null, scalar writes are refused unless explicitly allow-listed
  4. the declared type must match
"""
from __future__ import annotations
import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from superradiant_assistant.config import CONFIG


class SafetyViolation(Exception):
    """A write was refused because it would violate a configured safety limit."""


# Globals with null min/max that may still be written as a scalar. Everything else
# with an undeclared range is refused, because a range-less parameter has no
# checkable safety envelope.
SCALAR_WRITE_ALLOWED_WITHOUT_RANGE: frozenset[str] = frozenset({"TD_loading"})

# Bookkeeping parameters written by orchestration code, not by the model. They are
# deliberately absent from config.json's `globals`, so they never appear in a tool
# enum and the model cannot name them — but code paths still need to write them,
# so they get their own ranges here.
_INTERNAL_GLOBALS: Dict[str, "GlobalSpec"] = {}

_LINSPACE_RE = re.compile(
    r"^\s*np\.linspace\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,)]+?)\s*\)\s*$"
)

MAX_SWEEP_POINTS = 101
MIN_SWEEP_POINTS = 2


@dataclass(frozen=True)
class GlobalSpec:
    name: str
    lo: Optional[float]
    hi: Optional[float]
    type: str
    description: str = ""
    #: Set for a parameter that wraps -- 360 for a phase in degrees. Its declared
    #: maximum then sits just below the period (359.9 here, because the driver
    #: rejects 360 exactly), which makes the obvious request, "sweep the phase
    #: from 0 to 360", fail by a tenth of a degree. See `wrap_sweep`.
    period: Optional[float] = None

    @property
    def has_range(self) -> bool:
        return self.lo is not None and self.hi is not None

    @property
    def is_periodic(self) -> bool:
        return self.period is not None and self.period > 0

    def describe_range(self) -> str:
        base = f"[{self.lo}, {self.hi}]" if self.has_range else "(no declared range)"
        # Saying so here puts it in the confirmation prompt and the tool schema,
        # which is where the operator and the model each need it.
        return base + (f", wraps at {self.period:g}" if self.is_periodic else "")


_INTERNAL_GLOBALS.update({
    "delta_duration": GlobalSpec(
        name="delta_duration", lo=0.0, hi=1000.0, type="float",
        description="Sweep-group tag, incremented per sweep so shots can be grouped.",
    ),
})


def load_global_specs() -> Dict[str, GlobalSpec]:
    """The model-visible parameter contract, straight from config.json."""
    specs: Dict[str, GlobalSpec] = {}
    for g in CONFIG.experiment_globals:
        name = g.get("name")
        if not name:
            continue
        lo, hi = g.get("min"), g.get("max")
        period = g.get("period")
        specs[name] = GlobalSpec(
            name=name,
            lo=float(lo) if lo is not None else None,
            hi=float(hi) if hi is not None else None,
            type=g.get("type", "float"),
            description=g.get("description", ""),
            period=float(period) if period else None,
        )
    return specs


def _writable_specs() -> Dict[str, GlobalSpec]:
    """Everything a write may target: model-visible globals plus internal bookkeeping."""
    return {**load_global_specs(), **_INTERNAL_GLOBALS}


def tunable_global_names() -> List[str]:
    """Globals with a declared range — the auto-optimization search space."""
    return [n for n, s in load_global_specs().items() if s.has_range]


def all_global_names() -> List[str]:
    return list(load_global_specs().keys())


# --------------------------------------------------------------------------
# Sequence files
# --------------------------------------------------------------------------

def allowed_sequence_files() -> List[str]:
    """Sequence files that may be loaded into runmanager, from config.json.

    An explicit file list, not a directory: a sibling file dropped next to a
    configured sequence is still refused, because nobody vetted it.
    """
    return [s["file"] for s in CONFIG.sequences if s.get("file")]


def _canonical_path(p: str) -> str:
    """Comparable form of a path. Windows spellings of the same file differ by
    separator and case, and config.json uses forward slashes."""
    return os.path.normcase(os.path.normpath(str(Path(p))))


def sequence_files_match(a: Optional[str], b: Optional[str]) -> bool:
    """Whether two paths name the same sequence file."""
    if not a or not b:
        return False
    return _canonical_path(a) == _canonical_path(b)


def validate_sequence_file(path: str) -> str:
    """Resolve `path` to its allow-listed config.json entry, or raise.

    Returns the spelling from config.json rather than the caller's, so runmanager
    always receives the vetted string.
    """
    allowed = allowed_sequence_files()
    if not allowed:
        raise SafetyViolation(
            "config.json lists no sequences, so no sequence file may be loaded."
        )
    want = _canonical_path(path)
    for entry in allowed:
        if _canonical_path(entry) == want:
            return entry
    raise SafetyViolation(
        f"'{path}' is not a configured sequence. Allowed: {allowed}"
    )


def _require_spec(name: str) -> GlobalSpec:
    spec = _writable_specs().get(name)
    if spec is None:
        raise SafetyViolation(
            f"'{name}' is not a configured global. Allowed: {sorted(load_global_specs())}"
        )
    return spec


def _coerce_typed(spec: GlobalSpec, value: Any) -> Any:
    if spec.type == "bool":
        if not isinstance(value, bool):
            raise SafetyViolation(
                f"'{spec.name}' is declared bool but got {type(value).__name__} ({value!r})"
            )
        return value
    if isinstance(value, bool):
        raise SafetyViolation(
            f"'{spec.name}' is declared {spec.type} but got a bool ({value!r})"
        )
    try:
        return float(value)
    except (TypeError, ValueError):
        raise SafetyViolation(
            f"'{spec.name}' is declared {spec.type} but got a non-numeric value ({value!r})"
        )


def validate_scalar(name: str, value: Any) -> Any:
    """Validate a single scalar write. Returns the coerced value, or raises."""
    spec = _require_spec(name)
    coerced = _coerce_typed(spec, value)

    if not spec.has_range:
        if name not in SCALAR_WRITE_ALLOWED_WITHOUT_RANGE:
            raise SafetyViolation(
                f"'{name}' has no declared min/max, so a scalar write cannot be range-checked "
                f"and is refused. Add min/max to config.json, or use a sweep if this is a "
                f"per-shot list parameter."
            )
        return coerced

    if not (spec.lo <= coerced <= spec.hi):
        raise SafetyViolation(
            f"'{name}' = {coerced} is outside the safe range {spec.describe_range()}"
        )
    return coerced


def sweep_base_spec(name: str) -> GlobalSpec:
    """The spec whose range bounds a sweep over `name`.

    Per-shot list parameters are named `<base>_list` and carry no range of their
    own; their values are bounded by the scalar parameter they sweep around.
    """
    if name.endswith("_list"):
        base = name[: -len("_list")]
        specs = _writable_specs()
        if base in specs and specs[base].has_range:
            return specs[base]
    return _require_spec(name)


def sweep_is_bounded(name: str) -> bool:
    """Whether a sweep over `name` can be range-checked at all.

    False means config.json declares no min/max for the parameter or its base
    scalar, so the only thing standing between a bad sweep and the hardware is
    the operator confirmation prompt. Callers should say so out loud.
    """
    try:
        return sweep_base_spec(name).has_range
    except SafetyViolation:
        return False


def wrap_sweep(name: str, start: float, end: float,
               n_points: int) -> tuple[float, float, int, str]:
    """Drop the duplicate endpoint of a full turn around a periodic parameter.

    `phase` runs 0-359.9, because the driver rejects 360 exactly. "Sweep the
    phase from 0 to 360" is then refused by a tenth of a degree -- and it is the
    obvious way to ask for a full rotation, so it was asked for twice in a row
    and refused twice, after which the agent gave up on `run_sweep` and stepped
    through the phases one shot at a time with a confirmation for each.

    For a periodic parameter the endpoint IS the start point, so 13 points from
    0 to 360 means 12 distinct phases 30 degrees apart. Returning that is what
    the request meant; refusing it is pedantry about a duplicate.

    Returns (start, end, n_points, note); `note` is empty when nothing changed.
    """
    try:
        spec = sweep_base_spec(name)
    except SafetyViolation:
        return start, end, n_points, ""
    if not (spec.is_periodic and spec.has_range) or n_points < 2:
        return start, end, n_points, ""

    span = end - start
    # Only a full turn, and only when the endpoint is what pushes it out of range.
    if abs(abs(span) - spec.period) > 1e-6 or spec.lo <= end <= spec.hi:
        return start, end, n_points, ""

    step = span / (n_points - 1)
    new_end = end - step
    new_n = n_points - 1
    if not (spec.lo <= new_end <= spec.hi) or new_n < MIN_SWEEP_POINTS:
        return start, end, n_points, ""

    return start, new_end, new_n, (
        f"'{name}' wraps at {spec.period:g}, so {start:g}..{end:g} in {n_points} "
        f"points repeats its first value at the end. Swept {start:g}..{new_end:g} "
        f"in {new_n} points instead — the same {abs(step):g} spacing, one full "
        f"turn, no duplicate. Report the range actually swept."
    )


def validate_sweep(name: str, start: float, end: float, n_points: int) -> None:
    """Validate a sweep before it is expanded into shots. Raises on violation."""
    _require_spec(name)

    if not isinstance(n_points, int) or isinstance(n_points, bool):
        raise SafetyViolation(f"n_points must be an integer, got {n_points!r}")
    if not (MIN_SWEEP_POINTS <= n_points <= MAX_SWEEP_POINTS):
        raise SafetyViolation(
            f"n_points={n_points} is outside [{MIN_SWEEP_POINTS}, {MAX_SWEEP_POINTS}]"
        )
    try:
        start_f, end_f = float(start), float(end)
    except (TypeError, ValueError):
        raise SafetyViolation(f"sweep bounds must be numeric, got {start!r}..{end!r}")
    if start_f == end_f:
        raise SafetyViolation(f"sweep start and end are both {start_f} — nothing to sweep")

    bound = sweep_base_spec(name)
    if bound.has_range:
        for label, v in (("start", start_f), ("end", end_f)):
            if not (bound.lo <= v <= bound.hi):
                raise SafetyViolation(
                    f"sweep {label}={v} for '{name}' is outside the safe range "
                    f"{bound.describe_range()} of '{bound.name}'"
                )


def parse_linspace(expr: str) -> Optional[tuple[float, float, int]]:
    """Parse an `np.linspace(a, b, n)` expression string into (a, b, n)."""
    m = _LINSPACE_RE.match(expr)
    if not m:
        return None
    try:
        a = float(ast.literal_eval(m.group(1)))
        b = float(ast.literal_eval(m.group(2)))
        n = int(ast.literal_eval(m.group(3)))
    except (ValueError, SyntaxError, TypeError):
        return None
    return a, b, n


def validate_write(name: str, value: Any) -> Any:
    """Validate any runmanager write, scalar or sweep expression.

    This is the chokepoint every write path funnels through. Sweep expressions are
    validated by their endpoints against the range of the parameter they sweep.
    """
    if isinstance(value, str):
        parsed = parse_linspace(value)
        if parsed is None:
            raise SafetyViolation(
                f"'{name}': refusing to write the raw expression {value!r}. Only "
                f"np.linspace(start, end, n) sweep expressions and plain scalars are allowed."
            )
        start, end, n = parsed
        validate_sweep(name, start, end, n)
        return value

    if isinstance(value, (list, tuple)):
        bound = sweep_base_spec(name)
        if not bound.has_range:
            raise SafetyViolation(
                f"'{name}': cannot range-check a list write against '{bound.name}' "
                f"(no declared min/max)"
            )
        if not (MIN_SWEEP_POINTS <= len(value) <= MAX_SWEEP_POINTS):
            raise SafetyViolation(
                f"'{name}': list of {len(value)} points is outside "
                f"[{MIN_SWEEP_POINTS}, {MAX_SWEEP_POINTS}]"
            )
        for v in value:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                raise SafetyViolation(f"'{name}': non-numeric list entry {v!r}")
            if not (bound.lo <= fv <= bound.hi):
                raise SafetyViolation(
                    f"'{name}': list value {fv} is outside the safe range "
                    f"{bound.describe_range()} of '{bound.name}'"
                )
        return list(value)

    return validate_scalar(name, value)


def validate_writes(globals_to_set: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a whole batch. All-or-nothing: one violation refuses the batch."""
    return {k: validate_write(k, v) for k, v in globals_to_set.items()}
