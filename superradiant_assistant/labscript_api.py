"""The wrapper the agent talks to instead of touching labscript directly.

Every hardware read/write the agent can perform goes through this object, and
every write is validated against the ranges in config.json before it reaches
runmanager. When the labscript GUIs aren't running this raises rather than
returning mock data — silently faking a successful write would let an operator
believe the machine was reconfigured when it wasn't.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from superradiant_assistant.config import CONFIG
from superradiant_assistant.safety import (
    SafetyViolation,
    load_global_specs,
    sequence_files_match,
    validate_scalar,
    validate_sequence_file,
    validate_sweep,
    validate_writes,
)


class LabscriptUnavailable(RuntimeError):
    """The labscript suite isn't reachable (GUIs not running / bridge failed)."""


class LabscriptAPI:
    """Range-enforced facade over RunmanagerInterface."""

    def __init__(self, sequence_file: Optional[str] = None,
                 output_folder: Optional[str] = None):
        self._sequence_file = sequence_file
        self._output_folder = output_folder or str(CONFIG.historical_data_root)
        self._rm = None

    @property
    def runmanager(self):
        if self._rm is None:
            from superradiant_assistant.interfaces.runmanager_iface import RunmanagerInterface
            try:
                self._rm = RunmanagerInterface(
                    sequence_file=self._sequence_file,
                    output_folder=self._output_folder,
                )
            except Exception as e:
                raise LabscriptUnavailable(
                    "labscript suite is not reachable. Open runmanager, BLACS and lyse "
                    f"(scripts\\launch_lab.bat) and retry. Underlying error: {e}"
                ) from e
        return self._rm

    def get_labscript_file(self) -> Optional[str]:
        """The sequence file runmanager currently has loaded."""
        try:
            return self.runmanager.get_labscript_file()
        except LabscriptUnavailable:
            raise
        except Exception as e:
            raise LabscriptUnavailable(
                f"could not read the loaded sequence from runmanager: {e}"
            ) from e

    def set_labscript_file(self, path: str) -> Dict[str, Any]:
        """Load an allow-listed sequence into runmanager, then verify it took.

        The read-back is the point. Setting without verifying would let the agent
        believe it had switched experiments when it hadn't, and every later shot
        would run the old sequence under the new sequence's parameter names —
        silent, and expensive in machine time.

        Globals are deliberately left alone: the new sequence may read a
        different set, and guessing values for them is not this method's job.
        """
        vetted = validate_sequence_file(path)

        old = None
        try:
            old = self.runmanager.get_labscript_file()
        except LabscriptUnavailable:
            raise
        except Exception:
            pass  # best-effort; the write is still validated and verified below

        if sequence_files_match(old, vetted):
            return {"old": old, "new": vetted, "changed": False}

        self.runmanager.set_labscript_file(vetted)

        actual = self.get_labscript_file()
        if not sequence_files_match(actual, vetted):
            raise RuntimeError(
                f"runmanager still reports {actual!r} after being told to load "
                f"{vetted!r} — not proceeding. The GUI may be busy, or the path "
                f"may not exist on the runmanager machine."
            )
        return {"old": old, "new": vetted, "changed": True}

    def describe_globals(self) -> List[Dict[str, Any]]:
        """The parameter contract: what may be set, and within what bounds."""
        return [
            {
                "name": s.name,
                "min": s.lo,
                "max": s.hi,
                "type": s.type,
                "description": s.description,
                "tunable": s.has_range,
            }
            for s in load_global_specs().values()
        ]

    def get_globals(self, names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Read current values, filtered to the configured whitelist."""
        values, _ = self.read_globals(names)
        return values

    def read_globals(
        self, names: Optional[List[str]] = None
    ) -> tuple[Dict[str, Any], List[str]]:
        """Read current values, reporting what the whitelist filtered out.

        Returns (values, dropped): `values` holds only globals described in
        config.json, `dropped` names everything else runmanager reported. Callers
        surface `dropped` so an empty result can be attributed correctly — "the
        GUI has no globals loaded" and "the GUI's globals belong to a different
        apparatus than config.json describes" are very different problems, and
        collapsing both into "read 0 globals" sends the reader hunting in the
        wrong place.
        """
        specs = load_global_specs()
        if names:
            unknown = [n for n in names if n not in specs]
            if unknown:
                raise SafetyViolation(
                    f"unknown globals {unknown}. Allowed: {sorted(specs)}"
                )
        try:
            current = self.runmanager.get_globals()
        except LabscriptUnavailable:
            raise
        except Exception as e:
            raise LabscriptUnavailable(f"could not read globals from runmanager: {e}") from e

        wanted = set(names) if names else set(specs)
        values = {k: v for k, v in current.items() if k in wanted}
        dropped = sorted(k for k in current if k not in wanted)
        return values, dropped

    def set_global(self, name: str, value: Any) -> Dict[str, Any]:
        """Validated single-parameter write. Returns {name, old, new}."""
        coerced = validate_scalar(name, value)
        old = None
        try:
            old = self.runmanager.get_globals().get(name)
        except LabscriptUnavailable:
            raise
        except Exception:
            pass  # reading the prior value is best-effort; the write still validates
        self.runmanager.set_globals({name: coerced})
        return {"name": name, "old": old, "new": coerced}

    def set_globals(self, globals_to_set: Dict[str, Any]) -> Dict[str, Any]:
        """Validated batch write. One violation refuses the whole batch."""
        validated = validate_writes(globals_to_set)
        self.runmanager.set_globals(validated)
        return validated

    def queue_sweep(self, name: str, start: float, end: float, n_points: int) -> str:
        """Validated sweep. runmanager expands the linspace into n_points shots."""
        validate_sweep(name, start, end, n_points)
        expr = f"np.linspace({start}, {end}, {n_points})"
        self.runmanager.set_globals({name: expr})
        return expr

    def engage(self) -> None:
        self.runmanager.engage()

    def n_shots(self) -> int:
        return self.runmanager.n_shots()
