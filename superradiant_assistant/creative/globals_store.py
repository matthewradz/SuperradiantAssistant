"""Create a new experiment parameter, once the operator has approved it.

A global has to exist in two places to be usable: runmanager's globals HDF5
file (so a shot can read it) and the agent's config.json (so the safety layer
knows its range). Creating only the first would give the model a parameter it
can drive with no range check at all, so both writes happen here or neither
does.

runmanager.remote deliberately has no "create global" command — it can only set
names that already exist — so the file itself is edited through runmanager's own
file API, in the environment where that library lives.
"""
from __future__ import annotations
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from superradiant_assistant.config import CONFIG, REPO_ROOT

CONFIG_PATH = REPO_ROOT / "config.json"

# Where runmanager keeps its globals. Discovered from the GUI when possible;
# this is the fallback for a fresh session.
DEFAULT_GLOBALS_FILE = r"C:\Experiments\Cesium\globals.h5"
DEFAULT_GROUP = "AWG"


@dataclass
class GlobalProposal:
    name: str
    minimum: float
    maximum: float
    initial: float
    units: str = ""
    description: str = ""
    reason: str = ""
    group: str = DEFAULT_GROUP
    globals_file: str = DEFAULT_GLOBALS_FILE

    def preview(self) -> str:
        """Exactly what will be created, for the y/N prompt."""
        return "\n".join([
            f"name        : {self.name}",
            f"range       : [{self.minimum:g}, {self.maximum:g}] {self.units}".rstrip(),
            f"initial     : {self.initial:g} {self.units}".rstrip(),
            f"description : {self.description or '(none)'}",
            f"reason      : {self.reason or '(none given)'}",
            f"writes to   : {self.globals_file}  group '{self.group}'",
            f"              {CONFIG_PATH}  (range enforced from here)",
        ])


def validate(p: GlobalProposal) -> Optional[str]:
    """Reject a proposal that cannot safely become a parameter."""
    if not p.name or not p.name.isidentifier():
        return f"'{p.name}' is not a valid Python identifier"
    if p.name in {g.get("name") for g in CONFIG.experiment_globals}:
        return f"'{p.name}' is already declared in config.json"
    try:
        lo, hi, init = float(p.minimum), float(p.maximum), float(p.initial)
    except (TypeError, ValueError):
        return "min, max and initial must all be numeric"
    if lo >= hi:
        return f"min ({lo}) must be below max ({hi})"
    if not (lo <= init <= hi):
        return f"initial value {init} is outside [{lo}, {hi}]"
    return None


def _add_to_config(p: GlobalProposal) -> None:
    """Append the declaration to config.json, preserving formatting elsewhere."""
    text = io.open(CONFIG_PATH, encoding="utf-8").read()
    cfg = json.loads(text)
    cfg.setdefault("globals", []).append({
        "name": p.name,
        "min": float(p.minimum),
        "max": float(p.maximum),
        "type": "float",
        "description": (p.description or "").strip()
                       + (f" Units: {p.units}." if p.units else "")
                       + " Added in creative mode.",
    })
    io.open(CONFIG_PATH, "w", encoding="utf-8").write(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"
    )


def _add_to_runmanager(p: GlobalProposal) -> Dict[str, Any]:
    from superradiant_assistant.interfaces.runmanager_iface import _call
    return _call("new_global", file=p.globals_file, group=p.group,
                 name=p.name, value=p.initial, units=p.units)


def create(p: GlobalProposal) -> str:
    """Create the global in both places. Call only after operator approval.

    config.json is written first: if the runmanager write then fails, the agent
    is left knowing about a parameter it cannot set, which surfaces as a clear
    error. The reverse order would leave a live, unbounded parameter.
    """
    problem = validate(p)
    if problem:
        return f"refused: {problem}"

    _add_to_config(p)
    try:
        result = _add_to_runmanager(p)
    except Exception as e:
        return (f"partial: '{p.name}' was declared in config.json but could NOT be "
                f"created in runmanager ({type(e).__name__}: {e}). Add it by hand in "
                f"the GUI, or remove the config.json entry.")

    return (f"created '{p.name}' = {p.initial:g}{(' ' + p.units) if p.units else ''} "
            f"in group '{p.group}', range [{p.minimum:g}, {p.maximum:g}] enforced. "
            f"RELOAD the globals file in runmanager (File -> Revert/reload) before "
            f"the next shot, or the GUI will not see it. "
            f"The agent needs a RESTART to pick up the new range. Detail: {result}")
