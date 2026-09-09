"""Put an analysis routine in front of lyse.

Writing a routine to disk does nothing on its own: lyse only runs routines that
are in its Singleshot list, and its web API accepts exactly four requests --
'hello', 'get dataframe', and two forms of "add this shot". There is no remote
call to add or enable a routine, so the agent would write a perfectly good
analysis and then sit waiting for results that were never going to come.

What lyse does support is loading its whole state from a config file: the
routine list lives under `lyse_state/singleshot` as (filepath, checked) pairs.
So the agent writes that file, and the operator loads it from File -> Load
configuration. One click instead of hand-adding each routine, and the agent can
say precisely what it is asking for.
"""
from __future__ import annotations
import ast
import configparser
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from superradiant_assistant.config import CONFIG

CONFIG_DIR = (Path(CONFIG.labscript_suite_root) / "app_saved_configs"
              / "Cesium" / "lyse")
AGENT_CONFIG = CONFIG_DIR / "lyse-agent.ini"

Routine = Tuple[str, bool]


def _read_state(path: Path) -> dict:
    """Parse an existing lyse config, or return an empty state."""
    if not path.is_file():
        return {}
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(path, encoding="utf-8")
    except (OSError, configparser.Error):
        return {}
    if "lyse_state" not in parser:
        return {}
    state = {}
    for key, raw in parser["lyse_state"].items():
        try:
            state[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            continue
    return state


def current_routines(path: Optional[Path] = None,
                     kind: str = "singleshot") -> List[Routine]:
    """The routines of one kind recorded in a lyse config file."""
    state = _read_state(path or AGENT_CONFIG)
    return [tuple(r) for r in state.get(kind, [])]


def _write_state(path: Path, state: dict) -> None:
    from pprint import pformat
    parser = configparser.ConfigParser(interpolation=None)
    parser["lyse_state"] = {k: pformat(v) for k, v in state.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        parser.write(f)


def build_config(routines: Sequence[Routine],
                 path: Optional[Path] = None,
                 analysis_paused: bool = False,
                 kind: str = "singleshot") -> Path:
    """Write a lyse config whose `kind` list is exactly `routines`.

    Existing state in the file (window geometry, folders, the other routine
    list) is preserved so loading it does not disturb the rest of the setup.
    """
    target = path or AGENT_CONFIG
    state = _read_state(target)
    state[kind] = [(str(p), bool(active)) for p, active in routines]
    other = "multishot" if kind == "singleshot" else "singleshot"
    state.setdefault(other, [])
    state["analysis_paused"] = bool(analysis_paused)
    state.setdefault("lastsingleshotfolder", str(
        Path(CONFIG.labscript_suite_root) / "userlib" / "analysislib"
        / "Cesium" / "singleshot_routines"))
    _write_state(target, state)
    return target


def _lyse_client():
    """A client for lyse's ZMQ server, using the port from the lab config."""
    import lyse
    return lyse


#: A lyse that has not loaded the routine-management patch falls through to its
#: "assume it's a filepath" branch and answers as if a shot had been submitted.
#: Treating that as success would look like the routines had been set.
_STALE_LYSE_REPLIES = ("experiment added successfully", "added successfully")


def _is_stale_lyse(reply) -> bool:
    return isinstance(reply, str) and reply.strip().lower() in _STALE_LYSE_REPLIES


_RESTART_HINT = (
    "The running lyse does not have the routine-management request: it replied "
    "as though a shot had been submitted. lyse must be CLOSED AND REOPENED to "
    "pick up the patched communication.py. Until then, routines can only be "
    "changed in the lyse window."
)


def live_routines(kind: str = "singleshot") -> Optional[List[str]]:
    """Routines lyse is running right now, or None if it cannot be asked."""
    try:
        from superradiant_assistant.interfaces.runmanager_iface import _call
        reply = _call("lyse_get_routines", kind=kind)
    except Exception:
        return None
    if _is_stale_lyse(reply) or not isinstance(reply, list):
        return None
    return reply


def set_live_routines(paths: Sequence[str], replace: bool = True,
                      kind: str = "singleshot") -> str:
    """Set one of lyse's routine lists directly, without the GUI.

    `kind` is "singleshot" (runs per shot) or "multishot" (runs once over the
    whole sequence). Putting a whole-sweep plot in the singleshot box makes it
    re-run for every shot, which stalls the GUI.
    """
    from superradiant_assistant.interfaces.runmanager_iface import _call

    if kind not in ("singleshot", "multishot"):
        return f"refused: kind must be 'singleshot' or 'multishot', got {kind!r}"

    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        return f"refused: these routine files do not exist: {missing}"

    # What this call is about to displace. `replace=True` silently discarded
    # routines the operator had loaded by hand, and the only trace was a shorter
    # list in some later `get_lyse_routines`. Naming them forces it into the
    # tool result, where the model has to account for it.
    before = live_routines(kind) or [] if replace else []
    keeping = {Path(p).name for p in paths}
    dropped = [Path(p).name for p in before if Path(p).name not in keeping]

    entries = [[str(Path(p).resolve()), True] for p in paths]
    try:
        reply = _call("lyse_set_routines", routines=entries, replace=replace,
                      kind=kind)
    except Exception as e:
        return f"error talking to lyse: {type(e).__name__}: {e}\n{_RESTART_HINT}"

    _NOT_DONE = ("\nThis step is NOT complete. Do not mark it done, and do not "
                 "queue shots that depend on this routine until it is loaded — "
                 "they would produce files with no results.")
    if _is_stale_lyse(reply):
        return f"FAILED — routines were NOT changed. {_RESTART_HINT}{_NOT_DONE}"
    if isinstance(reply, str) and "not supported" in reply.lower():
        return f"FAILED — lyse refused: {reply}\n{_RESTART_HINT}{_NOT_DONE}"

    # The change is applied on lyse's main thread, so it is not in effect the
    # instant the request returns. Read the list back rather than assuming.
    import time
    wanted = {Path(p).resolve() for p in paths}
    for _ in range(10):
        time.sleep(0.4)
        now = live_routines(kind)
        if now is not None and wanted <= {Path(p).resolve() for p in now}:
            listing = "\n".join(f"    {Path(p).name}" for p in now)
            note = ""
            if dropped:
                note = (f"\nREPLACED and no longer running: {', '.join(dropped)}. "
                        f"The operator may have loaded those deliberately — say "
                        f"in your reply that you removed them.")
            return f"lyse now runs these {kind} routines:\n{listing}{note}"

    now = live_routines(kind)
    return (f"lyse accepted the request ({reply}) but its {kind} routine list "
            f"still reads {[Path(p).name for p in (now or [])]}. Check lyse.")


def request_routine(routine_path: str, active: bool = True,
                    replace: bool = False, kind: str = "singleshot") -> str:
    """Prepare a lyse config that includes `routine_path`, and say what to do.

    Used when the running lyse cannot be told directly. Returns operator-facing
    text: this cannot take effect on its own, and pretending otherwise is how a
    sweep ends up waiting on analysis lyse was never told to run.
    """
    p = Path(routine_path)
    if not p.is_file():
        return (f"refused: {p} does not exist. Write the analysis routine first, "
                f"then ask for it to be loaded.")

    existing = [] if replace else current_routines(kind=kind)
    kept = [(f, a) for f, a in existing if Path(f).resolve() != p.resolve()]
    routines = kept + [(str(p.resolve()), active)]

    # Carry the live singleshot list into the file too, so loading the config
    # to add a multishot routine does not silently drop the other box.
    if kind == "multishot":
        live_single = live_routines("singleshot") or []
        if live_single:
            build_config([(f, True) for f in live_single], kind="singleshot")
    target = build_config(routines, kind=kind)

    listing = "\n".join(
        f"    {'[x]' if a else '[ ]'} {Path(f).name}" for f, a in routines)
    return (
        f"This lyse does not accept {kind} routines over its API, so a "
        f"configuration has been written instead:\n"
        f"    {target}\n"
        f"{kind} routines it will activate:\n{listing}\n"
        f"ASK THE OPERATOR to load it in lyse: File -> Load configuration -> "
        f"select that file. Until they do, this routine will not run and no "
        f"plot or results will appear. Do not wait for results before they "
        f"confirm, and do not mark the step complete."
    )
