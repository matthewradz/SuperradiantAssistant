"""Runmanager interface that talks to the running GUI via a subprocess bridge."""
from __future__ import annotations
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from superradiant_assistant.safety import validate_writes

_BRIDGE = Path(__file__).resolve().parents[2] / "scripts" / "runmanager_bridge.py"

# Prefer the environment variable, otherwise fall back to the current Python
# environment, so a hard-coded path cannot break the import.
_CONDA_ENV = os.environ.get("CONDA_PREFIX",
                            os.path.expanduser(r"~\anaconda3\envs\python38"))
_YBCLOCK_PYTHON = os.path.join(_CONDA_ENV, "python.exe")
_LIB_BIN = os.path.join(_CONDA_ENV, "Library", "bin")


def _call(action: str, **kwargs) -> Any:
    cmd = {"action": action, **kwargs}
    full_env = os.environ.copy()
    full_env["PATH"] = _LIB_BIN + ";" + full_env.get("PATH", "")

    result = subprocess.run(
        [_YBCLOCK_PYTHON, str(_BRIDGE)],
        input=json.dumps(cmd),
        capture_output=True,
        text=True,
        env=full_env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"runmanager bridge error:\n{result.stderr.strip() or result.stdout.strip()}"
        )
    out = result.stdout.strip()
    if out:
        try:
            data = json.loads(out)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"runmanager bridge invalid JSON output: {out}") from e

        if not data.get("ok"):
            raise RuntimeError(f"runmanager bridge error: {data.get('error')}")
        return data.get("result")
    return None


class RunmanagerInterface:
    def __init__(
        self,
        sequence_file: Optional[str] = None,
        output_folder: Optional[str] = None,
    ):
        if sequence_file:
            _call("set_labscript_file", path=sequence_file)
        if output_folder:
            _call("set_shot_output_folder", path=output_folder)

    def set_globals(self, globals_to_set: Dict[str, Any]) -> None:
        _call("set_globals", globals=validate_writes(globals_to_set))

    def set_labscript_file(self, path: str) -> None:
        """Point runmanager at a different sequence file.

        Separate from the constructor so a session can switch experiments; setting
        it once at startup meant a multi-sequence run had to be clicked through
        in the GUI.
        """
        _call("set_labscript_file", path=path)

    def get_labscript_file(self) -> Optional[str]:
        return _call("get_labscript_file")

    def engage(self) -> None:
        err = _call("error_in_globals")
        if err:
            raise RuntimeError("runmanager reports errors in globals — not engaging")
        _call("engage")

    def set_globals_and_engage(self, globals_to_set: Dict[str, Any]) -> None:
        _call("set_globals_and_engage", globals=validate_writes(globals_to_set))

    def get_globals(self) -> Dict[str, Any]:
        res = _call("get_globals")
        # Debug print, to see in the terminal exactly what the bridge returned.
        # print(f"[DEBUG Bridge Raw Globals]: {res}") 
        return res or {}

    def n_shots(self) -> int:
        return _call("n_shots") or 0

    def compile_check(self, labscript_file: Optional[str] = None,
                       globals_file: str = r"C:\Experiments\Cesium\globals.h5"
                       ) -> Dict[str, Any]:
        """Compile a sequence in a throwaway process and report any traceback.

        The remote API exposes no compile status, and runmanager pipes compile
        errors to its GUI output pane without ever writing them to a log. This
        is therefore the only way to obtain the real reason a shot failed to
        build instead of inferring it from the absence of output files.
        """
        target = labscript_file or self.get_labscript_file()
        if not target:
            return {"compiles": False, "error": "no labscript file is loaded"}
        try:
            return _call("compile_check", labscript_file=target,
                         globals_file=globals_file) or {}
        except Exception as e:
            return {"compiles": None, "error": f"compile check unavailable: {e}"}