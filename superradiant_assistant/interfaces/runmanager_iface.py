"""Runmanager interface that talks to the running GUI via a subprocess bridge.

The bridge (scripts/runmanager_bridge.py) runs in the ybclock_3_11_24 conda env
which has runmanager.remote installed. This file runs in the superradiant env.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

_BRIDGE = Path(__file__).resolve().parents[2] / "scripts" / "runmanager_bridge.py"
_YBCLOCK_PYTHON = r"C:\Users\radzi\AppData\Local\Anaconda3\envs\ybclock_3_11_24\python.exe"
_LIB_BIN = r"C:\Users\radzi\AppData\Local\Anaconda3\envs\ybclock_3_11_24\Library\bin"


def _call(action: str, **kwargs) -> Any:
    cmd = {"action": action, **kwargs}
    env_patch = {"PATH": f"{_LIB_BIN};"}  # prepend so DLLs resolve
    import os
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
        data = json.loads(out)
        if not data.get("ok"):
            raise RuntimeError(f"runmanager bridge: {data.get('error')}")
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
        _call("set_globals", globals=globals_to_set)

    def engage(self) -> None:
        err = _call("error_in_globals")
        if err:
            raise RuntimeError("runmanager reports errors in globals — not engaging")
        _call("engage")

    def set_globals_and_engage(self, globals_to_set: Dict[str, Any]) -> None:
        _call("set_globals_and_engage", globals=globals_to_set)

    def get_globals(self) -> Dict[str, Any]:
        return _call("get_globals") or {}

    def n_shots(self) -> int:
        return _call("n_shots") or 0
