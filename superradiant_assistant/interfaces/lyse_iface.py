"""Interface to the running lyse GUI — adds shots via ZMQ bridge."""
from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path
from typing import List

_BRIDGE = Path(__file__).resolve().parents[2] / "scripts" / "lyse_bridge.py"
_YBCLOCK_PYTHON = r"C:\Users\radzi\AppData\Local\Anaconda3\envs\ybclock_3_11_24\python.exe"
_LIB_BIN = r"C:\Users\radzi\AppData\Local\Anaconda3\envs\ybclock_3_11_24\Library\bin"


def _call(action: str, **kwargs):
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
        raise RuntimeError(f"lyse bridge error:\n{result.stderr.strip() or result.stdout.strip()}")
    out = result.stdout.strip()
    if out:
        data = json.loads(out)
        if not data.get("ok"):
            raise RuntimeError(f"lyse bridge: {data.get('error')}")
        return data.get("result")
    return None


def is_lyse_running() -> bool:
    try:
        _call("hello")
        return True
    except Exception:
        return False


def add_shots(filepaths: List[str]) -> int:
    """Add shot HDF5 files to the lyse filebox. Returns number successfully added."""
    result = _call("add_shots", filepaths=filepaths)
    return int(result) if result is not None else 0
