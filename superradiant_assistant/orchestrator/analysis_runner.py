"""Analysis script runner — edits and runs lyse analysis scripts after shots complete.

For improved_cost_clean.py: edits parameter_str and y_op_string then runs it.
For calibrate_larmor_frequency_clean.py: runs it and extracts the correction value.
"""
from __future__ import annotations
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple

_YBCLOCK_PYTHON = r"C:\Users\radzi\AppData\Local\Anaconda3\envs\ybclock_3_11_24\python.exe"
_LABSCRIPT_ROOT = Path(
    r'C:\Users\radzi\Documents\labscript-suite\labscript-suite\userlib\labscriptlib\ybclock'
)


def edit_script_inplace(src_path: Path, edits: dict) -> None:
    """Edit the analysis script in-place so lyse picks up the changes."""
    txt = src_path.read_text(encoding="utf-8", errors="replace")
    for field, new_value in edits.items():
        if field == "delta_list":
            # Replace delta_list=[...] with the new value
            txt = re.sub(
                r'delta_list\s*=\s*\[[^\]]*\]',
                f'delta_list = [{new_value}]',
                txt,
            )
        else:
            pattern = rf'({re.escape(field)}\s*=\s*)["\']([^"\']*)["\']'
            new_txt = re.sub(pattern, rf'\g<1>"{new_value}"', txt)
            if new_txt == txt:
                pattern2 = rf'({re.escape(field)}\s*=\s*)([^\n#]+)'
                new_txt = re.sub(pattern2, rf'\g<1>"{new_value}"', txt, count=1)
            txt = new_txt
    src_path.write_text(txt, encoding="utf-8")


def _edit_script(src_path: Path, edits: dict) -> Path:
    """Copy script to a temp file and apply string substitutions for editable fields."""
    txt = src_path.read_text(encoding="utf-8", errors="replace")

    for field, new_value in edits.items():
        # Replace assignment: field = "old_value" or field = 'old_value'
        pattern = rf'({re.escape(field)}\s*=\s*)["\']([^"\']*)["\']'
        replacement = rf'\g<1>"{new_value}"'
        new_txt = re.sub(pattern, replacement, txt)
        if new_txt == txt:
            # Try without quotes (numeric or bare value)
            pattern2 = rf'({re.escape(field)}\s*=\s*)([^\n#]+)'
            replacement2 = rf'\g<1>"{new_value}"'
            new_txt = re.sub(pattern2, replacement2, txt, count=1)
        txt = new_txt

    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w",
                                       encoding="utf-8")
    tmp.write(txt)
    tmp.close()
    return Path(tmp.name)


def run_analysis_script(
    script_rel_path: str,
    edits: Optional[dict] = None,
    timeout: int = 60,
) -> Tuple[bool, str]:
    """Run an analysis script in the ybclock env and return (success, stdout)."""
    import os
    src = _LABSCRIPT_ROOT / script_rel_path
    if not src.exists():
        return False, f"Script not found: {src}"

    run_path = _edit_script(src, edits or {}) if edits else src

    env = os.environ.copy()
    _env_root = r"C:\Users\radzi\AppData\Local\Anaconda3\envs\ybclock_3_11_24"
    env["PATH"] = f"{_env_root}\\Library\\bin;" + env.get("PATH", "")
    env["QT_PLUGIN_PATH"] = f"{_env_root}\\Library\\plugins"

    try:
        result = subprocess.run(
            [_YBCLOCK_PYTHON, str(run_path)],
            capture_output=True, text=True, timeout=timeout, env=env
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        output = f"Script timed out after {timeout}s"
        success = False
    finally:
        if edits and run_path != src:
            try:
                run_path.unlink()
            except Exception:
                pass

    return success, output


def extract_larmor_correction(output: str) -> Optional[float]:
    """Parse the correction value from calibrate_larmor_frequency_clean.py output."""
    # Looks for: "Larmor Correction Needed: X" or "correction: X" or just a float
    patterns = [
        r'[Ll]armor\s+[Cc]orrection\s+[Nn]eeded[:\s]+([+-]?\d+(?:\.\d+)?)',
        r'[Cc]orrection[:\s]+([+-]?\d+(?:\.\d+)?)',
        r'[Cc]orrect\s+by[:\s]+([+-]?\d+(?:\.\d+)?)',
    ]
    for p in patterns:
        m = re.search(p, output)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None
