"""Parameter-file loader.

params.txt format (whitespace-separated):
    # name        min     max     init
    green_mot_frequency       47.5   49.0   48.3
    LS_blue_mot_intensity     0.20   0.45   0.38
    green_molasses_intensity  0.05   0.20   0.11
    spin_b_field_z            0.70   1.00   0.885

Lines starting with '#' or empty lines are ignored.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
from pydantic import BaseModel


class ParamSpec(BaseModel):
    name: str
    lo: float
    hi: float
    init: float

    def clip(self, v: float) -> float:
        return max(self.lo, min(self.hi, v))


def load_params(path: Path) -> List[ParamSpec]:
    specs: List[ParamSpec] = []
    text = Path(path).read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad params line: {raw!r}")
        name, lo, hi, init = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        specs.append(ParamSpec(name=name, lo=lo, hi=hi, init=init))
    return specs