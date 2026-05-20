"""Compact result signals returned to the orchestrator after each shot/batch."""
from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ShotSignal(BaseModel):
    """Compact summary of a single shot for the orchestrator."""
    shot_id: str
    shot_path: str
    sequence_name: Optional[str] = None
    timestamp: Optional[str] = None

    # Atom-loading metrics (from results/cavity_scan_analysis attrs)
    Neta_1: Optional[float] = None
    Neta_2: Optional[float] = None
    Neta_3: Optional[float] = None
    Neta_4: Optional[float] = None
    Neta_5: Optional[float] = None

    # Per-scan fit quality
    chi_square_2: Optional[float] = None
    r_sq_2: Optional[float] = None

    # Globals snapshot
    atom_loading_globals: Dict[str, Any] = Field(default_factory=dict)
    all_globals: Dict[str, Any] = Field(default_factory=dict)  # all groups merged

    success: bool = True
    notes: str = ""


class BatchSignal(BaseModel):
    """Compact summary of a batch of shots."""
    batch_id: str
    n_shots: int
    n_valid: int
    mean_metric: Optional[float] = None
    std_metric: Optional[float] = None
    metric_name: str = "Neta_2"
    notes: str = ""