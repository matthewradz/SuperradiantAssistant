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

    # Every lyse result found in the shot, keyed by the name the analysis script
    # saved it under. The named fields above are one experiment's vocabulary
    # (ybclock's cavity scan); this carries whatever the current analysis emits
    # — CH3_mean, CH3_std, anything — so a new experiment needs no code change.
    metrics: Dict[str, float] = Field(default_factory=dict)

    # Globals snapshot
    atom_loading_globals: Dict[str, Any] = Field(default_factory=dict)
    all_globals: Dict[str, Any] = Field(default_factory=dict)    # all groups merged
    requested_globals: Dict[str, Any] = Field(default_factory=dict)  # what the coder requested

    success: bool = True
    notes: str = ""

    def __getattr__(self, name: str):
        """Fall back to `metrics` so dynamic results read like named fields.

        The orchestrator resolves a stage's target with
        `getattr(sig, stage.target_metric, None)`. Without this, any metric that
        is not one of the hard-coded ybclock fields resolves to None, and a
        threshold is silently never evaluated.
        """
        try:
            metrics = object.__getattribute__(self, "__dict__").get("metrics")
        except AttributeError:
            metrics = None
        if metrics and name in metrics:
            return metrics[name]
        raise AttributeError(name)

    def metric_value(self, name: str) -> Optional[float]:
        """Look up a metric by name, whether it is a named field or dynamic."""
        return getattr(self, name, None)

    def available_metrics(self) -> Dict[str, float]:
        """Every metric on this shot that actually has a value."""
        named = {
            k: getattr(self, k)
            for k in ("Neta_1", "Neta_2", "Neta_3", "Neta_4", "Neta_5",
                      "chi_square_2", "r_sq_2")
            if getattr(self, k) is not None
        }
        return {**named, **(self.metrics or {})}


class BatchSignal(BaseModel):
    """Compact summary of a batch of shots."""
    batch_id: str
    n_shots: int
    n_valid: int
    mean_metric: Optional[float] = None
    std_metric: Optional[float] = None
    metric_name: str = "Neta_2"
    notes: str = ""