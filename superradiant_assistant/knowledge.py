"""Registry of sequences and analysis scripts the system knows about.

Phase 0: hard-coded.
Phase 1: an LLM can search this when planning.
Phase 2: auto-discover by scanning labscriptlib + analysis/scripts.
"""
from __future__ import annotations
from typing import Dict, List
from pydantic import BaseModel


class SequenceEntry(BaseModel):
    name: str            # sequence filename (without path)
    path: str            # full or relative path inside labscriptlib
    description: str
    tags: List[str] = []
    # Globals groups this sequence reads from runmanager
    globals_groups: List[str] = []


class AnalysisEntry(BaseModel):
    name: str            # analysis script filename
    path: str
    description: str
    # Where in HDF5 it writes its outputs
    results_subpath: str
    # Which scalar metrics it produces
    metrics: List[str] = []


SEQUENCES: Dict[str, SequenceEntry] = {
    "recycling_on_clock_transition_release_recap_FPGA_yellow_FNC.py": SequenceEntry(
        name="recycling_on_clock_transition_release_recap_FPGA_yellow_FNC.py",
        path="ybclock/sequences/cooling/recycling_on_clock_transition_release_recap_FPGA_yellow_FNC.py",
        description="Recycling sequence on clock transition with release-recap, used for atom loading and squeezing studies.",
        tags=["loading", "cavity", "clock", "squeezing"],
        globals_groups=["Atom Loading", "Squeezing", "Cavity Scan Parameters"],
    ),
}

ANALYSES: Dict[str, AnalysisEntry] = {
    "cavity_scan_analysis": AnalysisEntry(
        name="cavity_scan_analysis.py",
        path="ybclock/analysis/scripts/cavity_scan_analysis.py",
        description="Fits five cavity scans per shot; produces Neta_1..Neta_5 atom-number estimates.",
        results_subpath="cavity_scan_analysis",
        metrics=["Neta_1", "Neta_2", "Neta_3", "Neta_4", "Neta_5",
                 "r_sq_1", "r_sq_2", "r_sq_3", "r_sq_4", "r_sq_5",
                 "chi_square_2"],
    ),
    "cavity_photon_count_analysis": AnalysisEntry(
        name="cavity_photon_count_analysis.py",
        path="ybclock/analysis/scripts/cavity_photon_count_analysis.py",
        description="Photon count analysis for cavity probing.",
        results_subpath="cavity_photon_count_analysis",
        metrics=[],
    ),
    "empty_cavity_helper": AnalysisEntry(
        name="empty_cavity_helper",
        path="ybclock/analysis/scripts/(internal)",
        description="Empty cavity reference fit for frequency calibration.",
        results_subpath="empty_cavity_helper/fitted_exp_cavity_frequency_parameters",
        metrics=["fcavity_1", "kappa_1"],
    ),
}


def find_metric_owner(metric: str) -> AnalysisEntry | None:
    for a in ANALYSES.values():
        if metric in a.metrics:
            return a
    return None