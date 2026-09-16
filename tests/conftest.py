from __future__ import annotations
import pytest


# The parameter contract the safety tests validate against. Pinned here rather
# than read from the operator's config.json: these tests check the *validation
# logic*, so they must not start passing vacuously (or failing spuriously) just
# because the machine is currently configured for a different apparatus.
FIXTURE_GLOBALS = [
    {"name": "green_mot_frequency", "min": 47.5, "max": 49.5, "type": "float",
     "description": "Green MOT frequency on handoff to cavity (MHz)."},
    {"name": "y_bias_field_loading", "min": 0.0, "max": 0.5, "type": "float",
     "description": "Y-axis bias field during loading."},
    # Deliberately no `clock_pi_resonance_frequency` scalar: the *_list parameter
    # below must stay unbounded so the "sweep cannot be range-checked" case is
    # still covered. The test for base-range inheritance builds its own pair.
    {"name": "clock_pi_resonance_frequency_list", "min": None, "max": None, "type": "float",
     "description": "Per-shot clock frequency list (MHz); sweep-only."},
    {"name": "TD_loading", "min": None, "max": None, "type": "bool",
     "description": "True = 2D transverse loading."},
]


@pytest.fixture(autouse=True)
def pin_global_specs(monkeypatch):
    """Make the safety suite independent of whichever config.json is installed."""
    from superradiant_assistant.config import CONFIG
    monkeypatch.setattr(CONFIG, "experiment_globals", FIXTURE_GLOBALS)


@pytest.fixture(autouse=True)
def isolate_memory(tmp_path, monkeypatch):
    """Keep tests out of the real memory store.

    run_loop appends every shot to MEMORY, so without this the suite would write
    into the operator's actual lab notes.
    """
    from superradiant_assistant.memory import MEMORY
    monkeypatch.setattr(MEMORY, "root", tmp_path / "memory")
    monkeypatch.setattr(MEMORY, "memory_file", tmp_path / "memory" / "MEMORY.md")
    monkeypatch.setattr(MEMORY, "user_file", tmp_path / "memory" / "USER.md")
    monkeypatch.setattr(MEMORY, "history_file", tmp_path / "memory" / "history.jsonl")
