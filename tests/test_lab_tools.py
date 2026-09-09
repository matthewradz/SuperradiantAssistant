"""Tool-layer tests: permission whitelist, gate behaviour, and input validation."""
from __future__ import annotations
import json

import pytest

from superradiant_assistant.hooks import ToolGate
from superradiant_assistant.tools.lab_tools import (
    build_tool_specs, run_optimization, run_sweep, set_runmanager_global,
    analyze_results, METRIC_NAMES,
)
from superradiant_assistant.tools.registry import ToolRegistry, ToolSpec


@pytest.fixture
def gate(tmp_path):
    return ToolGate(
        audit_path=tmp_path / "audit.jsonl",
        session_id="test",
        require_confirm=frozenset({"run_optimization", "run_sweep"}),
        always_confirm=frozenset({"set_runmanager_global"}),
        audit=frozenset({"run_optimization", "set_runmanager_global"}),
        auto_approve=True,
    )


@pytest.fixture
def registry(gate):
    return ToolRegistry(build_tool_specs(), gate=gate)


class TestPermissionMatrix:
    """docs/tool-schema.md §5 — the answer agent must never reach hardware."""

    def test_answer_agent_has_no_hardware_tools(self, registry):
        allowed = set(registry.names_for("answer"))
        assert allowed == {"search_lab_knowledge", "load_skill", "read_shot_results"}
        for forbidden in ("run_optimization", "run_sweep", "set_runmanager_global"):
            assert forbidden not in allowed

    def test_only_lead_may_write_globals(self, registry):
        for agent in ("planner", "coder", "answer"):
            assert "set_runmanager_global" not in registry.names_for(agent)
        assert "set_runmanager_global" in registry.names_for("lead")

    def test_planner_cannot_queue_shots(self, registry):
        allowed = set(registry.names_for("planner"))
        assert "run_optimization" not in allowed
        assert "run_sweep" not in allowed

    def test_dispatch_enforces_whitelist_not_just_schema(self, registry):
        # The whitelist is checked at dispatch, so a model that names a tool it
        # was never offered still cannot run it.
        result = registry.dispatch("answer", "set_runmanager_global",
                                    {"name": "green_mot_frequency", "value": 48.0,
                                     "reason": "test"})
        assert "not permitted" in result


class TestGate:
    def test_denied_call_does_not_reach_handler(self, tmp_path):
        called = []
        spec = ToolSpec(
            name="dangerous", description="", parameters={"type": "object", "properties": {}},
            handler=lambda: called.append(1) or "ran", allowed_agents=frozenset({"lead"}),
        )
        gate = ToolGate(audit_path=tmp_path / "a.jsonl", session_id="t",
                        always_confirm=frozenset({"dangerous"}), auto_approve=False)
        reg = ToolRegistry([spec], gate=gate)
        result = reg.dispatch("lead", "dangerous", {})
        assert "refused" in result
        assert not called, "handler ran despite the gate denying the call"

    def test_non_interactive_run_fails_closed(self, tmp_path):
        gate = ToolGate(audit_path=tmp_path / "a.jsonl", session_id="t",
                        always_confirm=frozenset({"x"}), auto_approve=False)
        decision = gate.before_tool_call("x", {})
        assert decision.denied

    def test_audit_record_written(self, tmp_path):
        gate = ToolGate(audit_path=tmp_path / "audit.jsonl", session_id="sess-1",
                        audit=frozenset({"tracked"}), auto_approve=True)
        gate.after_tool_call("tracked", {"name": "x", "value": 1}, result="ok")
        lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
        record = json.loads(lines[0])
        assert record["tool"] == "tracked"
        assert record["session_id"] == "sess-1"
        assert record["args"] == {"name": "x", "value": 1}
        assert "ts" in record

    def test_untracked_tool_not_audited(self, tmp_path):
        gate = ToolGate(audit_path=tmp_path / "audit.jsonl", session_id="s",
                        audit=frozenset({"other"}), auto_approve=True)
        gate.after_tool_call("readonly_thing", {}, result="ok")
        assert not (tmp_path / "audit.jsonl").exists()


class TestSetGlobalValidation:
    def test_out_of_range_refused_before_any_hardware_call(self):
        r = set_runmanager_global("green_mot_frequency", 99.0, "test")
        assert r.startswith("refused")
        assert "outside the safe range" in r

    def test_unknown_parameter_refused(self):
        r = set_runmanager_global("nonexistent_param", 1.0, "test")
        assert r.startswith("error")
        assert "not a settable global" in r

    def test_rangeless_list_param_refused(self):
        r = set_runmanager_global("clock_pi_resonance_frequency_list", 100.0, "test")
        assert r.startswith("refused")

    def test_internal_bookkeeping_param_not_settable_by_model(self):
        # delta_duration is writable by code but must not be model-addressable.
        r = set_runmanager_global("delta_duration", 1.0, "test")
        assert "not a settable global" in r


class TestRunOptimizationValidation:
    def test_bad_metric_refused(self):
        r = run_optimization("not_a_metric", 700, "seq.py", "spec", "reason")
        assert r.startswith("error")

    def test_bad_sequence_refused(self):
        r = run_optimization("Neta_2", 700, "definitely/not/a/sequence.py", "spec", "reason")
        assert r.startswith("error")
        assert "sequence_file" in r

    @pytest.mark.parametrize("threshold", [float("inf"), float("nan"), float("-inf")])
    def test_non_finite_threshold_refused(self, threshold):
        from superradiant_assistant.config import CONFIG
        seq = CONFIG.sequences[0]["file"]
        r = run_optimization("Neta_2", threshold, seq, "spec", "reason")
        assert r.startswith("error")
        assert "finite" in r

    def test_bad_threshold_op_refused(self):
        from superradiant_assistant.config import CONFIG
        seq = CONFIG.sequences[0]["file"]
        r = run_optimization("Neta_2", 700, seq, "spec", "reason", threshold_op="=~")
        assert r.startswith("error")


class TestRunSweepValidation:
    def _seq(self):
        from superradiant_assistant.config import CONFIG
        return CONFIG.sequences[0]["file"]

    def test_unknown_sweep_param_refused(self):
        r = run_sweep("not_a_param", "explicit", self._seq(), "spec", "reason",
                      start=1, end=2, n_points=11)
        assert r.startswith("error")

    def test_centered_mode_requires_range_and_step(self):
        r = run_sweep("green_mot_frequency", "centered", self._seq(), "spec", "reason")
        assert r.startswith("error")
        assert "range_mhz" in r

    def test_non_positive_step_refused(self):
        r = run_sweep("green_mot_frequency", "centered", self._seq(), "spec", "reason",
                      range_mhz=1.0, step_mhz=0)
        assert r.startswith("error")

    def test_too_many_points_refused(self):
        r = run_sweep("green_mot_frequency", "centered", self._seq(), "spec", "reason",
                      range_mhz=1.0, step_mhz=0.0001)
        assert r.startswith("error")
        assert "points" in r

    def test_explicit_mode_zero_width_refused(self):
        r = run_sweep("green_mot_frequency", "explicit", self._seq(), "spec", "reason",
                      start=48.0, end=48.0, n_points=11)
        assert r.startswith("error")

    def test_explicit_mode_point_count_bounded(self):
        r = run_sweep("green_mot_frequency", "explicit", self._seq(), "spec", "reason",
                      start=47.6, end=49.4, n_points=5000)
        assert r.startswith("error")
        assert "n_points" in r

    def test_bad_mode_refused(self):
        r = run_sweep("green_mot_frequency", "wobble", self._seq(), "spec", "reason")
        assert r.startswith("error")


class TestAnalyzeValidation:
    def test_resonance_requires_sweep_param_and_ratio(self):
        assert analyze_results("resonance").startswith("error")
        assert analyze_results("resonance", sweep_param="green_mot_frequency").startswith("error")

    def test_bad_plot_ratio_format_refused(self):
        r = analyze_results("resonance", sweep_param="green_mot_frequency",
                            plot_ratio="Neta_5 over Neta_4")
        assert r.startswith("error")

    def test_plot_ratio_must_use_real_metrics(self):
        r = analyze_results("resonance", sweep_param="green_mot_frequency",
                            plot_ratio="foo/bar")
        assert r.startswith("error")

    def test_unknown_analysis_type_refused(self):
        assert analyze_results("astrology").startswith("error")


class TestSchemaGeneration:
    def test_enums_come_from_config_not_prompt(self):
        from superradiant_assistant.safety import all_global_names
        specs = {s.name: s for s in build_tool_specs()}
        param_enum = specs["set_runmanager_global"].parameters["properties"]["name"]["enum"]
        assert set(param_enum) == set(all_global_names())

    def test_metric_enums_match_shot_signal_fields(self):
        from superradiant_assistant.signals import ShotSignal
        for m in METRIC_NAMES:
            assert m in ShotSignal.model_fields
