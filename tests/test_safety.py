"""Safety validation tests — the layer that stops an out-of-range value reaching hardware."""
from __future__ import annotations
import pytest

from superradiant_assistant.safety import (
    SafetyViolation, validate_scalar, validate_sweep, validate_write, validate_writes,
    parse_linspace, load_global_specs, tunable_global_names,
)


class TestScalarWrites:
    def test_in_range_value_accepted(self):
        assert validate_scalar("green_mot_frequency", 48.5) == 48.5

    def test_boundary_values_accepted(self):
        assert validate_scalar("y_bias_field_loading", 0.0) == 0.0
        assert validate_scalar("y_bias_field_loading", 0.5) == 0.5

    @pytest.mark.parametrize("value", [47.4, 49.6, 0.0, -100.0, 1e9])
    def test_out_of_range_refused(self, value):
        with pytest.raises(SafetyViolation, match="outside the safe range"):
            validate_scalar("green_mot_frequency", value)

    def test_unknown_parameter_refused(self):
        with pytest.raises(SafetyViolation, match="not a configured global"):
            validate_scalar("definitely_not_a_real_global", 1.0)

    def test_rangeless_list_param_refused(self):
        # clock_pi_resonance_frequency_list has null min/max: a scalar write to it
        # cannot be range-checked, so it must be refused rather than passed through.
        with pytest.raises(SafetyViolation, match="no declared min/max"):
            validate_scalar("clock_pi_resonance_frequency_list", 100.0)

    def test_rangeless_bool_allowed_when_allowlisted(self):
        assert validate_scalar("TD_loading", True) is True
        assert validate_scalar("TD_loading", False) is False

    def test_bool_param_rejects_number(self):
        with pytest.raises(SafetyViolation, match="declared bool"):
            validate_scalar("TD_loading", 1)

    def test_float_param_rejects_bool(self):
        with pytest.raises(SafetyViolation, match="got a bool"):
            validate_scalar("green_mot_frequency", True)

    def test_non_numeric_refused(self):
        with pytest.raises(SafetyViolation, match="non-numeric"):
            validate_scalar("green_mot_frequency", "48.5; rm -rf /")


class TestSweepValidation:
    def test_valid_sweep_accepted(self):
        validate_sweep("green_mot_frequency", 47.6, 49.4, 21)

    def test_endpoint_out_of_range_refused(self):
        with pytest.raises(SafetyViolation, match="outside the safe range"):
            validate_sweep("green_mot_frequency", 47.6, 60.0, 21)
        with pytest.raises(SafetyViolation, match="outside the safe range"):
            validate_sweep("green_mot_frequency", 10.0, 49.0, 21)

    @pytest.mark.parametrize("n", [1, 0, -5, 102, 10_000])
    def test_bad_point_count_refused(self, n):
        with pytest.raises(SafetyViolation, match="n_points"):
            validate_sweep("green_mot_frequency", 47.6, 49.4, n)

    def test_zero_width_sweep_refused(self):
        with pytest.raises(SafetyViolation, match="nothing to sweep"):
            validate_sweep("green_mot_frequency", 48.0, 48.0, 11)

    def test_unknown_sweep_parameter_refused(self):
        with pytest.raises(SafetyViolation, match="not a configured global"):
            validate_sweep("not_a_real_param_list", 1.0, 2.0, 11)

    def test_list_param_inherits_base_parameter_range(self, monkeypatch):
        # A *_list parameter carries no range; it is bounded by the scalar it
        # sweeps around. No such pair exists in the shipped config, so build one.
        import superradiant_assistant.safety as safety
        specs = safety.load_global_specs()
        specs["clock_pi_resonance_frequency"] = safety.GlobalSpec(
            name="clock_pi_resonance_frequency", lo=99.0, hi=101.0, type="float",
        )
        monkeypatch.setattr(safety, "load_global_specs", lambda: specs)

        safety.validate_sweep("clock_pi_resonance_frequency_list", 99.5, 100.5, 21)
        with pytest.raises(SafetyViolation, match="outside the safe range"):
            safety.validate_sweep("clock_pi_resonance_frequency_list", 99.5, 500.0, 21)

    def test_sweep_without_declared_range_is_reported_as_unbounded(self):
        # Documents a real gap: config.json gives neither
        # clock_pi_resonance_frequency_list nor a base scalar a min/max, so this
        # sweep cannot be range-checked and only the confirmation prompt guards it.
        from superradiant_assistant.safety import sweep_is_bounded
        assert sweep_is_bounded("clock_pi_resonance_frequency_list") is False
        assert sweep_is_bounded("green_mot_frequency") is True


class TestExpressionWrites:
    def test_linspace_parsed(self):
        assert parse_linspace("np.linspace(1.0, 2.0, 11)") == (1.0, 2.0, 11)
        assert parse_linspace("  np.linspace(-0.11, -0.09, 5)  ") == (-0.11, -0.09, 5)

    def test_non_linspace_expression_not_parsed(self):
        assert parse_linspace("np.arange(0, 10)") is None
        assert parse_linspace("48.5") is None

    def test_valid_linspace_write_accepted(self):
        expr = "np.linspace(47.6, 49.4, 11)"
        assert validate_write("green_mot_frequency", expr) == expr

    def test_out_of_range_linspace_refused(self):
        with pytest.raises(SafetyViolation, match="outside the safe range"):
            validate_write("green_mot_frequency", "np.linspace(47.6, 99.0, 11)")

    def test_arbitrary_expression_refused(self):
        # The bridge evaluates expression strings, so anything that isn't a
        # recognized sweep must not reach it.
        with pytest.raises(SafetyViolation, match="refusing to write the raw expression"):
            validate_write("green_mot_frequency", "__import__('os').system('calc')")

    def test_list_write_range_checked(self):
        assert validate_write("green_mot_frequency", [47.6, 48.0, 49.0]) == [47.6, 48.0, 49.0]
        with pytest.raises(SafetyViolation, match="outside the safe range"):
            validate_write("green_mot_frequency", [47.6, 48.0, 99.0])


class TestBatchWrites:
    def test_valid_batch_accepted(self):
        out = validate_writes({"green_mot_frequency": 48.5, "y_bias_field_loading": 0.2})
        assert out == {"green_mot_frequency": 48.5, "y_bias_field_loading": 0.2}

    def test_one_bad_entry_refuses_whole_batch(self):
        with pytest.raises(SafetyViolation):
            validate_writes({"green_mot_frequency": 48.5, "y_bias_field_loading": 99.0})

    def test_internal_bookkeeping_param_writable_by_code(self):
        # delta_duration tags sweep groups; it is absent from config.json so the
        # model can't name it, but the sweep coder must still be able to write it.
        assert validate_writes({"delta_duration": 1.6}) == {"delta_duration": 1.6}


class TestConfigContract:
    def test_tunable_globals_all_have_ranges(self):
        specs = load_global_specs()
        for name in tunable_global_names():
            assert specs[name].has_range

    def test_rangeless_globals_excluded_from_tunable(self):
        assert "TD_loading" not in tunable_global_names()
        assert "clock_pi_resonance_frequency_list" not in tunable_global_names()
