"""Goal-mode tests.

Goal mode is the expensive setting: it keeps re-planning and re-shooting until
the threshold is met. Switched off, an optimize stage takes exactly one shot and
reports whether the threshold was met. These tests pin that difference.
"""
from __future__ import annotations

import pytest

from superradiant_assistant.goal import Goal, Stage
from superradiant_assistant.orchestrator.developer import ShotRequest
from superradiant_assistant.orchestrator.loop import run_loop
from superradiant_assistant.signals import ShotSignal


class CountingExecutor:
    """Returns a fixed metric value and counts how many shots were taken."""

    def __init__(self, metric_value: float, metric: str = "Neta_2"):
        self.metric_value = metric_value
        self.metric = metric
        self.shots = 0

    def execute(self, req: ShotRequest) -> ShotSignal:
        self.shots += 1
        return ShotSignal(
            shot_id=f"shot_{self.shots}", shot_path="",
            requested_globals=dict(req.globals_to_set),
            **{self.metric: self.metric_value},
        )


def _stage(threshold: float = 700.0, max_iterations: int = 10) -> Stage:
    return Stage(
        name="opt", task_type="optimize", stage_kind="optimize",
        target_metric="Neta_2", threshold=threshold, threshold_op=">",
        sequence_file="seq.py", max_iterations=max_iterations,
    )


def _run(goal_mode: bool, metric_value: float, tmp_path, max_iterations: int = 10):
    goal = Goal(stages=[_stage(max_iterations=max_iterations)], goal_mode=goal_mode)
    ex = CountingExecutor(metric_value)
    summary = run_loop(goal=goal, data_root=tmp_path, repo_root=tmp_path, executor=ex)
    return goal.stages[0], ex, summary


class TestGoalModeDefault:
    def test_defaults_to_on(self):
        assert Goal().goal_mode is True

    def test_can_be_switched_off(self):
        assert Goal(goal_mode=False).goal_mode is False


class TestGoalModeOff:
    def test_takes_exactly_one_shot_when_threshold_missed(self, tmp_path):
        stage, ex, _ = _run(False, 100.0, tmp_path, max_iterations=10)
        assert ex.shots == 1, "goal mode off must not iterate"
        assert stage.result["stop_reason"] == "single_shot_done"

    def test_missing_the_threshold_is_not_a_failure(self, tmp_path):
        # The run did what was asked; the target simply wasn't reached.
        stage, _, _ = _run(False, 100.0, tmp_path)
        assert stage.status == "complete"
        assert stage.result["threshold_met"] is False

    def test_reports_when_the_threshold_was_met(self, tmp_path):
        stage, ex, _ = _run(False, 900.0, tmp_path)
        assert ex.shots == 1
        assert stage.result["threshold_met"] is True

    def test_high_max_iterations_is_still_capped_at_one(self, tmp_path):
        _, ex, _ = _run(False, 100.0, tmp_path, max_iterations=50)
        assert ex.shots == 1


class TestGoalModeOn:
    def test_iterates_until_the_iteration_cap(self, tmp_path):
        stage, ex, _ = _run(True, 100.0, tmp_path, max_iterations=5)
        assert ex.shots == 5, "goal mode should keep shooting toward the target"
        assert stage.result["stop_reason"] == "max_iters_exhausted"
        assert stage.status == "failed"

    def test_stops_early_once_the_threshold_is_met(self, tmp_path):
        stage, ex, _ = _run(True, 900.0, tmp_path, max_iterations=10)
        assert ex.shots == 1, "should stop as soon as the target is reached"
        assert stage.result["stop_reason"] == "target_met"
        assert stage.status == "complete"

    def test_costs_more_shots_than_goal_mode_off(self, tmp_path):
        _, ex_on, _ = _run(True, 100.0, tmp_path, max_iterations=5)
        _, ex_off, _ = _run(False, 100.0, tmp_path, max_iterations=5)
        assert ex_on.shots > ex_off.shots


class TestStageModel:
    @pytest.mark.parametrize("op,value,threshold,expected", [
        (">", 701.0, 700.0, True),
        (">", 700.0, 700.0, False),
        (">=", 700.0, 700.0, True),
        ("<", 699.0, 700.0, True),
        ("<=", 700.0, 700.0, True),
        ("==", 700.0, 700.0, True),
    ])
    def test_threshold_comparison(self, op, value, threshold, expected):
        s = Stage(name="s", threshold=threshold, threshold_op=op)
        assert s.threshold_met(value) is expected

    def test_no_threshold_is_never_met(self):
        assert Stage(name="s", threshold=None).threshold_met(1e9) is False
