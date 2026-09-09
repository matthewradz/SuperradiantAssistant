"""Sequence-file loading: the allow-list, the read-back, and the stage hook.

The read-back is what most of this file is about. Telling runmanager to load a
sequence and assuming it worked would let the agent queue shots against the wrong
experiment under the right parameter names — a failure that produces plausible
numbers and wastes machine time, so every path that can silently skip the switch
is pinned here.
"""
from __future__ import annotations

import pytest

from superradiant_assistant.config import CONFIG
from superradiant_assistant.goal import Stage
from superradiant_assistant.safety import (
    SafetyViolation, allowed_sequence_files, sequence_files_match,
    validate_sequence_file,
)

SEQ_A = "C:/lab/labscriptlib/apparatus/experiment_a.py"
SEQ_B = "C:/lab/labscriptlib/apparatus/experiment_b.py"


@pytest.fixture(autouse=True)
def pin_sequences(monkeypatch):
    """Two configured sequences, independent of the installed config.json.

    Same reasoning as conftest's pin_global_specs: these tests check the
    validation logic, so they must not pass vacuously on a machine that happens
    to have one sequence configured.
    """
    monkeypatch.setattr(CONFIG, "sequences", [
        {"file": SEQ_A, "description": "first"},
        {"file": SEQ_B, "description": "second"},
    ])


class FakeRunmanager:
    """Stands in for the GUI. `obey=False` models a runmanager that ignores the write."""

    def __init__(self, current=None, obey=True):
        self.current = current
        self.obey = obey
        self.set_calls = []

    def get_labscript_file(self):
        return self.current

    def set_labscript_file(self, path):
        self.set_calls.append(path)
        if self.obey:
            self.current = path


class FakeExecutor:
    def __init__(self, runmanager):
        self.runmanager = runmanager


class TestAllowList:
    def test_lists_configured_files(self):
        assert allowed_sequence_files() == [SEQ_A, SEQ_B]

    def test_accepts_a_configured_file(self):
        assert validate_sequence_file(SEQ_A) == SEQ_A

    def test_returns_the_config_spelling_not_the_callers(self):
        # Windows names the same file several ways; runmanager should receive the
        # vetted string, so downstream read-back comparisons have one form to match.
        weird = r"C:\lab\labscriptlib\APPARATUS\experiment_a.py"
        assert validate_sequence_file(weird) == SEQ_A

    def test_refuses_an_unlisted_sibling(self):
        # A file list, not a directory allow-list: dropping a script next to a
        # configured sequence must not make it runnable.
        with pytest.raises(SafetyViolation):
            validate_sequence_file("C:/lab/labscriptlib/apparatus/unvetted.py")

    def test_refuses_path_traversal(self):
        with pytest.raises(SafetyViolation):
            validate_sequence_file(
                "C:/lab/labscriptlib/apparatus/../../../../evil.py"
            )

    def test_refuses_everything_when_nothing_is_configured(self, monkeypatch):
        monkeypatch.setattr(CONFIG, "sequences", [])
        with pytest.raises(SafetyViolation):
            validate_sequence_file(SEQ_A)

    def test_skips_entries_without_a_file_key(self, monkeypatch):
        monkeypatch.setattr(CONFIG, "sequences", [
            {"description": "malformed, no file"},
            {"file": SEQ_A},
        ])
        assert allowed_sequence_files() == [SEQ_A]


class TestPathMatching:
    def test_separator_and_case_insensitive(self):
        assert sequence_files_match("C:/a/B.py", r"C:\A\b.py")

    def test_different_files_do_not_match(self):
        assert not sequence_files_match(SEQ_A, SEQ_B)

    def test_none_never_matches(self):
        # runmanager reports None when no sequence is loaded; that must not read
        # as "already correct" and skip the switch.
        assert not sequence_files_match(None, SEQ_A)
        assert not sequence_files_match(SEQ_A, None)
        assert not sequence_files_match(None, None)


class TestStageSequenceLoading:
    """`_load_stage_sequence` returns None on success, an error string on failure."""

    def _load(self, executor, sequence_file):
        from superradiant_assistant.orchestrator.loop import _load_stage_sequence
        return _load_stage_sequence(
            executor, Stage(name="s", sequence_file=sequence_file)
        )

    def test_switches_when_a_different_file_is_loaded(self):
        rm = FakeRunmanager(current=SEQ_B)
        assert self._load(FakeExecutor(rm), SEQ_A) is None
        assert rm.set_calls == [SEQ_A]

    def test_loads_when_nothing_is_loaded_yet(self):
        rm = FakeRunmanager(current=None)
        assert self._load(FakeExecutor(rm), SEQ_A) is None
        assert rm.set_calls == [SEQ_A]

    def test_does_not_rewrite_when_already_correct(self):
        rm = FakeRunmanager(current=SEQ_A)
        assert self._load(FakeExecutor(rm), SEQ_A) is None
        assert rm.set_calls == []

    def test_fails_when_runmanager_ignores_the_write(self):
        rm = FakeRunmanager(current=SEQ_B, obey=False)
        err = self._load(FakeExecutor(rm), SEQ_A)
        assert err and "still reports" in err

    def test_refuses_an_unlisted_sequence(self):
        rm = FakeRunmanager(current=SEQ_A)
        err = self._load(FakeExecutor(rm), "C:/lab/evil.py")
        assert err and "SafetyViolation" in err
        assert rm.set_calls == []

    def test_reports_bridge_failure_instead_of_raising(self):
        # A stage must fail cleanly and let run_loop move on, not crash the session.
        class BrokenRunmanager:
            def get_labscript_file(self):
                raise RuntimeError("bridge down")

            def set_labscript_file(self, path):
                raise AssertionError("must not write when the read failed")

        err = self._load(FakeExecutor(BrokenRunmanager()), SEQ_A)
        assert err and "bridge down" in err

    def test_offline_executor_is_a_no_op(self):
        # OfflineReplayExecutor has no `runmanager`; replay runs must not fail
        # just because they name a sequence.
        class Offline:
            pass

        assert self._load(Offline(), SEQ_A) is None

    def test_stage_without_a_sequence_is_a_no_op(self):
        rm = FakeRunmanager(current=SEQ_A)
        assert self._load(FakeExecutor(rm), "") is None
        assert rm.set_calls == []


class TestLoadSequenceTool:
    @pytest.fixture
    def api(self, monkeypatch):
        """A LabscriptAPI wired to a fake runmanager, installed as the session API."""
        from superradiant_assistant.labscript_api import LabscriptAPI
        import superradiant_assistant.tools as tools_pkg

        api = LabscriptAPI()
        rm = FakeRunmanager(current=SEQ_B)
        api._rm = rm
        api.fake_rm = rm
        monkeypatch.setattr(tools_pkg, "get_lab_api", lambda: api)
        return api

    def test_switch_reports_both_files(self, api):
        from superradiant_assistant.tools.lab_tools import load_sequence
        out = load_sequence(SEQ_A, "moving to the next experiment")
        assert "experiment_a.py" in out
        assert "experiment_b.py" in out
        assert api.fake_rm.current == SEQ_A

    def test_switch_warns_that_globals_are_untouched(self, api):
        # The new sequence may read a different set of globals; the agent needs to
        # know its cached values no longer describe what will run.
        from superradiant_assistant.tools.lab_tools import load_sequence
        assert "globals untouched" in load_sequence(SEQ_A, "why")

    def test_no_op_when_already_loaded(self, api):
        from superradiant_assistant.tools.lab_tools import load_sequence
        out = load_sequence(SEQ_B, "why")
        assert "already loaded" in out
        assert api.fake_rm.set_calls == []

    def test_unlisted_file_is_refused_not_errored(self, api):
        from superradiant_assistant.tools.lab_tools import load_sequence
        out = load_sequence("C:/lab/evil.py", "why")
        assert out.startswith("refused:")
        assert api.fake_rm.set_calls == []

    def test_disobedient_runmanager_surfaces_as_an_error(self, api):
        from superradiant_assistant.tools.lab_tools import load_sequence
        api.fake_rm.obey = False
        out = load_sequence(SEQ_A, "why")
        assert out.startswith("error:")
        assert "still reports" in out


class TestToolPermissions:
    @pytest.fixture
    def registry(self, tmp_path):
        from superradiant_assistant.hooks import ToolGate
        from superradiant_assistant.tools.lab_tools import build_tool_specs
        from superradiant_assistant.tools.registry import ToolRegistry
        return ToolRegistry(build_tool_specs(), gate=ToolGate(
            audit_path=tmp_path / "audit.jsonl", session_id="test",
            auto_approve=True,
        ))

    def test_only_the_coder_may_switch_sequences(self, registry):
        # Loading a sequence is implementing a decision, not making one -- that
        # moved from the lead to the coder alongside write_shot/write_analysis.
        assert "load_sequence" in registry.names_for("coder")
        for agent in ("lead", "planner", "answer"):
            assert "load_sequence" not in registry.names_for(agent)

    def test_schema_enum_is_the_configured_list(self, registry):
        from superradiant_assistant.tools.lab_tools import build_tool_specs
        spec = next(s for s in build_tool_specs() if s.name == "load_sequence")
        assert spec.parameters["properties"]["file"]["enum"] == [SEQ_A, SEQ_B]

    def test_switching_requires_a_reason(self, registry):
        from superradiant_assistant.tools.lab_tools import build_tool_specs
        spec = next(s for s in build_tool_specs() if s.name == "load_sequence")
        assert set(spec.parameters["required"]) == {"file", "reason"}

    def test_has_a_confirmation_preview(self, registry):
        # Switching experiments changes what the next engage fires; it must be
        # previewable rather than silent.
        from superradiant_assistant.tools.lab_tools import build_tool_specs
        spec = next(s for s in build_tool_specs() if s.name == "load_sequence")
        assert spec.preview is not None
