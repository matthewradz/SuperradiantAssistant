"""Memory store tests: persistence, context assembly, and compaction."""
from __future__ import annotations
import json

import pytest

from superradiant_assistant.memory.store import (
    MemoryStore, _extract_tag, _shrink_memory, MEMORY_FILE_MAX_CHARS,
    _COMPACT_SYSTEM,
)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "mem")


class TestHistory:
    def test_appends_one_json_line_per_turn(self, store):
        store.append_history("user", "hello")
        store.append_history("assistant", "hi")
        lines = store.history_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["role"] == "user"
        assert json.loads(lines[1])["content"] == "hi"

    def test_records_carry_a_timestamp(self, store):
        store.append_history("user", "x")
        assert "ts" in json.loads(store.history_file.read_text(encoding="utf-8").strip())

    def test_structured_content_survives_roundtrip(self, store):
        entry = {"shot_id": "abc", "Neta_2": 716.7, "params": {"green_mot_frequency": 48.9}}
        store.append_history("shot", entry)
        assert store.read_history()[0]["content"] == entry

    def test_read_history_on_missing_file(self, store):
        assert store.read_history() == []

    def test_read_history_respects_limit(self, store):
        for i in range(10):
            store.append_history("user", f"turn {i}")
        recent = store.read_history(limit=3)
        assert len(recent) == 3
        assert recent[-1]["content"] == "turn 9"

    def test_corrupt_line_skipped(self, store):
        store.append_history("user", "good")
        with store.history_file.open("a", encoding="utf-8") as f:
            f.write("{not json\n")
        assert len(store.read_history()) == 1


class TestContextBlock:
    def test_empty_when_nothing_stored(self, store):
        assert store.build_context_block() == ""

    def test_includes_each_populated_section(self, store):
        store._write(store.memory_file, "green_mot_frequency works best near 48.9")
        store._write(store.user_file, "Prefers terse numeric reports.")
        store.append_episode("Ran a resonance scan.")
        block = store.build_context_block()
        assert "48.9" in block
        assert "terse numeric" in block
        assert "resonance scan" in block

    def test_reflects_edits_made_after_construction(self, store):
        # Memory is re-read per turn, so hand-editing MEMORY.md takes effect
        # without restarting the agent.
        assert store.build_context_block() == ""
        store._write(store.memory_file, "later addition")
        assert "later addition" in store.build_context_block()


class TestEpisodes:
    def test_appends_under_a_timestamped_heading(self, store):
        store.append_episode("First thing.")
        store.append_episode("Second thing.")
        text = store.episode_path().read_text(encoding="utf-8")
        assert "First thing." in text and "Second thing." in text
        # One `### HH:MM` per entry. Counting "## " instead also matched the
        # page's own Summary / Next steps / Log headings -- and matched the
        # "## " inside each entry's "### " -- so it read 5 once the day's page
        # gained a template.
        assert text.count("### ") == 2

    def test_written_to_a_dated_file(self, store):
        store.append_episode("x", day="2026-07-31")
        assert (store.root / "2026-07-31.md").exists()


class TestCompaction:
    def test_extract_tag(self):
        assert _extract_tag("<episode>hi</episode>", "episode") == "hi"
        assert _extract_tag("no tags here", "episode") is None
        assert _extract_tag("<episode>a\nb</episode>", "episode") == "a\nb"

    def test_writes_all_three_outputs(self, store):
        class FakeLLM:
            def generate(self, prompt, system=None, temperature=0.2):
                class R:
                    text = ("<episode>Scanned resonance at 82.465 MHz.</episode>"
                            "<updated_memory>Resonance sits near 82.465 MHz.</updated_memory>"
                            "<updated_user>Prefers numbers over prose.</updated_user>")
                return R()

        assert store.compact([{"role": "user", "content": "find resonance"}], FakeLLM())
        assert "82.465" in store.read_memory()
        assert "numbers over prose" in store.read_user()
        assert "82.465" in store.read_today_episode()

    def test_unchanged_operator_profile_not_overwritten(self, store):
        store._write(store.user_file, "original profile")

        class FakeLLM:
            def generate(self, prompt, system=None, temperature=0.2):
                class R:
                    text = ("<episode>e</episode><updated_memory>m</updated_memory>"
                            "<updated_user>UNCHANGED</updated_user>")
                return R()

        store.compact([{"role": "user", "content": "x"}], FakeLLM())
        assert store.read_user() == "original profile"

    def test_llm_failure_is_non_fatal(self, store):
        class ExplodingLLM:
            def generate(self, *a, **k):
                raise RuntimeError("API down")

        assert store.compact([{"role": "user", "content": "x"}], ExplodingLLM()) is False

    def test_no_turns_is_a_noop(self, store):
        class ShouldNotBeCalled:
            def generate(self, *a, **k):
                raise AssertionError("compaction called with no turns")

        assert store.compact([], ShouldNotBeCalled()) is False

    def test_existing_memory_passed_to_curator(self, store):
        store._write(store.memory_file, "PRIOR FACT")
        seen = {}

        class CapturingLLM:
            def generate(self, prompt, system=None, temperature=0.2):
                seen["prompt"] = prompt
                class R:
                    text = "<updated_memory>new</updated_memory>"
                return R()

        store.compact([{"role": "user", "content": "x"}], CapturingLLM())
        assert "PRIOR FACT" in seen["prompt"]


class TestMemoryShrink:
    """MEMORY.md has a size budget now: compact() enforces it, remember() does
    not (that path stays a fast, synchronous append with no LLM call)."""

    def test_under_budget_is_returned_unchanged(self):
        class ShouldNotBeCalled:
            def generate(self, *a, **k):
                raise AssertionError("shrink called on text already under budget")

        text = "short"
        assert _shrink_memory(text, ShouldNotBeCalled(), limit=100) == text

    def test_one_pass_gets_it_under_budget(self):
        class FakeLLM:
            def generate(self, prompt, system=None, temperature=0.2):
                class R:
                    text = "x" * 40
                return R()

        out = _shrink_memory("x" * 200, FakeLLM(), limit=50)
        assert len(out) == 40

    def test_retries_when_still_over_after_one_pass(self):
        calls = []

        class ImprovingLLM:
            def generate(self, prompt, system=None, temperature=0.2):
                calls.append(prompt)
                class R:
                    text = "x" * (80 if len(calls) == 1 else 30)
                return R()

        out = _shrink_memory("x" * 200, ImprovingLLM(), limit=50, max_passes=2)
        assert len(calls) == 2, "one retry after the first pass was still over budget"
        assert len(out) == 30

    def test_gives_up_after_max_passes_without_looping_forever(self):
        calls = []

        class NeverShrinksLLM:
            def generate(self, prompt, system=None, temperature=0.2):
                calls.append(prompt)
                class R:
                    text = "x" * 200        # never actually gets shorter
                return R()

        out = _shrink_memory("x" * 200, NeverShrinksLLM(), limit=50, max_passes=2)
        assert len(calls) == 2, "stops at max_passes rather than looping forever"
        assert len(out) == 200, "returns its best (still over-budget) attempt, not an error"

    def test_llm_failure_returns_the_original_text(self):
        class ExplodingLLM:
            def generate(self, *a, **k):
                raise RuntimeError("API down")

        original = "x" * 200
        assert _shrink_memory(original, ExplodingLLM(), limit=50) == original

    def test_empty_reply_returns_the_original_text(self):
        class EmptyLLM:
            def generate(self, *a, **k):
                class R:
                    text = "   "
                return R()

        original = "x" * 200
        assert _shrink_memory(original, EmptyLLM(), limit=50) == original

    def test_compact_shrinks_an_over_budget_curator_output(self, store):
        big = "y" * (MEMORY_FILE_MAX_CHARS + 500)

        class OverBudgetLLM:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt, system=None, temperature=0.2):
                self.calls += 1
                class R: pass
                r = R()
                if self.calls == 1:
                    r.text = f"<updated_memory>{big}</updated_memory>"
                else:
                    r.text = "z" * (MEMORY_FILE_MAX_CHARS - 100)
                return r

        llm = OverBudgetLLM()
        store.compact([{"role": "user", "content": "x"}], llm)
        assert llm.calls == 2, "compact's own call, then exactly one shrink pass"
        assert len(store.read_memory()) <= MEMORY_FILE_MAX_CHARS
        assert "z" in store.read_memory(), "the shrunk text is what got written, not the original"

    def test_curator_is_told_memory_is_not_a_day_log(self):
        # Two things this catches: the curator was writing dated, per-sequence
        # entries ("2026-08-17 seq 0069: theta0=17.68; 2026-08-18 seq 0003:
        # theta0=17.51") straight into MEMORY.md -- a notebook entry wearing a
        # memory-file disguise, unconditionally injected into every agent's
        # context on every turn. This lives only in prompt text, so a future
        # edit that drops it degrades silently instead of failing loudly.
        low = _COMPACT_SYSTEM.lower()
        assert "not a day-by-day log" in low
        assert "shot id" in low and "sequence number" in low
        assert "one current line per fact" in low or "one line per fact" in low

    def test_compact_does_not_shrink_when_already_under_budget(self, store):
        class OneCallLLM:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt, system=None, temperature=0.2):
                self.calls += 1
                if self.calls > 1:
                    raise AssertionError("shrink pass ran on output already under budget")
                class R:
                    text = "<updated_memory>short and fine</updated_memory>"
                return R()

        llm = OneCallLLM()
        store.compact([{"role": "user", "content": "x"}], llm)
        assert llm.calls == 1
        assert store.read_memory() == "short and fine"
