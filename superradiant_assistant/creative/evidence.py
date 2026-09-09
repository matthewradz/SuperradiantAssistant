"""What this session actually measured.

A report is a claim about an experiment that happened. Nothing in the system
checked that one had: with the sweep declined and no shot ever fired, the model
still produced a report with invented shot IDs and an invented passband, and
every guard passed it — because the guards looked at the model's chat replies,
while the report travelled to disk inside a tool argument.

This module is the record the report is checked against. It holds only values
that came back from real shot files.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class SessionEvidence:
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    shot_ids: Set[str] = field(default_factory=set)
    #: Names of the columns that were swept, in the order they were measured.
    sweep_params: List[str] = field(default_factory=list)
    sweeps_queued: int = 0
    sweeps_with_data: int = 0
    #: How many of `measurements` were read off disk rather than measured here.
    rows_from_disk: int = 0
    #: Things worth remembering that are not measurements: a script written, a
    #: parameter added, a shot executed. Memory is distilled on exit only when
    #: the session did something -- writing MEMORY.md after two lines of chat
    #: costs an API call and dilutes the real notes with nothing.
    actions: List[str] = field(default_factory=list)

    #: Reports written this session, as (path, the shot ids they covered). Used to
    #: refuse a second report about the same measurement: `write_report` is
    #: available to both `lead` and `coder`, and each keeps its own plan, so a
    #: delegated sweep ends with the coder writing a report, telling the lead, and
    #: the lead writing another one from the same 23 shots.
    reports: List[Any] = field(default_factory=list)

    def record_report(self, path: str) -> None:
        self.reports.append((str(path), frozenset(self.shot_ids)))

    def report_for_same_shots(self) -> Optional[str]:
        """The path of an existing report covering exactly today's shots."""
        current = frozenset(self.shot_ids)
        for path, covered in self.reports:
            if covered == current:
                return path
        return None

    def record_action(self, what: str) -> None:
        if what and what not in self.actions:
            self.actions.append(what)

    @property
    def did_substantive_work(self) -> bool:
        # Reading shots off disk is not by itself worth an API call to distil:
        # inspecting old data and asking a question is a conversation, and a
        # memory note written from it dilutes the real ones. Writing a report or
        # a script does count -- those go through `record_action`.
        return bool(self.measured_here or self.actions or self.sweeps_queued)

    def work_summary(self) -> str:
        parts = []
        if self.measured_here:
            parts.append(f"{len(self.measurements) - self.rows_from_disk} "
                         f"measured shot(s)")
        if self.rows_from_disk:
            parts.append(f"{self.rows_from_disk} shot(s) read from disk")
        if self.sweeps_queued:
            parts.append(f"{self.sweeps_queued} run(s) queued")
        if self.actions:
            parts.append(", ".join(self.actions[:6]))
        return " | ".join(parts) or "nothing"

    def record_read_rows(self, rows: List[Dict[str, Any]],
                         sweep_param: str = "") -> None:
        """Rows read back from shot files this session did not itself queue.

        Reporting on them is legitimate — a report for a run the operator
        watched yesterday, or a week's worth of data — but it is not the same
        claim as "I measured this", so it is counted separately and the report
        says which it was. Before this, `write_report` refused every such
        request outright: the guard could not tell "no data exists" from "the
        data exists on disk but a different process produced it".
        """
        if not rows:
            return
        # Reading back shots this session already measured is the normal case --
        # the agent runs a sweep and then reads the results. Absorbing them again
        # doubled everything: 23 shots became `points: 46` in the frontmatter and
        # every point was plotted twice.
        fresh = [r for r in rows
                 if str(r.get("shot_id")) not in self.shot_ids]
        if not fresh:
            return
        before = len(self.measurements)
        self._absorb(fresh, sweep_param)
        self.rows_from_disk += len(self.measurements) - before

    def _absorb(self, rows: List[Dict[str, Any]], sweep_param: str) -> None:
        if sweep_param and sweep_param not in self.sweep_params:
            self.sweep_params.append(sweep_param)
        for r in rows:
            self.measurements.append(dict(r))
            sid = r.get("shot_id")
            if sid:
                self.shot_ids.add(str(sid))

    @property
    def measured_here(self) -> bool:
        """Whether any row came from a sweep this session actually ran."""
        return len(self.measurements) > self.rows_from_disk

    def record_rows(self, rows: List[Dict[str, Any]],
                    sweep_param: str = "") -> None:
        """Take rows read back from shot files after analysis.

        `sweep_param` is kept so a report can plot these rows without having to
        guess which column was the independent variable -- the alternative was
        relying on dict insertion order, which is true today and silently wrong
        the day the row is built differently.
        """
        if not rows:
            return
        self.sweeps_with_data += 1
        self._absorb(rows, sweep_param)

    def record_queue(self) -> None:
        self.sweeps_queued += 1

    @property
    def has_measurements(self) -> bool:
        """Whether there is any data to report, measured here or read from disk."""
        return bool(self.measurements)

    def summary(self) -> str:
        if not self.measurements:
            return (f"NO MEASUREMENTS THIS SESSION "
                    f"({self.sweeps_queued} sweep(s) queued, none produced data)")
        if not self.measured_here:
            return (f"RETROSPECTIVE: {self.rows_from_disk} shot(s) read from "
                    f"disk, none measured in this session")
        return (f"{len(self.measurements)} measured shot(s) from "
                f"{self.sweeps_with_data} sweep(s)")


_EVIDENCE = SessionEvidence()


def get_evidence() -> SessionEvidence:
    return _EVIDENCE


def reset_evidence() -> None:
    """Only for tests — a session's evidence should never be cleared mid-run."""
    global _EVIDENCE
    _EVIDENCE = SessionEvidence()
