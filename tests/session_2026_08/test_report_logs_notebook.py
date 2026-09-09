"""Every report writes its own notebook entry, stamped when it was written.

The log used to be written only by the memory curator, which runs on exit or
when the context grows -- so 2026-08-17 carried two entries covering six
experiments, stamped 14:52 and 16:57, the times compaction happened.
"""
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from superradiant_assistant.creative import tools as T
from superradiant_assistant.creative.evidence import SessionEvidence
import superradiant_assistant.creative.evidence as E
from superradiant_assistant.memory import MEMORY
from superradiant_assistant.memory.store import parse_episode, day_key

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


TMP = Path(__file__).parent / "_log_tmp"
if TMP.exists():
    shutil.rmtree(TMP)
(TMP / "reports").mkdir(parents=True)
(TMP / "notebook").mkdir(parents=True)

T.REPORT_DIR = TMP / "reports"
T.FIGURE_DIR = T.REPORT_DIR / "figures"
saved_root = MEMORY.root
MEMORY.root = TMP / "notebook"


def evidence(n=6, measured=True):
    ev = SessionEvidence()
    rows = [{"shot_id": "2026-08-17_%04d_x_0" % i, "shot_path": "",
             "waveplate_angle": float(10 * i),
             "CHAN1_mean": 0.4 + 0.01 * i} for i in range(n)]
    if measured:
        ev.record_queue()
        ev.record_rows(rows, sweep_param="waveplate_angle")
    else:
        ev.record_read_rows(rows, sweep_param="waveplate_angle")
    return ev


def write(title, ev):
    saved = E._EVIDENCE
    E._EVIDENCE = ev
    try:
        return T.write_report(title, f"# {title}\n\nBody.\n", reason="test")
    finally:
        E._EVIDENCE = saved


try:
    print("\n=== 1. writing a report writes a log entry ===")
    out = write("Waveplate sweep 0 to 50 deg", evidence())
    page = MEMORY.read_episode(day_key())
    _, _, log = parse_episode(page)
    print("    " + "\n    ".join(log.strip().splitlines()))
    check("logged in the notebook" in out, "the tool says it logged")
    check(log.strip(), "the day's log is not empty")
    check(re.search(r"^### \d{2}:\d{2}$", log, re.M),
          "the entry is stamped with a time")
    check("Waveplate sweep 0 to 50 deg" in log, "and names the experiment")
    check("6 shot(s) measured" in log, "and how many shots it measured")
    check("waveplate_angle 0 to 50" in log, "and the range that was swept")

    print("\n=== 2. the entry names the report file ===")
    md = re.search(r"report: `([^`]+)`", log)
    print(f"    {md.group(1) if md else None}")
    check(md is not None, "the entry carries a report path")
    check((T.REPORT_DIR / md.group(1)).is_file(),
          "and the path resolves, relative to the reports root")
    check(md.group(1).startswith(T._report_day_dir().name),
          "written as <day>/<file>.md, so it is findable from the notebook")

    print("\n=== 3. one report, one entry ===")
    write("Second experiment of the day", evidence(n=4))
    _, _, log = parse_episode(MEMORY.read_episode(day_key()))
    stamps = re.findall(r"^### \d{2}:\d{2}$", log, re.M)
    reports = re.findall(r"report: `([^`]+)`", log)
    print(f"    {len(stamps)} stamp(s), {len(reports)} report(s) named")
    check(len(stamps) == 2, "a second report adds a second entry, not a rewrite")
    check(len(reports) == 2, "each entry names its own report")
    check(len(set(reports)) == 2, "and they are different files")
    check("Waveplate sweep 0 to 50 deg" in log
          and "Second experiment of the day" in log,
          "the first entry survives the second")

    print("\n=== 4. a refused report writes nothing ===")
    before = MEMORY.read_episode(day_key())
    refused = write("No data at all", SessionEvidence())
    check(refused.startswith("refused:"), "an empty-evidence report is refused")
    check(MEMORY.read_episode(day_key()) == before,
          "and the notebook is untouched -- no entry without a report")

    print("\n=== 5. a duplicate report writes nothing either ===")
    ev = evidence(n=5)
    write("Third experiment", ev)
    before = MEMORY.read_episode(day_key())
    again = write("Third experiment written twice", ev)
    check(again.startswith("already reported:"), "the duplicate guard fires")
    check(MEMORY.read_episode(day_key()) == before,
          "and it adds no second entry for the same shots")

    print("\n=== 6. a retrospective report says so in the log ===")
    write("Read back from yesterday", evidence(n=3, measured=False))
    _, _, log = parse_episode(MEMORY.read_episode(day_key()))
    last = log.strip().split("### ")[-1]
    print("    " + " | ".join(last.splitlines()[1:]))
    check("read back from disk" in last,
          "the entry distinguishes measured here from read off disk")

    print("\n=== 7. the summary and next steps are left alone ===")
    MEMORY.write_day_summary("A summary someone wrote.", "- do the next thing")
    write("Fourth experiment", evidence(n=7))
    summary, nxt, log = parse_episode(MEMORY.read_episode(day_key()))
    check(summary.strip() == "A summary someone wrote.", "summary untouched")
    check(nxt.strip() == "- do the next thing", "next steps untouched")
    check(len(re.findall(r"^### \d{2}:\d{2}$", log, re.M)) == 5,
          "and the log kept every entry")
finally:
    MEMORY.root = saved_root
    shutil.rmtree(TMP, ignore_errors=True)

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
