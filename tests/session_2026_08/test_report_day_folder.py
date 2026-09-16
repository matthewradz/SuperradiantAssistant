"""A report and its figures land in reports/YYYY-MM-DD/, links intact."""
import io
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from superradiant_assistant.creative import tools as T
from superradiant_assistant.creative.evidence import SessionEvidence

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


TMP = Path(__file__).parent / "_day_folder_tmp"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True)
T.REPORT_DIR = TMP
T.FIGURE_DIR = TMP / "figures"

WHEN = datetime(2026, 8, 17, 18, 40, 0)


print("\n=== 1. the day folder is the date, not the title ===")
day = T._report_day_dir(WHEN)
print(f"    {day}")
check(day.name == "2026-08-17", "folder is YYYY-MM-DD")
check(day.parent == TMP, "directly under the reports root")

print("\n=== 2. the stem still carries the date and never collides ===")
day.mkdir(parents=True, exist_ok=True)
s1 = T._report_stem("Filter response", WHEN)
(day / f"{s1}.md").write_text("x", encoding="utf-8")
s2 = T._report_stem("Filter response", WHEN)
print(f"    {s1} then {s2}")
check(s1 == "Filter_response_20260817", f"dated stem ({s1})")
check(s2 == s1 + "_2", "a second report the same day gets _2, not an overwrite")
check(T._report_stem("Filter response_20260817", WHEN) == s2,
      "a title the model already dated is not dated twice")

print("\n=== 3. a report written before the folders existed still blocks ===")
# Reports live flat in REPORT_DIR from before this change; a new one must not
# take a name one of those already has.
(TMP / "Legacy_report_20260817.md").write_text("x", encoding="utf-8")
check(T._report_stem("Legacy report", WHEN) == "Legacy_report_20260817_2",
      "the flat folder is checked too")

print("\n=== 4. write_report files everything under the day ===")
ev = SessionEvidence()
ev.record_queue()
rows = [{"shot_id": "2026-08-17_%04d_x_0" % i, "shot_path": "",
         "waveplate_angle": float(10 * i),
         "CHAN1_mean": 0.4 + 0.01 * i} for i in range(6)]
ev.record_rows(rows, sweep_param="waveplate_angle")

real_get = T.get_evidence if hasattr(T, "get_evidence") else None
import superradiant_assistant.creative.evidence as E
saved = E._EVIDENCE
E._EVIDENCE = ev
try:
    out = T.write_report("Day folder probe", "# Day folder probe\n\nBody text.\n",
                         reason="test")
finally:
    E._EVIDENCE = saved
print("    " + out.splitlines()[0])

today = T._report_day_dir()
# Only this report's file: section 2 already parked collision fodder in the
# same day folder on purpose.
mds = sorted(today.glob("Day_folder_probe*.md")) if today.is_dir() else []
check(len(mds) == 1, f"the report is in {today.name}/ ({len(mds)} file)")
check(not list(TMP.glob("Day_folder_probe*.md")),
      "and nothing left flat in the reports root")

print("\n=== 5. the figure sits in <day>/figures and the link resolves ===")
pngs = sorted((today / "figures").glob("*.png")) if (today / "figures").is_dir() else []
print(f"    {[p.name for p in pngs]}")
check(len(pngs) == 1, "the sweep figure is in the day's figures/")
text = mds[0].read_text(encoding="utf-8")
links = re.findall(r"!\[[^\]]*\]\(figures/([^)]+)\)", text)
print(f"    links: {links}")
check(len(links) == 1, "the report links one image")
check((mds[0].parent / "figures" / links[0]).is_file(),
      "and the link resolves relative to the .md, so the folder moves whole")

print("\n=== 6. the confirmation preview shows the same paths ===")
E._EVIDENCE = ev
try:
    prev = T._preview_write_report({"title": "Day folder probe",
                                    "markdown": "# x\n"})
finally:
    E._EVIDENCE = saved
file_line = [l for l in prev.splitlines() if l.startswith("file ")][0]
fig_line = [l for l in prev.splitlines() if l.startswith("figure ")][0]
print("    " + file_line.strip())
print("    " + fig_line.strip())
check(T._report_day_dir().name in file_line,
      "the previewed .md path is inside the day folder")
check(T._report_day_dir().name in fig_line and "figures" in fig_line,
      "and so is the previewed figure path")

shutil.rmtree(TMP, ignore_errors=True)

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
