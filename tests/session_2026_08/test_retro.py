"""A report about shots already on disk must be possible, and must say so."""
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


import superradiant_assistant.creative.tools as T
import superradiant_assistant.creative.evidence as E
from superradiant_assistant.creative.evidence import SessionEvidence

ROWS = [{"shot_id": f"2026-08-11_0000_filter_scan_ch2_{i:02d}",
         "sine_frequency": 100.0 + i * 996.0,
         "transfer_ratio": 0.9 / (1 + (i / 15.0) ** 8)}
        for i in range(26)]

print("\n=== 1. reading from disk is not the same claim as measuring ===")
ev = SessionEvidence()
ev.record_read_rows(ROWS, sweep_param="sine_frequency")
check(ev.has_measurements, "there is data to report")
check(not ev.measured_here, "but nothing was measured in this session")
check(ev.rows_from_disk == 26, f"26 rows counted as read ({ev.rows_from_disk})")
check("RETROSPECTIVE" in ev.summary(), f"summary says so: {ev.summary()}")
check(ev.sweeps_with_data == 0, "a disk read is not counted as a sweep")

print("\n=== 2. a real sweep still reads as measured ===")
ev2 = SessionEvidence()
ev2.record_rows(ROWS, sweep_param="sine_frequency")
check(ev2.measured_here, "measured_here is True")
check(ev2.rows_from_disk == 0, "nothing attributed to disk")
check("RETROSPECTIVE" not in ev2.summary(), f"summary: {ev2.summary()}")
check(ev2.sweeps_with_data == 1, "counted as one sweep")

print("\n=== 3. both together: measured wins ===")
ev3 = SessionEvidence()
ev3.record_read_rows(ROWS[:5], sweep_param="sine_frequency")
ev3.record_rows(ROWS[5:], sweep_param="sine_frequency")
check(ev3.measured_here, "a session that measured anything is not retrospective")
check(ev3.rows_from_disk == 5, f"the read rows are still counted ({ev3.rows_from_disk})")
check(len(ev3.shot_ids) == 26, "all shots available for citation")

print("\n=== 4. reading alone does not trigger a memory flush ===")
check(not ev.did_substantive_work,
      "reading old data is a conversation, not work worth an API call")
ev.record_action("wrote report")
check(ev.did_substantive_work, "but writing a report is")
check(ev2.did_substantive_work, "and so is running a sweep")

print("\n=== 5. the report is written, not refused ===")
tmp = Path(tempfile.mkdtemp(prefix="retro_"))
T.REPORT_DIR, T.FIGURE_DIR = tmp, tmp / "figures"
ev4 = SessionEvidence()
ev4.record_read_rows(ROWS, sweep_param="sine_frequency")
E._EVIDENCE = ev4
MD = """# CH2 Low-Pass Filter

## Measured Data
| Frequency (Hz) | Ratio |
| :--- | :--- |
| 100 | 0.900 |

Shot `2026-08-11_0000_filter_scan_ch2_00` is the first point.
"""
out = T.write_report("CH2 Low-Pass Filter Frequency Response Scan", MD, "backfill")
print("    " + out.replace("\n", "\n    "))
check("refused" not in out, "not refused")
# rglob: reports live in a per-day folder now, tmp/YYYY-MM-DD/.
md_file = next(iter(tmp.rglob("*.md")), None)
check(md_file is not None, "a file exists on disk")
doc = md_file.read_text(encoding="utf-8")

print("\n=== 6. and it says where the numbers came from ===")
check("source: shot files on disk" in doc, "frontmatter names the source")
check("- retrospective" in doc, "tagged retrospective for the vault")
check("[!info] Retrospective" in doc, "a callout at the top of the body")
check("this session did not" in doc, "states plainly that it did not measure")
check("figures/" in doc, "the figure is still drawn and embedded")
# The figures directory is inside the day folder, beside the .md, which is
# what makes the `figures/...` links in the report resolve.
check((md_file.parent / "figures").is_dir(), "and the file is there, beside the report")

print("\n=== 7. a measured report is NOT marked retrospective ===")
tmp2 = Path(tempfile.mkdtemp(prefix="meas_"))
T.REPORT_DIR, T.FIGURE_DIR = tmp2, tmp2 / "figures"
ev5 = SessionEvidence()
ev5.record_rows(ROWS, sweep_param="sine_frequency")
E._EVIDENCE = ev5
T.write_report("Measured Now", MD, "live")
doc2 = next(iter(tmp2.rglob("*.md"))).read_text(encoding="utf-8")
check("source: measured this session" in doc2, "frontmatter says measured")
check("retrospective" not in doc2, "no retrospective tag or callout")

print("\n=== 8. an invented shot id is still refused, retrospective or not ===")
E._EVIDENCE = ev4
T.REPORT_DIR, T.FIGURE_DIR = tmp, tmp / "figures"
liar = MD + "\nAlso shot `2026-08-11_0099_never_happened_0`.\n"
out2 = T.write_report("Lying Report", liar, "test")
check(out2.startswith("refused"), f"refused: {out2[:80]}")
check("never_happened" in out2, "and it names the invented id")

print("\n=== 9. an empty session is still refused, with usable advice ===")
E._EVIDENCE = SessionEvidence()
out3 = T.write_report("Nothing", MD, "test")
check(out3.startswith("refused"), "refused when nothing was read at all")
check("read_shot_results" in out3,
      f"and it says how to fix it:\n      {out3}")

print("\n=== 10. the swept parameter is inferred from a disk read ===")
from superradiant_assistant.tools.lab_tools import _varying_global
names = {"sine_frequency", "amplitude", "phase"}
rows = [{"sine_frequency": 100.0 * i, "amplitude": 0.5, "phase": 0.0}
        for i in range(1, 6)]
check(_varying_global(rows, names) == "sine_frequency",
      f"picked the column that moves: {_varying_global(rows, names)!r}")
flat = [{"sine_frequency": 1000.0, "amplitude": 0.5} for _ in range(4)]
check(_varying_global(flat, names) == "", "nothing varies -> no sweep parameter")
check(_varying_global([], names) == "", "no rows -> no sweep parameter")

shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(tmp2, ignore_errors=True)
print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
