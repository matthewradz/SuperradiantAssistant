"""The report: a figure from the same rows, a tidy .md, provenance from code."""
import re
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
from superradiant_assistant.creative.evidence import SessionEvidence

print("\n=== 3. tables are aligned in the raw file ===")
ragged = """| Frequency (Hz) | Ratio | dB |
| :--- | :--- | :--- |
| 100 | 5.015 | +14.01 |
| 25000 | 0.029 | -30.65 |"""
aligned = T._align_tables(ragged)
print("    " + aligned.replace("\n", "\n    "))
rows = aligned.splitlines()
check(len({len(r) for r in rows}) == 1,
      f"every row the same width: {sorted({len(r) for r in rows})}")
check(":---" not in aligned, "the :--- separator is normalised")
check("5.015" in aligned and "-30.65" in aligned, "no data lost")
check(T._align_tables("no table here") == "no table here", "non-tables untouched")

print("\n=== 4. the figure is drawn from the reported rows ===")
ev = SessionEvidence()
rows = [{"shot_id": f"2026-08-10_0008_filter_scan_ch2_{i:02d}",
         "sine_frequency": 100.0 + i * 996.0,
         "transfer_ratio": 0.9 / (1 + (i / 15.0) ** 8),
         "transfer_db": -i * 0.4,
         "CHAN1_amp": 0.5, "CHAN2_amp": 0.45}
        for i in range(26)]
ev.record_rows(rows, sweep_param="sine_frequency")
check(ev.sweep_params == ["sine_frequency"], f"sweep param recorded: {ev.sweep_params}")

tmp = Path(tempfile.mkdtemp(prefix="rep_"))
T.REPORT_DIR, T.FIGURE_DIR = tmp, tmp / "figures"
fig = T._sweep_figure("probe", ev.measurements, "sine_frequency")
check(fig is not None and fig.exists(), f"figure written: {fig}")
check(fig.stat().st_size > 5000, f"and it is a real image ({fig.stat().st_size} bytes)")
check(fig.parent.name == "figures", "figures live in their own folder")

print("\n=== 5. panels: the interesting quantity first, diagnostics after ===")
order = T._panel_order(["CHAN1_amp", "transfer_db", "shot_count", "transfer_ratio"])
print(f"    {order}")
check(order.index("transfer_ratio") < order.index("shot_count"),
      "a ratio outranks an unrecognised column")
check(order.index("transfer_db") < order.index("shot_count"), "so does a dB column")
check(len(T._panel_order([f"m{i}" for i in range(20)])) == 20, "ordering keeps all keys")

print("\n=== 6. no numeric column, no figure, no crash ===")
ev2 = SessionEvidence()
ev2.record_rows([{"shot_id": "a", "note": "text"}], sweep_param="phase")
check(T._sweep_figure("x", ev2.measurements, "phase") is None,
      "nothing plottable returns None")
check(T._sweep_figure("x", [], "phase") is None, "no rows returns None")
check(T._sweep_figure("x", ev.measurements, "") is None,
      "no sweep param returns None")

print("\n=== 7. the whole report ===")
import superradiant_assistant.creative.evidence as E
E._EVIDENCE = ev
MD = r"""# CH2 Filter Report

## Method
Fitted $H(f) = \frac{A}{\sqrt{1+(f/f_c)^{2n}}}$ using `filter_scan_ch2.py`.

| Frequency (Hz) | Ratio |
| :--- | :--- |
| 100 | 5.015 |
| 25000 | 0.029 |

## Conclusion
Cutoff $f_c \approx 14.6\text{ kHz}$.
"""
out = T.write_report("CH2 Filter Frequency Response & Cutoff", MD, "test")
print("    " + out.replace("\n", "\n    "))
# rglob: reports live in a per-day folder now, tmp/YYYY-MM-DD/.
md_file = next(p for p in tmp.rglob("*.md"))
doc = md_file.read_text(encoding="utf-8")
print("\n    --- the .md ---\n    " + doc.replace("\n", "\n    "))

check(doc.count("\n# ") + doc.startswith("# ") * 1 == 1
      or len(re.findall(r"^# ", doc, re.M)) == 1,
      f"exactly one H1: {len(re.findall(r'^# ', doc, re.M))}")
print("    --- provenance, written by code not by the model ---")
check(doc.startswith("---\ntitle:"), "opens with YAML frontmatter")
check("shots: 26" in doc, "shot count stated from the record")
check("swept: sine_frequency" in doc, "frontmatter names the swept parameter")
check("swept_from: 100" in doc and "swept_to: 25000" in doc, "swept range stated")
check("points: 26" in doc, "point count stated")
check("shot_first: \"2026-08-10_0008_filter_scan_ch2_00\"" in doc
      and "shot_last: \"2026-08-10_0008_filter_scan_ch2_25\"" in doc,
      "frontmatter gives the shot range, not 26 property rows")
check("shot_ids:" not in doc, "the full list is not in the properties panel")
check("## Shots" in doc and doc.index("## Shots") > doc.index("## Conclusion"),
      "the full shot list is an appendix at the end")
check(doc.count("2026-08-10_0008_filter_scan_ch2_13") == 1,
      "every shot listed exactly once")
check("tags:" in doc, "frontmatter carries tags for the vault")
check(doc.count("---") >= 2, "the frontmatter block is closed")

check("figures/" in doc, "the figure is embedded in the markdown")
check("![" in doc, "as a markdown image, which Obsidian renders inline")
# LaTeX is deliberately preserved: the .md is read in Obsidian, which renders
# it. The no-LaTeX rule applies to the terminal and to memory, not here.
check("$" in doc and "\\frac" in doc, "LaTeX kept for Obsidian to render")
check("| 100   | 5.015 |" in doc or re.search(r"\| 100\s+\| 5\.015", doc),
      "the table was aligned")
check(not list(tmp.glob("*.html")), "no HTML is written any more")

print("\n=== 9. a failed plot does not lose the report ===")
real = T._sweep_figure
T._sweep_figure = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
# Fresh shots, or the duplicate guard refuses this before the figure is even
# attempted -- a second report about the same measurement is now a refusal.
ev.record_rows([{"shot_id": "2026-08-10_0009_filter_scan_ch2_00",
                 "sine_frequency": 26000.0,
                 "filter_ch2_singleshot": 0.01,
                 "transfer_ratio": 0.01}], sweep_param="sine_frequency")
try:
    out2 = T.write_report("Second Report", MD, "test")
finally:
    T._sweep_figure = real
check("wrote" in out2, f"the report was still written: {out2.splitlines()[0]}")
check(len(list(tmp.rglob("*.md"))) == 2, "and it is a separate file")

shutil.rmtree(tmp, ignore_errors=True)
print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
