"""A report must never silently replace the previous one."""
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


import superradiant_assistant.creative.tools as T

WHEN = datetime(2026, 8, 11, 14, 30)

print("\n=== 1. the titles that used to collapse to 'report' ===")
cases = [
    ("AWG CH2 Filter Frequency Response & Cutoff Determination",
     "AWG_CH2_Filter_Frequency_Response_Cutoff_Determination"),
    ("2:3 Lissajous phase sweep", "2_3_Lissajous_phase_sweep"),
    ("Filter response (100 Hz - 25 kHz)", "Filter_response_100_Hz_25_kHz"),
    ("未知滤波器频率响应测试报告", "未知滤波器频率响应测试报告"),
]
for title, want in cases:
    got = T._slug(title)
    check(got == want, f"{title!r}\n           -> {got}")
    check(T._safe_stem(title) is None or " " not in title,
          f"(and _safe_stem would have rejected it: {T._safe_stem(title)!r})")

print("\n=== 2. slugs are usable filenames ===")
for title, _ in cases:
    s = T._slug(title)
    check(s and "/" not in s and "\\" not in s and ":" not in s
          and not s.startswith("_") and not s.endswith("_"),
          f"{s!r} is a bare, trimmed stem")
    check(len(s) <= T._MAX_SLUG, f"{len(s)} chars, within {T._MAX_SLUG}")
long_title = "A " * 80
check(len(T._slug(long_title)) <= T._MAX_SLUG,
      f"a very long title is capped at {len(T._slug(long_title))}")
check(T._slug("!!!") == "report", f"a title with no usable characters -> "
                                  f"{T._slug('!!!')!r}")
check(T._slug("") == "report", "an empty title falls back")

print("\n=== 3. the stem carries the date ===")
stem = T._report_stem("Filter response", WHEN)
check(stem == "Filter_response_20260811", f"dated: {stem}")
check(T._report_stem("Filter response_20260811", WHEN) == stem,
      "a title the model already dated is not dated twice")

print("\n=== 4. writing twice in a day does not overwrite ===")
import tempfile
tmp = Path(tempfile.mkdtemp(prefix="reports_"))
T.REPORT_DIR = tmp
s1 = T._report_stem("Filter response", WHEN)
(tmp / f"{s1}.md").write_text("first", encoding="utf-8")
s2 = T._report_stem("Filter response", WHEN)
check(s2 != s1, f"second call gives a different stem: {s1} -> {s2}")
check(s2 == f"{s1}_2", f"and it is numbered: {s2}")
(tmp / f"{s2}.md").write_text("second", encoding="utf-8")
s3 = T._report_stem("Filter response", WHEN)
check(s3 == f"{s1}_3", f"third: {s3}")
check((tmp / f"{s1}.md").read_text(encoding="utf-8") == "first",
      "the first report is still on disk, untouched")

print("\n=== 5. preview and writer agree on the path ===")
# The preview showing a path the writer does not use is how a confirmation
# prompt starts lying; it happened once already with write_analysis.
import inspect
src_w = inspect.getsource(T.write_report)
src_p = inspect.getsource(T._preview_write_report)
check("_report_stem" in src_w, "writer uses _report_stem")
check("_report_stem" in src_p, "preview uses _report_stem")
check("_safe_stem" not in src_w and "_safe_stem" not in src_p,
      "neither still uses the rejecting version")

print("\n=== 6. against a folder of existing reports: nothing would collide ===")
# A temp folder holding the names the old code produced, rather than whichever
# reports happen to sit in one machine's install -- pointed at a real folder,
# this asserted nothing on any other checkout.
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    T.REPORT_DIR = Path(tmp)
    for stem in ("report", "report_1", "AWG_CH2_Filter_Response_20260812"):
        (T.REPORT_DIR / f"{stem}.md").write_text("", encoding="utf-8")
    existing = sorted(p.stem for p in T.REPORT_DIR.glob("*.md"))
    print(f"    on disk: {existing}")
    new = T._report_stem("AWG CH2 Filter Frequency Response & Cutoff Determination")
    check(new not in existing, f"a new report would be written as {new}")
    check(new != "report", "no longer collapses to the shared 'report' name")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
