"""The reply panel: markdown gone, framed, correct width."""
import re
import sys
from pathlib import Path

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from superradiant_assistant import splash as S
from preview_ansi import render

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


def plain(s):
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s)


REPLY = """BLACS is **not connected / not executing shots**.

A test shot was queued (`Test_Speed.py`), but BLACS never executed it (compiled shot files exist, but `/data/traces` is missing). Please check that BLACS is open and its device workers (RigolScope / RigolDG1022) are connected and tab-restarted if needed.

### Queued Shots & Parameters
- `2026-08-09_0002_filter_scan_0`: `sine_frequency` = **2000.0 Hz**
- `2026-08-09_0002_filter_scan_1`: `sine_frequency` = **2500.0 Hz**

1. Fit Malus's law $P(\\theta) = P_0\\cos^2(\\theta-\\theta_0)$ to extract the offset.
2. *Note:* no lyse metrics were reported before the timeout.

---
Done."""

panel = S.reply_panel(REPLY, width=100)
Path(SCRATCH / "reply.ansi").write_text(panel, encoding="utf-8")
render([Path(SCRATCH / "reply.ansi")], SCRATCH / "reply.png")

text = plain(panel)
print("\n--- rendered ---")
print(text)

print("\n=== markdown markers are gone ===")
for marker, what in (("**", "bold asterisks"), ("`", "backticks"),
                     ("###", "heading hashes"), ("$", "LaTeX dollars")):
    check(marker not in text, f"no {what} survive")
check("- `2026" not in text, "list dashes replaced")
check("·" in text, "bullets became a middle dot")

print("\n=== content survived ===")
for want in ("not connected / not executing shots", "Test_Speed.py",
             "/data/traces", "Queued Shots & Parameters", "2000.0 Hz",
             "P_0\\cos^2(\\theta-\\theta_0)".replace("\\\\", "\\")):
    check(want.replace("\\\\", "\\") in text, f"kept: {want[:36]}")
check("1. Fit Malus" in text or "1." in text, "numbered list kept its numbers")

print("\n=== emphasis became real styling ===")
check("\x1b[1m" in panel, "** turned into ANSI bold")
check(S.c(S.ACCENT) in panel, "code spans got the accent colour")

print("\n=== geometry ===")
rows = panel.split("\n")
widths = {S._plain_len(r) for r in rows}
check(widths == {100}, f"every row is exactly 100 columns: {sorted(widths)}")
check(rows[0].lstrip().startswith("\x1b") or "┌" in rows[0], "top border present")
check("└" in rows[-1], "bottom border present")
check("assistant" in plain(rows[0]), "titled")

print("\n=== narrow terminals ===")
for w in (48, 72, 100, 140):
    p = S.reply_panel(REPLY, width=w)
    ws = {S._plain_len(r) for r in p.split("\n")}
    check(ws == {w}, f"width {w}: all rows {sorted(ws)}")

print("\n=== degenerate input ===")
check("(no reply)" in plain(S.reply_panel("")), "empty reply does not crash")
check(S._plain_len(S.reply_panel("hi", width=60).split("\n")[0]) == 60,
      "one-word reply still framed")

print("\n" + "=" * 58)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
