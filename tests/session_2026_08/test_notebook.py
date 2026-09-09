"""The daily notebook: fixed format, rewritten summary, browsable by day."""
import sys
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


from superradiant_assistant.memory.store import (
    MemoryStore, day_key, parse_episode, render_episode,
    SUMMARY_HEADING, NEXT_HEADING, LOG_HEADING, DAY_ROLLOVER_HOUR,
)

tmp = Path(tempfile.mkdtemp(prefix="notebook_"))
M = MemoryStore(root=tmp)

print(f"\n=== 1. the day boundary is {DAY_ROLLOVER_HOUR:02d}:00 ===")
# Back to plain calendar dates. The 02:06 and 03:47 entries that suggested a
# habit of working past midnight were a timezone artefact: the machine was on
# UTC+8 while the operator was in US Eastern, so afternoon work was stamped as
# the small hours of the next day.
check(DAY_ROLLOVER_HOUR == 0, f"rollover is midnight, not {DAY_ROLLOVER_HOUR}:00")
for when, want in ((datetime(2026, 8, 10, 0, 1), "2026-08-10"),
                   (datetime(2026, 8, 10, 2, 6), "2026-08-10"),
                   (datetime(2026, 8, 10, 14, 6), "2026-08-10"),
                   (datetime(2026, 8, 10, 23, 59), "2026-08-10"),
                   (datetime(2026, 8, 11, 0, 0), "2026-08-11")):
    got = day_key(when)
    check(got == want, f"{when:%Y-%m-%d %H:%M} -> {got}")

print("\n=== 2. opening the agent starts a page ===")
p = M.ensure_episode("2026-08-11")
check(p.exists(), f"created {p.name}")
text = p.read_text(encoding="utf-8")
for h in ("# 2026-08-11", SUMMARY_HEADING, NEXT_HEADING, LOG_HEADING):
    check(h in text, f"page has {h!r}")
before = text
M.ensure_episode("2026-08-11")
check(p.read_text(encoding="utf-8") == before, "opening again does not clobber it")

print("\n=== 3. exits append to the log, never to the summary ===")
M.write_day_summary("First summary.", "- do a thing", "2026-08-11")
for i in range(3):
    M.append_episode(f"- session {i} did something", "2026-08-11")
text = p.read_text(encoding="utf-8")
s, n, log = parse_episode(text)
check(text.count(SUMMARY_HEADING) == 1, "exactly one Summary section")
check(text.count(NEXT_HEADING) == 1, "exactly one Next steps section")
check(s == "First summary.", f"summary survived three appends: {s!r}")
check(all(f"session {i}" in log for i in range(3)), "all three entries in the log")
check(log.count("###") == 3, f"three timestamped entries ({log.count('###')})")

print("\n=== 4. the summary is replaced, not accumulated ===")
M.write_day_summary("Second summary.", "- do a better thing", "2026-08-11")
s2, n2, log2 = parse_episode(p.read_text(encoding="utf-8"))
check(s2 == "Second summary.", "summary replaced")
check("First summary." not in p.read_text(encoding="utf-8"),
      "the old summary is gone, not stacked above the new one")
check(n2 == "- do a better thing", "next steps replaced too")
check(log2 == log, "the log is untouched by a summary rewrite")

print("\n=== 5. six exits leave the same page as one ===")
q = Path(tempfile.mkdtemp(prefix="notebook2_"))
M2 = MemoryStore(root=q)
M2.ensure_episode("2026-08-12")
for i in range(6):
    M2.append_episode(f"- entry {i}", "2026-08-12")
    M2.write_day_summary(f"Summary after exit {i}.", f"- next after {i}",
                         "2026-08-12")
t = M2.episode_path("2026-08-12").read_text(encoding="utf-8")
check(t.count(SUMMARY_HEADING) == 1, "still one Summary after six exits")
check("Summary after exit 5." in t and "Summary after exit 4." not in t,
      "only the last summary remains")
check(t.count("###") == 6, f"all six log entries kept ({t.count('###')})")

print("\n=== 6. a page written before this format still parses ===")
old = tmp / "2026-08-09.md"
old.write_text("\n## 10:26\n- did an old thing\n\n## 21:22\n- and another\n",
               encoding="utf-8")
s3, n3, log3 = parse_episode(old.read_text(encoding="utf-8"))
check(s3 == "" and n3 == "", "no summary claimed for a legacy page")
check("did an old thing" in log3 and "and another" in log3,
      "the whole legacy file is treated as log")
M.write_day_summary("Reconstructed.", "- carry on", "2026-08-09")
t3 = old.read_text(encoding="utf-8")
check(t3.startswith("# 2026-08-09"), "legacy page upgraded to the format")
check("did an old thing" in t3, "legacy content preserved through the upgrade")

print("\n=== 7. days can be listed and read back ===")
days = M.episode_days()
check(days == sorted(days), f"sorted oldest first: {days}")
check("2026-08-09" in days and "2026-08-11" in days, f"both pages listed: {days}")
check("Reconstructed." in M.read_episode("2026-08-09"), "a past day reads back")
check(M.read_episode("1999-01-01") == "", "a day with no page reads as empty")
(tmp / "MEMORY.md").write_text("core", encoding="utf-8")
check("MEMORY" not in " ".join(days) and "USER" not in " ".join(days),
      f"only dated files count as days: {M.episode_days()}")

print("\n=== 8. the summary is generated from the whole day, not one session ===")
seen = {}


class FakeLLM:
    def generate(self, prompt, system=None, temperature=None):
        seen["prompt"], seen["system"] = prompt, system
        return type("R", (), {"text": "<summary>All of it.</summary>"
                              "<next_steps>- go on</next_steps>"})()


check(M.summarise_day(FakeLLM(), "2026-08-11"), "summarise_day reports success")
for i in range(3):
    check(f"session {i}" in seen["prompt"],
          f"the whole day's log reached the prompt (session {i})")
check("No LaTeX" in seen["system"], "the prompt forbids LaTeX")
check("actionable tomorrow" in seen["system"], "next steps must be actionable")
s4, n4, _ = parse_episode(M.read_episode("2026-08-11"))
check(s4 == "All of it." and n4 == "- go on", "both blocks written back")

print("\n=== 9. an empty day is not summarised ===")
M.ensure_episode("2026-08-20")
check(not M.summarise_day(FakeLLM(), "2026-08-20"),
      "a page with no log costs no API call")

print("\n=== 10. the weekly page is built from the daily pages ===")
class WeekLLM:
    def generate(self, prompt, system=None, temperature=None):
        seen["week_prompt"], seen["week_system"] = prompt, system
        return type("R", (), {"text": "## What was measured\nthings"})()


out = M.weekly_summary(WeekLLM(), days=["2026-08-09", "2026-08-11"])
check(out is not None and out.exists(), f"wrote {out}")
check(out.parent.name == "weekly", "weekly pages live in their own folder")
check("2026-08-09" in seen["week_prompt"] and "All of it." in seen["week_prompt"],
      "the daily summaries are what the week is built from")
check("Still open" in seen["week_system"], "the week records what is unresolved")
check("2026-08-09" not in [d for d in M.episode_days() if "/" in d],
      "weekly files do not pollute the day list")
check(M.episode_days() == sorted(M.episode_days()), "day list still clean")

print("\n=== 11. round-trip is stable ===")
r = render_episode("2026-09-01", "S", "N", "### 10:00\n- x")
a, b, cc = parse_episode(r)
check((a, b, cc) == ("S", "N", "### 10:00\n- x"), f"parse(render(x)) == x: {(a,b,cc)}")
r2 = render_episode("2026-09-01", *parse_episode(r))
check(r2 == r, "rendering twice changes nothing")
empty = render_episode("2026-09-02", "", "", "")
check(parse_episode(empty) == ("", "", ""), "placeholders read back as empty")

shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(q, ignore_errors=True)

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
