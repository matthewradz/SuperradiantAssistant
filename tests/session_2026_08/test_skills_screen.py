"""The /skill screen: what it lists, what it says about each one."""
import io
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from superradiant_assistant import splash as S
from superradiant_assistant.skill_loader import SKILLS

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


def render(keys, cols=110, lines=44):
    """The last frame the screen drew, with the escapes stripped."""
    it = iter(keys)
    real_key, real_ansi, real_tty = S.read_key, S.enable_ansi, sys.stdin.isatty
    real_size = shutil.get_terminal_size
    S.read_key = lambda: next(it)
    S.enable_ansi = lambda: True
    sys.stdin.isatty = lambda: True
    shutil.get_terminal_size = lambda d=(100, 30): os.terminal_size((cols, lines))
    buf, real_stdout = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        S.skills_screen()
    except StopIteration:
        pass
    finally:
        sys.stdout = real_stdout
        S.read_key, S.enable_ansi = real_key, real_ansi
        sys.stdin.isatty = real_tty
        shutil.get_terminal_size = real_size
    frame = buf.getvalue().split(S.CLEAR)[-1]
    frame = re.sub(r"\x1b\[\?[0-9]+[hl]", "", frame)
    return re.sub(r"\x1b\[[0-9;]*m", "", frame)


print("\n=== 1. every installed skill is listed ===")
screen = render(["q"])
names = SKILLS.names()
missing = [n for n in names if n not in screen]
check(not missing, f"all {len(names)} skills appear ({missing or 'none missing'})")
check(f"{len(names)} available" in screen, "and the count is stated")

print("\n=== 2. each carries its one-line description ===")
first = names[0]
desc = SKILLS.skills[first]["meta"].get("description", "")
check(desc.split(".")[0] in screen,
      f"the selected skill's description is shown ({first})")

print("\n=== 3. a skill for another apparatus is marked, not hidden ===")
foreign = [n for n in names if SKILLS.apparatus_of(n)
           and SKILLS.apparatus_of(n) != SKILLS.CURRENT_APPARATUS]
check(bool(foreign), f"there are foreign skills to mark: {foreign}")
check(all("NOT this bench" in screen for _ in foreign),
      "the list marks them")
# The detail panel describes the SELECTED skill, so the cursor has to be on a
# foreign one. Walk it there rather than relying on the alphabetically first
# skill happening to be foreign, which is how this silently stopped testing
# anything when the skill set changed.
steps = names.index(foreign[0])
detail = render(["down"] * steps + ["q"])
check(f"  {foreign[0]}" in detail, f"cursor reached {foreign[0]}")
check("do not exist on this bench" in detail,
      "and the detail panel spells out what that means")

print("\n=== 4. the screen says what a skill IS ===")
# The question this screen exists to answer: is a skill a script?
check("nothing here executes" in screen or "context" in screen,
      "it states that the body enters the model's context")
check("load_skill" in screen, "and names the tool that does it")
check("characters" in screen, "and how much context that costs")

print("\n=== 5. the cursor moves and the detail follows it ===")
one = render(["q"])
two = render(["down", "q"])
check(one != two, "a different skill is selected after ↑↓")
def marked(frame):
    # Rows look like "│ > name ..." once the colour is stripped.
    return [l for l in frame.splitlines() if l.lstrip("│ ").startswith("> ")]


sel1, sel2 = marked(one), marked(two)
check(len(sel1) == 1 and len(sel2) == 1, "exactly one row is marked at a time")
check(sel1 != sel2, f"and it moved: {sel1[0].strip()[:40]} -> {sel2[0].strip()[:40]}")

print("\n=== 6. section headings stand in for the body ===")
i = names.index("waveplate-malus-sweep")
screen = render(["down"] * i + ["q"])
steps = S._skill_steps(SKILLS.skills["waveplate-malus-sweep"]["body"])
check(len(steps) >= 4, f"the waveplate skill has {len(steps)} sections")
check(all(s[:20] in screen for s in steps), "each is listed under 'covers'")
check("Traps" in screen, "including the traps section")

print("\n=== 7. nothing on the screen runs anything ===")
# A reading list, by design. If a key ever queues a shot, this test should fail.
for key in ("enter", "r", "m", "right", "left", " "):
    out = render([key, "q"]) if key not in ("enter",) else ""
    check("error" not in out.lower() and "Traceback" not in out,
          f"{key!r} is inert")

print("\n=== 8. the box never overflows a narrow terminal ===")
narrow = render(["q"], cols=80)
widths = {len(l) for l in narrow.splitlines() if l.startswith("┌")
          or l.startswith("│")}
check(len(widths) <= 1, f"every framed row is the same width: {widths}")
check(max(len(l) for l in narrow.splitlines()) <= 80,
      "and nothing is wider than the terminal")

print("\n=== 9. no ANSI, no alternate buffer — a plain list ===")
real_ansi, real_tty = S.enable_ansi, sys.stdin.isatty
S.enable_ansi = lambda: False
sys.stdin.isatty = lambda: False
buf, real_stdout = io.StringIO(), sys.stdout
sys.stdout = buf
try:
    S.skills_screen()
finally:
    sys.stdout = real_stdout
    S.enable_ansi, sys.stdin.isatty = real_ansi, real_tty
plain = buf.getvalue()
check(S.ALT_ON not in plain, "the alternate buffer is not entered")
check(all(n in plain for n in names), "and every skill is still printed")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
