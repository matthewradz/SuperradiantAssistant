"""The per-agent model field on the team screen, driven by scripted keys."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")

from superradiant_assistant import splash as S
from superradiant_assistant.splash import model_problem

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


print("\n=== 1. a name is checked for its provider, not against a list ===")
check(model_problem("claude-opus-5") == "", "claude-opus-5 is accepted")
check(model_problem("gemini-3.6-flash") == "", "gemini-3.6-flash is accepted")
check(model_problem("gemini-9.9-whatever-ships-next") == "",
      "and so is a model that does not exist yet")
check(model_problem("gwmini-3.6-flash") != "", "the typo that cost a session is caught")
print(f"    -> {model_problem('gwmini-3.6-flash')}")
check(model_problem("  ") != "", "so is an empty entry")


class FakeAgent:
    def __init__(self, name, model, thinking="low"):
        self.name, self.model, self.thinking_level = name, model, thinking


class FakeRegistry:
    def names_for(self, agent):
        return ["a", "b", "c"]


def drive(keys, team):
    """Run the screen against a scripted keyboard and a real terminal size."""
    it = iter(keys)
    real_key, real_ansi, real_tty = S.read_key, S.enable_ansi, sys.stdin.isatty
    S.read_key = lambda: next(it)
    S.enable_ansi = lambda: True
    sys.stdin.isatty = lambda: True
    devnull = open(os.devnull, "w", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = devnull
    try:
        return S.team_screen(team, FakeRegistry())
    finally:
        sys.stdout = real_stdout
        devnull.close()
        S.read_key, S.enable_ansi = real_key, real_ansi
        sys.stdin.isatty = real_tty


def fresh():
    return {n: FakeAgent(n, "gemini-3.6-flash")
            for n in ("lead", "planner", "coder", "answer")}


print("\n=== 2. one agent's model can be typed on its own ===")
# down to 'coder', open the box, clear it, type a Claude model, apply, leave.
keys = (["down", "down", "m"] + ["\x08"] * 40
        + list("claude-opus-5") + ["enter", "q"])
levels, models = drive(keys, fresh())
print(f"    {models}")
check(models["coder"] == "claude-opus-5", "the coder moved")
check([models[n] for n in ("lead", "planner", "answer")]
      == ["gemini-3.6-flash"] * 3, "and nobody else did")
check(levels["coder"] == "low", "the thinking level is untouched")

print("\n=== 3. the box is prefilled, so a version bump is a few keys ===")
keys = ["m"] + ["\x08"] * 5 + list("5-pro") + ["enter", "q"]
_, models = drive(keys, fresh())
check(models["lead"] == "gemini-3.6-5-pro",
      f"edited from the current name: {models['lead']}")

print("\n=== 4. a typo is refused at the box, not at the next message ===")
keys = (["m"] + ["\x08"] * 40 + list("gwmini") + ["enter"]   # refused, box stays
        + ["\x08"] * 6 + list("gemini-2.5-pro") + ["enter", "q"])
_, models = drive(keys, fresh())
check(models["lead"] == "gemini-2.5-pro",
      f"the box stayed open and took the correction: {models['lead']}")

keys = ["m"] + ["\x08"] * 40 + list("gwmini") + ["enter", "esc", "q"]
_, models = drive(keys, fresh())
check(models["lead"] == "gemini-3.6-flash", "esc leaves it on what it had")

print("\n=== 5. typing a model does not trip the screen's own keys ===")
# 'm' inside "gemini", 'q' and 'j'/'k' inside a name must all be plain text.
keys = ["m"] + ["\x08"] * 40 + list("gemini-jkq-claude") + ["enter", "q"]
_, models = drive(keys, fresh())
check(models["lead"] == "gemini-jkq-claude",
      f"every character landed in the box: {models['lead']}")

print("\n=== 6. quitting without touching anything changes nothing ===")
levels, models = drive(["down", "up", "q"], fresh())
check(set(models.values()) == {"gemini-3.6-flash"}, "models unchanged")
check(set(levels.values()) == {"low"}, "levels unchanged")

print("\n=== 7. a mixed team is a legal state ===")
team = fresh()
team["coder"].model = "claude-opus-5"
_, models = drive(["q"], team)
check(models["coder"] == "claude-opus-5" and models["lead"] == "gemini-3.6-flash",
      "the screen reports each agent's own model")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
