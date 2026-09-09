"""The reply must be in the language the operator just used."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


from superradiant_assistant.orchestrator.agents import language_directive as ld

print("\n=== 1. the two messages that went wrong ===")
en = ("Investigate the unknown filter on CH2 up to 25 kHz and fit the "
      "amplitude ratio vs frequency curve")
check("in English" in ld(en), f"English request -> {ld(en).strip()[:60]}")
zh = "帮我把示波器切到 XY 模式看李萨如图形"
check("in Chinese" in ld(zh), f"Chinese request -> {ld(zh).strip()[:60]}")

print("\n=== 2. mixed text follows the majority ===")
mostly_en = "run a sweep from 100 Hz to 25 kHz 谢谢"
check("in English" in ld(mostly_en), f"mostly English -> {ld(mostly_en).strip()[:52]}")
mostly_zh = "把 phase 从 0 扫到 330,画出李萨如图形族,一共 12 个点"
check("in Chinese" in ld(mostly_zh), f"mostly Chinese -> {ld(mostly_zh).strip()[:52]}")

print("\n=== 3. no signal, no directive ===")
# Slash commands are not in this list: agent.py handles them before send() is
# reached, so they never carry a directive regardless of what this returns.
for t in ("", "  ", "26", "0-330", "y", "  \n "):
    check(ld(t) == "", f"no directive for {t!r}")
check(ld("run it") != "", "a short but real English request still gets one")

print("\n=== 4. other scripts are named correctly ===")
for text, want in ((" オシロスコープを設定して", "Japanese"),
                   ("스코프를 설정해 주세요", "Korean"),
                   ("настрой осциллограф", "Russian")):
    check(want in ld(text), f"{want}: {ld(text).strip()[:50]}")

print("\n=== 5. it is appended for the model, not stored in memory ===")
import superradiant_assistant.orchestrator.agents as A


class FakeSession:
    def __init__(self):
        self.saw = None

    def send_message(self, text):
        self.saw = text
        return type("R", (), {"text": "ok"})()


remembered = []
A.MEMORY.append_history = lambda k, v: remembered.append((k, v))

ag = A.Agent.__new__(A.Agent)
ag.name, ag._session, ag._evidence, ag._turns = "lead", FakeSession(), [], 0
ag.cost_tracker, ag._beat = None, None
out = A.Agent.send(ag, "run the sweep", ld("run the sweep"))
check("[The operator wrote this message in English" in ag._session.saw,
      "the directive reached the model")
check(ag._session.saw.startswith("run the sweep"),
      "the operator's words come first, unmodified")
stored = [v for k, v in remembered if k == "lead:in"]
check(stored == ["run the sweep"],
      f"history stored the request without the directive: {stored}")
check(ag._evidence == ["run the sweep"],
      "the directive is not treated as evidence")
check(out == "ok", "the reply is returned unchanged")

print("\n=== 6. send() still works with no directive ===")
ag2 = A.Agent.__new__(A.Agent)
ag2.name, ag2._session, ag2._evidence, ag2._turns = "coder", FakeSession(), [], 0
ag2.cost_tracker, ag2._beat = None, None
A.Agent.send(ag2, "plain message")
check(ag2._session.saw == "plain message", "unchanged when no directive given")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
