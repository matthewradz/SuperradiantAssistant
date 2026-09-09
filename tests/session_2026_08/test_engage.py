"""Verify the engage_shot verification and the thinking config."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


print("\n=== 1. _shot_state tells compiled from executed ===")
from superradiant_assistant.tools.lab_tools import _shot_state

DATA = Path.home() / "Desktop" / "Labscript agent ver2" / "local_data"
never_ran = DATA / "2026-08-08_0007_Lissajous_0.h5"
did_run = DATA / "2026-08-08_0005_Lissajous_0.h5"
check(_shot_state(never_ran) == "compiled",
      f"0007 (queued, never executed) -> {_shot_state(never_ran)!r}")
check(_shot_state(did_run) == "analysed",
      f"0005 (ran + lyse results)     -> {_shot_state(did_run)!r}")
check(_shot_state(DATA / "nope.h5") == "unreadable", "missing file -> 'unreadable'")

print("\n=== 2. engage_shot reports FAILED when nothing executes ===")
import superradiant_assistant.tools.lab_tools as lt


class FakeAPI:
    def n_shots(self): return 1
    def get_labscript_file(self): return "C:/x/Lissajous.py"
    def engage(self): pass          # succeeds, but nothing ever runs


import superradiant_assistant.tools as tools_pkg
tools_pkg.get_lab_api = lambda: FakeAPI()
lt._lyse_readiness_warning = lambda: ""

# No new shot files will appear, so this is the "queued but never ran" case.
out = lt.engage_shot(reason="test", wait_seconds=3)
print("   ---\n   " + out.replace("\n", "\n   ") + "\n   ---")
check(out.startswith("FAILED"), "starts with FAILED, not 'engaged'")
check("NEVER EXECUTED" in out, "says NEVER EXECUTED")
check("This step is NOT complete" in out, "tells the model not to mark it done")
check("BLACS is running" in out, "names the thing to check")

print("\n=== 3. wait_seconds=0 opts out but says so ===")
out0 = lt.engage_shot(reason="test", wait_seconds=0)
check("execution NOT verified" in out0, f"unverified path is labelled: {out0}")

print("\n=== 4. engage failure still reported ===")


class BrokenAPI(FakeAPI):
    def engage(self): raise RuntimeError("runmanager gone")


tools_pkg.get_lab_api = lambda: BrokenAPI()
out2 = lt.engage_shot(reason="test", wait_seconds=3)
check(out2.startswith("engage failed"), f"engage exception surfaces: {out2}")

print("\n=== 5. thinking config builds against the installed SDK ===")
from google.genai import types as gt
from superradiant_assistant.llm.client import _thinking_config, _THINKING_LEVELS

for lvl in _THINKING_LEVELS:
    cfg = _thinking_config(gt, lvl)
    check(cfg is not None, f"level {lvl!r} -> {cfg}")
check(_thinking_config(gt, "default") is None, "'default' returns None (SDK decides)")
check(_thinking_config(gt, "nonsense") is not None, "unknown level falls back to low")

print("\n=== 6. team wiring: coder thinks harder than lead ===")
from superradiant_assistant.tools import build_registry
from superradiant_assistant.hooks import default_gate
from superradiant_assistant.orchestrator.agents import build_team

team = build_team(build_registry(gate=default_gate(), creative=True),
                  model="gemini-3.6-flash", thinking="low")
levels = {n: a.thinking_level for n, a in team.items()}
print("   ", levels)
check(levels["lead"] == "low", "lead runs low")
check(levels["coder"] == "medium", "coder runs one level higher")

print("\n=== 7. session still constructs with the new kwargs ===")
import inspect
from superradiant_assistant.llm.client import GeminiAgentSession
sig = inspect.signature(GeminiAgentSession.__init__)
check("thinking_level" in sig.parameters, "GeminiAgentSession takes thinking_level")
src = Path(REPO / "superradiant_assistant" / "llm" / "client.py").read_text(encoding="utf-8")
check(src.count("http_options=genai_types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)") >= 1
      or src.count("HttpOptions(timeout=REQUEST_TIMEOUT_MS)") >= 2,
      "the agent session now gets the request timeout too")

print("\n=== 8. schema still builds ===")
reg = build_registry(gate=default_gate(), creative=True)
decls = reg.declarations_for("lead")
eng = [d for d in decls if d.name == "engage_shot"][0]
props = set(eng.parameters.properties or {})
check("wait_seconds" in props, f"engage_shot exposes wait_seconds: {sorted(props)}")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)

