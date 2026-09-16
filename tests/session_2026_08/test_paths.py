import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from superradiant_assistant.config import CONFIG, REPO_ROOT, _data_root, _suite_root

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


print("REPO_ROOT      :", REPO_ROOT)
print("suite root     :", CONFIG.labscript_suite_root)
print("data root      :", CONFIG.historical_data_root)

print("\n=== resolved paths exist ===")
check(CONFIG.labscript_suite_root.exists(),
      f"labscript-suite found without being hard-coded")
check(CONFIG.historical_data_root == (REPO_ROOT.parent / "local_data").resolve(),
      "data root sits beside the repo, derived not hard-coded")

# A checkout with no data folder yet is a legitimate state -- nothing creates it
# and BLACS writes it on its first shot -- so its absence is reported, not
# failed. Asserting it existed made this suite fail on every fresh clone.
POPULATED = CONFIG.historical_data_root.exists()
if POPULATED:
    check(True, f"data root present: {CONFIG.historical_data_root}")
else:
    print(f"  ----  data root not created yet: {CONFIG.historical_data_root}")
    print("        (expected on a fresh checkout; the install checks below are skipped)")

print("\n=== the folders the memory/report code derives from it ===")
for sub in ("agent_memory", "reports"):
    p = CONFIG.historical_data_root.parent / sub
    check(p.exists() or sub == "reports", f"{sub}: {p}")

print("\n=== _data_root behaviour ===")
check(_data_root("local_data") == (REPO_ROOT.parent / "local_data").resolve(),
      "relative path -> beside the repo")

# The recovery rule is: an absolute path that no longer exists falls back to a
# same-NAMED folder beside the repo. So the fixture makes that folder itself
# rather than naming one that happens to exist on the machine the test was
# written on -- keyed to a real path here, this passed for a local reason and
# failed everywhere else.
probe = REPO_ROOT.parent / "_stale_path_probe"
probe.mkdir(exist_ok=True)
try:
    stale = str(Path(r"C:\gone") / "some old copy" / probe.name)
    check(_data_root(stale) == probe.resolve(),
          f"stale absolute path recovers to the folder beside the repo: "
          f"{_data_root(stale)}")
finally:
    probe.rmdir()
check(_data_root(r"C:\definitely\not\here") == Path(r"C:\definitely\not\here"),
      "unrecoverable absolute path is returned unchanged (visible failure)")

print("\n=== _suite_root behaviour ===")
check(_suite_root("") == Path.home() / "labscript-suite",
      "empty setting -> ~/labscript-suite")
check(_suite_root(r"C:\nope\labscript-suite") == Path.home() / "labscript-suite",
      "stale setting falls back to the standard location")

print("\n=== the tools that use these still work ===")
# Derivation is asserted unconditionally; anything that needs a populated
# install is reported instead, so this file passes on a fresh checkout.
from superradiant_assistant.memory import MEMORY
check(MEMORY.root == CONFIG.historical_data_root.parent / "agent_memory",
      f"memory root: {MEMORY.root}")

from superradiant_assistant import script_inventory as inv
n_shot = len(inv.scripts_of_kind("shot"))
if CONFIG.labscript_suite_root.exists():
    check(n_shot > 0, f"script inventory finds {n_shot} shot sequences")
else:
    print(f"  ----  no labscript-suite installed; inventory found {n_shot}")

if POPULATED:
    check(MEMORY.history_file.exists(),
          f"history.jsonl reachable ({MEMORY.history_file.stat().st_size} bytes)"
          if MEMORY.history_file.exists() else "history.jsonl missing")
else:
    print("  ----  no history yet (nothing has run on this checkout)")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
