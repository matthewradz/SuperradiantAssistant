"""Skill frontmatter, apparatus labelling, and the rewritten search skill."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Which skills get labelled "NOT this bench" depends on which apparatus is open,
# and the default is `testbench`. This file's expectations are written from the
# cesium bench, so say so instead of relying on a default -- the labelling used
# to come from a hardcoded constant and this was implicit.
os.environ["AGENT_APPARATUS"] = "cesium"

from superradiant_assistant.skill_loader import SKILLS, SkillLoader, _parse_frontmatter

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


print("\n=== 1. every skill still parses (no BOM casualties) ===")
idx = SKILLS.get_descriptions()
print(idx)
check("(no description)" not in idx, "no skill lost its description")
# Not a fixed count: `save_experiment_skill` lets the agent add its own, and it
# has. What must hold is that the hand-written ones are all still loadable.
BUILTIN = {"filter-response-measurement", "hardware-safety",
           "lab-knowledge-search", "writing-experiment-code"}
missing = BUILTIN - set(SKILLS.names())
check(not missing, f"every built-in skill present (missing {missing or 'none'}); "
                   f"{len(SKILLS.names())} total: {SKILLS.names()}")

print("\n=== 2. a BOM no longer breaks the frontmatter ===")
doc = "---\nname: x\ndescription: hello\n---\n\nbody"
for label, text in (("plain", doc),
                    ("with BOM", "\ufeff" + doc),
                    ("CRLF", doc.replace("\n", "\r\n")),
                    ("BOM+CRLF", "\ufeff" + doc.replace("\n", "\r\n"))):
    meta, body = _parse_frontmatter(text)
    check(meta.get("description") == "hello" and body.strip() == "body",
          f"{label}: parsed -> {meta.get('description')!r}")

print("\n=== 3. a foreign apparatus's skills are labelled in the menu ===")
CES = ["writing-experiment-code", "lab-knowledge-search",
       "waveplate-malus-sweep"]
# The two filter skills are the test bench's: they sweep the AWG's sine
# frequency, and the Cesium table has no AWG.
BENCH = ["filter-response-measurement", "filter-response-ch2-3k-25k"]
for n in BENCH:
    check(SKILLS.apparatus_of(n) == "testbench",
          f"{n} declares apparatus: testbench")
    line = [l for l in idx.split("\n") if l.strip().startswith(f"- {n}")][0]
    check("[testbench, NOT this bench]" in line,
          f"{n} is labelled foreign from the cesium bench")
for n in CES:
    line = [l for l in idx.split("\n") if l.strip().startswith(f"- {n}")][0]
    check("NOT this bench" not in line, f"{n} is NOT labelled")

print("\n=== 4. loading one carries the warning into context ===")
body = SKILLS.get_content("filter-response-measurement")
check("NOT the bench in front of you" in body, "a foreign skill body warns")
check("do not tell the operator you can run it" in body, "and says not to offer it")
own = SKILLS.get_content("writing-experiment-code")
check("NOT the bench" not in own, "a cesium skill carries no warning")

print("\n=== 5. the search skill describes the pipeline that exists ===")
s = SKILLS.get_content("lab-knowledge-search")
for gone, why in (("keyword scoring, not semantic search", "the old claim"),
                  ("There is no embedding model", "the old claim"),
                  ("count triple", "the old scoring rule")):
    check(gone not in s, f"removed: {why} ({gone!r})")
for want in ("dense", "BM25", "fusion(dense+bm25)", "rerank='llm'",
             "answer=true", "gemini-embedding-001"):
    check(want in s, f"documents the real pipeline: {want}")
check("none of\nthem anywhere in the corpus" in s.lower()
      and "the failure has happened here" in s,
      "warns about the fabrication that happened")
check("list_scripts" in s
      and "not guaranteed to describe the bench you are running" in s.lower(),
      "redirects apparatus questions to the right tools")

print("\n=== 6. schema and startup unaffected ===")
from superradiant_assistant.tools import build_registry
from superradiant_assistant.hooks import default_gate
reg = build_registry(gate=default_gate(), creative=True)
d = [x for x in reg.declarations_for("lead") if x.name == "load_skill"][0]
enum = list(d.parameters.properties["skill_name"].enum or [])
check(sorted(enum) == sorted(SKILLS.names()), f"load_skill enum matches: {len(enum)}")
check(len(SKILLS.get_descriptions()) < 1400,
      f"index still small: {len(SKILLS.get_descriptions())} chars")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
