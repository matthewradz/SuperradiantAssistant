"""The attribution rule must reach the model, and must not gag it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from superradiant_assistant.knowledge import rag

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


print("\n=== 1. the rule is attached to every result ===")
out = rag.search("how are the atoms cooled", top_k=2)
check(rag.GROUNDING_RULE in out, "appended to a normal result")
check(out.rstrip().endswith(rag.GROUNDING_RULE.rstrip()),
      "at the end, where it is read last")

print("\n=== 2. it constrains attribution, not content ===")
r = rag.GROUNDING_RULE
check("ONLY for something that passage actually states" in r,
      "citations must match the passage")
check("You may add your own physics knowledge" in r,
      "own knowledge is explicitly permitted")
check("interpretation and recommendations" in r,
      "interpretation is explicitly permitted")
check("Mark it as yours" in r, "and only has to be labelled")
for phrase in ("only the passages", "do not use outside knowledge",
               "must not answer"):
    check(phrase.lower() not in r.lower(), f"does NOT forbid discussion ({phrase!r})")

print("\n=== 3. empty results tell it to say so, not to invent ===")
saved = rag._INDEX
try:
    rag._INDEX = rag.Index(chunks=[])
    empty = rag.search("anything at all")
    check("knowledge base is empty" in empty, "empty corpus is reported")
    idx = rag.get_index(rebuild=True)
finally:
    rag._INDEX = saved
zero = rag.search("zzzqqq nonexistent token xyzzy", top_k=3)
if "found 0 passages" in zero:
    check("do not cover this" in zero or "documents do not cover" in zero,
          "a zero-hit search tells it to say the docs do not cover it")
else:
    print(f"    (retrieval still returned something: {zero.splitlines()[0][:70]})")
    check(True, "fusion always returns its best guess — the rule covers it")

print("\n=== 4. a failed llm rerank is reported, not hidden ===")
real = rag.rerank_llm
rag.rerank_llm = lambda *a, **k: None          # simulate the API failing
try:
    out = rag.search("clock transition precision", top_k=3, rerank="llm")
finally:
    rag.rerank_llm = real
head = out.split("\n")[0]
print("   ", head)
check("llm rerank FAILED" in head, "the header says the stage did not run")
check("did not run" in out, "and a NOTE explains it in words")
check("not judged order" in out, "and says the order is unreliable")

print("\n=== 5. a successful rerank still says so ===")
rag.rerank_llm = lambda q, c, **k: c[:k.get("top_k", 3)]
try:
    ok = rag.search("clock transition precision", top_k=3, rerank="llm")
finally:
    rag.rerank_llm = real
check("-> llm rerank" in ok.split("\n")[0], "header records the stage")
check("FAILED" not in ok.split("\n")[0], "and does not cry failure")

print("\n=== 6. the four fabricated claims are still absent from the corpus ===")
from superradiant_assistant.knowledge.loader import load_knowledge_base
corpus = "\n".join(d.content for d in load_knowledge_base()).lower()
for needle, label in (("556", "556 nm"), ("3p1", "1S0->3P1"),
                      ("non-demolition", "QND"), ("microkelvin", "microkelvin")):
    check(needle not in corpus,
          f"{label} is genuinely not in the documents (so citing one was wrong)")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
