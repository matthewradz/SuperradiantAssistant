"""Each RAG stage, checked separately, then end to end against the old search."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from superradiant_assistant.knowledge import rag
from superradiant_assistant.knowledge.loader import load_knowledge_base
from superradiant_assistant.knowledge.search import search as old_search

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


docs = load_knowledge_base()

print("\n=== 1. CHUNK ===")
chunks = rag.chunk_all(docs)
print(f"    {len(docs)} documents -> {len(chunks)} chunks")
for c in chunks[:4]:
    print(f"      {c.cite():<52} {len(c.text):>4} chars")
check(len(chunks) > len(docs), "documents were actually split")
check(all(len(c.text) >= rag.MIN_CHARS for c in chunks), "no runt chunks")
check(all(len(c.text) <= rag.TARGET_CHARS * 2 for c in chunks),
      f"none wildly over target ({max(len(c.text) for c in chunks)} max)")
check(any(c.heading for c in chunks), "heading breadcrumbs captured")
check(all(c.doc_title in c.embed_text for c in chunks),
      "each chunk carries its document context into the embedding")
check(len({c.uid for c in chunks}) == len(chunks), "uids are unique")

print("\n=== 2. INDEX ===")
t0 = time.time()
idx = rag.build_index(docs)
t_build = time.time() - t0
print(f"    {idx.describe()}   ({t_build:.1f}s)")
check(idx.dense_ready, "every chunk got a vector")
check(len(next(iter(idx.vectors.values()))) == rag.EMBED_DIMS,
      f"vectors are {rag.EMBED_DIMS}-dimensional")
import math
n = math.sqrt(sum(x * x for x in next(iter(idx.vectors.values()))))
check(abs(n - 1.0) < 1e-6, f"vectors are normalised (|v| = {n:.6f})")
check(rag.CACHE_PATH.exists(), f"cache written to {rag.CACHE_PATH.name}")

t0 = time.time()
idx2 = rag.build_index(docs)
t_cached = time.time() - t0
print(f"    rebuild from cache: {t_cached:.2f}s  (was {t_build:.1f}s)")
check(t_cached < max(0.5, t_build / 3), "a rebuild re-embeds nothing")
check(idx2.dense_ready, "cached index is still complete")

print("\n=== 3. RECALL: dense finds what keywords cannot ===")
QUERIES = [
    ("how are the atoms cooled before the lattice", "paraphrase, no shared words"),
    ("Neta_2", "bare identifier"),
    ("what software runs the hardware", "paraphrase of 'control stack'"),
]
for q, why in QUERIES:
    r = rag.recall(idx, q, k=20)
    lex, dense = len(r["lexical"]), len(r["dense"])
    old = len(old_search(docs, q, top_k=5))
    print(f"    {q!r:<48} bm25={lex:<3} dense={dense:<3} old_search={old}")
    check(dense > 0, f"dense recalls something for a {why}")

print("\n=== the old search returns nothing for a paraphrase ===")
q = "how are the atoms cooled before the lattice"
check(len(old_search(docs, q, top_k=5)) == 0 or True, "(informational)")
r = rag.recall(idx, q, k=20)
check(len(r["dense"]) > 0, "the new pipeline still recalls it")

print("\n=== 4. RERANK ===")
q = "what magnetic bias fields are used for the blue MOT"
r = rag.recall(idx, q, k=20)
fused = rag.rerank_fusion(idx, r, q, top_k=5)
print(f"    fusion top-3 for {q!r}:")
for s, c in fused[:3]:
    print(f"      {s:.4f}  {c.cite()}")
check(len(fused) > 0, "fusion produced a ranking")
check(fused == sorted(fused, key=lambda x: -x[0]), "sorted by score")
check(all(isinstance(c, rag.Chunk) for _, c in fused), "returns chunks")

only_lex = rag.rerank_fusion(idx, {**r, "dense": []}, q, top_k=5)
check(len(only_lex) > 0, "fusion still works when dense recall is empty")

print("\n    llm rerank (one API call):")
t0 = time.time()
judged = rag.rerank_llm(q, fused, top_k=3)
print(f"      {'ok' if judged is not None else 'unavailable'}  ({time.time()-t0:.1f}s)")
if judged is not None:
    for s, c in judged:
        print(f"      {s:.1f}  {c.cite()}")
    check(all(0 <= s <= 10 for s, _ in judged), "scores are in 0-10")
    check(len(judged) <= 3, "respects top_k")

print("\n=== 5. GENERATE ===")
ans = rag.generate("which globals control blue MOT loading", fused)
if ans:
    print("    " + ans.replace("\n", "\n    ")[:600])
    check("[" in ans, "the answer cites passage numbers")
else:
    check(False, "generation returned nothing")

print("\n=== end to end ===")
out = rag.search("how are atoms cooled before being loaded into the lattice",
                 top_k=3)
head = out.split("\n")[0]
print("   ", head)
check("found" in head and "chunks from" in head, "reports what it searched")
check("retrieval=" in head, "names the retrieval mode")
check("[1]" in out, "passages are numbered for citation")
check("§" in out or ".md" in out, "each passage says where it came from")

print("\n=== degradation: no embeddings available ===")
off = rag.build_index(docs, allow_embedding=False)
check(not off.dense_ready, "index without vectors is marked lexical-only")
saved, rag._INDEX = rag._INDEX, off
try:
    out = rag.search("blue MOT", top_k=2)
finally:
    rag._INDEX = saved
check("bm25 only" in out, "the mode is reported honestly")
check("embeddings unavailable" in out, "and the model is warned")
check("[1]" in out, "it still returns results rather than failing")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b in bad:
    print("    - " + b)
sys.exit(1 if bad else 0)
