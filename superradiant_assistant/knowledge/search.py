"""Simple keyword search over loaded Documents.

Phase 1: title/summary/tag matches weighted 3x over body matches.
Phase 2: swap in embedding-based cosine similarity.
"""
from __future__ import annotations
from typing import List, Optional
from superradiant_assistant.knowledge.loader import Document


def search(
    docs: List[Document],
    query: str,
    top_k: int = 5,
    kinds: Optional[List[str]] = None,
) -> List[Document]:
    terms = [t for t in query.lower().split() if len(t) > 2]
    if not terms:
        return docs[:top_k]

    scored: List[tuple[float, Document]] = []
    for d in docs:
        if kinds and d.kind not in kinds:
            continue
        header = f"{d.title} {d.summary} {' '.join(d.tags)}".lower()
        body = d.content.lower()
        score = sum(header.count(t) * 3 + body.count(t) for t in terms)
        if score > 0:
            scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


def search_for_role(
    docs: List[Document],
    query: str,
    role: str = "planner",
    top_k: int = 5,
) -> List[Document]:
    """Role-aware search: planner gets high-level docs; coder gets API docs."""
    kind_prefs = {
        "planner":  ["manual", "sequence", "subsequence", "code_example"],
        "coder":    ["analysis", "class", "connection", "device", "code_example"],
        "answer":   None,  # all kinds
    }
    preferred = kind_prefs.get(role)
    results = search(docs, query, top_k=top_k, kinds=preferred)
    if len(results) < top_k:
        extra = search(docs, query, top_k=top_k)
        seen = {d.path for d in results}
        for d in extra:
            if d.path not in seen:
                results.append(d)
                if len(results) == top_k:
                    break
    return results