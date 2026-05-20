"""Simple keyword search over loaded Documents.

Phase 1: title/summary/tag matches weighted 3x over body matches.
Phase 2: swap in embedding-based cosine similarity.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Optional
from superradiant_assistant.knowledge.loader import Document


def extract_function_body(path: str, func_name: str, max_lines: int = 80) -> Optional[str]:
    """Read the full file and return the source of `func_name` (up to max_lines)."""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None

    # Find the def line
    start = None
    indent = None
    for i, line in enumerate(lines):
        if re.match(rf'^(\s*)def\s+{re.escape(func_name)}\s*\(', line):
            start = i
            indent = len(line) - len(line.lstrip())
            break
    if start is None:
        return None

    # Collect lines until we return to the same or lower indent (end of function)
    body_lines = [lines[start]]
    for line in lines[start + 1:]:
        stripped = line.rstrip()
        if stripped == "":
            body_lines.append(line)
            continue
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= indent and stripped:
            break
        body_lines.append(line)
        if len(body_lines) >= max_lines:
            body_lines.append(f"    # ... [truncated after {max_lines} lines]")
            break

    return "\n".join(body_lines)


def augment_with_function_excerpts(docs: List[Document], query: str) -> List[Document]:
    """If the query mentions a function name, inject its body into the relevant doc."""
    # Extract function-like names from query
    func_names = re.findall(r'\b([a-z][a-z0-9_]+(?:_[a-z0-9]+)+)\b', query.lower())
    if not func_names:
        return docs

    augmented = []
    for doc in docs:
        if not doc.path.endswith(".py"):
            augmented.append(doc)
            continue
        extra_excerpts = []
        for fn in func_names:
            body = extract_function_body(doc.path, fn)
            if body:
                extra_excerpts.append(f"\n## Function `{fn}` (full body):\n```python\n{body}\n```")
        if extra_excerpts:
            import copy
            d = copy.copy(doc)
            d.content = "\n".join(extra_excerpts) + "\n\n" + doc.content
            augmented.append(d)
        else:
            augmented.append(doc)
    return augmented


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