"""Knowledge base loader.

Walks the folders configured in CONFIG.knowledge_sources() and turns
each file into a Document. Lab-agnostic: any lab can plug in their own
KnowledgeSource entries.
"""
from __future__ import annotations
import ast
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

from superradiant_assistant.config import CONFIG, KnowledgeSource


# ---------- Document model ----------

@dataclass
class Document:
    title: str                      # short, human-readable
    path: str                       # absolute path on disk
    kind: str                       # "device" | "sequence" | "analysis" | ...
    content: str                    # text used for embeddings + LLM context
    summary: str = ""               # short auto-generated summary
    tags: List[str] = field(default_factory=list)
    mtime: float = 0.0              # modification time (for cache invalidation)
    char_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------- Per-file extraction ----------

def _read_text(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"<could not read file: {e}>"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n# ... [truncated at {max_chars} chars]\n"
    return text


def _python_summary(source: str) -> str:
    """Extract a short summary from a Python source file:
    module docstring + top-level defs + class names.
    Falls back to first non-empty lines if AST parsing fails.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Pre-3.x or messy files: use first ~20 non-empty lines as a fallback.
        lines = [ln for ln in source.splitlines() if ln.strip()]
        return "\n".join(lines[:20])

    parts: List[str] = []
    mdoc = ast.get_docstring(tree)
    if mdoc:
        parts.append(mdoc.strip())

    func_names: List[str] = []
    class_names: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_names.append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            func_names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            class_names.append(node.name)

    if class_names:
        parts.append("Classes: " + ", ".join(class_names))
    if func_names:
        parts.append("Top-level functions: " + ", ".join(func_names))

    return "\n".join(parts).strip()


def _markdown_summary(source: str) -> str:
    """For markdown, the summary is the first heading + the first paragraph."""
    lines = source.splitlines()
    title_line = ""
    paragraph: List[str] = []
    for ln in lines:
        if not title_line and ln.startswith("#"):
            title_line = ln.lstrip("#").strip()
            continue
        if title_line:
            if ln.strip() == "":
                if paragraph:
                    break
                else:
                    continue
            paragraph.append(ln.strip())
    summary = title_line
    if paragraph:
        summary = (summary + " — " + " ".join(paragraph)) if summary else " ".join(paragraph)
    return summary[:500]


def _make_title(path: Path, kind: str) -> str:
    return f"[{kind}] {path.name}"


def _file_to_document(path: Path, kind: str, max_chars: int) -> Optional[Document]:
    if not path.is_file():
        return None
    text = _read_text(path, max_chars)
    if path.suffix.lower() == ".py":
        summary = _python_summary(text)
    elif path.suffix.lower() in (".md", ".markdown"):
        summary = _markdown_summary(text)
    else:
        summary = ""

    return Document(
        title=_make_title(path, kind),
        path=str(path.resolve()),
        kind=kind,
        content=text,
        summary=summary,
        tags=[kind, path.stem],
        mtime=path.stat().st_mtime,
        char_count=len(text),
    )


# ---------- Source iteration ----------

def _iter_files(src: KnowledgeSource) -> List[Path]:
    base = Path(src.path)
    if not base.exists():
        return []
    if src.recursive:
        candidates = list(base.rglob(src.glob))
    else:
        candidates = list(base.glob(src.glob))

    out: List[Path] = []
    for p in candidates:
        if any(ex in str(p) for ex in src.exclude):
            continue
        if src.include and p.name not in src.include:
            continue
        out.append(p)
    return sorted(out)


# ---------- Public API ----------

def load_knowledge_base() -> List[Document]:
    """Load all configured knowledge sources into a list of Documents."""
    docs: List[Document] = []
    seen_paths: set[str] = set()
    for src in CONFIG.knowledge_sources():
        for f in _iter_files(src):
            if str(f.resolve()) in seen_paths:
                continue
            d = _file_to_document(f, src.kind, CONFIG.embedding_max_chars_per_doc)
            if d is None:
                continue
            seen_paths.add(d.path)
            docs.append(d)
    return docs


def summarize_knowledge_base(docs: List[Document]) -> Dict[str, Any]:
    """Quick summary stats for printing / debugging."""
    by_kind: Dict[str, int] = {}
    total_chars = 0
    for d in docs:
        by_kind[d.kind] = by_kind.get(d.kind, 0) + 1
        total_chars += d.char_count
    return {
        "n_docs": len(docs),
        "by_kind": by_kind,
        "total_chars": total_chars,
        "avg_chars": int(total_chars / max(1, len(docs))),
    }