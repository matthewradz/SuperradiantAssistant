"""Knowledge base loader.

Walks the folders configured in CONFIG.knowledge_sources() and turns
each file into a Document. Lab-agnostic: any lab can plug in their own
KnowledgeSource entries.
"""
from __future__ import annotations
import ast
import os
import re
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


# Names that are definitely not runmanager globals
_BUILTINS = frozenset({
    'True', 'False', 'None', 'self', 'cls',
    'print', 'range', 'len', 'max', 'min', 'sum', 'abs', 'int', 'float',
    'str', 'list', 'dict', 'set', 'tuple', 'bool', 'type', 'isinstance',
    'hasattr', 'getattr', 'setattr', 'enumerate', 'zip', 'map', 'filter',
    'sorted', 'reversed', 'open', 'super', 'staticmethod', 'classmethod',
    'property', 'Exception', 'ValueError', 'TypeError', 'RuntimeError',
    'np', 'os', 'sys', 'json', 'time', 'Path',
    't', 'ms', 'us', 'ns', 'duration', 'samplerate',
})

# Prefixes that indicate a runmanager global in this lab's naming convention
_GLOBAL_PREFIX = re.compile(
    r'^(LS_|'
    r'green_mot_|blue_mot_|yellow_|red_mot_|'
    r'x_bias_|y_bias_|z_bias_|spin_b_field|'
    r'clock_|recycling_|TD_|cavity_|lattice_|'
    r'yellow_doublepass|mot_coil|green_frequency|'
    r'green_molasses_|atom_number_)'
)


def _extract_runmanager_globals(tree: ast.AST) -> List[str]:
    """Walk the full AST and return names that are:
    - used as values (Load context),
    - never locally assigned or declared as a function argument,
    - match the lab's runmanager global naming patterns.
    These are free variables that must come from runmanager globals.
    """
    # Collect every name that is defined locally in the file
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            all_args = (node.args.args + node.args.posonlyargs +
                        node.args.kwonlyargs +
                        ([node.args.vararg] if node.args.vararg else []) +
                        ([node.args.kwarg] if node.args.kwarg else []))
            for arg in all_args:
                defined.add(arg.arg)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split('.')[0])

    # Collect Name nodes that look like runmanager globals
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name = node.id
            if (name not in _BUILTINS and
                    name not in defined and
                    '_' in name and
                    not name.startswith('__') and
                    _GLOBAL_PREFIX.match(name)):
                found.add(name)

    return sorted(found)


def _python_summary(full_source: str) -> str:
    """Extract a short summary from a Python source file.

    Uses the full (non-truncated) source so that globals defined deep in
    the file still appear in the summary for keyword search.
    """
    try:
        tree = ast.parse(full_source)
    except SyntaxError:
        lines = [ln for ln in full_source.splitlines() if ln.strip()]
        return "\n".join(lines[:20])

    parts: List[str] = []
    mdoc = ast.get_docstring(tree)
    if mdoc:
        parts.append(mdoc.strip()[:400])

    func_names: List[str] = []
    class_names: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            class_names.append(node.name)

    if class_names:
        parts.append("Classes: " + ", ".join(class_names))
    if func_names:
        parts.append("Top-level functions: " + ", ".join(func_names[:40]))

    globals_used = _extract_runmanager_globals(tree)
    if globals_used:
        parts.append("Runmanager globals used: " + ", ".join(globals_used))

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
        # Read full source for summary so globals deep in the file are captured,
        # even when the content field is truncated for token budget.
        try:
            full_source = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            full_source = text
        summary = _python_summary(full_source)
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