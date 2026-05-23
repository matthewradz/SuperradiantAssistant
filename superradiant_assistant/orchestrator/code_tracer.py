"""Code tracer — follows imports and function calls through labscript sequences
and subsequences to answer questions like 'where is global X used?' or
'what does function Y do?'.

Used by the answer task type.
"""
from __future__ import annotations
import ast
import re
from pathlib import Path
from typing import List, Dict, Set, Optional


_LABSCRIPT_ROOT = Path(
    r'C:\Users\radzi\Documents\labscript-suite\labscript-suite\userlib\labscriptlib\ybclock'
)


def _resolve_import(module: str) -> Optional[Path]:
    """Try to resolve a labscriptlib import to a file path."""
    # Convert module path to file path: labscriptlib.ybclock.subsequences.X -> subsequences/X.py
    parts = module.split(".")
    if "ybclock" in parts:
        idx = parts.index("ybclock")
        rel = Path(*parts[idx + 1:]).with_suffix(".py")
        candidate = _LABSCRIPT_ROOT / rel
        if candidate.exists():
            return candidate
        # Try as directory __init__.py
        candidate2 = _LABSCRIPT_ROOT / Path(*parts[idx + 1:]) / "__init__.py"
        if candidate2.exists():
            return candidate2
    return None


def _find_function_in_file(path: Path, func_name: str, max_lines: int = 60) -> Optional[str]:
    """Extract a function's source from a file."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    start = None
    base_indent = None
    for i, line in enumerate(lines):
        m = re.match(rf'^(\s*)def\s+{re.escape(func_name)}\s*\(', line)
        if m:
            start = i
            base_indent = len(m.group(1))
            break
    if start is None:
        return None
    body = [lines[start]]
    for line in lines[start + 1:]:
        stripped = line.rstrip()
        if stripped == "":
            body.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base_indent and stripped:
            break
        body.append(line)
        if len(body) >= max_lines:
            body.append("    # ... [truncated]")
            break
    return "\n".join(body)


def _find_global_uses(path: Path, global_name: str, context: int = 2) -> List[Dict]:
    """Find all lines in a file where global_name appears, with surrounding context."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    results = []
    for i, line in enumerate(lines):
        if global_name in line and not line.strip().startswith("#"):
            start = max(0, i - context)
            end = min(len(lines), i + context + 1)
            snippet = "\n".join(
                f"{'>>>' if j == i else '   '} {j+1:>4}: {lines[j].rstrip()}"
                for j in range(start, end)
            )
            results.append({"line": i + 1, "snippet": snippet})
    return results


def trace_global(global_name: str, search_roots: Optional[List[Path]] = None) -> str:
    """Trace where a global is used across sequences and subsequences."""
    if search_roots is None:
        search_roots = [
            _LABSCRIPT_ROOT / "sequences",
            _LABSCRIPT_ROOT / "subsequences",
        ]

    report_lines = [f"# Usages of `{global_name}` across sequences and subsequences\n"]
    total = 0

    for root in search_roots:
        if not root.exists():
            continue
        for py_file in sorted(root.rglob("*.py")):
            uses = _find_global_uses(py_file, global_name)
            if uses:
                rel = py_file.relative_to(_LABSCRIPT_ROOT)
                report_lines.append(f"\n## {rel}")
                for u in uses[:6]:  # cap per file
                    report_lines.append(u["snippet"])
                    report_lines.append("")
                total += len(uses)

    if total == 0:
        report_lines.append(f"No usages of `{global_name}` found in sequences or subsequences.")
    else:
        report_lines.append(f"\nTotal: {total} usages across files.")

    return "\n".join(report_lines)


def trace_function(func_name: str, start_file: Optional[Path] = None) -> str:
    """Trace a function's definition and what it calls."""
    # Search for the function in all subsequence files
    search_dirs = [
        _LABSCRIPT_ROOT / "subsequences",
        _LABSCRIPT_ROOT / "sequences",
        _LABSCRIPT_ROOT / "connection_functions",
    ]
    if start_file:
        search_dirs.insert(0, start_file.parent)

    report_lines = [f"# Definition of `{func_name}`\n"]

    for d in search_dirs:
        if not d.exists():
            continue
        for py_file in sorted(d.rglob("*.py")):
            body = _find_function_in_file(py_file, func_name)
            if body:
                rel = py_file.relative_to(_LABSCRIPT_ROOT)
                report_lines.append(f"Found in: `{rel}`\n")
                report_lines.append(f"```python\n{body}\n```")
                return "\n".join(report_lines)

    report_lines.append(f"Function `{func_name}` not found in subsequences or sequences.")
    return "\n".join(report_lines)


def build_answer_context(query: str) -> str:
    """Given a Q&A query, extract relevant code context for the LLM to answer."""
    chunks = []

    # Extract global names (snake_case, 2+ segments)
    globals_mentioned = re.findall(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', query.lower())

    # Extract function names (look for words followed by () in query)
    funcs_mentioned = re.findall(r'\b([a-z][a-z0-9_]+)\s*(?:\(|function|subsequence)', query.lower())

    for fn in funcs_mentioned[:3]:
        result = trace_function(fn)
        if "not found" not in result:
            chunks.append(result)

    for gn in globals_mentioned[:3]:
        if len(gn) > 8:  # skip short common words
            result = trace_global(gn)
            if "No usages" not in result:
                chunks.append(result)

    return "\n\n---\n\n".join(chunks) if chunks else ""
