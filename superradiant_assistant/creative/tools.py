"""Creative-mode tools: the model writes experiment code, the operator approves it.

These are the only tools in the system that write files or add parameters, and
they exist solely in creative mode. Every one of them is registered as
`always_confirm`, so `--yes` cannot skip the prompt — the whole point is that a
human reads the code before it reaches hardware.

Two things guard each write:

  1. `code_guard` parses the source and refuses anything that imports outside the
     whitelist, reaches for eval/exec/open, or calls a device method directly
     instead of going through the range-checked wrapper. This runs *before* the
     operator is asked, so obviously-bad code never wastes their attention.
  2. The confirmation prompt shows the full source. That is the real boundary;
     the guard is a filter in front of it, not a substitute for it.
"""
from __future__ import annotations
import io
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from superradiant_assistant import script_inventory
from superradiant_assistant.config import CONFIG, REPO_ROOT
from superradiant_assistant.creative import code_guard
from superradiant_assistant.creative.globals_store import (
    GlobalProposal, create as create_global, validate as validate_global,
)
from superradiant_assistant.hooks import ChoiceOption, ChoicePrompt
from superradiant_assistant.tools.registry import ToolSpec

SHOT_DIR = script_inventory.SHOT_DIR
ANALYSIS_DIR = script_inventory.ANALYSIS_DIR
MULTISHOT_DIR = script_inventory.MULTISHOT_DIR
SKILLS_DIR = REPO_ROOT / "superradiant_assistant" / "skills"
REPORT_DIR = Path(CONFIG.historical_data_root).parent / "reports"

CREATIVE_AGENTS = frozenset({"lead", "coder"})

#: Writing code and loading it into runmanager is the coder's job now, not the
#: lead's -- the lead decides and reports, the coder implements. `save_experiment_
#: skill` stays on CREATIVE_AGENTS: writing up a working procedure is the same
#: kind of decision as `write_report`, which has always been lead-only.
CODE_AGENTS = frozenset({"coder"})


def _safe_stem(name: str) -> Optional[str]:
    """A filename the model chose is still a path — keep it to a bare stem."""
    stem = Path(str(name)).stem
    if not stem or not stem.replace("_", "").replace("-", "").isalnum():
        return None
    return stem


#: Longer than this and the path wraps the confirmation box, which is how a
#: prompt stops being readable.
_MAX_SLUG = 56


def _slug(title: str) -> str:
    """A report title turned into a filename, rather than rejected as one.

    `_safe_stem` refuses anything containing a space, so every English title the
    model produced -- "AWG CH2 Filter Frequency Response & Cutoff Determination"
    -- fell back to the literal stem `report` and overwrote the previous one.
    Chinese titles survived only because CJK characters are `isalnum()` and
    those titles happened to have no spaces.
    """
    out, prev_sep = [], False
    for ch in str(title).strip():
        if ch.isalnum():
            out.append(ch)
            prev_sep = False
        elif not prev_sep and out:
            out.append("_")
            prev_sep = True
    stem = "".join(out).strip("_")[:_MAX_SLUG].strip("_")
    return stem or "report"


def _report_day_dir(when: Optional[datetime] = None) -> Path:
    """One folder per day of measurements: reports/YYYY-MM-DD/.

    A day's reports belong together -- they are the same apparatus state, the
    same alignment, often the same question asked three ways -- and a flat folder
    of thirty files named after their titles gives no way to see that. The
    figures go in `<day>/figures/`, which keeps the `figures/...` links inside
    each report relative to the report itself, so a day's folder can be moved or
    sent whole and the images still resolve.
    """
    when = when or datetime.now()
    return REPORT_DIR / f"{when:%Y-%m-%d}"


def _report_stem(title: str, when: Optional[datetime] = None,
                 day: Optional[Path] = None) -> str:
    """A dated, collision-free stem, so a report is never silently replaced.

    A report records one particular run. Overwriting it destroys the record of
    a measurement that cannot be reproduced from the file that replaced it, so
    unlike a script there is no case for reuse -- it always gets its own name.

    Collisions are checked inside the day's folder, and also against the flat
    REPORT_DIR: reports written before the folders existed still live there.
    """
    when = when or datetime.now()
    day = day if day is not None else _report_day_dir(when)
    stem = _slug(title)
    suffix = f"{when:%Y%m%d}"
    if stem.endswith(f"_{suffix}"):          # the model often dates its own
        stem = stem[: -(len(suffix) + 1)] or "report"
    base = f"{stem}_{suffix}"

    def taken(name: str) -> bool:
        return (day / f"{name}.md").exists() or (REPORT_DIR / f"{name}.md").exists()

    if not taken(base):
        return base
    for n in range(2, 1000):
        if not taken(f"{base}_{n}"):
            return f"{base}_{n}"
    return f"{base}_{when:%H%M%S}"


def _numbered(source: str) -> str:
    return "\n".join(f"  {i:3d} | {line}"
                     for i, line in enumerate(source.splitlines(), 1))


# --------------------------------------------------------------------------
# add / replace / reuse -- the three-way question in front of every script write
# --------------------------------------------------------------------------

def _script_choices(kind: str, args: Dict[str, Any]) -> Optional[ChoicePrompt]:
    """Ask whether to add, replace, or reuse -- instead of a bare y/N.

    The model treats these three as interchangeable and picks whichever its
    phrasing fell into, so a fresh idea silently replaced a working script and a
    trivial variation became the ninth near-duplicate in the folder. None of that
    is visible in a y/N prompt, which asks only "run this call?" and never "was
    this the right one of the three?".
    """
    stem = _safe_stem(args.get("filename", ""))
    if stem is None:
        return None                     # the handler will refuse it anyway

    existing = script_inventory.find(kind, stem)
    dated = script_inventory.dated_stem(stem, kind)
    directory = script_inventory.KIND_DIRS[kind]
    others = script_inventory.compact_list(kind)

    context = []
    if existing is not None:
        origin = ("written by creative mode" if existing.generated
                  else "NOT written by creative mode -- hand-written work")
        context += [
            f"'{stem}.py' ALREADY EXISTS in {directory}",
            f"    {existing.n_lines} lines, last changed "
            f"{existing.modified:%Y-%m-%d %H:%M}, {origin}",
        ]
    else:
        context.append(f"No '{stem}.py' exists yet in {directory}")
    if others:
        context.append("")
        context.append(f"{kind} scripts that already exist:")
        context += [f"    {line}" for line in others]

    replace_detail = (f"replace {stem}.py -- its {existing.n_lines} lines are lost"
                      if existing is not None
                      else f"write {stem}.py under the name the agent chose")

    return ChoicePrompt(
        question="Add a new script, replace this one, or reuse what exists?",
        context="\n".join(context),
        options=[
            ChoiceOption(
                key="1", label="new",
                detail=f"write a dated new file: {dated}.py",
                apply=lambda a: {**a, "filename": dated},
            ),
            ChoiceOption(
                key="2", label="replace", detail=replace_detail,
                apply=lambda a: a,
            ),
            ChoiceOption(
                key="3", label="reuse",
                detail="do not write; use a script that already exists",
                apply=None,
                deny_reason=(
                    "the operator wants an existing script reused instead of a new "
                    "one. Do NOT call write_" + ("shot" if kind == "shot" else "analysis")
                    + " again for this. Call list_scripts, read the closest match with "
                    "read_lab_file, and use it as it is — adjusting globals rather than "
                    "code if that is enough. Existing " + kind + " scripts:\n    "
                    + "\n    ".join(others or ["(none)"])
                ),
            ),
        ],
    )


# --------------------------------------------------------------------------
# C1  propose_global
# --------------------------------------------------------------------------

def propose_global(name: str, minimum: float, maximum: float, initial: float,
                   description: str, reason: str, units: str = "") -> str:
    p = GlobalProposal(name=name, minimum=float(minimum), maximum=float(maximum),
                       initial=float(initial), units=units,
                       description=description, reason=reason)
    problem = validate_global(p)
    if problem:
        return f"refused: {problem}"
    return create_global(p)


def _preview_propose_global(args: Dict[str, Any]) -> str:
    try:
        p = GlobalProposal(
            name=args.get("name", "?"),
            minimum=float(args.get("minimum", 0)),
            maximum=float(args.get("maximum", 0)),
            initial=float(args.get("initial", 0)),
            units=args.get("units", ""),
            description=args.get("description", ""),
            reason=args.get("reason", ""),
        )
    except (TypeError, ValueError) as e:
        return f"(cannot preview: {e})"
    problem = validate_global(p)
    body = p.preview()
    if problem:
        body += f"\n\nWARNING: this will be refused — {problem}"
    return body


# --------------------------------------------------------------------------
# C2  write_shot
# --------------------------------------------------------------------------

def write_shot(filename: str, source: str, description: str, reason: str,
               key_globals: Optional[List[str]] = None) -> str:
    stem = _safe_stem(filename)
    if stem is None:
        return f"refused: '{filename}' is not a usable script name"

    report = code_guard.check_shot(source)
    if not report.ok:
        return ("refused by code guard — fix these and call again:\n"
                + report.summary())

    path = SHOT_DIR / f"{stem}.py"
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    io.open(path, "w", encoding="utf-8").write(source.rstrip() + "\n")

    from superradiant_assistant.creative.evidence import get_evidence
    get_evidence().record_action(
        f"{'replaced' if existed else 'wrote'} shot {path.name}")

    registered = _register_sequence(str(path).replace("\\", "/"), description,
                                    key_globals or [])
    return (f"{'overwrote' if existed else 'wrote'} {path}\n"
            f"{registered}\n"
            f"The agent must RESTART before this sequence appears in its tool schema.")


def _register_sequence(file_path: str, description: str,
                       key_globals: List[str]) -> str:
    """Add the shot to config.json so the safety layer will accept it."""
    cfg_path = REPO_ROOT / "config.json"
    cfg = json.loads(io.open(cfg_path, encoding="utf-8").read())
    seqs = cfg.setdefault("sequences", [])
    for s in seqs:
        if str(s.get("file", "")).replace("\\", "/").lower() == file_path.lower():
            s["description"] = description
            s["key_globals"] = key_globals
            io.open(cfg_path, "w", encoding="utf-8").write(
                json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
            return "updated its entry in config.json"
    seqs.append({
        "file": file_path,
        "description": description,
        "use_case": "written in creative mode",
        "key_globals": key_globals,
    })
    io.open(cfg_path, "w", encoding="utf-8").write(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    return "registered it in config.json"


def _overwrite_warning(path: Path) -> List[str]:
    """Loud, specific warning when a write would replace existing work.

    Burying "overwrote" in the result string is not enough: the operator is
    reading the diff to judge the new code, and needs to know at that moment
    that saying yes destroys something.
    """
    if not path.exists():
        return []
    try:
        n_lines = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        n_lines = -1

    warn = [
        "",
        "!! OVERWRITES AN EXISTING FILE !!",
        f"   {path}",
        f"   current file is {n_lines} lines" if n_lines >= 0 else "   current file unreadable",
    ]
    if _was_generated(path):
        warn.append("   (this file was written in creative mode — replacing its own "
                    "earlier version)")
    else:
        warn.append("   This file was NOT written by creative mode — it may be "
                    "hand-written work. Saying yes replaces it.")
    return warn


def _was_generated(path: Path) -> bool:
    """Whether creative mode wrote this file, per config.json's registration.

    Provenance lives in config.json, not in the file body — an earlier version
    searched the source text for a marker that is never written there, so every
    overwrite claimed to be destroying hand-written work. A warning that always
    fires is a warning nobody reads.
    """
    try:
        cfg = json.loads(io.open(REPO_ROOT / "config.json", encoding="utf-8").read())
    except (OSError, ValueError):
        return False
    target = str(path).replace("\\", "/").lower()
    for s in cfg.get("sequences", []):
        if str(s.get("file", "")).replace("\\", "/").lower() == target:
            return s.get("use_case") == "written in creative mode"
    return False


def _preview_write_shot(args: Dict[str, Any]) -> str:
    source = args.get("source", "")
    stem = _safe_stem(args.get("filename", ""))
    report = code_guard.check_shot(source)
    lines = [
        f"file      : {SHOT_DIR / ((stem or '?') + '.py')}",
        f"globals   : {', '.join(args.get('key_globals') or []) or '(none declared)'}",
        f"reason    : {args.get('reason', '(none)')}",
        f"guard     : {'passed' if report.ok else 'FAILED'}",
    ]
    if report.device_calls:
        lines.append(f"device    : {', '.join(sorted(set(report.device_calls)))}")
    if not report.ok:
        lines += [f"            {v}" for v in report.violations]
    lines += _overwrite_warning(SHOT_DIR / ((stem or "_") + ".py"))
    lines.append("")
    lines.append("--- source ---")
    lines.append(_numbered(source))
    lines.append("--- end ---")
    lines.append("This writes a file and the shot will run on real hardware.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# C3  write_analysis
# --------------------------------------------------------------------------

def write_analysis(filename: str, source: str, description: str, reason: str,
                   kind: str = "singleshot") -> str:
    if kind not in ("singleshot", "multishot"):
        return f"refused: kind must be 'singleshot' or 'multishot', got {kind!r}"

    stem = _safe_stem(filename)
    if stem is None:
        return f"refused: '{filename}' is not a usable script name"

    report = code_guard.check_analysis(source, multishot=(kind == "multishot"))
    if not report.ok:
        return ("refused by code guard — fix these and call again:\n"
                + report.summary())

    path = (MULTISHOT_DIR if kind == "multishot" else ANALYSIS_DIR) / f"{stem}.py"
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    io.open(path, "w", encoding="utf-8").write(source.rstrip() + "\n")

    from superradiant_assistant.creative.evidence import get_evidence
    get_evidence().record_action(
        f"{'replaced' if existed else 'wrote'} {kind} routine {path.name}")

    # Writing the file is only half the job: lyse runs the routines in its own
    # list, and nothing puts a new file there automatically. Try to set it
    # directly; fall back to preparing a config the operator can load.
    try:
        from superradiant_assistant.interfaces.lyse_iface import (
            set_live_routines, request_routine,
        )
        loading = set_live_routines([str(path)], replace=True, kind=kind)
        if loading.startswith("FAILED") or loading.startswith("error"):
            loading += "\n\n" + request_routine(str(path), kind=kind)
    except Exception as e:
        loading = (f"(could not reach lyse: {e}) "
                   f"Ask the operator to add this routine in lyse by hand.")

    note = ("Whatever it passes to save_result becomes a readable metric."
            if kind == "singleshot" else
            "This runs once after the sequence, over the whole dataframe.")
    return (f"{'overwrote' if existed else 'wrote'} {kind} routine {path}\n"
            f"{note}\n\n{loading}")


def _preview_write_analysis(args: Dict[str, Any]) -> str:
    # `kind` decides both the directory and which guard rules apply. Ignoring it
    # here showed the operator a singleshot path and a bogus "guard: FAILED" for
    # every multishot write -- wrong information at exactly the moment they were
    # being asked to judge it.
    source = args.get("source", "")
    kind = args.get("kind", "singleshot")
    multishot = kind == "multishot"
    stem = _safe_stem(args.get("filename", ""))
    report = code_guard.check_analysis(source, multishot=multishot)
    lines = [
        f"kind   : {kind}",
        f"file   : {(MULTISHOT_DIR if multishot else ANALYSIS_DIR) / ((stem or '?') + '.py')}",
        f"reason : {args.get('reason', '(none)')}",
        f"guard  : {'passed' if report.ok else 'FAILED'}",
    ]
    if not report.ok:
        lines += [f"         {v}" for v in report.violations]
    lines += _overwrite_warning(
        (MULTISHOT_DIR if multishot else ANALYSIS_DIR) / ((stem or "_") + ".py"))
    lines += ["", "--- source ---", _numbered(source), "--- end ---",
              "This runs once per sweep, over the whole dataframe." if multishot
              else "lyse will run this on every shot once it is loaded."]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# C4  save_experiment_skill
# --------------------------------------------------------------------------

def save_experiment_skill(skill_name: str, description: str, body: str,
                          tags: str = "") -> str:
    from superradiant_assistant.creative.evidence import get_evidence

    stem = _safe_stem(skill_name)
    if stem is None:
        return f"refused: '{skill_name}' is not a usable skill name"

    # A skill is a claim that this procedure works. Saving one from a procedure
    # that was never run would hand the same fiction to every future session.
    evidence = get_evidence()
    if not evidence.has_measurements:
        return (f"refused: this session produced no measurements, so the procedure "
                f"is unproven. {evidence.summary()}. Run it end to end first.")

    path = SKILLS_DIR / stem / "SKILL.md"
    existed = path.exists()
    front = "\n".join([
        "---",
        f"name: {stem}",
        f"description: {description.strip()}",
        f"tags: {tags.strip()}" if tags.strip() else "tags: creative-mode",
        "---",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    io.open(path, "w", encoding="utf-8").write(front + body.rstrip() + "\n")
    return (f"{'overwrote' if existed else 'wrote'} {path}\n"
            f"It loads on the next agent start, and works with creative mode OFF.")


def _preview_save_skill(args: Dict[str, Any]) -> str:
    stem = _safe_stem(args.get("skill_name", ""))
    return "\n".join([
        f"skill  : {stem}",
        f"file   : {SKILLS_DIR / ((stem or '?')) / 'SKILL.md'}",
        f"summary: {args.get('description', '(none)')}",
        "",
        "--- content ---",
        args.get("body", ""),
        "--- end ---",
        "Saving makes this procedure reusable when creative mode is off.",
    ])


# --------------------------------------------------------------------------
# C5  write_report
# --------------------------------------------------------------------------

def _unverified_ids(text: str) -> List[str]:
    """Shot identifiers in `text` that this session never actually measured.

    Written artefacts travel as tool arguments, which the reply-level identifier
    guard never sees — a report went to disk quoting four shot IDs from a sweep
    the operator had declined.
    """
    from superradiant_assistant.creative.evidence import get_evidence
    from superradiant_assistant.orchestrator.id_guard import find_ids

    measured = {s.lower() for s in get_evidence().shot_ids}
    bad = []
    for tok in find_ids(text or ""):
        low = tok.lower()
        if not any(low in m or m in low for m in measured):
            bad.append(tok)
    return bad


FIGURE_DIR = REPORT_DIR / "figures"

#: Anything more than this and the figure is a wall of thumbnails rather than a
#: result. The primary quantities come first, so the cut falls on diagnostics.
_MAX_PANELS = 6

#: Plotted first when present: these are what a sweep is usually measuring.
#: Order matters twice over -- it decides which panel leads, and because
#: duplicates are dropped, which of two identical columns keeps its name.
#: `transfer` outranks `ratio` so `transfer_ratio` survives and the alphabetical
#: tie-break does not hand the panel to `ch1_ch2_ratio`.
_PRIMARY_HINTS = ("transfer", "ratio", "_db", "eta", "amp", "vpp")


def _panel_order(keys: List[str]) -> List[str]:
    """Measured quantities, the ones a reader wants first at the front."""
    def rank(k: str) -> tuple:
        low = k.lower()
        for i, hint in enumerate(_PRIMARY_HINTS):
            if hint in low:
                return (i, k)
        return (len(_PRIMARY_HINTS), k)
    return sorted(keys, key=rank)


def _sweep_figure(stem: str, measurements: List[Dict[str, Any]],
                  sweep_param: str, dest: Optional[Path] = None) -> Optional[Path]:
    """Plot the reported rows, one panel per measured quantity.

    Drawn from `evidence.measurements` -- the same rows the report's table is
    written from -- so the figure cannot disagree with the text. A report whose
    plot and prose describe different fits is worse than one with no plot, and
    that is exactly what happened on 2026-08-10.
    """
    if not measurements or not sweep_param:
        return None
    xs_all = [m.get(sweep_param) for m in measurements]
    if sum(1 for x in xs_all if isinstance(x, (int, float))) < 2:
        return None

    candidates = []
    for m in measurements:
        for k, v in m.items():
            if k in ("shot_id", sweep_param) or k in candidates:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                candidates.append(k)

    def series(k):
        return tuple(round(float(m[k]), 12) for m in measurements
                     if isinstance(m.get(k), (int, float)))

    # A panel earns its place by saying something the others do not. Without
    # this the filter sweep drew six panels of which `ch1_ch2_ratio` was pixel
    # for pixel `transfer_ratio`, and `CHAN1_vpp` was the constant 1.12.
    keys, seen = [], {}
    for k in _panel_order(candidates):
        ys = series(k)
        if len(set(ys)) < 2:
            continue                            # constant: nothing to plot
        if ys in seen:
            continue                            # identical to a panel already kept
        seen[ys] = k
        keys.append(k)
    keys = keys[:_MAX_PANELS]
    if not keys:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    cols = 1 if len(keys) == 1 else 2
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.4 * rows),
                             squeeze=False)
    for ax, key in zip([a for row in axes for a in row], keys):
        pts = [(m[sweep_param], m[key]) for m in measurements
               if isinstance(m.get(sweep_param), (int, float))
               and isinstance(m.get(key), (int, float))]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=4, lw=1.2)
        ax.set_xlabel(sweep_param)
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
    # Blank out any unused cell rather than leaving an empty framed box.
    for ax in [a for row in axes for a in row][len(keys):]:
        ax.axis("off")
    fig.suptitle(f"{len(measurements)} shot(s) vs {sweep_param}", fontsize=10)
    fig.tight_layout()

    # `dest` is the day folder's own figures/ directory; FIGURE_DIR is the flat
    # one, kept as the default so a caller that predates the day folders still
    # works.
    figdir = dest if dest is not None else FIGURE_DIR
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / f"{stem}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _align_tables(md: str) -> str:
    """Pad pipe-table cells so the raw Markdown is readable as text.

    The model emits `| :--- |` separators and ragged cells; rendered it is fine,
    but the .md is the file that gets opened in an editor and pasted into a
    thesis, and unaligned pipes there are unreadable.
    """
    out, block = [], []

    def flush():
        if not block:
            return
        grids = [[c.strip() for c in row.strip().strip("|").split("|")]
                 for row in block]
        width = max(len(g) for g in grids)
        grids = [g + [""] * (width - len(g)) for g in grids]
        sep_rows = {i for i, g in enumerate(grids)
                    if all(re.fullmatch(r":?-{2,}:?", c or "-") for c in g)}
        sizes = [max(len(g[i]) for j, g in enumerate(grids) if j not in sep_rows)
                 for i in range(width)]
        sizes = [max(3, s) for s in sizes]
        for i, g in enumerate(grids):
            if i in sep_rows:
                cells = ["-" * s for s in sizes]
            else:
                cells = [c.ljust(s) for c, s in zip(g, sizes)]
            out.append("| " + " | ".join(cells) + " |")
        block.clear()

    for line in md.splitlines():
        if line.strip().startswith("|") and line.strip().endswith("|"):
            block.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def _yaml(value: str) -> str:
    """Quote a scalar so a colon or a `#` in it cannot break the frontmatter."""
    s = str(value)
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _frontmatter(title: str, evidence) -> str:
    """Obsidian properties: where the numbers came from, stated by code.

    Every field here is one the model has got wrong at least once -- the shot
    count, the swept range, which shots were actually read. Reading them off the
    record costs nothing and makes the report searchable in the vault.
    """
    retro = not getattr(evidence, "measured_here", True)
    lines = ["---",
             f"title: {_yaml(title)}",
             f"date: {datetime.now():%Y-%m-%d}",
             f"recorded: {datetime.now():%Y-%m-%d %H:%M}",
             # Stated as a property, not buried in prose: a report written from
             # data on disk is a different claim from one written from a run the
             # assistant just performed, and the difference must survive being
             # skimmed a month later.
             f"source: {'shot files on disk' if retro else 'measured this session'}",
             f"shots: {len(evidence.shot_ids)}",
             f"sweeps: {evidence.sweeps_with_data}"]
    for param in getattr(evidence, "sweep_params", []):
        xs = sorted(m[param] for m in evidence.measurements
                    if isinstance(m.get(param), (int, float)))
        if xs:
            lines += [f"swept: {param}",
                      f"swept_from: {xs[0]:g}",
                      f"swept_to: {xs[-1]:g}",
                      f"points: {len(xs)}"]
    ids = sorted(evidence.shot_ids)
    if ids:
        # First and last only. Obsidian shows frontmatter in a properties panel,
        # and twenty-six list entries there bury the four fields worth reading.
        # The full list goes at the end of the body instead.
        lines += [f"shot_first: {_yaml(ids[0])}", f"shot_last: {_yaml(ids[-1])}"]
    lines += ["tags:", "  - lab/report", "  - apparatus/cesium"]
    if retro:
        lines.append("  - retrospective")
    lines.append("---")
    return "\n".join(lines)


def _shot_appendix(evidence) -> str:
    """The shots behind the report, listed once at the end.

    Provenance a reader can check, without it being the first thing they see.
    """
    ids = sorted(evidence.shot_ids)
    if not ids:
        return ""
    listed = ", ".join(f"`{i}`" for i in ids)
    return (f"\n## Shots\n\n{len(ids)} shot(s) read back for this report:\n\n"
            f"{listed}\n")


#: How many per-shot lyse figures to embed. A three-point phase sweep wants all
#: of them; a 23-point filter sweep would otherwise put 23 traces above the text.
_MAX_SHOT_FIGURES = 6


def _shot_figures(evidence) -> List[Path]:
    """The figures the lyse routines saved beside their shot files.

    lyse draws into its own window; unless the routine calls `savefig` the plot
    the operator watched exists nowhere else, and the report could only show
    `_sweep_figure`'s scalar-vs-parameter plot. A Lissajous curve is an X-Y plot
    of two waveforms *within* one shot, so no amount of plotting the results
    table can reproduce it -- it has to come from lyse.

    Convention: `<shot>__<figure>.png` next to the shot's .h5.

    ONE figure per kind, and the newest one. lyse re-runs every multishot routine
    after every shot, so a routine that names its output after the newest shot
    leaves one file per shot, each a partial version of the same plot drawn
    before the run had finished. A twenty-shot run left nineteen; the report
    embedded six of the unfinished ones and none of the final one, twice, with
    two different routines. The kind is the part after `__` -- the routine's own
    label for the plot -- so grouping on it and keeping the newest file gives
    exactly the last state of each distinct figure.
    """
    newest: Dict[str, Path] = {}
    for m in evidence.measurements:
        shot = m.get("shot_path")
        if not shot:
            continue
        p = Path(shot)
        for f in p.parent.glob(f"{p.stem}__*.png"):
            kind = f.stem.partition("__")[2]
            try:
                when = f.stat().st_mtime
            except OSError:
                continue
            best = newest.get(kind)
            if best is None or when > best.stat().st_mtime:
                newest[kind] = f
    # Sorted by kind, so the order does not depend on which shot was walked first.
    return [newest[k] for k in sorted(newest)][:_MAX_SHOT_FIGURES]


def _figure_section(figure: Optional[Path],
                    shot_figures: Optional[List[Path]] = None) -> str:
    parts = []
    if figure:
        parts.append(f"![{figure.stem}](figures/{figure.name})\n\n"
                     f"*Plotted from the same rows the table below is written "
                     f"from.*")
    for f in shot_figures or []:
        shot, _, name = f.stem.partition("__")
        parts.append(f"![{f.stem}](figures/{f.name})\n\n"
                     f"*{name.replace('_', ' ')} — `{shot}`, drawn by lyse.*")
    if not parts:
        return ""
    return "## Measured data\n\n" + "\n\n".join(parts) + "\n"


def _log_to_notebook(title: str, md_path: Path, evidence) -> Optional[str]:
    """One notebook log entry per report, written when the report is.

    The log used to be written only by the memory curator, which runs on exit or
    when the context grows -- so a day's page carried two entries covering six
    experiments, stamped with the times compaction happened rather than the times
    the measurements finished. Writing it here makes the stamp real and makes the
    correspondence exact: one report, one entry, and the entry names the file.

    Returns the day it wrote to, or None if the notebook could not be reached --
    a failure here must not lose the report, which is already on disk.
    """
    from superradiant_assistant.memory import MEMORY

    params = ", ".join(getattr(evidence, "sweep_params", []) or [])
    span = ""
    rows = [r for r in evidence.measurements if params.split(",")[0].strip() in r]
    if rows and params:
        first = params.split(",")[0].strip()
        try:
            xs = sorted(float(r[first]) for r in rows if r.get(first) is not None)
            span = f", {first} {xs[0]:g} to {xs[-1]:g}" if xs else ""
        except (TypeError, ValueError):
            span = ""

    # Relative to the reports root, which is what makes it findable from the
    # notebook without hard-coding either folder's absolute path.
    try:
        where = md_path.relative_to(REPORT_DIR).as_posix()
    except ValueError:
        where = md_path.as_posix()

    kind = ("read back from disk" if not getattr(evidence, "measured_here", True)
            else "measured")
    entry = (f"**{title}** — {len(evidence.shot_ids)} shot(s) {kind}{span}\n"
             f"report: `{where}`")
    try:
        MEMORY.append_episode(entry)
        return MEMORY.episode_path().name
    except Exception as e:
        print(f"  [report] written, but the notebook entry failed: "
              f"{type(e).__name__}: {e}")
        return None


def write_report(title: str, markdown: str, reason: str = "") -> str:
    from superradiant_assistant.creative.evidence import get_evidence

    evidence = get_evidence()
    if not evidence.has_measurements:
        return (
            "refused: no data has been read in this session, so there is nothing "
            f"to report. {evidence.summary()}. Either run the measurement, or "
            "call `read_shot_results` to read the shots you want to report on — "
            "a report about shots already on disk is fine and will be marked "
            "retrospective. What is refused is a report written from "
            "expectations or from what the apparatus 'should' do."
        )

    bogus = _unverified_ids(markdown)
    if bogus:
        return (
            f"refused: the report cites shot identifiers this session never "
            f"measured: {', '.join(bogus[:8])}. Measured shots are: "
            f"{', '.join(sorted(evidence.shot_ids)[:8]) or '(none)'}. "
            f"Quote only shots you actually read back."
        )

    # One measurement, one report. `write_report` belongs to both `lead` and
    # `coder`, and each keeps its own plan, so a delegated sweep produced two:
    # the coder wrote one, reported back, and the lead — its own report step
    # still open — wrote a second from the same 23 shots.
    already = evidence.report_for_same_shots()
    if already:
        return (
            f"already reported: {already}\n"
            f"That report covers exactly these {len(evidence.shot_ids)} shot(s), "
            f"so this would be a duplicate. Point the operator at the existing "
            f"file and mark the step done. Write another only after measuring "
            f"something new."
        )

    day = _report_day_dir()
    day.mkdir(parents=True, exist_ok=True)
    figdir = day / "figures"
    stem = _report_stem(title, day=day)

    figure = None
    try:
        figure = _sweep_figure(stem, evidence.measurements,
                              (getattr(evidence, "sweep_params", []) or [""])[0],
                              dest=figdir)
    except Exception as e:                      # a failed plot must not lose the report
        print(f"  [report] could not draw the figure: {type(e).__name__}: {e}")

    # Copied rather than linked: the report is meant to survive being moved into
    # a vault or sent to someone, and the shot folder will not travel with it.
    shot_figs = []
    for src in _shot_figures(evidence):
        figdir.mkdir(parents=True, exist_ok=True)
        dst = figdir / src.name
        shutil.copy2(src, dst)
        shot_figs.append(dst)

    # LaTeX is left alone: the .md is read in Obsidian, which renders `$...$`.
    # The rule against LaTeX applies to the terminal and to memory, where
    # nothing renders it -- not here.
    body = _align_tables(markdown.rstrip())
    # The model supplies its own `# Title`; keep exactly one, and put the figure
    # between it and the model's prose.
    body = re.sub(r"^#\s+.*\n?", "", body, count=1).lstrip("\n")
    note = ""
    if not getattr(evidence, "measured_here", True):
        note = ("> [!info] Retrospective\n"
                "> Written from shot files already on disk; this session did not\n"
                "> run the measurement. The numbers come from the shots listed at\n"
                "> the end, not from anything queued while writing this.\n\n")
    doc = (f"{_frontmatter(title, evidence)}\n\n# {title}\n\n{note}"
           f"{_figure_section(figure, shot_figs)}\n{body}\n"
           f"{_shot_appendix(evidence)}")

    md_path = day / f"{stem}.md"
    io.open(md_path, "w", encoding="utf-8").write(doc)

    evidence.record_report(md_path)
    logged = _log_to_notebook(title, md_path, evidence)
    out = [f"wrote {md_path}"]
    if logged:
        out.append(f"logged in the notebook: {logged}")
    if figure:
        out.append(f"wrote {figure}  (embedded in the report)")
    else:
        out.append("no figure: the measured rows had no numeric column to plot")
    if shot_figs:
        out.append(f"embedded {len(shot_figs)} lyse figure(s) saved beside the "
                   f"shots")
    else:
        out.append("no lyse figures found beside the shots — a routine that "
                   "draws a plot must also save it as "
                   "`<shot>__<figure>.png`, or only this session sees it")
    return "\n".join(out)


def _preview_write_report(args: Dict[str, Any]) -> str:
    from superradiant_assistant.creative.evidence import get_evidence

    day = _report_day_dir()
    stem = _report_stem(args.get("title", ""), day=day)
    md = args.get("markdown", "")
    head = md if len(md) < 1500 else md[:1500] + "\n... (truncated in preview)"
    evidence = get_evidence()

    lines = [
        f"title    : {args.get('title')}",
        f"file     : {day / (stem + '.md')}",
        f"figure   : {day / 'figures' / (stem + '.png')}"
        + ("" if getattr(evidence, "sweep_params", []) else
           "   (none: no swept parameter was recorded)"),
        f"evidence : {evidence.summary()}",
    ]
    if not evidence.has_measurements:
        lines += [
            "",
            "!! NO DATA HAS BEEN READ IN THIS SESSION !!",
            "   Nothing has been measured and no shots have been read back.",
            "   This will be refused.",
        ]
    elif not getattr(evidence, "measured_here", True):
        lines += [
            "",
            "RETROSPECTIVE — written from shot files already on disk.",
            "   This session ran no measurement. The report will say so, and be",
            "   tagged `retrospective` in the vault.",
        ]
        bogus = _unverified_ids(md)
        if bogus:
            lines += ["", "!! CITES SHOTS NOT AMONG THOSE READ !!",
                      f"   {', '.join(bogus[:8])}"]
    else:
        bogus = _unverified_ids(md)
        if bogus:
            lines += [
                "",
                "!! CITES SHOTS THIS SESSION DID NOT MEASURE !!",
                f"   {', '.join(bogus[:8])}",
            ]
    lines += ["", "--- report ---", head, "--- end ---"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def build_creative_tool_specs() -> List[ToolSpec]:
    """Tools added only when creative mode is on."""
    return [
        ToolSpec(
            name="propose_global",
            description=(
                "Propose a NEW experiment parameter. The operator must approve it. "
                "Give a min/max you can defend on hardware grounds — the range you "
                "set here is what every later write is checked against. Only for "
                "parameters that do not exist yet; read the current ones first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Python identifier."},
                    "minimum": {"type": "number"},
                    "maximum": {"type": "number"},
                    "initial": {"type": "number", "description": "Starting value, inside [min, max]."},
                    "units": {"type": "string", "description": "e.g. Hz, V, s"},
                    "description": {"type": "string", "description": "What it controls, physically."},
                    "reason": {"type": "string", "description": "Why this experiment needs it."},
                },
                "required": ["name", "minimum", "maximum", "initial", "description", "reason"],
            },
            handler=propose_global,
            allowed_agents=CODE_AGENTS,
            preview=_preview_propose_global,
        ),
        ToolSpec(
            name="write_shot",
            description=(
                "Write a labscript shot script. Call "
                "load_skill('writing-experiment-code') first for the required "
                "skeleton and this apparatus's conventions — guessing them fails "
                "the code guard."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Bare name, no path or extension."},
                    "source": {"type": "string", "description": "Complete Python source."},
                    "description": {"type": "string", "description": "What the shot does."},
                    "key_globals": {"type": "array", "items": {"type": "string"},
                                     "description": "Globals the shot reads."},
                    "reason": {"type": "string"},
                },
                "required": ["filename", "source", "description", "reason"],
            },
            handler=write_shot,
            allowed_agents=CODE_AGENTS,
            preview=_preview_write_shot,
            choices=lambda a: _script_choices("shot", a),
        ),
        ToolSpec(
            name="write_analysis",
            description=(
                "Write a lyse analysis routine. `kind='singleshot'` runs on every "
                "shot and saves per-shot metrics; `kind='multishot'` runs once over "
                "the whole sequence and is the ONLY correct place for a curve across "
                "a sweep. Putting sweep-wide plotting in a singleshot routine "
                "re-fetches the dataframe and redraws on every shot, which stalls "
                "lyse. Call load_skill('writing-experiment-code') first. Start the "
                "source with a one-line module docstring saying what the routine "
                "measures — that line is how it appears in list_scripts, and a "
                "routine nobody can identify gets rewritten instead of reused."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Bare name, no path or extension."},
                    "source": {"type": "string", "description": "Complete Python source."},
                    "description": {"type": "string"},
                    "reason": {"type": "string"},
                    "kind": {"type": "string", "enum": ["singleshot", "multishot"],
                              "description": "Per-shot metrics, or once-per-sequence plotting."},
                },
                "required": ["filename", "source", "description", "reason"],
            },
            handler=write_analysis,
            allowed_agents=CODE_AGENTS,
            preview=_preview_write_analysis,
            choices=lambda a: _script_choices(
                "multishot" if a.get("kind") == "multishot" else "singleshot", a),
        ),
        ToolSpec(
            name="save_experiment_skill",
            description=(
                "Save a working experiment as a reusable SKILL.md. Do this only after "
                "the experiment has actually run and produced results. Write the "
                "procedure so it can be repeated with creative mode OFF: name the "
                "sequence and analysis files, the globals and their values, the wiring, "
                "and what the result should look like."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "kebab-case, e.g. lissajous-ratio"},
                    "description": {"type": "string", "description": "One line, used for skill matching."},
                    "body": {"type": "string", "description": "Markdown procedure."},
                    "tags": {"type": "string", "description": "Comma-separated."},
                },
                "required": ["skill_name", "description", "body"],
            },
            handler=save_experiment_skill,
            allowed_agents=CREATIVE_AGENTS,
            preview=_preview_save_skill,
        ),
        ToolSpec(
            name="write_report",
            description=(
                "Write the experiment report as Markdown and HTML. Include purpose, "
                "method, the actual measured numbers in a table, and what they mean. "
                "Quote only values you read back from shot files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "markdown": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["title", "markdown"],
            },
            handler=write_report,
            # The lead only. It is the agent that talks to the operator and holds
            # the whole picture, and the report is the deliverable. When the coder
            # had it too, each kept its own plan and a delegated sweep produced
            # two reports from the same 23 shots: the coder wrote one, reported
            # back, and the lead's own report step was still open.
            allowed_agents=frozenset({"lead"}),
            preview=_preview_write_report,
        ),
    ]


CREATIVE_TOOL_NAMES = frozenset({
    "propose_global", "write_shot", "write_analysis",
    "save_experiment_skill", "write_report",
})
