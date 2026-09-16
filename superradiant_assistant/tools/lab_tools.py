"""The lab tools (docs/tool-schema.md §2).

Each is a thin wrapper over existing code; the logic lives in the modules they
call. What they add is validation, a fixed data root, and a one-line signal
string in the shape "<conclusion> | <key number> | <cost>" — signals get replayed
into planning context, so they must be short and carry numbers. Failures return a
signal too: "no dip found | min ratio 0.981 | widen the sweep" is more useful to
the next planning turn than an exception string.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

from superradiant_assistant.config import CONFIG
from superradiant_assistant.safety import (
    SafetyViolation, load_global_specs, all_global_names, validate_scalar,
    sweep_is_bounded, allowed_sequence_files, MIN_SWEEP_POINTS, MAX_SWEEP_POINTS,
)
from superradiant_assistant.skill_loader import SKILLS
from superradiant_assistant.tools.registry import ToolSpec

LEGACY_METRIC_NAMES = ["Neta_1", "Neta_2", "Neta_3", "Neta_4", "Neta_5",
                       "chi_square_2", "r_sq_2"]

# How many recent shots to inspect when discovering what the analysis emits.
_METRIC_DISCOVERY_SHOTS = 12


def discovered_metric_names() -> List[str]:
    """Metric names this apparatus actually produces, newest shots first.

    The set is whatever lyse saved, so it follows the analysis scripts rather
    than a list baked in here — a hard-coded list is why an experiment emitting
    CH3_mean reported nothing but `None` for every metric.
    """
    from superradiant_assistant.interfaces.hdf5_reader import list_shots, read_shot

    found: List[str] = []
    try:
        paths = list_shots(CONFIG.historical_data_root)[-_METRIC_DISCOVERY_SHOTS:]
    except OSError:
        return []
    for p in reversed(paths):
        try:
            for name in read_shot(p).available_metrics():
                if name not in found:
                    found.append(name)
        except Exception:
            continue
    return found


def metric_names() -> List[str]:
    """Everything the model may name as a metric: declared, discovered, legacy."""
    names: List[str] = []
    for name in (list(CONFIG.metrics) + discovered_metric_names()
                 + LEGACY_METRIC_NAMES):
        if name and name not in names:
            names.append(name)
    return names


# Kept as a module-level list for callers that only need the legacy vocabulary.
METRIC_NAMES = LEGACY_METRIC_NAMES

#: The agents that ACT. The advisor is deliberately not in here, so that adding a
#: tool to `ALL_AGENTS` never quietly hands a write to the one agent that is
#: supposed to be unable to change anything.
ALL_AGENTS = frozenset({"lead", "planner", "coder"})

#: `ALL_AGENTS` plus the advisor: put this on a spec only if it is read-only.
ADVISED = ALL_AGENTS | frozenset({"advisor"})

#: Loading a sequence or wiring up lyse routines is part of implementing the
#: experiment, not deciding it -- the lead hands that off to the coder now.
CODER_ONLY = frozenset({"coder"})

#: `set_lyse_routines` was ALL_AGENTS before the handoff, and the planner's
#: access was not part of what moved -- only the lead's was.
ALL_AGENTS_MINUS_LEAD = ALL_AGENTS - frozenset({"lead"})


def _sequence_files() -> List[str]:
    return allowed_sequence_files()


def _lyse_readiness_warning() -> str:
    """Flag a sweep that will produce shots nothing is going to analyse."""
    try:
        from superradiant_assistant.interfaces import lyse_iface
        single = lyse_iface.live_routines("singleshot")
    except Exception:
        return ""
    if single is None:
        return ("\nCOULD NOT CHECK LYSE — if it is not running, these shots will "
                "produce no analysis results at all.")
    if not single:
        return ("\nWARNING: lyse has NO singleshot routines loaded. Every shot in "
                "this sweep will run the hardware and save traces, but nothing "
                "will compute any metric from them. Load a routine with "
                "set_lyse_routines before reading results.")
    return ""


def _sequence_uses_global(sequence_file: str, name: str) -> Optional[bool]:
    """Does `sequence_file` actually reference the global `name`?

    The model picks the sequence itself, and a wrong pick is silent: sweeping
    `sine_frequency` against a sequence that never reads it queues shots that
    all run identical settings. Returns None when the file cannot be read, so
    an unreadable path never blocks a sweep on its own.
    """
    try:
        src = Path(sequence_file).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    import re
    return re.search(rf"\b{re.escape(name)}\b", src) is not None


# --------------------------------------------------------------------------
# T1  search_lab_knowledge
# --------------------------------------------------------------------------

def search_lab_knowledge(query: str, role: str = "planner", top_k: int = 6,
                          rerank: str = "fusion", answer: bool = False) -> str:
    """Retrieve passages from the lab documents.

    Chunk / index / recall / rerank, from `knowledge.rag`. The previous version
    counted term occurrences in whole files, which returned every document or
    none of them and could not match a paraphrase at all.

    `rerank='llm'` adds a cross-encoder pass. It is not the default because it
    measured at 76 s against this corpus, and this tool gets called several
    times per turn.
    """
    from superradiant_assistant.knowledge import rag

    top_k = max(1, min(int(top_k), 10))
    if rerank not in ("fusion", "llm"):
        rerank = "fusion"
    try:
        return rag.search(query, top_k=top_k, rerank=rerank, synthesise=bool(answer))
    except Exception as e:
        # Retrieval must never take the turn down with it.
        from superradiant_assistant.knowledge.search import search_for_role
        from superradiant_assistant.tools import get_knowledge
        docs = search_for_role(get_knowledge(), query, role=role, top_k=top_k)
        head = (f"retrieval fell back to keyword search "
                f"({type(e).__name__}: {e})")
        if not docs:
            return f"{head}\nfound 0 docs for '{query}'"
        body = "\n\n".join(f"### {d.title}\n{d.content[:1200]}" for d in docs)
        return f"{head}\nfound {len(docs)} docs\n\n{body}"


# --------------------------------------------------------------------------
# T2  load_skill
# --------------------------------------------------------------------------

def load_skill(skill_name: str) -> str:
    return SKILLS.get_content(skill_name)


# --------------------------------------------------------------------------
# T3  read_shot_results
# --------------------------------------------------------------------------

def read_shot_results(limit: int = 20, metrics: Optional[List[str]] = None,
                       include_globals: bool = False) -> str:
    from superradiant_assistant.interfaces.hdf5_reader import list_shots, read_shot

    limit = max(1, min(int(limit), 200))
    known = metric_names()
    wanted = [m for m in (metrics or known) if m in known] or known

    # The data root is fixed by config, never passed in — otherwise the model
    # could read arbitrary paths off this machine.
    paths = list_shots(CONFIG.historical_data_root)[-limit:]
    if not paths:
        return f"read 0 shots | root={CONFIG.historical_data_root} | no .h5 files found"

    rows, best_val, best_id = [], None, None
    allowed_globals = set(all_global_names())
    # Recorded so a report can be written about data read back from disk. The
    # report guard used to see an empty evidence record and refuse outright,
    # which made "write the report for the run I watched yesterday" impossible.
    recorded: List[Dict[str, Any]] = []
    for p in paths:
        try:
            sig = read_shot(p)
        except Exception as e:
            rows.append(f"{p.stem}: unreadable ({e})")
            continue
        parts = [f"{m}={_fmt(getattr(sig, m, None))}" for m in wanted]
        g = {k: v for k, v in sig.all_globals.items() if k in allowed_globals}
        if include_globals:
            parts.append(f"globals={g}")
        rows.append(f"{sig.shot_id}: " + "  ".join(parts))

        got = {m: getattr(sig, m, None) for m in wanted}
        got = {k: v for k, v in got.items() if v is not None}
        if got:
            recorded.append({"shot_id": sig.shot_id, "shot_path": sig.shot_path,
                             **g, **got})

        primary = getattr(sig, wanted[0], None)
        if primary is not None and (best_val is None or primary > best_val):
            best_val, best_id = primary, sig.shot_id

    if recorded:
        try:
            from superradiant_assistant.creative.evidence import get_evidence
            get_evidence().record_read_rows(
                recorded, sweep_param=_varying_global(recorded, allowed_globals))
        except Exception:
            pass

    header = (f"read {len(paths)} shots | best {wanted[0]}={_fmt(best_val)}"
              f" ({best_id}) | root={CONFIG.historical_data_root.name}")
    return header + "\n" + "\n".join(rows)


def remember(fact: str, kind: str = "instruction", reason: str = "") -> str:
    """Write one fact into memory now.

    Until this existed, "remember that" could only be answered with a sentence.
    The assistant said "I've noted this rule" and noted nothing, because memory
    was written only by the compaction curator, which is told to keep parameter
    values and calibration results — not instructions about how to work.
    """
    from superradiant_assistant.memory import MEMORY

    if kind not in ("instruction", "apparatus"):
        return (f"error: kind must be 'instruction' (how the operator wants work "
                f"done, goes to USER.md) or 'apparatus' (a fact about the "
                f"hardware, goes to MEMORY.md), not {kind!r}")
    try:
        return MEMORY.remember(fact, kind)
    except Exception as e:
        return f"could not write memory: {type(e).__name__}: {e}"


def _preview_remember(args: Dict[str, Any]) -> str:
    from superradiant_assistant.memory import MEMORY
    from superradiant_assistant.memory.store import NOTE_TARGETS, CONFIG_APPARATUS

    kind = args.get("kind", "instruction")
    target, heading = NOTE_TARGETS.get(kind, NOTE_TARGETS["instruction"])
    # Asked of the store, not re-derived here. Computing the path twice is how
    # a confirmation box ends up naming a file the write does not touch.
    path = MEMORY._note_path(target)
    fact = " ".join(str(args.get("fact", "")).split())
    scope = ("shared by every apparatus" if target == "labscript"
             else f"this apparatus only ({CONFIG_APPARATUS()})")
    return "\n".join([
        f"kind    : {kind}",
        f"scope   : {scope}",
        f"file    : {path}",
        f"section : {heading}",
        f"reason  : {args.get('reason') or '(none given)'}",
        "",
        "--- to be appended ---",
        f"- {fact}",
        "--- end ---",
    ])


def _varying_global(rows: List[Dict[str, Any]], names: set) -> str:
    """Which global was swept across `rows`, judged by which one moved.

    `run_sweep` knows its own swept parameter and says so; a read from disk has
    to work it out. The global taking the most distinct numeric values is the one
    the shots differ in, and it is what a report should be plotted against.
    """
    best, best_n = "", 1
    for name in names:
        vals = {r[name] for r in rows
                if isinstance(r.get(name), (int, float))
                and not isinstance(r.get(name), bool)}
        if len(vals) > best_n:
            best, best_n = name, len(vals)
    return best


def _fmt(v: Any) -> str:
    if v is None:
        return "None"
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


# --------------------------------------------------------------------------
# T4  analyze_results
# --------------------------------------------------------------------------

def analyze_results(analysis_type: str, sweep_param: Optional[str] = None,
                     plot_ratio: Optional[str] = None) -> str:
    from superradiant_assistant.orchestrator.standalone_analysis import (
        analyze_resonance_sweep, analyze_larmor_calibration,
    )

    if analysis_type == "resonance":
        if not sweep_param or not plot_ratio:
            return "error: resonance analysis needs both sweep_param and plot_ratio"
        if sweep_param not in all_global_names():
            return f"error: unknown sweep_param '{sweep_param}'"
        if "/" not in plot_ratio:
            return f"error: plot_ratio must look like 'Neta_5/Neta_4', got '{plot_ratio}'"
        num, den = (s.strip() for s in plot_ratio.split("/", 1))
        known = metric_names()
        if num not in known or den not in known:
            return f"error: plot_ratio must use metric names {known}"

        r = analyze_resonance_sweep(CONFIG.historical_data_root, sweep_param, plot_ratio)
        if r.get("error"):
            return f"analysis failed | {r['error']} | no fit"
        ratio = r.get("min_ratio")
        if not r.get("found"):
            return (f"no dip found | min ratio {_fmt(ratio)} | widen the sweep")
        freq = r.get("fit_resonance_freq_mhz") or r.get("resonance_freq_mhz")
        lw = r.get("fit_linewidth_mhz")
        lw_txt = f"{lw*1e6:.0f} Hz" if lw else "unknown"
        return (f"resonance at {freq:.6f} MHz | linewidth {lw_txt} | "
                f"min ratio {_fmt(ratio)}")

    if analysis_type == "calibration":
        r = analyze_larmor_calibration(CONFIG.historical_data_root)
        if r.get("error"):
            return f"analysis failed | {r['error']} | no fit"
        c = r.get("correction_hz")
        if c is None:
            return "calibration fit failed | no correction | check the data"
        return (f"correction {c:+.2f} Hz | fit freq "
                f"{r.get('fit_freq_khz', 0)*1000:.1f} Hz | apply to rf_larmor_frequency")

    return f"error: analysis_type must be 'resonance' or 'calibration', got '{analysis_type}'"


# --------------------------------------------------------------------------
# T5  get_runmanager_globals
# --------------------------------------------------------------------------

def get_runmanager_globals(names: Optional[List[str]] = None) -> str:
    from superradiant_assistant.tools import get_lab_api

    try:
        current, dropped = get_lab_api().read_globals(names)
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"
    if not current:
        if dropped:
            sample = ", ".join(dropped[:8]) + ("..." if len(dropped) > 8 else "")
            return (f"read 0 configured globals | runmanager has {len(dropped)} globals "
                    f"but none are described in config.json | present: {sample}")
        return "read 0 globals | runmanager returned nothing | is a sequence loaded?"
    body = "\n".join(f"  {k} = {v}" for k, v in sorted(current.items()))
    tail = (f"\n({len(dropped)} further globals are loaded but absent from config.json)"
            if dropped else "")
    return f"read {len(current)} globals\n{body}{tail}"


# --------------------------------------------------------------------------
# T6  run_optimization
# --------------------------------------------------------------------------

def run_optimization(target_metric: str, threshold: float, sequence_file: str,
                      signal_spec: str, reason: str, threshold_op: str = ">",
                      max_iterations: int = 10,
                      goal_mode: Optional[bool] = None) -> str:
    import math
    from superradiant_assistant.goal import Goal, Stage
    from superradiant_assistant.orchestrator.loop import run_loop
    from superradiant_assistant.tools import get_session_context

    if target_metric not in metric_names():
        return f"error: target_metric must be one of {metric_names()}"
    if sequence_file not in _sequence_files():
        return f"error: sequence_file must be one of {_sequence_files()}"
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return f"error: threshold must be numeric, got {threshold!r}"
    if not math.isfinite(threshold):
        return f"error: threshold must be finite, got {threshold}"
    if threshold_op not in (">", ">=", "<", "<=", "=="):
        return f"error: bad threshold_op '{threshold_op}'"
    max_iterations = max(1, min(int(max_iterations), 50))

    ctx = get_session_context()
    effective_goal_mode = ctx.goal_mode if goal_mode is None else bool(goal_mode)

    stage = Stage(
        name="optimize", description=reason, task_type="optimize",
        stage_kind="optimize", target_metric=target_metric, threshold=threshold,
        threshold_op=threshold_op, sequence_file=sequence_file,
        max_iterations=max_iterations, notes=signal_spec,
    )
    goal = Goal(stages=[stage], goal_mode=effective_goal_mode, raw_prompt=reason)

    summary = run_loop(
        goal=goal, data_root=CONFIG.historical_data_root,
        repo_root=Path(__file__).resolve().parents[2],
        llm_client=ctx.llm_client, knowledge=ctx.knowledge, executor=ctx.executor,
    )
    res = stage.result or {}
    mode_note = "goal" if effective_goal_mode else "single-shot"

    # Every point that was measured, in the order it was measured. The model
    # reads this verbatim and reports from it; without it, it went and read the
    # shot files itself and tabulated ten of fifteen, leaving out the best one.
    lines = []
    for row in res.get("shot_rows") or []:
        params = ", ".join(f"{k}={_fmt(v)}"
                           for k, v in (row.get("params") or {}).items())
        lines.append(f"  iter {row.get('iteration')}: {params} -> "
                     f"{target_metric}={_fmt(row.get(target_metric))}"
                     f"  [{row.get('shot_id')}]")

    # The best value with its coordinates. A best value on its own is not a
    # result anyone can act on, and it is what let three different "best"
    # numbers coexist in one report.
    best_at = ", ".join(f"{k}={_fmt(v)}"
                        for k, v in (res.get("best_params") or {}).items())
    head = (f"{res.get('stop_reason', 'unknown')} | best {target_metric}="
            f"{_fmt(res.get('best_value'))}"
            + (f" at {best_at}" if best_at else "")
            + (f" [{res.get('best_shot_id')}]" if res.get("best_shot_id") else "")
            + f" | {res.get('iterations', 0)} shots ({mode_note} mode)")
    if not lines:
        return head
    return (head + f"\n\nEvery point measured ({len(lines)}) — report from these, "
            f"do not go reading the shot files yourself:\n" + "\n".join(lines))


# --------------------------------------------------------------------------
# T7  run_sweep
# --------------------------------------------------------------------------

def run_sweep(sweep_param: str, mode: str, sequence_file: str, signal_spec: str,
               reason: str, range_mhz: Optional[float] = None,
               step_mhz: Optional[float] = None, start: Optional[float] = None,
               end: Optional[float] = None, n_points: int = 21,
               plot_ratio: Optional[str] = None) -> str:
    from superradiant_assistant.goal import Goal, Stage
    from superradiant_assistant.orchestrator.loop import run_loop
    from superradiant_assistant.tools import get_session_context

    if sweep_param not in all_global_names():
        return f"error: sweep_param must be one of {all_global_names()}"
    if sequence_file not in _sequence_files():
        return f"error: sequence_file must be one of {_sequence_files()}"

    # Guard the pairing, not just each argument: a sweep is only meaningful if
    # the sequence actually reads the parameter being swept.
    if _sequence_uses_global(sequence_file, sweep_param) is False:
        return (f"error: sequence '{Path(sequence_file).name}' never references "
                f"'{sweep_param}', so sweeping it there would queue "
                f"{n_points} identical shots. Pick the sequence that uses "
                f"{sweep_param}, or ask the operator which one they meant. "
                f"Candidates: {_sequence_files()}")

    stage_kwargs: Dict[str, Any] = {}
    wrap_note = ""
    if mode == "centered":
        if range_mhz is None or step_mhz is None:
            return "error: centered mode needs both range_mhz and step_mhz"
        if step_mhz <= 0:
            return f"error: step_mhz must be positive, got {step_mhz}"
        n = int(round(range_mhz / step_mhz)) + 1
        if not (MIN_SWEEP_POINTS <= n <= MAX_SWEEP_POINTS):
            return (f"error: range/step gives {n} points, outside "
                    f"[{MIN_SWEEP_POINTS}, {MAX_SWEEP_POINTS}]")
        stage_kwargs.update(sweep_range_mhz=float(range_mhz),
                            sweep_step_mhz=float(step_mhz), max_iterations=n)
    elif mode == "explicit":
        if start is None or end is None:
            return "error: explicit mode needs both start and end"
        if start == end:
            return f"error: start and end are both {start} — nothing to sweep"
        n = int(n_points)
        if not (MIN_SWEEP_POINTS <= n <= MAX_SWEEP_POINTS):
            return (f"error: n_points={n} outside "
                    f"[{MIN_SWEEP_POINTS}, {MAX_SWEEP_POINTS}]")
        # A full turn around a periodic parameter ends where it began; drop the
        # repeat rather than refusing the sweep over a tenth of a degree.
        from superradiant_assistant.safety import wrap_sweep
        start, end, n, wrap_note = wrap_sweep(sweep_param, float(start),
                                              float(end), n)
        stage_kwargs.update(sweep_start=float(start), sweep_end=float(end),
                            max_iterations=n)
    else:
        return f"error: mode must be 'centered' or 'explicit', got '{mode}'"

    # Queueing shots that nothing will analyse costs real machine time and
    # produces a pile of files with no results -- which then reads as "the
    # experiment failed" rather than "nobody was listening".
    warning = _lyse_readiness_warning()
    if wrap_note:
        warning = f"\n{wrap_note}{warning}"

    ctx = get_session_context()
    stage = Stage(
        name="sweep", description=reason, task_type="resonance", stage_kind="sweep",
        sweep_param=sweep_param, plot_ratio=plot_ratio,
        sequence_file=sequence_file, notes=signal_spec, **stage_kwargs,
    )
    goal = Goal(stages=[stage], raw_prompt=reason)

    run_loop(
        goal=goal, data_root=CONFIG.historical_data_root,
        repo_root=Path(__file__).resolve().parents[2],
        llm_client=ctx.llm_client, knowledge=ctx.knowledge, executor=ctx.executor,
    )
    res = stage.result or {}
    span = (f"{start}→{end}" if mode == "explicit"
            else f"±{range_mhz/2:.4f} MHz around current")

    # The model reads this string verbatim and will repeat whatever it claims.
    # Saying "queued" after an offline replay is what taught it to invent shot
    # IDs, so state the mode first and never hand it IDs it cannot verify.
    live = getattr(ctx.executor, "runmanager", None) is not None
    if not live:
        # The wrap note has to survive every return path: it says what range was
        # actually swept, and reporting the requested one instead is wrong in
        # exactly the way this whole change exists to prevent.
        return ((f"{wrap_note}\n" if wrap_note else "")
                + f"OFFLINE REPLAY — NOTHING WAS QUEUED ON HARDWARE. "
                f"{res.get('stop_reason', 'unknown')} | {sweep_param} {span} "
                f"over {stage.max_iterations} points was replayed against historical "
                f"shots only. No new shots exist and no shot IDs were produced. "
                f"Report this to the operator as a replay, not as an executed sweep; "
                f"tell them to restart with `agent.py --live`.")
    # "queued" is a claim about what reached BLACS, so it has to be gated on an
    # engage actually having happened. Reporting the requested point count here
    # regardless of outcome is how an executor error got relayed as a success.
    stop_reason = res.get("stop_reason", "unknown")
    engages = res.get("iterations", 0)
    if engages == 0 or stop_reason in ("executor_error", "sequence_load_failed"):
        return ((f"{wrap_note}\n" if wrap_note else "")
                + f"FAILED — NOTHING WAS QUEUED. stop_reason={stop_reason}, "
                f"engage calls={engages}. The sweep of {sweep_param} {span} did not "
                f"reach BLACS. Report this as a failure and quote the stop_reason; "
                f"do not describe any shots as queued, executed, or completed.")
    head = (f"{warning}\n" if warning else "") + (
            f"LIVE | {stop_reason} | queued {stage.max_iterations} "
            f"shots: {sweep_param} {span} | {engages} engage calls. "
            f"Shot IDs are assigned by BLACS and are NOT available here — do not "
            f"state any shot ID or filename in your reply.")

    # The measurement, not just the queueing. Reporting only "queued N shots"
    # is what made every sweep end without the agent learning anything.
    sr = res.get("sweep_results")
    if sr:
        from superradiant_assistant.orchestrator.loop import summarise_sweep_results
        head += ("\nMEASURED RESULTS (read back from the shot files after lyse "
                 "analysed them):\n" + summarise_sweep_results(sr, sweep_param))
        if not sr.get("complete"):
            head += (f"\nNOTE: only {sr.get('collected')} of {sr.get('expected')} "
                     f"shots were analysed before the timeout — say so rather than "
                     f"describing the sweep as fully measured.")
        head += ("\nInterpret these numbers for the operator: state the trend, the "
                 "best point, and whether it answers what they asked.")
    return head


# --------------------------------------------------------------------------
# T8  set_runmanager_global
# --------------------------------------------------------------------------

def set_runmanager_global(name: str, value: Any, reason: str) -> str:
    from superradiant_assistant.tools import get_lab_api

    # The model may only write model-visible globals, never internal bookkeeping ones.
    if name not in load_global_specs():
        return f"error: '{name}' is not a settable global. Allowed: {all_global_names()}"
    try:
        validate_scalar(name, value)
        r = get_lab_api().set_global(name, value)
    except SafetyViolation as e:
        return f"refused: {e}"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"
    return f"{name}: {_fmt(r['old'])} -> {_fmt(r['new'])} | reason: {reason}"


# --------------------------------------------------------------------------
# T10  load_sequence
# --------------------------------------------------------------------------

def load_sequence(file: str, reason: str) -> str:
    """Switch which sequence runmanager will compile and run.

    Kept separate from `engage_shot` for the same reason writing a global is:
    changing the experiment and firing it are two decisions, and the operator
    should confirm them one at a time.
    """
    from superradiant_assistant.tools import get_lab_api

    try:
        r = get_lab_api().set_labscript_file(file)
    except SafetyViolation as e:
        return f"refused: {e}"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"

    new = Path(r["new"]).name
    if not r["changed"]:
        return f"already loaded | {new} | nothing changed | reason: {reason}"
    old = Path(r["old"]).name if r["old"] else "(nothing)"
    return (f"loaded {new} | was {old} | globals untouched — re-read them, this "
            f"sequence may use a different set | reason: {reason}")


def _preview_load_sequence(args: Dict[str, Any]) -> str:
    from superradiant_assistant.tools import get_lab_api

    lines = [f"load      : {args.get('file')}"]
    try:
        lines.append(f"currently : {get_lab_api().get_labscript_file()}")
    except Exception as e:
        lines.append(f"currently : (could not read: {e})")
    lines.append(f"reason    : {args.get('reason', '(none given)')}")
    lines.append("NOTE: changes which sequence engage_shot will run. Globals are "
                 "not touched, and the new sequence may read a different set.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# T9  engage_shot
# --------------------------------------------------------------------------

#: A shot file appears the moment runmanager COMPILES it -- globals, connection
#: table, script, and nothing measured. BLACS adds `front_panel` and
#: `data/traces` when it actually executes. Distinguishing the two is the only
#: way to tell a real run from a recipe on disk.
def _shot_state(path: Path) -> str:
    """'executed' | 'analysed' | 'compiled' | 'unreadable'."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            traces = f.get("data/traces")
            if traces is None or not len(traces):
                return "compiled"
            return "analysed" if f.get("results") is not None else "executed"
    except Exception:
        return "unreadable"


def _await_execution(before: set, expected: Optional[int],
                     timeout_s: float) -> Dict[str, Any]:
    """Wait for the shots just queued to actually run, and report what happened.

    `engage()` returns as soon as runmanager hands the queue to BLACS, which is
    a claim about queueing and nothing else. Reporting that as success is how a
    session with no hardware connected ended with a plan marked 4/5 complete and
    a report being written about a shot BLACS never executed.
    """
    import time
    from superradiant_assistant.interfaces.hdf5_reader import list_shots

    root = Path(CONFIG.historical_data_root)
    deadline = time.time() + timeout_s
    new: List[Path] = []
    states: Dict[str, str] = {}

    while time.time() < deadline:
        time.sleep(2.0)
        try:
            new = [p for p in list_shots(root) if p not in before]
        except Exception:
            continue
        states = {p.name: _shot_state(p) for p in new}
        ran = [n for n, s in states.items() if s in ("executed", "analysed")]
        done = [n for n, s in states.items() if s == "analysed"]
        # Stop as soon as everything expected has been analysed; otherwise keep
        # waiting -- a shot that has executed may still be in lyse's queue.
        if expected and len(done) >= expected:
            break
        if expected and len(ran) >= expected and time.time() > deadline - 20:
            break

    return {
        "new": [p.name for p in new],
        "states": states,
        "executed": [n for n, s in states.items() if s in ("executed", "analysed")],
        "analysed": [n for n, s in states.items() if s == "analysed"],
        "compiled_only": [n for n, s in states.items() if s == "compiled"],
    }


_NOT_COMPLETE = (" This step is NOT complete. Do not mark it done, do not write "
                 "a report, and do not describe the shot as executed.")


def engage_shot(reason: str, wait_seconds: float = 120.0) -> str:
    """Queue whatever runmanager currently holds, then verify it actually ran.

    Deliberately takes no parameters that change the experiment: setting a value
    and firing a shot are separate decisions, and keeping them in separate tools
    means the confirmation prompt for each states exactly one thing.
    """
    from superradiant_assistant.interfaces.hdf5_reader import list_shots
    from superradiant_assistant.tools import get_lab_api

    api = get_lab_api()
    try:
        n = api.n_shots()
    except Exception:
        n = None
    # Name the sequence in the result: the shot runs whatever the GUI holds, so
    # reporting "engaged" without saying which sequence hides a mismatch between
    # what the agent thinks is loaded and what actually fires.
    try:
        seq = Path(api.get_labscript_file() or "").name or "(unknown)"
    except Exception:
        seq = "(unknown)"

    warning = _lyse_readiness_warning()

    try:
        before = set(list_shots(Path(CONFIG.historical_data_root)))
    except Exception:
        before = set()

    try:
        api.engage()
    except Exception as e:
        return f"engage failed | {type(e).__name__}: {e} | nothing queued"

    count = f"{n} shot(s)" if n is not None else "an unknown number of shots"
    head = f"queued {count} of {seq} | reason: {reason}"

    from superradiant_assistant.creative.evidence import get_evidence
    get_evidence().record_queue()

    if wait_seconds <= 0:
        return f"engaged | {head} | execution NOT verified{warning}"

    got = _await_execution(before, n, wait_seconds)

    if not got["executed"]:
        detail = (f" {len(got['compiled_only'])} shot file(s) exist but contain no "
                  f"/data/traces: runmanager compiled them, BLACS never ran them."
                  if got["compiled_only"] else
                  " No new shot files appeared at all.")
        return (f"FAILED — the shots were queued but NEVER EXECUTED. {head}.{detail} "
                f"Check that BLACS is running and its device workers connected to "
                f"the hardware.{_NOT_COMPLETE}{warning}")

    lines = [f"engaged | {head}",
             f"executed: {len(got['executed'])}/{n if n else '?'} "
             f"({', '.join(got['executed'][:6])})"]

    if got["compiled_only"]:
        lines.append(f"NOT executed: {len(got['compiled_only'])} shot(s) still hold "
                     f"no traces — {', '.join(got['compiled_only'][:6])}. Say so "
                     f"rather than describing the run as complete.")
    if not got["analysed"]:
        lines.append("NO ANALYSIS RESULTS: the shots ran and saved traces, but no "
                     "lyse routine has written anything into /results. There are no "
                     "numbers to report yet." + _NOT_COMPLETE)
    else:
        lines.append(f"analysed by lyse: {len(got['analysed'])} shot(s). Read the "
                     f"values with read_shot_results before drawing any conclusion.")
    return "\n".join(lines) + warning


def _preview_engage_shot(args: Dict[str, Any]) -> str:
    from superradiant_assistant.tools import get_lab_api

    api = get_lab_api()
    lines = ["action    : engage runmanager — queue the currently loaded sequence"]
    try:
        lines.append(f"sequence  : {api.get_labscript_file()}")
    except Exception as e:
        lines.append(f"sequence  : (could not read: {e})")
    try:
        lines.append(f"shots     : {api.n_shots()}")
    except Exception as e:
        lines.append(f"shots     : (could not read: {e})")
    lines.append(f"reason    : {args.get('reason', '(none given)')}")
    lines.append("NOTE: runs the sequence and globals currently loaded in the GUI. "
                 "This tool does not modify either — use load_sequence to change "
                 "the sequence.")
    return "\n".join(lines)


def _preview_set_global(args: Dict[str, Any]) -> str:
    name = args.get("name", "?")
    spec = load_global_specs().get(name)
    lines = [f"parameter : {name}", f"new value : {args.get('value')}"]
    if spec:
        lines.append(f"allowed   : {spec.describe_range()}")
    lines.append(f"reason    : {args.get('reason', '(none given)')}")
    return "\n".join(lines)


def _preview_run_optimization(args: Dict[str, Any]) -> str:
    return (f"optimize {args.get('target_metric')} {args.get('threshold_op', '>')} "
            f"{args.get('threshold')}\nsequence  : {args.get('sequence_file')}\n"
            f"max shots : {args.get('max_iterations', 10)}\n"
            f"reason    : {args.get('reason', '(none)')}")


def _preview_run_sweep(args: Dict[str, Any]) -> str:
    if args.get("mode") == "explicit":
        span = f"{args.get('start')} → {args.get('end')} in {args.get('n_points', 21)} points"
    else:
        span = f"±{args.get('range_mhz', 0)/2} MHz, step {args.get('step_mhz')} MHz"
    param = args.get("sweep_param", "?")
    lines = [f"sweep     : {param}", f"range     : {span}",
             f"sequence  : {args.get('sequence_file')}",
             f"reason    : {args.get('reason', '(none)')}",
             "NOTE: this queues every point as a real shot."]
    # The limit this process will enforce — not necessarily what config.json
    # says right now, since it is only read at startup.
    spec = load_global_specs().get(param)
    if spec is not None and spec.has_range:
        lines.append(f"enforced  : {param} must stay within {spec.describe_range()} "
                     f"(loaded at startup; restart to pick up config.json edits)")
    if not sweep_is_bounded(param):
        lines.append(f"WARNING: config.json declares no min/max for '{param}', so these "
                      f"bounds were NOT range-checked. Check them yourself.")
    return "\n".join(lines)


def _print_plan(p) -> None:
    """Frame the plan so it is findable in a scrolling transcript.

    It was four bare lines among hundreds of tool logs, and the operator is
    meant to be able to glance at it and see where the agent has got to.
    """
    from superradiant_assistant import splash as S

    lines = [l.strip() for l in p.render().split("\n")]
    body = [l for l in lines[1:] if l]                 # drop the goal header
    print()
    print(S.framed(getattr(p, "goal", "") or "Plan", "\n".join(body),
                   colour=S.ACCENT))
    print()


def _plan_specs() -> List[ToolSpec]:
    """The visible plan. Printed on every change so the operator can follow along."""
    from superradiant_assistant.orchestrator import plan as plan_mod

    def set_plan(goal: str, steps: List[str]) -> str:
        if not steps:
            return "error: a plan needs at least one step"
        p = plan_mod.set_plan(goal, list(steps))
        _print_plan(p)
        # Terse: the rendered plan is already on the operator's screen, and
        # echoing it back into the conversation grows the context every turn
        # for no new information.
        return (f"plan set with {len(p.steps)} steps. Start step 1 with "
                f"update_plan(step=1, status='active').")

    def update_plan(step: int, status: str, note: str = "",
                    next_step: int = 0) -> str:
        step, next_step = int(step), int(next_step)

        # `update_plan(step=1, status='active', next_step=1)` was the model's
        # first move in essentially every session: the hint says to activate
        # step 1, and it filled in next_step out of habit. Returning an error
        # cost a full round trip every time and taught it nothing, because the
        # next session starts fresh. Activating a step and then "moving to" the
        # same step is not ambiguous -- it means activate it.
        if next_step == step:
            next_step = 0

        err = plan_mod.update_step(step, status, note)
        if err:
            return f"error: {err}"
        # Closing one step and opening the next in a single call halves the
        # number of model turns spent on bookkeeping.
        if next_step:
            err = plan_mod.update_step(next_step, "active")
            if err:
                return f"step {step} updated, but next_step failed: {err}"

        p = plan_mod.get_plan()
        _print_plan(p)
        done = sum(1 for s in p.steps if s.status in ("done", "skipped"))
        active = p.active

        # Point out steps left behind. Marking a later step active while an
        # earlier one is still untouched is usually an accident, and silently
        # allowing it produces a plan that claims progress it never made.
        if active:
            stranded = [i for i, s in enumerate(p.steps, 1)
                        if s.status == "pending" and i < p.steps.index(active) + 1]
            if stranded:
                return (f"{done}/{len(p.steps)} done; now on step: {active.text}\n"
                        f"NOTE: step(s) {stranded} are still pending and were "
                        f"skipped over. Close them properly (done or skipped with "
                        f"a reason) rather than leaving them open.")
        return (f"{done}/{len(p.steps)} done"
                + (f"; now on step: {active.text}" if active else "; no step active"))

    return [
        ToolSpec(
            name="set_plan",
            description=(
                "Write down the steps for a request that HAS several steps, before "
                "touching anything. Each step should be one checkable action, in the "
                "order you will do them, ending with reading the results back and "
                "reporting. A plan written after the fact is worthless — its purpose "
                "is to force the design decisions before the hardware moves.\n"
                "Do NOT plan a request that is one action. 'Set the angle to 280 deg' "
                "is one tool call; wrapping it in a three-step plan cost five extra "
                "model turns and over half the tokens of the whole task, and told the "
                "operator nothing they could not see.\n"
                "If the operator ASKS for a plan, write one however small the task is — "
                "then the plan is the deliverable, not overhead."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "What the operator asked for, in one line."},
                    "steps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["goal", "steps"],
            },
            handler=set_plan,
            # The lead only. The plan exists so the operator can see where the work
            # is, and the lead is the only agent that talks to them. A subagent's
            # plan is rendered but owned by nobody: the coder's plan once carried its
            # own "write the report" step, so one measurement produced two reports
            # from the same 23 shots. A consulted planner spent 5 of its 15 calls
            # keeping a plan that nothing ever executed.
            allowed_agents=frozenset({"lead"}),
        ),
        ToolSpec(
            name="update_plan",
            description=(
                "Mark a step done/failed and start the next one in ONE call via "
                "next_step — separate calls cost an extra model turn each. If a step "
                "turns out to be wrong, mark it skipped with the reason."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "step": {"type": "integer", "description": "1-based step number."},
                    "status": {"type": "string",
                                "enum": ["pending", "active", "done", "failed", "skipped"]},
                    "note": {"type": "string", "description": "What was learned, briefly."},
                    "next_step": {"type": "integer",
                                   "description": "Step to make active in the same call."},
                },
                "required": ["step", "status"],
            },
            handler=update_plan,
            allowed_agents=frozenset({"lead"}),
        ),
    ]


def _lyse_specs() -> List[ToolSpec]:
    """Control which analysis routines lyse runs."""
    from superradiant_assistant.interfaces import lyse_iface

    _ANALYSIS_ROOT = (Path(CONFIG.labscript_suite_root) / "userlib" / "analysislib"
                      / "Cesium")

    def _resolve_routine(name: str) -> str:
        """Accept a bare filename by looking in both routine directories.

        Resolving only against the singleshot directory made every multishot
        filename come back as "does not exist".
        """
        p = Path(name)
        if p.is_absolute():
            return str(p)
        for sub in ("singleshot_routines", "multishot_routines"):
            candidate = _ANALYSIS_ROOT / sub / p.name
            if candidate.is_file():
                return str(candidate)
        return str(_ANALYSIS_ROOT / "singleshot_routines" / p.name)

    def set_lyse_routines(routines: List[str], replace: bool = True,
                          kind: str = "singleshot") -> str:
        if not routines:
            return "error: give at least one routine path"
        resolved = [_resolve_routine(r) for r in routines]

        # A multishot routine dropped into the singleshot box runs once per
        # shot: it refetches the whole dataframe and redraws every time, which
        # is what stalls lyse. Catch the mismatch rather than let it through.
        misplaced = [p for p in resolved
                     if ("multishot_routines" in p) != (kind == "multishot")]
        if misplaced:
            return (f"refused: {[Path(p).name for p in misplaced]} live in the "
                    f"{'singleshot' if kind == 'multishot' else 'multishot'} "
                    f"directory but you asked to load them as {kind}. Load each "
                    f"routine as the kind it was written for.")

        return lyse_iface.set_live_routines(resolved, replace=replace, kind=kind)

    def get_lyse_routines() -> str:
        lines = []
        for kind in ("singleshot", "multishot"):
            current = lyse_iface.live_routines(kind)
            if current is None:
                lines.append(f"{kind}: could not be read (is lyse running?)")
            elif not current:
                lines.append(f"{kind}: NONE loaded")
            else:
                lines.append(f"{kind}: " + ", ".join(Path(p).name for p in current))
        if all("NONE" in ln for ln in lines):
            lines.append("With no routines loaded, shots produce no results at all.")
        return "\n".join(lines)

    return [
        ToolSpec(
            name="set_lyse_routines",
            description=(
                "Choose which analysis routines lyse runs. Writing an analysis file "
                "does NOT make lyse run it — this does. lyse keeps TWO separate "
                "lists: singleshot (per shot) and multishot (once per sequence), so "
                "load each routine with the kind it was written for and call this "
                "twice if you have both. Verify with get_lyse_routines: a sweep whose "
                "routine was never loaded produces shot files and no results."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "routines": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Filenames or full paths of analysis routines.",
                    },
                    "replace": {
                        "type": "boolean",
                        "description": "True replaces that list; False appends.",
                    },
                    "kind": {
                        "type": "string", "enum": ["singleshot", "multishot"],
                        "description": "Which of lyse's two lists to set.",
                    },
                },
                "required": ["routines"],
            },
            handler=set_lyse_routines,
            allowed_agents=ALL_AGENTS_MINUS_LEAD,
        ),
        ToolSpec(
            name="get_lyse_routines",
            description="Which analysis routines lyse is currently running.",
            parameters={"type": "object", "properties": {}},
            handler=get_lyse_routines,
            allowed_agents=ADVISED,
        ),
    ]


def list_scripts(kind: str = "all") -> str:
    """The scripts that exist for this apparatus, grouped by kind."""
    from superradiant_assistant import script_inventory

    kinds = None if kind in ("all", "", None) else [kind]
    if kinds and kinds[0] not in script_inventory.KIND_DIRS:
        return (f"error: kind must be one of "
                f"{['all'] + sorted(script_inventory.KIND_DIRS)}, got {kind!r}")
    body = script_inventory.render_inventory(kinds)
    return (body + "\n\nReuse one of these if it fits. If you do write a new "
            "script, the operator is asked whether to add, replace or reuse — "
            "so say in your message which you are proposing and why.")


def _introspection_specs() -> List[ToolSpec]:
    """Read-only tools. No confirmation: they cannot change anything."""
    from superradiant_assistant.tools.introspect import (
        read_lab_file, list_lab_files, inspect_shot,
    )
    return [
        ToolSpec(
            name="remember",
            description=(
                "Write one fact into long-term memory, now. Use it whenever the "
                "operator says to remember something, corrects you about how to "
                "work, or tells you never to do something again — and also when "
                "you discover a durable fact about the apparatus that cost time "
                "to find. Saying 'I have noted that' without calling this tool "
                "records NOTHING: memory is a file, and this is the only tool "
                "that writes it. One fact per call, stated so it is actionable "
                "next session without this conversation: 'start frequency sweeps "
                "at 3 kHz, not 100 Hz — below ~2.1 kHz the 600 us scope window "
                "holds under one cycle and the amplitude fit fails'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact, in one or two sentences, "
                                       "including why it matters.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["instruction", "apparatus"],
                        "description": "'instruction' for how the operator wants "
                                       "work done (USER.md); 'apparatus' for a "
                                       "fact about the hardware or the data "
                                       "(MEMORY.md).",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this is worth remembering.",
                    },
                },
                "required": ["fact"],
            },
            handler=remember,
            allowed_agents=ALL_AGENTS,
            preview=_preview_remember,
        ),
        ToolSpec(
            name="list_scripts",
            description=(
                "The scripts that already exist for THIS apparatus: shot sequences, "
                "singleshot routines and multishot routines, with dates, which ones "
                "creative mode wrote, and which are loaded in lyse right now. Call "
                "this BEFORE writing any script. Most tasks need no new script at "
                "all — an existing one with different globals is faster, already "
                "debugged, and does not leave the folder full of near-duplicates. "
                "Unlike list_lab_files this excludes drivers and the previous "
                "experiment's hundreds of analyses."
            ),
            parameters={
                "type": "object",
                "properties": {"kind": {
                    "type": "string",
                    "enum": ["all", "shot", "singleshot", "multishot"],
                    "description": "Which list to show. Default 'all'.",
                }},
            },
            handler=list_scripts,
            allowed_agents=ADVISED,
        ),
        ToolSpec(
            name="inspect_shot",
            description=(
                "Show a shot file's actual structure: HDF5 groups, dataset columns, "
                "globals and saved analysis results. Use this FIRST when you need to "
                "know how data is laid out — do not guess a format and fire a shot to "
                "find out. Note /data/traces is a group INSIDE the shot's HDF5 file, "
                "not a directory on disk."
            ),
            parameters={
                "type": "object",
                "properties": {"shot": {
                    "type": "string",
                    "description": "'latest', a filename, or part of one.",
                }},
            },
            handler=inspect_shot,
            # Not the advisor. On 2026-08-18 it inspected two shots to compare
            # their scope scaling: 2,092 characters each, two model round trips,
            # and every number it took away (`theta0_deg`, `rms_mV`,
            # `CHAN1_codes`, `CHAN1_ptp`) was already a named metric --
            # `read_shot_results` returned the same comparison for both shots in
            # 429 characters and one call. What this tool adds over that is the
            # LAYOUT: trace dataset names, column tuples, the globals subgroup.
            # That is what you need to WRITE an analysis routine, and the advisor
            # does not write anything.
            allowed_agents=ALL_AGENTS,
        ),
        ToolSpec(
            name="read_lab_file",
            description=(
                "Read an existing sequence, analysis routine or device driver. Read a "
                "working example before writing a new one — the conventions that work "
                "on this apparatus are in those files, not in general labscript "
                "knowledge."
            ),
            parameters={
                "type": "object",
                "properties": {"path": {
                    "type": "string",
                    "description": "Full path, inside the experiment directories.",
                }},
                "required": ["path"],
            },
            handler=read_lab_file,
            allowed_agents=ADVISED,
        ),
        ToolSpec(
            name="list_lab_files",
            description=(
                "List the sequences, analysis routines and drivers that already exist, "
                "so you can read one instead of inventing a convention."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "subdir": {"type": "string", "description": "Optional subdirectory."},
                    "pattern": {"type": "string", "description": "Glob, default '*.py'."},
                },
            },
            handler=list_lab_files,
            allowed_agents=ADVISED,
        ),
    ]


def build_tool_specs() -> List[ToolSpec]:
    """Tool declarations. Enums come from config.json so the model cannot name
    a parameter, sequence or metric that isn't on the list."""
    metric_enum = {"type": "string", "enum": metric_names()}
    seq_enum = {"type": "string", "enum": _sequence_files()}
    all_globals = all_global_names()

    return _plan_specs() + _introspection_specs() + _lyse_specs() + [
        ToolSpec(
            name="search_lab_knowledge",
            description=(
                "Retrieve passages from the lab documentation. Ask in your own "
                "words — retrieval is semantic as well as keyword, so a "
                "paraphrase works and you do not have to guess the document's "
                "wording. Each passage is returned with the file and heading it "
                "came from; cite those rather than paraphrasing from memory. "
                "The corpus is whatever documents this lab has added, and it may "
                "describe a different apparatus than the one in front of you — "
                "check before carrying a number across. For THIS apparatus's own "
                "code and data use list_scripts, read_lab_file and inspect_shot."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language query."},
                    "role": {"type": "string", "enum": ["planner", "coder"],
                              "description": "Unused by retrieval; kept for callers."},
                    "top_k": {"type": "integer", "description": "1-10, default 6."},
                    "rerank": {
                        "type": "string", "enum": ["fusion", "llm"],
                        "description": "'fusion' (default, instant). 'llm' judges "
                                       "each passage against the query and takes "
                                       "about a minute — only for a question the "
                                       "fast path answered badly.",
                    },
                    "answer": {
                        "type": "boolean",
                        "description": "Also synthesise a cited answer from the "
                                       "passages. Costs one extra call.",
                    },
                },
                "required": ["query"],
            },
            handler=search_lab_knowledge,
            allowed_agents=ADVISED,
        ),
        ToolSpec(
            name="load_skill",
            description="Load the full procedure for a lab workflow. Call this before "
                        "carrying out an unfamiliar procedure.",
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "enum": SKILLS.names(),
                                    "description": "Which skill to load."},
                },
                "required": ["skill_name"],
            },
            handler=load_skill,
            allowed_agents=ADVISED,
        ),
        ToolSpec(
            name="read_shot_results",
            description="Read metrics from many recent shots at once (HDF5). Use "
                        "this to read numbers back — one call covers up to 200 "
                        "shots. `inspect_shot` shows one shot's LAYOUT (groups, "
                        "column names, units) and is for finding out how data is "
                        "stored, not for collecting values: calling it once per "
                        "shot costs one model round trip each.",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "1-200, newest first, default 20."},
                    "metrics": {"type": "array", "items": metric_enum,
                                 "description": "Which metrics to report. Default: all."},
                    "include_globals": {"type": "boolean",
                                          "description": "Also report the parameter snapshot."},
                },
                "required": [],
            },
            handler=read_shot_results,
            # The coder was left out on the grounds that reporting numbers is the
            # lead's job. But the coder is the one that runs the sweeps, and
            # without a bulk reader it read them back with `inspect_shot` — once
            # per shot, one model round trip each. Twelve shots that way cost
            # ~123k tokens of fixed overhead on 2026-08-17; this tool returns the
            # same numbers in one call of 753 characters.
            allowed_agents=ADVISED,
        ),
        ToolSpec(
            name="analyze_results",
            description="Fit recent shot data: Lorentzian dip for resonance, Ramsey "
                        "fringes for Larmor calibration.",
            parameters={
                "type": "object",
                "properties": {
                    "analysis_type": {"type": "string", "enum": ["resonance", "calibration"]},
                    "sweep_param": {"type": "string", "enum": all_globals,
                                     "description": "Required for resonance: the swept parameter."},
                    "plot_ratio": {"type": "string",
                                    "description": "Required for resonance, e.g. 'Neta_5/Neta_4'."},
                },
                "required": ["analysis_type"],
            },
            handler=analyze_results,
            allowed_agents=ADVISED,
        ),
        ToolSpec(
            name="get_runmanager_globals",
            description="Read the experiment's current parameter values. Requires the "
                        "labscript GUIs to be running.",
            parameters={
                "type": "object",
                "properties": {
                    "names": {"type": "array",
                               "items": {"type": "string", "enum": all_globals},
                               "description": "Which parameters to read. Omit for all."},
                },
                "required": [],
            },
            handler=get_runmanager_globals,
            allowed_agents=ADVISED,
        ),
        ToolSpec(
            name="run_optimization",
            description="Run the optimization loop toward a metric threshold. The "
                        "optimizer chooses the parameter values, not you.",
            parameters={
                "type": "object",
                "properties": {
                    "target_metric": metric_enum,
                    "threshold": {"type": "number"},
                    "threshold_op": {"type": "string", "enum": [">", ">=", "<", "<=", "=="]},
                    "max_iterations": {"type": "integer", "description": "1-50."},
                    "sequence_file": seq_enum,
                    "signal_spec": {"type": "string",
                                     "description": "What the result should report back."},
                    "reason": {"type": "string", "description": "Why — recorded in the audit log."},
                },
                "required": ["target_metric", "threshold", "sequence_file",
                              "signal_spec", "reason"],
            },
            handler=run_optimization,
            allowed_agents=frozenset({"lead", "coder"}),
            preview=_preview_run_optimization,
        ),
        ToolSpec(
            name="run_sweep",
            description="Sweep one parameter and queue every point as a shot.",
            parameters={
                "type": "object",
                "properties": {
                    "sweep_param": {"type": "string", "enum": all_globals},
                    "mode": {"type": "string", "enum": ["centered", "explicit"],
                              "description": "centered: around the current value (MHz). "
                                             "explicit: from start to end."},
                    "range_mhz": {"type": "number", "description": "centered mode: total width."},
                    "step_mhz": {"type": "number", "description": "centered mode: step size."},
                    "start": {"type": "number", "description": "explicit mode: first value."},
                    "end": {"type": "number", "description": "explicit mode: last value."},
                    "n_points": {"type": "integer",
                                  "description": f"explicit mode: {MIN_SWEEP_POINTS}-{MAX_SWEEP_POINTS}."},
                    "plot_ratio": {"type": "string", "description": "e.g. 'Neta_5/Neta_4'."},
                    "sequence_file": seq_enum,
                    "signal_spec": {"type": "string"},
                    "reason": {"type": "string", "description": "Why — recorded in the audit log."},
                },
                "required": ["sweep_param", "mode", "sequence_file", "signal_spec", "reason"],
            },
            handler=run_sweep,
            allowed_agents=frozenset({"lead", "coder"}),
            preview=_preview_run_sweep,
        ),
        ToolSpec(
            name="set_runmanager_global",
            description="Write a single experiment parameter. Changes the physical "
                        "apparatus; always requires operator confirmation.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": all_globals},
                    "value": {"type": "number"},
                    "reason": {"type": "string", "description": "Why — recorded in the audit log."},
                },
                "required": ["name", "value", "reason"],
            },
            handler=set_runmanager_global,
            allowed_agents=frozenset({"lead"}),
            preview=_preview_set_global,
        ),
        ToolSpec(
            name="load_sequence",
            description="Switch which sequence runmanager will run. Use this when "
                        "moving to a different experiment. Does not change any "
                        "parameter, and the new sequence may read a different set "
                        "of globals — re-read them afterwards.",
            parameters={
                "type": "object",
                "properties": {
                    "file": seq_enum,
                    "reason": {"type": "string", "description": "Why — recorded in the audit log."},
                },
                "required": ["file", "reason"],
            },
            handler=load_sequence,
            allowed_agents=CODER_ONLY,
            preview=_preview_load_sequence,
        ),
        ToolSpec(
            name="engage_shot",
            description="Queue the sequence currently loaded in runmanager as a real "
                        "shot in BLACS, using the globals already set in the GUI, then "
                        "WAIT and verify it actually executed. Does not change any "
                        "parameter — call set_runmanager_global first if a value needs "
                        "to change. The result distinguishes queued, executed and "
                        "analysed: a shot file exists as soon as runmanager compiles "
                        "it, so its existence proves nothing. Read the result carefully "
                        "— if it says the shot was never executed, that is a failure, "
                        "not a step you can mark done.",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why — recorded in the audit log."},
                    "wait_seconds": {
                        "type": "number",
                        "description": "How long to wait for execution and analysis. "
                                       "Default 120. Raise it for long sequences.",
                    },
                },
                "required": ["reason"],
            },
            handler=engage_shot,
            allowed_agents=frozenset({"lead", "coder"}),
            preview=_preview_engage_shot,
        ),
    ]

