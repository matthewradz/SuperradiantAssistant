"""LLM-based prompt parser and stage decomposer.

Used when --llm is set. The LLM reads the full prompt and returns a
structured multi-stage Goal. It has access to the knowledge base so it
can reference real sequence files, analysis outputs, and global names.
"""
from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Optional

from superradiant_assistant.goal import Goal, Stage
from superradiant_assistant.knowledge.loader import Document
from superradiant_assistant.knowledge.search import search_for_role
from superradiant_assistant.llm.client import LLMClient


def _build_system_instruction() -> str:
    """Build the system instruction dynamically, grounding it in config.json."""
    from superradiant_assistant.config import CONFIG

    seq_lines = "\n".join(
        f"  - {s['file']} | use_case={s['use_case']} | {s['description']}"
        for s in CONFIG.sequences
    )
    analysis_lines = "\n".join(
        f"  - {s['file']} | use_case={s.get('use_case',[])} | {s['description']}"
        for s in CONFIG.analysis_scripts
    )
    globals_lines = "\n".join(
        f"  - {g['name']} ({g['type']}) range [{g.get('min','?')} – {g.get('max','?')}]: {g['description']}"
        for g in CONFIG.experiment_globals
    )

    return f"""You are a research planning assistant for an AMO physics lab.
Parse the user's goal into one or more stages. Use ONLY the sequences, analysis scripts,
and globals listed below — do not invent others.

## Available sequences
{seq_lines}

## Available analysis scripts
{analysis_lines}

## Globals the assistant may edit
{globals_lines}

## Task types
- answer    : Q&A / code tracing — no shots needed
- calibrate : run a Ramsey/calibration sequence, run analysis script, read correction, update a global
- resonance : sweep a parameter, plot a ratio (e.g. Neta_5/Neta_4) to find a peak/dip
- optimize  : iterate shots to maximize/minimize a metric above/below a threshold

Return ONLY valid JSON, no markdown:
{{
  "stages": [
    {{
      "name": "<short_snake_case_name>",
      "description": "<one sentence>",
      "task_type": "<answer | calibrate | resonance | optimize>",
      "sequence_file": "<filename.py from the sequences list, or empty for answer>",
      "analysis_script": "<filename.py from analysis list, or empty>",
      "analysis_y_op": "<y-axis expression e.g. Neta_5/Neta_4, or empty>",
      "analysis_param_str": "<x-axis global name for improved_cost_clean.py, or empty>",
      "target_metric": "<Neta_1..Neta_5 | Neta_2 default>",
      "threshold": <float or null>,
      "threshold_op": "<> | >= | < | <= | ==>",
      "max_iterations": <int>,
      "stage_kind": "<optimize | sweep>",
      "sweep_param": "<exact global name or empty>",
      "plot_ratio": "<e.g. Neta_5/Neta_4 or empty>",
      "sweep_range_mhz": <float or null>,
      "sweep_step_mhz": <float or null>
    }}
  ],
  "timeout_seconds": <int>,
  "rationale": "<one sentence>"
}}

Rules:
- Use ONLY sequences and analysis scripts from the lists above
- sweep_param must be VERBATIM from the user's prompt or the globals list
- For calibrate: task_type=calibrate, stage_kind=sweep, use the calibration sequence + analysis script
- For resonance: task_type=resonance, stage_kind=sweep, use the RABI sequence + improved_cost_clean.py
- For optimize: task_type=optimize, stage_kind=optimize, pick any loading-capable sequence
- For answer: task_type=answer, no sequence or shots needed
- Sweeps have no threshold (null); optimize stages have threshold
- For frequency sweeps (resonance): convert range/step to MHz for sweep_range_mhz/sweep_step_mhz; max_iterations = round(sweep_range_mhz / sweep_step_mhz) + 1
- For calibrate (calibration_precession_time sweep): set sweep_param="calibration_precession_time", sweep_start=0.5, sweep_end=20.0, max_iterations=20, sweep_range_mhz=null, sweep_step_mhz=null
- sweep_start and sweep_end are in the native units of the parameter (ms for time, MHz for frequency)
- Default timeout_seconds = 3600
"""


def _to_mhz(value: float, unit: str) -> float:
    u = unit.lower().strip()
    if u == "hz":   return value / 1_000_000
    if u == "khz":  return value / 1_000
    if u == "mhz":  return value
    return value


def _extract_sweep_overrides(prompt: str) -> Dict[str, Any]:
    """Deterministically extract sweep parameters the user stated explicitly.
    These override whatever the LLM returns so values are never lost/truncated."""
    out: Dict[str, Any] = {}

    # Parameter name — longest snake_case identifier (3+ segments)
    candidates = re.findall(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+){2,})\b', prompt)
    if candidates:
        out["sweep_param"] = candidates[0]   # first one mentioned

    # Range — "across/over/of N Hz/kHz/MHz"
    m = re.search(
        r'(?:across|over|of|range)\s+(\d+(?:\.\d+)?)\s*(Hz|kHz|MHz)',
        prompt, re.I
    )
    if m:
        out["sweep_range_mhz"] = _to_mhz(float(m.group(1)), m.group(2))

    # Step — "in steps of N Hz/kHz/MHz" or "N Hz steps"
    m = re.search(
        r'(?:steps?\s+of|in\s+steps?\s+of)\s+(\d+(?:\.\d+)?)\s*(Hz|kHz|MHz)'
        r'|(\d+(?:\.\d+)?)\s*(Hz|kHz|MHz)\s+steps?',
        prompt, re.I
    )
    if m:
        if m.group(1):
            out["sweep_step_mhz"] = _to_mhz(float(m.group(1)), m.group(2))
        else:
            out["sweep_step_mhz"] = _to_mhz(float(m.group(3)), m.group(4))

    # Plot ratio — "Neta5/Neta4", "Neta_5/Neta_4", "neta5/neta4"
    m = re.search(r'[Nn]eta_?(\d)\s*/\s*[Nn]eta_?(\d)', prompt)
    if m:
        out["plot_ratio"] = f"Neta_{m.group(1)}/Neta_{m.group(2)}"
        out["target_metric"] = f"Neta_{m.group(1)}"

    # Sequence file
    m = re.search(r'([A-Za-z0-9_\-]+\.py)', prompt)
    if m:
        out["sequence_file"] = m.group(1)

    return out


def parse_prompt_with_llm(
    prompt: str,
    client: LLMClient,
    knowledge: List[Document],
    default_params: str = "examples/params.txt",
) -> Goal:
    # Extract any snake_case identifiers from the prompt that look like parameter names
    # (multi-segment names the user typed explicitly — must be preserved verbatim)
    explicit_params = re.findall(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+){2,})\b', prompt)
    param_hint = ""
    if explicit_params:
        param_hint = (
            f"\nIMPORTANT — the user explicitly named these parameters; "
            f"use them VERBATIM (do not abbreviate or rename): "
            + ", ".join(dict.fromkeys(explicit_params))  # deduplicated, order-preserved
        )

    # Get relevant docs to help the LLM understand available sequences/analyses
    docs = search_for_role(knowledge, prompt, role="planner", top_k=6)
    doc_titles = "\n".join(f"- {d.title}: {d.summary[:120]}" for d in docs)

    user_prompt = f"""Parse this experimental goal into stages:

"{prompt}"
{param_hint}

Available sequences and analyses (use these filenames):
{doc_titles}

Default params file: {default_params}
"""

    overrides = _extract_sweep_overrides(prompt)

    resp = client.generate(user_prompt, system=_build_system_instruction(), temperature=0.1)
    goal = _parse_llm_goal(resp.text, default_params, prompt, explicit_params)

    # Hard-override with deterministically extracted values — LLM must not lose these
    for stage in goal.stages:
        # For calibrate stages: always set precession time sweep
        if stage.task_type == "calibrate":
            stage.stage_kind = "sweep"
            stage.sweep_param = "calibration_precession_time"
            if stage.sweep_start is None:
                stage.sweep_start = 0.5
            if stage.sweep_end is None:
                stage.sweep_end = 20.0
            if stage.max_iterations < 5:
                stage.max_iterations = 20

        if overrides.get("sweep_param") and stage.task_type != "calibrate":
            stage.sweep_param = overrides["sweep_param"]
            stage.stage_kind = "sweep"
        if overrides.get("sweep_range_mhz") is not None:
            stage.sweep_range_mhz = overrides["sweep_range_mhz"]
        if overrides.get("sweep_step_mhz") is not None:
            stage.sweep_step_mhz = overrides["sweep_step_mhz"]
            # Recompute max_iterations from range/step
            if stage.sweep_range_mhz and stage.sweep_step_mhz:
                stage.max_iterations = max(
                    int(round(stage.sweep_range_mhz / stage.sweep_step_mhz)) + 1, 1
                )
        if overrides.get("plot_ratio"):
            stage.plot_ratio = overrides["plot_ratio"]
        # For sweeps, don't set target_metric from plot_ratio — sweeps have no optimization target.
        # target_metric defaults to Neta_2 (used only for replay filtering, not displayed).
        if overrides.get("sequence_file") and not stage.sequence_file:
            stage.sequence_file = overrides["sequence_file"]

    return goal


def _pick_sweep_param(llm_param: str | None, explicit_params: List[str]) -> str | None:
    """If LLM truncated/renamed the sweep param, restore the user's original name."""
    if not llm_param:
        return explicit_params[0] if explicit_params else None
    # If the LLM param is a substring/prefix of an explicit param, prefer the explicit one
    for ep in explicit_params:
        if llm_param in ep or ep.startswith(llm_param):
            return ep
    return llm_param or None


def _parse_llm_goal(text: str, default_params: str, raw_prompt: str,
                    explicit_params: List[str] | None = None) -> Goal:
    raw = text.strip()
    # strip markdown fences if present
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[llm_preprocessor] JSON parse error: {e}")
        print(f"[llm_preprocessor] raw response: {text[:500]}")
        # fall back to a single default stage
        return Goal(
            stages=[Stage(
                name="optimize",
                description=raw_prompt,
                target_metric="Neta_2",
                params_file=default_params,
            )],
            raw_prompt=raw_prompt,
        )

    explicit_params = explicit_params or []
    stages: List[Stage] = []
    for s in data.get("stages", []):
        raw_kind = s.get("stage_kind", "optimize")
        raw_task = s.get("task_type", "optimize")
        valid_tasks = {"answer", "calibrate", "resonance", "optimize"}
        stage = Stage(
            name=s.get("name", "stage"),
            description=s.get("description", ""),
            task_type=raw_task if raw_task in valid_tasks else "optimize",
            target_metric=_validate_metric(s.get("target_metric", "Neta_2")),
            threshold=_safe_float(s.get("threshold")),
            threshold_op=s.get("threshold_op") or ">",
            sequence_file=s.get("sequence_file", ""),
            analysis_script=s.get("analysis_script") or None,
            analysis_y_op=s.get("analysis_y_op") or None,
            analysis_param_str=s.get("analysis_param_str") or None,
            params_file=None,  # params come from config.json globals, not a file
            max_iterations=int(s.get("max_iterations") or 20),
            notes=s.get("hook_description", ""),
            stage_kind=raw_kind if raw_kind in ("optimize", "sweep") else "optimize",
            sweep_param=_pick_sweep_param(s.get("sweep_param"), explicit_params),
            plot_ratio=s.get("plot_ratio") or None,
            sweep_range_mhz=_safe_float(s.get("sweep_range_mhz")),
            sweep_step_mhz=_safe_float(s.get("sweep_step_mhz")),
            sweep_start=_safe_float(s.get("sweep_start")),
            sweep_end=_safe_float(s.get("sweep_end")),
        )
        stages.append(stage)

    if not stages:
        stages = [Stage(
            name="optimize",
            description=raw_prompt,
            target_metric="Neta_2",
            params_file=default_params,
        )]

    return Goal(
        stages=stages,
        timeout_seconds=int(data.get("timeout_seconds") or 3600),
        raw_prompt=raw_prompt,
    )


def _validate_metric(m: str) -> str:
    valid = {"Neta_1", "Neta_2", "Neta_3", "Neta_4", "Neta_5",
             "Sz", "contrast", "N_atoms"}
    return m if m in valid else "Neta_2"


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None