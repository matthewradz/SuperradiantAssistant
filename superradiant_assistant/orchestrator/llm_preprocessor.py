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


SYSTEM_INSTRUCTION = """You are a research planning assistant for an AMO physics lab.
Your job is to parse a user's experimental goal into a structured list of stages.

Each stage is one logical sub-task, executed in order. For a simple goal like
"optimize atom loading", there is one stage. For a complex goal like "measure
clock coherence time with and without fiber noise cancellation", there are
multiple stages (e.g. stabilize loading, scan clock frequency, Ramsey sweep,
FNC-off repeat).

Return ONLY valid JSON with this schema — no markdown, no explanation:
{
  "stages": [
    {
      "name": "<short_snake_case_name>",
      "description": "<one sentence describing what this stage does>",
      "target_metric": "<e.g. Neta_2 | Sz | contrast | N_atoms>",
      "threshold": <float or null>,
      "threshold_op": "<> | >= | < | <= | ==>",
      "sequence_file": "<filename.py or empty string>",
      "params_file": "<e.g. examples/params.txt or empty string if not needed>",
      "max_iterations": <int>,
      "needs_user_hook": <true if the user must intervene before this stage>,
      "hook_description": "<what the user must do, or empty string>",
      "stage_kind": "<optimize | sweep>",
      "sweep_param": "<global variable name being swept, or empty string>",
      "plot_ratio": "<numerator/denominator metric ratio to plot, e.g. Neta_5/Neta_4, or empty string>",
      "sweep_range_mhz": <total sweep range converted to MHz, e.g. 0.003 for 3kHz, or null>,
      "sweep_step_mhz": <step size converted to MHz, e.g. 0.00025 for 250Hz, or null>
    }
  ],
  "timeout_seconds": <int>,
  "rationale": "<one sentence summarising the overall plan>"
}

Rules:
- target_metric must be one of: Neta_1, Neta_2, Neta_3, Neta_4, Neta_5, Sz, contrast, N_atoms
- If a metric is not clearly specified, default to Neta_2 for loading stages
- For sweep stages with a plot_ratio (e.g. "Neta_5/Neta_4"), set target_metric to the NUMERATOR metric (e.g. Neta_5)
- If no threshold is specified, set threshold to null
- If no sequence is specified, use the most relevant one from the documents
- Default max_iterations is 20 per stage
- Default timeout_seconds is 3600
- For stages that require user intervention (e.g. turn FNC on/off), set needs_user_hook=true
- stage_kind="sweep" for parameter scans; stage_kind="optimize" for threshold-based optimizations
- For sweep stages, set sweep_param to the EXACT global variable name as it appears in runmanager (do NOT abbreviate or paraphrase it)
- For sweep stages that plot a ratio (e.g. "plot neta5/neta4"), set plot_ratio accordingly (e.g. "Neta_5/Neta_4")
- For a sweep, set threshold to null and max_iterations to the desired number of scan points (default 15)
- For a sweep with an explicit range/step (e.g. "3kHz in steps of 250Hz"), convert both to MHz: sweep_range_mhz=0.003, sweep_step_mhz=0.00025. Set max_iterations = round(sweep_range_mhz / sweep_step_mhz) + 1
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

    resp = client.generate(user_prompt, system=SYSTEM_INSTRUCTION, temperature=0.1)
    goal = _parse_llm_goal(resp.text, default_params, prompt, explicit_params)

    # Hard-override with deterministically extracted values — LLM must not lose these
    for stage in goal.stages:
        if overrides.get("sweep_param"):
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
        stage = Stage(
            name=s.get("name", "stage"),
            description=s.get("description", ""),
            target_metric=_validate_metric(s.get("target_metric", "Neta_2")),
            threshold=_safe_float(s.get("threshold")),
            threshold_op=s.get("threshold_op") or ">",
            sequence_file=s.get("sequence_file", ""),
            params_file=s.get("params_file") or default_params or None,
            max_iterations=int(s.get("max_iterations", 20)),
            notes=s.get("hook_description", ""),
            stage_kind=raw_kind if raw_kind in ("optimize", "sweep") else "optimize",
            sweep_param=_pick_sweep_param(s.get("sweep_param"), explicit_params),
            plot_ratio=s.get("plot_ratio") or None,
            sweep_range_mhz=_safe_float(s.get("sweep_range_mhz")),
            sweep_step_mhz=_safe_float(s.get("sweep_step_mhz")),
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
        timeout_seconds=int(data.get("timeout_seconds", 3600)),
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