"""LLM-based prompt parser and stage decomposer.

Used when --llm is set. The LLM reads the full prompt and returns a
structured multi-stage Goal. It has access to the knowledge base so it
can reference real sequence files, analysis outputs, and global names.
"""
from __future__ import annotations
import json
import re
from typing import List, Dict, Any

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
      "params_file": "<e.g. examples/params.txt>",
      "max_iterations": <int>,
      "needs_user_hook": <true if the user must intervene before this stage>,
      "hook_description": "<what the user must do, or empty string>",
      "stage_kind": "<optimize | sweep>",
      "sweep_param": "<global variable name being swept, or empty string>",
      "plot_ratio": "<numerator/denominator metric ratio to plot, e.g. Neta_5/Neta_4, or empty string>"
    }
  ],
  "timeout_seconds": <int>,
  "rationale": "<one sentence summarising the overall plan>"
}

Rules:
- target_metric must be one of: Neta_1, Neta_2, Neta_3, Neta_4, Neta_5, Sz, contrast, N_atoms
- If a metric is not clearly specified, default to Neta_2 for loading stages
- If no threshold is specified, set threshold to null
- If no sequence is specified, use the most relevant one from the documents
- Default max_iterations is 20 per stage
- Default timeout_seconds is 3600
- For stages that require user intervention (e.g. turn FNC on/off), set needs_user_hook=true
- stage_kind="sweep" for parameter scans; stage_kind="optimize" for threshold-based optimizations
- For sweep stages, set sweep_param to the global variable name being varied
- For sweep stages that plot a ratio (e.g. "plot neta5/neta4"), set plot_ratio accordingly
- For a sweep, set threshold to null and max_iterations to the desired number of scan points (default 15)
"""


def parse_prompt_with_llm(
    prompt: str,
    client: LLMClient,
    knowledge: List[Document],
    default_params: str = "examples/params.txt",
) -> Goal:
    # Get relevant docs to help the LLM understand available sequences/analyses
    docs = search_for_role(knowledge, prompt, role="planner", top_k=6)
    doc_titles = "\n".join(f"- {d.title}: {d.summary[:120]}" for d in docs)

    user_prompt = f"""Parse this experimental goal into stages:

"{prompt}"

Available sequences and analyses (use these filenames):
{doc_titles}

Default params file: {default_params}
"""

    resp = client.generate(user_prompt, system=SYSTEM_INSTRUCTION, temperature=0.1)
    return _parse_llm_goal(resp.text, default_params, prompt)


def _parse_llm_goal(text: str, default_params: str, raw_prompt: str) -> Goal:
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

    stages: List[Stage] = []
    for s in data.get("stages", []):
        raw_kind = s.get("stage_kind", "optimize")
        stage = Stage(
            name=s.get("name", "stage"),
            description=s.get("description", ""),
            target_metric=_validate_metric(s.get("target_metric", "Neta_2")),
            threshold=_safe_float(s.get("threshold")),
            threshold_op=s.get("threshold_op", ">"),
            sequence_file=s.get("sequence_file", ""),
            params_file=s.get("params_file", default_params),
            max_iterations=int(s.get("max_iterations", 20)),
            notes=s.get("hook_description", ""),
            stage_kind=raw_kind if raw_kind in ("optimize", "sweep") else "optimize",
            sweep_param=s.get("sweep_param") or None,
            plot_ratio=s.get("plot_ratio") or None,
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