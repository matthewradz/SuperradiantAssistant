"""LLM-backed coder — turns a PlanStep into a ShotRequest."""
from __future__ import annotations
import json
import re
from typing import List, Dict, Any

from superradiant_assistant.goal import Stage
from superradiant_assistant.signals import ShotSignal
from superradiant_assistant.params import ParamSpec
from superradiant_assistant.knowledge.loader import Document
from superradiant_assistant.knowledge.search import search_for_role
from superradiant_assistant.llm.client import LLMClient
from superradiant_assistant.llm.prompts import coder_system_instruction, state2text
from superradiant_assistant.orchestrator.planner import PlanStep, _best_metric
from superradiant_assistant.orchestrator.developer import ShotRequest
from superradiant_assistant.optimizers.hill_climb import _shot_globals


class LLMCoder:
    def __init__(
        self,
        client: LLMClient,
        knowledge: List[Document],
        params: List[ParamSpec],
    ):
        self.client = client
        self.knowledge = knowledge
        self.params = params

    def make_shot_request(
        self,
        plan: PlanStep,
        stage: Stage,
        history: List[ShotSignal],
        state: Dict[str, Any],
    ) -> ShotRequest:

        query = plan.prompt_for_coder or stage.description or stage.target_metric
        docs = search_for_role(self.knowledge, query, role="coder", top_k=5)

        imports_block = "import numpy as np\nimport json"
        system = coder_system_instruction(docs, imports_block, state)

        # Sweep stages: only vary the sweep parameter
        if stage.stage_kind == "sweep" and stage.sweep_param:
            sweep_history = []
            for i, s in enumerate(history):
                val = _shot_globals(s).get(stage.sweep_param)
                metric = getattr(s, stage.target_metric, None)
                if val is not None:
                    sweep_history.append(f"  iter {i+1}: {stage.sweep_param}={val}, {stage.target_metric}={metric}")
            history_text = "\n".join(sweep_history) or "  (no shots yet)"

            user_prompt = f"""
{plan.prompt_for_coder}

This is a SWEEP stage. Set ONLY the parameter being swept.
Sweep parameter: {stage.sweep_param}
Plot ratio: {stage.plot_ratio or stage.target_metric}
Iteration: {len(history)+1} of {stage.max_iterations}

Values swept so far and their results:
{history_text}

Infer a reasonable next value from the knowledge base (check the current globals value and the
goal context). State your assumed sweep range in the rationale.

Return ONLY this JSON (no markdown):
{{
  "globals_to_set": {{
    "{stage.sweep_param}": <float_value>
  }},
  "rationale": "<value chosen and assumed sweep range>"
}}
""".strip()

        else:
            # Optimize stage: propose all relevant params
            if self.params:
                params_lines = "\n".join(
                    f"  {p.name}: range [{p.lo}, {p.hi}], typical {p.init}"
                    for p in self.params
                )
            else:
                params_lines = "  (no explicit bounds — infer from knowledge base and state assumptions)"

            best_val = _best_metric(history, stage.target_metric)
            best_params_text = ""
            if best_val is not None and history:
                for s in reversed(history):
                    v = getattr(s, stage.target_metric, None)
                    if v == best_val:
                        got = _shot_globals(s)
                        best_params_text = "\n".join(
                            f"  {p.name} = {got.get(p.name, p.init)}"
                            for p in self.params
                        )
                        break

            user_prompt = f"""
{plan.prompt_for_coder}

Goal: {stage.target_metric} {stage.threshold_op} {stage.threshold}
Sequence: {stage.sequence_file}
Iterations so far: {len(history)}
Best {stage.target_metric}: {best_val}
Best params so far:
{best_params_text or '  (none yet)'}

Available parameters and bounds:
{params_lines}

Propose the next set of parameter values as JSON.
If explicit bounds were provided, values MUST be within them.
If bounds were NOT provided, infer reasonable values from the knowledge base and goal context,
and include your assumed range in the rationale so the user can verify.

Return ONLY this JSON (no markdown):
{{
  "globals_to_set": {{
    "<param_name>": <float_value>,
    ...
  }},
  "rationale": "<one sentence — state assumed bounds if inferred>"
}}
""".strip()

        resp = self.client.generate(user_prompt, system=system, temperature=0.4)
        return _parse_shot_request(resp.text, stage.sequence_file, self.params,
                                   required_metric=stage.target_metric)


def _parse_shot_request(
    text: str,
    sequence_file: str,
    params: List[ParamSpec],
    required_metric: str = "Neta_2",
) -> ShotRequest:
    raw = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    try:
        data = json.loads(raw)
        globals_raw = data.get("globals_to_set", {})
    except json.JSONDecodeError:
        globals_raw = {}

    globals_to_set: Dict[str, float] = {}
    param_map = {p.name: p for p in params}
    for name, val in globals_raw.items():
        if name not in param_map:
            continue
        p = param_map[name]
        try:
            globals_to_set[name] = p.clip(float(val))
        except (TypeError, ValueError):
            globals_to_set[name] = p.init

    # fill any missing params with init
    for p in params:
        if p.name not in globals_to_set:
            globals_to_set[p.name] = p.init

    return ShotRequest(
        sequence_file=sequence_file,
        globals_to_set=globals_to_set,
        required_metric=required_metric,
        notes=data.get("rationale", "") if "data" in locals() else "",
    )