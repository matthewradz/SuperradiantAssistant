"""LLM-backed planner — replaces DeterministicPlanner."""
from __future__ import annotations
import json
import re
from typing import List, Dict, Any

from superradiant_assistant.goal import Stage
from superradiant_assistant.signals import ShotSignal
from superradiant_assistant.knowledge.loader import Document
from superradiant_assistant.knowledge.search import search_for_role
from superradiant_assistant.llm.client import LLMClient
from superradiant_assistant.llm.prompts import planner_system_instruction, history2text
from superradiant_assistant.orchestrator.planner import PlanStep, _best_metric


class LLMPlanner:
    def __init__(self, client: LLMClient, knowledge: List[Document]):
        self.client = client
        self.knowledge = knowledge

    def plan_next(
        self,
        stage: Stage,
        history: List[ShotSignal],
        state: Dict[str, Any],
        iteration: int,
    ) -> PlanStep:

        # retrieve relevant docs
        query = f"{stage.target_metric} {stage.description} {stage.sequence_file}"
        docs = search_for_role(self.knowledge, query, role="planner", top_k=5)

        best = _best_metric(history, stage.target_metric)
        hist_text = history2text(
            [{"iter": i+1,
              stage.target_metric: getattr(s, stage.target_metric),
              "shot_id": s.shot_id}
             for i, s in enumerate(history[-15:])
            ]
        )

        sweep_context = ""
        if stage.stage_kind == "sweep" and stage.sweep_param:
            sweep_context = f"""
## Sweep info
This is a SWEEP stage — do NOT ask the user for anything. Always return kind="propose_shot".
Sweep parameter: {stage.sweep_param}
Plot ratio: {stage.plot_ratio or stage.target_metric}
Step systematically through the parameter space. Infer a reasonable range from the knowledge
base if not specified. Each iteration propose a different value of {stage.sweep_param}.
"""

        user_prompt = f"""
## Current stage
Name: {stage.name}
Description: {stage.description}
Target: {stage.target_metric} {stage.threshold_op} {stage.threshold}
Sequence: {stage.sequence_file}
Iteration: {iteration+1} / {stage.max_iterations}
Best {stage.target_metric} so far: {best}
{sweep_context}
## Step history
{hist_text}
""".strip()

        system = planner_system_instruction(docs)
        resp = self.client.generate(user_prompt, system=system, temperature=0.2)

        return _parse_plan_step(resp.text)


def _parse_plan_step(text: str) -> PlanStep:
    raw = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return PlanStep(kind="propose_shot", rationale="LLM JSON parse error")

    kind = data.get("kind", "propose_shot")
    valid_kinds = {
        "propose_shot", "ask_user", "end",
        "stop_target_met", "stop_max_iters",
    }
    if kind not in valid_kinds:
        kind = "propose_shot"

    def _str(v) -> str:
        return v if isinstance(v, str) else ""

    return PlanStep(
        kind=kind,
        rationale=_str(data.get("rationale")),
        prompt_for_coder=_str(data.get("prompt_for_coder")),
        signal_description=_str(data.get("signal_description")),
        user_question=_str(data.get("user_question")),
    )