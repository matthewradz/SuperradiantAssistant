"""Regex-based prompt parser — used when --llm is NOT set."""
from __future__ import annotations
import re
from superradiant_assistant.goal import Goal, Stage


def parse_prompt(prompt: str, default_params: str = "examples/params.txt") -> Goal:
    """Parse a natural-language prompt into a Goal with one Stage."""

    # target metric
    target_metric = "Neta_2"
    m = re.search(r"\b(Neta_[1-5]|Sz|contrast|N_atoms)\b", prompt)
    if m:
        target_metric = m.group(1)

    # threshold
    threshold = None
    threshold_op = ">"
    m = re.search(
        r"\b(above|>=|>|at least|below|<=|<|at most)\s*([0-9]+(?:\.[0-9]+)?)",
        prompt, re.I,
    )
    if m:
        op_word = m.group(1).lower()
        threshold = float(m.group(2))
        threshold_op = {
            "above": ">", ">": ">",
            "at least": ">=", ">=": ">=",
            "below": "<", "<": "<",
            "at most": "<=", "<=": "<=",
        }[op_word]

    # sequence file
    sequence_file = ""
    m = re.search(r"([A-Za-z0-9_\-./\\]+\.py)", prompt)
    if m:
        sequence_file = m.group(1).split("/")[-1].split("\\")[-1]

    # params file
    params_file = default_params
    m = re.search(r"([A-Za-z0-9_\-./\\]+\.txt)", prompt)
    if m:
        params_file = m.group(1)

    # max iterations
    max_iterations = 50
    m = re.search(
        r"\b(?:for|in|max(?:imum)?)\s*([0-9]+)\s*(?:runs?|iter|iterations|shots?)\b",
        prompt, re.I,
    )
    if m:
        max_iterations = int(m.group(1))

    stage = Stage(
        name="optimize",
        description=prompt,
        target_metric=target_metric,
        threshold=threshold,
        threshold_op=threshold_op,
        sequence_file=sequence_file,
        params_file=params_file,
        max_iterations=max_iterations,
    )
    return Goal(stages=[stage], raw_prompt=prompt)