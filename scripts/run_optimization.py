"""Entry point for the HAL-style optimization loop."""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from superradiant_assistant.config import CONFIG
from superradiant_assistant.hooks import HookManager


def build_hooks(require_confirm: bool = False, live: bool = False) -> HookManager:
    hm = HookManager()

    def announce(goal, params, **_):
        print("=" * 70)
        for s in goal.stages:
            print(f"  stage         : {s.name} ({s.stage_kind})")
            print(f"  description   : {s.description[:70]}")
            if s.stage_kind == "sweep":
                print(f"  sweep         : {s.sweep_param}")
                if s.sweep_range_mhz and s.sweep_step_mhz:
                    print(f"  range/step    : ±{s.sweep_range_mhz/2*1000:.2f} kHz / "
                          f"{s.sweep_step_mhz*1000:.4f} kHz → {s.max_iterations} points")
                if s.plot_ratio:
                    print(f"  plot          : {s.plot_ratio}  [phase 3: via lyse]")
            else:
                print(f"  target        : {s.target_metric} {s.threshold_op} {s.threshold}")
            print(f"  sequence      : {s.sequence_file}")
            print(f"  max_iter      : {s.max_iterations}")
            print()
        print(f"  max_dollars   : ${goal.max_dollars}")
        print(f"  timeout (s)   : {goal.timeout_seconds}")
        if live:
            print("  mode          : LIVE (shots queued in BLACS)")
        print("=" * 70)

    def confirm_shot(request, iteration, **_):
        print(f"\n  [proposed params] {request.globals_to_set}")
        ans = input("  Run this shot? [Y/n]: ").strip().lower()
        return ans in ("", "y", "yes")

    def before_queue(request, iteration, **_):
        print(f"\n  [SAFETY] iter {iteration+1} — about to queue:")
        for k, v in request.globals_to_set.items():
            print(f"    {k} = {v}")
        try:
            ans = input("  Queue this shot in BLACS? [Y/n]: ").strip().lower()
        except EOFError:
            ans = ""
        proceed = ans not in ("n", "no")
        if not proceed:
            print("  Skipping shot.")
        return proceed

    hm.register("before_loop", announce)
    if live:
        hm.register("before_shot", before_queue)
    elif require_confirm:
        hm.register("before_shot", confirm_shot)
    return hm


def _parse_classification(text: str) -> dict:
    raw = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"kind": "query", "refined": text, "stages": []}


def _run_goal(goal, args, llm_client, knowledge):
    """Apply CLI overrides, build hooks, run loop, print summary."""
    from superradiant_assistant.orchestrator.loop import run_loop

    if args.max_iters:
        for s in goal.stages:
            s.max_iterations = args.max_iters
    goal.max_dollars = args.max_dollars
    goal.max_tokens = args.max_tokens

    live = getattr(args, "live", False)
    hooks = build_hooks(require_confirm=args.confirm, live=live)

    executor = None
    if live:
        from superradiant_assistant.interfaces.runmanager_iface import RunmanagerInterface
        from superradiant_assistant.orchestrator.live_executor import LiveExecutor
        sequence = goal.stages[0].sequence_file if goal.stages else ""
        rm = RunmanagerInterface(
            sequence_file=sequence or None,
            output_folder=str(CONFIG.historical_data_root),
        )
        executor = LiveExecutor(CONFIG.historical_data_root, rm)
        print(f"[live] RunmanagerInterface connected, LiveExecutor ready\n")

    summary = run_loop(
        goal=goal,
        data_root=CONFIG.historical_data_root,
        repo_root=REPO_ROOT,
        hooks=hooks,
        llm_client=llm_client,
        knowledge=knowledge,
        executor=executor,
    )
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2, default=str))
    if llm_client:
        print("\nLLM usage:", llm_client.cost_tracker.summary())
    return summary


def _run_interactive(args, llm_client, knowledge):
    """Q&A REPL — answers questions from KB, transitions to goal on a command."""
    from superradiant_assistant.knowledge.search import search_for_role
    from superradiant_assistant.llm.prompts import (
        answer_system_instruction,
        preprocess_system_instruction,
    )
    from superradiant_assistant.orchestrator.llm_preprocessor import parse_prompt_with_llm

    print("=" * 70)
    print("  SuperradiantAssistant — Interactive Mode  (--llm)")
    print("  Ask questions about the experiment, or state a goal to execute.")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Exiting]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break

        # Classify as COMMAND or QUERY
        cls_resp = llm_client.generate(
            user_input,
            system=preprocess_system_instruction(),
            temperature=0.0,
        )
        classification = _parse_classification(cls_resp.text)

        if classification.get("kind") == "command":
            refined = classification.get("refined") or user_input
            print(f"\n[Goal detected] {refined}")
            print("Parsing stages with LLM...")
            goal = parse_prompt_with_llm(refined, llm_client, knowledge)
            print(f"  {len(goal.stages)} stage(s):")
            for s in goal.stages:
                print(f"    - {s.name} ({s.stage_kind}): {s.description[:65]}")
            print()
            _run_goal(goal, args, llm_client, knowledge)
            print("\n[Goal complete — ask more questions or state a new goal.]")
        else:
            # Answer from KB — augment with function excerpts if code is mentioned
            from superradiant_assistant.knowledge.search import augment_with_function_excerpts
            docs = search_for_role(knowledge, user_input, role="answer", top_k=6)
            docs = augment_with_function_excerpts(docs, user_input)
            ans_resp = llm_client.generate(
                user_input,
                system=answer_system_instruction(docs),
            )
            print(f"\nAssistant: {ans_resp.text.strip()}")


def main():
    ap = argparse.ArgumentParser(
        description="Superradiant Assistant — HAL-style experiment orchestrator"
    )
    ap.add_argument("prompt", nargs="?", default=None,
                    help="Natural-language goal (omit with --llm to enter interactive mode)")
    ap.add_argument("--llm", action="store_true",
                    help="Use LLM for prompt parsing, planning, and coding")
    ap.add_argument("--confirm", action="store_true",
                    help="Require y/n confirmation before each shot")
    ap.add_argument("--live", action="store_true",
                    help="Queue shots in runmanager/BLACS (requires GUIs open); "
                         "results still come from offline replay")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="Override max iterations per stage")
    ap.add_argument("--max-dollars", type=float, default=100.0)
    ap.add_argument("--max-tokens", type=int, default=1_000_000)
    args = ap.parse_args()

    llm_client = None
    knowledge = None

    if args.llm:
        if not CONFIG.gemini_api_key:
            config_example = REPO_ROOT / "config.json.example"
            print("ERROR: GEMINI_API_KEY is not set.")
            print(f"  Copy {config_example} → {REPO_ROOT / 'config.json'}")
            print("  and fill in your Gemini API key.")
            return

        from superradiant_assistant.llm.client import make_client
        from superradiant_assistant.llm.cost_tracker import CostTracker
        from superradiant_assistant.knowledge.loader import load_knowledge_base

        tracker = CostTracker(max_dollars=args.max_dollars, max_tokens=args.max_tokens)
        llm_client = make_client("gemini", cost_tracker=tracker)
        print("Loading knowledge base...")
        knowledge = load_knowledge_base()
        print(f"  loaded {len(knowledge)} documents\n")

    # Interactive mode: --llm with no prompt
    if args.llm and not args.prompt:
        _run_interactive(args, llm_client, knowledge)
        return

    if not args.prompt:
        print("Provide a natural-language goal, or use --llm without a prompt for interactive mode.")
        print('  python scripts/run_optimization.py --llm')
        print('  python scripts/run_optimization.py "Optimize Neta_2 above 700" --llm')
        return

    if args.llm:
        print("Parsing goal with LLM...")
        from superradiant_assistant.orchestrator.llm_preprocessor import parse_prompt_with_llm
        goal = parse_prompt_with_llm(args.prompt, llm_client, knowledge)
        print(f"  identified {len(goal.stages)} stage(s):")
        for s in goal.stages:
            print(f"    - {s.name} ({s.stage_kind}): {s.description[:65]}")
        print()
    else:
        from superradiant_assistant.orchestrator.preprocessor import parse_prompt
        goal = parse_prompt(args.prompt)

    _run_goal(goal, args, llm_client, knowledge)


if __name__ == "__main__":
    main()