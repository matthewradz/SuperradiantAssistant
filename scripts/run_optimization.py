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

    # Answer tasks need no shots — handle immediately via code tracer + KB
    answer_stages = [s for s in goal.stages if s.task_type == "answer"]
    shot_stages   = [s for s in goal.stages if s.task_type != "answer"]

    if answer_stages and not shot_stages:
        _handle_answer_task(answer_stages, llm_client, knowledge)
        return {}

    executor = None
    if live:
        from superradiant_assistant.interfaces.runmanager_iface import RunmanagerInterface
        from superradiant_assistant.orchestrator.live_executor import LiveExecutor
        sequence = shot_stages[0].sequence_file if shot_stages else ""
        try:
            rm = RunmanagerInterface(
                sequence_file=sequence or None,
                output_folder=str(CONFIG.historical_data_root),
            )
            executor = LiveExecutor(CONFIG.historical_data_root, rm)
            print(f"[live] RunmanagerInterface connected, LiveExecutor ready")
            if sequence:
                print(f"[live] sequence: {sequence}\n")
        except Exception as e:
            print(f"\n[live] ERROR: labscript suite is not open or not reachable.")
            print(f"  Please open runmanager, BLACS, and lyse first:")
            print(f"    scripts\\launch_lab.bat")
            print(f"  Then retry this goal.")
            return {}

    summary = run_loop(
        goal=goal,
        data_root=CONFIG.historical_data_root,
        repo_root=REPO_ROOT,
        hooks=hooks,
        llm_client=llm_client,
        knowledge=knowledge,
        executor=executor,
    )

    # Post-shot: run analysis scripts for calibrate/resonance stages
    for stage in shot_stages:
        if stage.analysis_script and stage.status != "failed":
            _run_post_analysis(stage, live, llm_client)

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2, default=str))
    if llm_client:
        print("\nLLM usage:", llm_client.cost_tracker.summary())
    return summary


def _handle_answer_task(stages, llm_client, knowledge):
    """Handle Q&A / code-tracing without running any shots."""
    from superradiant_assistant.knowledge.search import search_for_role, augment_with_function_excerpts
    from superradiant_assistant.llm.prompts import answer_system_instruction
    from superradiant_assistant.orchestrator.code_tracer import build_answer_context

    for stage in stages:
        query = stage.description or stage.name
        print(f"\n[answer] {query}")

        # Code tracer: follow imports/globals in labscript files
        code_ctx = build_answer_context(query)

        # KB search
        docs = search_for_role(knowledge, query, role="answer", top_k=6)
        docs = augment_with_function_excerpts(docs, query)

        # Build system instruction with both KB docs and traced code
        system = answer_system_instruction(docs)
        if code_ctx:
            system += f"\n\n## Traced code context\n{code_ctx}"

        resp = llm_client.generate(query, system=system)
        print(f"\nAssistant: {resp.text.strip()}")


_SYNTHETIC_SWEEP_FOLDER  = r"C:\Users\radzi\Documents\data_synthetic_sweep"
_SYNTHETIC_LARMOR_FOLDER = r"C:\Users\radzi\Documents\data_synthetic_larmor"


def _read_delta_from_folder(folder) -> float | None:
    """Read delta_duration from the first HDF5 shot in a folder."""
    import h5py
    from pathlib import Path as _Path
    for p in sorted(_Path(folder).glob("*.h5")):
        try:
            with h5py.File(p, 'r') as f:
                if 'globals/recycling' in f:
                    raw = f['globals/recycling'].attrs.get('delta_duration')
                    if raw is not None:
                        return float(str(raw).split('#')[0].strip())
        except Exception:
            pass
    return None


def _run_post_analysis(stage, live: bool, llm_client):
    """Wait for user to confirm shots ran, then analyze and report."""
    from superradiant_assistant.orchestrator.standalone_analysis import (
        analyze_resonance_sweep, analyze_larmor_calibration
    )
    from pathlib import Path

    print(f"\n{'='*60}")
    print(f"  Shots are queued in BLACS.")
    print(f"  Run them, load the results into lyse, then press Enter.")
    print(f"{'='*60}")
    try:
        input("\n  [Press Enter when done] ")
    except EOFError:
        pass

    from superradiant_assistant.orchestrator.analysis_runner import edit_script_inplace

    data_folder = Path(_SYNTHETIC_SWEEP_FOLDER if stage.task_type == "resonance"
                       else _SYNTHETIC_LARMOR_FOLDER)

    # Read delta_duration from the synthetic data so the filter matches
    delta_last = _read_delta_from_folder(data_folder)

    # Edit the analysis script in-place so lyse sees the correct parameters
    if stage.analysis_script:
        edits = {}
        if stage.analysis_param_str:
            edits["parameter_str"] = stage.analysis_param_str
        if stage.analysis_y_op:
            edits["y_op_string"] = stage.analysis_y_op
        if delta_last is not None:
            edits["delta_list"] = str(delta_last)
        if edits:
            script_path = Path(stage.analysis_script)
            try:
                edit_script_inplace(script_path, edits)
                print(f"\n[analysis] Updated {script_path.name}:")
                for k, v in edits.items():
                    print(f"  {k} = {v}")
            except Exception as e:
                print(f"\n[analysis] Could not edit script: {e}")

    data_label = "resonance sweep" if stage.task_type == "resonance" else "calibration"
    print(f"\n[lyse] Clear lyse (File > Clear), then load the {data_label} shots.")
    print(f"       Click 'Run multishot analysis' in the lyse window.")
    try:
        input("  [Press Enter after multishot analysis has run] ")
    except EOFError:
        pass

    if stage.task_type == "resonance" and stage.sweep_param and stage.plot_ratio:
        print(f"\n[analysis] Analyzing {stage.plot_ratio} vs {stage.sweep_param}...")
        results = analyze_resonance_sweep(
            data_root=data_folder,
            sweep_param=stage.sweep_param,
            plot_ratio=stage.plot_ratio,
        )
        if llm_client:
            _interpret_resonance(stage, results, llm_client)

    elif stage.task_type == "calibrate":
        print(f"\n[analysis] Fitting Ramsey fringes...")
        results = analyze_larmor_calibration(data_root=data_folder)
        if llm_client:
            _interpret_calibration(stage, results, llm_client, live)


def _interpret_resonance(stage, results: dict, llm_client):
    """LLM interprets resonance sweep results and reports to user."""
    if results.get("error"):
        print(f"\n[assistant] Could not analyze: {results['error']}")
        return

    found      = results.get("found", False)
    freq       = results.get("resonance_freq_mhz")
    fit_freq   = results.get("fit_resonance_freq_mhz")
    linewidth  = results.get("fit_linewidth_mhz")
    ratio      = results.get("min_ratio")

    system = f"""You are a physics lab assistant reporting clock resonance findings.
The experiment swept {stage.sweep_param} and measured {stage.plot_ratio}.
On resonance, {stage.plot_ratio} dips from ~1.0 to ~0.2-0.3 (Lorentzian dip).
Report concisely in 2-3 sentences:
- If resonance found: state the fitted frequency and linewidth. Suggest updating clock_pi_resonance_frequency.
- If not found: say the ratio stayed flat and the resonance is outside this window — suggest widening the sweep."""

    if found and fit_freq:
        lw_hz = f"{linewidth*1e6:.0f} Hz" if linewidth else "unknown"
        prompt = (
            f"Lorentzian fit: resonance at {fit_freq:.6f} MHz, linewidth ~{lw_hz}, "
            f"min {stage.plot_ratio} = {ratio:.3f}."
        )
    elif found:
        prompt = (
            f"Minimum {stage.plot_ratio} = {ratio:.3f} at {freq:.6f} MHz. "
            f"Dip detected but Lorentzian fit failed."
        )
    else:
        prompt = (
            f"Min {stage.plot_ratio} = {ratio:.3f} — no significant dip. "
            f"Resonance likely outside the swept range."
        )
    try:
        resp = llm_client.generate(prompt, system=system)
        print(f"\n[assistant] {resp.text.strip()}")
        if results.get("plot_path"):
            print(f"  [plot saved: {results['plot_path']}]")
    except Exception as e:
        print(f"[assistant] {prompt}")


def _interpret_calibration(stage, results: dict, llm_client, live: bool):
    """LLM interprets Larmor calibration and offers to apply correction."""
    from superradiant_assistant.interfaces.runmanager_iface import RunmanagerInterface

    correction = results.get("correction_hz")
    if correction is None:
        print("\n[assistant] Calibration fit failed — could not determine correction.")
        return

    system = """You are a physics lab assistant reporting a Larmor frequency calibration result.
Report: what correction was found, and what it means physically (the RF is slightly off).
Be concise — 2 sentences."""

    prompt = (
        f"Ramsey fringe fit: fitted frequency = {results.get('fit_freq_khz', 0)*1000:.1f} Hz, "
        f"expected detuning = {results.get('larmor_detuning_hz', 100):.0f} Hz, "
        f"correction needed = {correction:+.2f} Hz."
    )
    try:
        resp = llm_client.generate(prompt, system=system)
        print(f"\n[assistant] {resp.text.strip()}")
        if results.get("plot_path"):
            print(f"  [plot saved: {results['plot_path']}]")
    except Exception:
        print(f"\n[assistant] Larmor Correction Needed: {correction:+.2f} Hz")

    # Offer to apply correction
    print(f"\n  Correction: {correction:+.2f} Hz to rf_larmor_frequency")
    try:
        ans = input("  Apply correction in runmanager? [Y/n]: ").strip().lower()
    except EOFError:
        ans = ""
    if ans not in ("n", "no") and live:
        try:
            rm = RunmanagerInterface()
            current = float(rm.get_globals().get("rf_larmor_frequency", 10157))
            new_val = current + correction
            rm.set_globals({"rf_larmor_frequency": new_val})
            print(f"  Updated rf_larmor_frequency: {current:.2f} -> {new_val:.2f} Hz")
        except Exception as e:
            print(f"  Could not update runmanager: {e}")




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

    last_exchange = ""  # remember last Q&A for context

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

        # Classify as COMMAND or QUERY — include last exchange for follow-up context
        classify_input = user_input
        if last_exchange:
            classify_input = f"[Previous exchange: {last_exchange[:300]}]\nUser: {user_input}"

        cls_resp = llm_client.generate(
            classify_input,
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
                tag = s.task_type
                print(f"    - {s.name} ({tag}/{s.stage_kind}): {s.description[:80]}")
            print()
            _run_goal(goal, args, llm_client, knowledge)
            print("\n[Goal complete — ask more questions or state a new goal.]")
        else:
            # Answer from KB — augment with function excerpts if code is mentioned
            from superradiant_assistant.knowledge.search import augment_with_function_excerpts
            from superradiant_assistant.orchestrator.code_tracer import build_answer_context
            docs = search_for_role(knowledge, user_input, role="answer", top_k=6)
            docs = augment_with_function_excerpts(docs, user_input)
            system = answer_system_instruction(docs)
            code_ctx = build_answer_context(user_input)
            if code_ctx:
                system += f"\n\n## Traced code context\n{code_ctx}"
            if last_exchange:
                system += f"\n\n## Previous exchange (for context)\n{last_exchange[:400]}"
            ans_resp = llm_client.generate(user_input, system=system)
            answer = ans_resp.text.strip()
            print(f"\nAssistant: {answer}")
            last_exchange = f"Q: {user_input}\nA: {answer[:200]}"


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