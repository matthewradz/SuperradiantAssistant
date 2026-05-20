"""Main orchestration loop — multi-stage, LLM-backed."""
from __future__ import annotations
import math
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from superradiant_assistant.goal import Goal, Stage
from superradiant_assistant.params import load_params
from superradiant_assistant.signals import ShotSignal
from superradiant_assistant.state import STATE
from superradiant_assistant.hooks import HookManager
from superradiant_assistant.orchestrator.executor import OfflineReplayExecutor


def run_loop(
    goal: Goal,
    data_root: Path,
    repo_root: Path,
    hooks: Optional[HookManager] = None,
    llm_client=None,
    knowledge=None,
    executor=None,
) -> Dict[str, Any]:
    hooks = hooks or HookManager()
    executor = executor or OfflineReplayExecutor(data_root)
    t0 = time.time()

    # build planners/coders per stage lazily
    def _make_planner_coder(stage: Stage):
        params = []
        if stage.params_file:
            params_path = (repo_root / stage.params_file).resolve()
            if params_path.exists():
                params = load_params(params_path)
                print(f"  [params] loaded {len(params)} bounds from {params_path.name}")
            else:
                print(f"  [params] {stage.params_file} not found — LLM will infer bounds from context")
        else:
            print("  [params] no params file specified — LLM will infer bounds from context")

        # Sweep stages with explicit range: use deterministic coder (no LLM needed)
        if stage.stage_kind == "sweep" and stage.sweep_param and stage.sweep_range_mhz:
            from superradiant_assistant.orchestrator.sweep_coder import DeterministicSweepCoder
            rm_iface = getattr(executor, "runmanager", None)
            sweep_coder = DeterministicSweepCoder(runmanager=rm_iface)
            if llm_client is not None and knowledge is not None:
                from superradiant_assistant.orchestrator.llm_planner import LLMPlanner
                planner = LLMPlanner(llm_client, knowledge)
            else:
                from superradiant_assistant.orchestrator.planner import DeterministicPlanner
                from superradiant_assistant.optimizers.hill_climb import HillClimbOptimizer
                planner = DeterministicPlanner(stage, params, HillClimbOptimizer(params, stage.target_metric))
            return planner, sweep_coder, params

        if llm_client is not None and knowledge is not None:
            from superradiant_assistant.orchestrator.llm_planner import LLMPlanner
            from superradiant_assistant.orchestrator.llm_coder import LLMCoder
            return LLMPlanner(llm_client, knowledge), LLMCoder(llm_client, knowledge, params), params
        else:
            from superradiant_assistant.orchestrator.planner import DeterministicPlanner
            from superradiant_assistant.orchestrator.developer import Developer
            from superradiant_assistant.optimizers.hill_climb import HillClimbOptimizer
            opt = HillClimbOptimizer(params, stage.target_metric)
            return DeterministicPlanner(stage, params, opt), Developer(stage.sequence_file), params

    all_stage_results: List[Dict[str, Any]] = []
    hooks.run("before_loop", goal=goal, params=[])

    for stage in goal.stages:
        stage.status = "active"
        STATE.set("current_stage", stage.name)
        history: List[ShotSignal] = []
        stop_reason = "unknown"

        planner, coder, params = _make_planner_coder(stage)

        print(f"\n{'='*70}")
        print(f"  STAGE: {stage.name} ({stage.stage_kind})")
        print(f"  {stage.description}")
        if stage.stage_kind == "sweep":
            print(f"  sweep: {stage.sweep_param}  |  plot: {stage.plot_ratio or '(none)'}")
            if stage.sweep_range_mhz and stage.sweep_step_mhz:
                print(f"  range: {stage.sweep_range_mhz*1000:.2f} kHz  "
                      f"step: {stage.sweep_step_mhz*1000:.4f} kHz  "
                      f"points: {stage.max_iterations}")
        else:
            print(f"  target: {stage.target_metric} {stage.threshold_op} {stage.threshold}")
        print(f"  max_iter: {stage.max_iterations}")
        print(f"{'='*70}")

        for it in range(stage.max_iterations):
            STATE.set("iteration", it + 1)

            if time.time() - t0 > goal.timeout_seconds:
                stop_reason = "timeout"
                break

            if hooks.any_true("should_abort", history=history, iteration=it):
                stop_reason = "aborted_by_hook"
                break

            hooks.run("before_propose", history=history, iteration=it)

            # --- Plan ---
            if hasattr(planner, "plan_next"):
                try:
                    if isinstance(planner, _get_llm_planner_type()):
                        plan = planner.plan_next(stage, history, STATE.to_dict(), it)
                    else:
                        plan = planner.plan_next(it, history)
                except Exception as e:
                    print(f"  [planner] error: {e} — skipping iteration")
                    continue
            else:
                break

            if plan.kind in ("end", "stop_target_met", "stop_max_iters"):
                stop_reason = plan.kind
                break

            # --- Ask user ---
            if plan.kind == "ask_user":
                print(f"\n[PLANNER ASKS] {plan.user_question}")
                user_resp = input("Your response: ").strip()
                STATE.set("user_response", user_resp)
                STATE.set("last_user_question", plan.user_question)
                continue  # re-plan with updated state

            # --- Develop ---
            try:
                if hasattr(coder, "make_shot_request") and _coder_takes_plan(coder):
                    req = coder.make_shot_request(plan, stage, history, STATE.to_dict())
                else:
                    req = coder.make_shot_request(plan.suggested_params)
            except Exception as e:
                print(f"  [coder] error: {e} — skipping iteration")
                continue

            # --- Before-shot hook ---
            if not hooks.all_true("before_shot", request=req, iteration=it):
                stop_reason = "rejected_by_before_shot"
                break

            # --- Execute ---
            try:
                sig = executor.execute(req)
            except Exception as e:
                print(f"[iter {it+1}] executor error: {e}")
                stop_reason = "executor_error"
                break

            history.append(sig)
            target_val = getattr(sig, stage.target_metric, None)
            best_val = max(
                (getattr(s, stage.target_metric) for s in history
                 if getattr(s, stage.target_metric) is not None),
                default=None,
            )

            if stage.stage_kind == "sweep":
                sweep_val = req.globals_to_set.get(stage.sweep_param, "?")
                print(
                    f"[{stage.name} iter {it+1:>2}/{stage.max_iterations}] "
                    f"{stage.sweep_param}={sweep_val} → queued | {sig.shot_id[:35]}"
                )
            else:
                print(
                    f"[{stage.name} iter {it+1:>2}/{stage.max_iterations}] "
                    f"{stage.target_metric}="
                    f"{f'{target_val:.1f}' if target_val is not None else 'None':>6} "
                    f"| best={f'{best_val:.1f}' if best_val is not None else 'None'} "
                    f"| {sig.shot_id[:40]}"
                )

            STATE.append_history({
                "stage": stage.name,
                "iteration": it + 1,
                "shot_id": sig.shot_id,
                stage.target_metric: target_val,
                "params": req.globals_to_set,
            })

            hooks.run("after_shot", signal=sig, history=history, iteration=it)

            if target_val is not None and stage.threshold_met(target_val):
                stop_reason = "target_met"
                break
        else:
            if stop_reason == "unknown":
                stop_reason = "max_iters_exhausted"

        # stage done
        best_val = max(
            (getattr(s, stage.target_metric) for s in history
             if getattr(s, stage.target_metric) is not None),
            default=None,
        )
        if stage.stage_kind == "sweep":
            stage.status = "complete" if history else "failed"
            if history:
                n = len(history)
                print(f"\n  [sweep] {n} shots queued for {stage.sweep_param}")
                if stage.plot_ratio:
                    print(f"  [sweep] plotting {stage.plot_ratio} vs {stage.sweep_param} "
                          f"from replay data...")
                    _plot_sweep(stage, history)
        else:
            stage.status = "complete" if stop_reason == "target_met" else "failed"
        stage.result = {
            "stop_reason": stop_reason,
            "iterations": len(history),
            "best_value": best_val,
        }
        all_stage_results.append({"stage": stage.name, **stage.result})
        hooks.run("on_stage_complete", stage=stage)

        if stop_reason in ("timeout", "aborted_by_hook", "executor_error"):
            break

    elapsed = time.time() - t0
    summary = {
        "elapsed_seconds": round(elapsed, 2),
        "stages": all_stage_results,
        "overall": "complete" if goal.all_complete else "incomplete",
    }
    hooks.run("on_finish", summary=summary)
    return summary


def _plot_sweep(stage, history: List[ShotSignal]) -> None:
    """Generate a plot after a sweep stage completes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[sweep] matplotlib not available, skipping plot")
        return

    xs, ys = [], []
    for sig in history:
        # Look in all_globals first (covers non-Atom-Loading params like clock frequency),
        # then fall back to atom_loading_globals
        combined = {**sig.atom_loading_globals, **sig.all_globals}
        x = combined.get(stage.sweep_param)
        if x is None:
            continue
        try:
            x = float(x) if not isinstance(x, list) else float(x[0])
        except (TypeError, ValueError, IndexError):
            continue

        if stage.plot_ratio and "/" in stage.plot_ratio:
            num_name, den_name = stage.plot_ratio.split("/", 1)
            num = getattr(sig, num_name.strip(), None)
            den = getattr(sig, den_name.strip(), None)
            if num is None or den is None or den == 0:
                continue
            if math.isnan(num) or math.isnan(den):
                continue
            y = num / den
        else:
            y = getattr(sig, stage.target_metric, None)
            if y is None or math.isnan(y):
                continue

        xs.append(x)
        ys.append(y)

    if not xs:
        print(f"[sweep] No valid data points to plot for sweep_param={stage.sweep_param}")
        return

    pairs = sorted(zip(xs, ys))
    xs_s, ys_s = zip(*pairs)

    fig, ax = plt.subplots()
    ax.plot(xs_s, ys_s, "o-")
    ax.set_xlabel(stage.sweep_param or "parameter")
    ylabel = stage.plot_ratio or stage.target_metric
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {stage.sweep_param}")

    out_path = Path(f"sweep_{stage.name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[sweep] Plot saved → {out_path.resolve()}")


def _coder_takes_plan(coder) -> bool:
    """True for LLMCoder and DeterministicSweepCoder (both take plan, stage, history, state)."""
    import inspect
    try:
        sig = inspect.signature(coder.make_shot_request)
        params = list(sig.parameters)
        return len(params) >= 4 and params[1] == "stage"
    except Exception:
        return False


# duck-type helpers to avoid circular imports
def _get_llm_planner_type():
    try:
        from superradiant_assistant.orchestrator.llm_planner import LLMPlanner
        return LLMPlanner
    except ImportError:
        return type(None)

def _get_llm_coder_type():
    try:
        from superradiant_assistant.orchestrator.llm_coder import LLMCoder
        return LLMCoder
    except ImportError:
        return type(None)