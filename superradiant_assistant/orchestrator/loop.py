"""Main orchestration loop — multi-stage, LLM-backed."""
from __future__ import annotations
from superradiant_assistant import splash as _S
import math
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from superradiant_assistant.goal import Goal, Stage
from superradiant_assistant.params import load_params, ParamSpec
from superradiant_assistant.signals import ShotSignal
from superradiant_assistant.state import STATE
from superradiant_assistant.memory import MEMORY
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
        # Build params list from config.json globals (replaces params.txt)
        params: List[ParamSpec] = []
        if stage.stage_kind != "sweep":
            from superradiant_assistant.config import CONFIG
            for g in CONFIG.experiment_globals:
                lo = g.get("min")
                hi = g.get("max")
                if lo is not None and hi is not None:
                    try:
                        params.append(ParamSpec(
                            name=g["name"],
                            lo=float(lo),
                            hi=float(hi),
                            init=(float(lo) + float(hi)) / 2,
                        ))
                    except (TypeError, ValueError):
                        pass
            if params:
                print(f"  {_S.tag('params')} using {len(params)} globals from config.json")

        # Sweep stages with explicit range: use deterministic coder (no LLM needed)
        if stage.stage_kind == "sweep" and stage.sweep_param and (
                stage.sweep_range_mhz or stage.sweep_start is not None):
            from superradiant_assistant.orchestrator.sweep_coder import DeterministicSweepCoder
            rm_iface = getattr(executor, "runmanager", None)
            sweep_coder = DeterministicSweepCoder(runmanager=rm_iface)
            if llm_client is not None and knowledge is not None:
                from superradiant_assistant.orchestrator.llm_planner import LLMPlanner
                planner = LLMPlanner(llm_client, knowledge)
            else:
                from superradiant_assistant.orchestrator.planner import DeterministicPlanner
                from superradiant_assistant.optimizers.hill_climb import HillClimbOptimizer
                planner = DeterministicPlanner(
                    stage, params,
                    HillClimbOptimizer(params, stage.target_metric,
                                       budget=stage.max_iterations))
            return planner, sweep_coder, params

        if llm_client is not None and knowledge is not None:
            from superradiant_assistant.orchestrator.llm_planner import LLMPlanner
            from superradiant_assistant.orchestrator.llm_coder import LLMCoder
            return LLMPlanner(llm_client, knowledge), LLMCoder(llm_client, knowledge, params), params
        else:
            from superradiant_assistant.orchestrator.planner import DeterministicPlanner
            from superradiant_assistant.orchestrator.developer import Developer
            from superradiant_assistant.optimizers.hill_climb import HillClimbOptimizer
            # The budget decides how much of the run goes on covering the range
            # before it contracts, so the optimizer has to know it.
            opt = HillClimbOptimizer(params, stage.target_metric,
                                     budget=stage.max_iterations)
            return DeterministicPlanner(stage, params, opt), Developer(stage.sequence_file), params

    all_stage_results: List[Dict[str, Any]] = []
    hooks.run("before_loop", goal=goal, params=[])

    for stage in goal.stages:
        stage.status = "active"
        STATE.set("current_stage", stage.name)
        history: List[ShotSignal] = []
        stop_reason = "unknown"

        # Point runmanager at this stage's sequence before anything is queued.
        # Without this a multi-sequence goal ran every stage against whichever
        # file happened to be open in the GUI.
        seq_error = _load_stage_sequence(executor, stage)
        if seq_error:
            print(f"  {_S.tag('sequence')} {seq_error}")
            stage.status = "failed"
            stage.result = {"stop_reason": "sequence_load_failed", "iterations": 0,
                            "best_value": None, "threshold_met": None,
                            "error": seq_error}
            all_stage_results.append({"stage": stage.name, **stage.result})
            hooks.run("on_stage_complete", stage=stage)
            continue

        planner, coder, params = _make_planner_coder(stage)

        # Without goal mode an optimize stage takes one shot and reports; it does
        # not keep iterating toward the threshold. Sweeps queue all points in a
        # single engage regardless, so the cap doesn't apply to them.
        max_iters = stage.max_iterations
        if stage.stage_kind != "sweep" and not goal.goal_mode:
            max_iters = 1

        _print_stage_banner(stage, max_iters, goal, _is_live(executor))

        iter_rows: List[Dict[str, Any]] = []
        for it in range(max_iters):
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

            # Whichever coder built this, the stage decides which metric matters.
            # Coders that default to a hard-coded name would otherwise send the
            # replay executor looking for the wrong result.
            if stage.target_metric:
                try:
                    req.required_metric = stage.target_metric
                except Exception:
                    pass

            # Sweep coder signals completion via notes field
            if getattr(req, "notes", "") == "sweep_complete":
                stop_reason = "sweep_complete"
                break

            # --- Before-shot hook ---
            if not hooks.all_true("before_shot", request=req, iteration=it):
                stop_reason = "rejected_by_before_shot"
                break

            # --- Execute ---
            # Stamped before the engage so the results collector can tell the
            # shots this sweep produced from everything already on disk.
            engage_ts = time.time()
            try:
                sig = executor.execute(req)
            except Exception as e:
                print(f"[iter {it+1}] executor error: {e}")
                stop_reason = "executor_error"
                break

            history.append(sig)
            target_val = sig.metric_value(stage.target_metric)
            best_val = max(
                (s.metric_value(stage.target_metric) for s in history
                 if s.metric_value(stage.target_metric) is not None),
                default=None,
            )

            if stage.stage_kind == "sweep":
                sweep_val = req.globals_to_set.get(stage.sweep_param, "?")
                # Say which of the two things actually happened. Printing
                # "queued" for a replayed shot is how a whole offline session
                # got mistaken for a live one.
                verb = "queued" if _is_live(executor) else "REPLAYED (offline, nothing queued)"
                print(
                    f"[{stage.name} iter {it+1:>2}/{max_iters}] "
                    f"{stage.sweep_param}={sweep_val} → {verb} | {sig.shot_id[:35]}"
                )
            else:
                print(
                    f"[{stage.name} iter {it+1:>2}/{max_iters}] "
                    f"{stage.target_metric}="
                    f"{f'{target_val:.1f}' if target_val is not None else 'None':>6} "
                    f"| best={f'{best_val:.1f}' if best_val is not None else 'None'} "
                    f"| {sig.shot_id[:40]}"
                )

            entry = {
                "stage": stage.name,
                "iteration": it + 1,
                "shot_id": sig.shot_id,
                stage.target_metric: target_val,
                "params": req.globals_to_set,
            }
            STATE.append_history(entry)
            MEMORY.append_history("shot", entry)
            # Carried out of the loop so the caller can report what was measured
            # at each point. Without it `run_optimization` returned only a best
            # value, and the agent went looking for the numbers itself: twelve
            # `inspect_shot` calls that found ten of fifteen shots and left out
            # the best one.
            iter_rows.append(entry)

            hooks.run("after_shot", signal=sig, history=history, iteration=it)

            if target_val is not None and stage.threshold_met(target_val):
                stop_reason = "target_met"
                break
        else:
            if stop_reason == "unknown":
                stop_reason = ("single_shot_done"
                                if max_iters == 1 and not goal.goal_mode
                                else "max_iters_exhausted")

        # stage done
        best_val = None if stage.stage_kind == "sweep" else max(
            (s.metric_value(stage.target_metric) for s in history
             if s.metric_value(stage.target_metric) is not None),
            default=None,
        )
        sweep_results: Optional[Dict[str, Any]] = None
        if stage.stage_kind == "sweep":
            stage.status = "complete" if history else "failed"
            if history:
                req_globals = history[0].requested_globals if history else {}
                sweep_val = req_globals.get(stage.sweep_param, "")
                if _is_live(executor):
                    _sweep_msg(f"queued: {stage.sweep_param} = {sweep_val}")
                    _sweep_msg("check BLACS — all shots should be in the queue")
                    # Queueing is not the measurement. Wait for the shots to run
                    # and for lyse to analyse them, then read the results back —
                    # otherwise the sweep ends with nothing learned.
                    print(f"  {_S.tag('sweep')} waiting for {stage.max_iterations} shots to run "
                          f"and lyse to analyse them...")
                    sweep_results = collect_sweep_results(
                        data_root=Path(data_root),
                        since_ts=engage_ts,
                        expected=stage.max_iterations,
                        sweep_param=stage.sweep_param,
                        timeout=stage.results_timeout,
                    )
                    # On failure, go and read the actual errors rather than
                    # handing the model a list of things that might be wrong.
                    if not sweep_results.get("complete"):
                        sweep_results["failure_detail"] = _real_failure_detail(
                            executor, engage_ts, sweep_results.get("diagnosis"))
                    print(f"  {_S.tag('sweep')} "
                          + summarise_sweep_results(sweep_results, stage.sweep_param))
                    # Record what was actually measured, so a later report can be
                    # checked against it rather than taken on trust.
                    try:
                        from superradiant_assistant.creative.evidence import get_evidence
                        ev = get_evidence()
                        ev.record_queue()
                        ev.record_rows(sweep_results.get("rows") or [],
                                       sweep_param=stage.sweep_param)
                    except Exception:
                        pass
                else:
                    _sweep_msg("OFFLINE REPLAY — nothing was queued on hardware")
                    print(f"  {_S.tag('sweep')} {stage.sweep_param} = {sweep_val} was replayed against "
                          f"historical shots only.")
                    _sweep_msg("restart with `agent.py --live` to queue real shots")
                # Reset the swept parameter to its scalar base value in runmanager
                if executor and hasattr(executor, "runmanager") and stage.sweep_param.endswith("_list"):
                    base_param = stage.sweep_param[:-5]  # strip "_list"
                    try:
                        g = executor.runmanager.get_globals()
                        base_val = g.get(base_param)
                        if base_val is not None:
                            executor.runmanager.set_globals({stage.sweep_param: base_val})
                            print(f"  {_S.tag('sweep')} Reset {stage.sweep_param} = {base_val} (scalar)")
                    except Exception as e:
                        print(f"  {_S.tag('sweep')} Could not reset {stage.sweep_param}: {e}")
        else:
            # A single-shot run that didn't reach the threshold isn't a failure —
            # it did exactly what was asked. Only goal mode treats a missed
            # threshold as an incomplete stage.
            stage.status = ("complete"
                             if stop_reason in ("target_met", "single_shot_done")
                             else "failed")
            # Evidence was recorded for sweeps only. An optimize run therefore
            # measured fifteen shots and then had its own report REFUSED for
            # "citing shots this session never measured", and the way round it
            # was to call read_shot_results until they counted as read from disk
            # -- which stamped the report RETROSPECTIVE, "this session ran no
            # measurement", above fifteen shots it had just fired.
            if _is_live(executor) and iter_rows:
                try:
                    from superradiant_assistant.creative.evidence import get_evidence
                    ev = get_evidence()
                    ev.record_queue()
                    # Flattened the way sweep rows are, so a report can plot
                    # these and cite them without knowing which loop made them.
                    paths = {s.shot_id: s.shot_path for s in history}
                    flat = []
                    for r in iter_rows:
                        params = r.get("params") or {}
                        flat.append({"shot_id": r.get("shot_id"),
                                     "shot_path": paths.get(r.get("shot_id"), ""),
                                     **params,
                                     stage.target_metric: r.get(stage.target_metric)})
                    # The independent variable, when there is exactly one. With
                    # several knobs moving there is no single x to plot against.
                    varied = {k for r in iter_rows for k in (r.get("params") or {})}
                    ev.record_rows(flat, sweep_param=(varied.pop() if len(varied) == 1
                                                       else ""))
                except Exception as e:
                    print(f"  [evidence] could not record this run: "
                          f"{type(e).__name__}: {e}")
        # The parameters of the best shot, not just its value. A best value with
        # no coordinates is not a result anyone can act on, and it is what let
        # three different "best" numbers coexist in one report.
        best_row = None
        if best_val is not None:
            best_row = next((r for r in iter_rows
                             if r.get(stage.target_metric) == best_val), None)

        stage.result = {
            "stop_reason": stop_reason,
            "iterations": len(history),
            "best_value": best_val,
            "best_params": (best_row or {}).get("params"),
            "best_shot_id": (best_row or {}).get("shot_id"),
            "threshold_met": bool(
                best_val is not None and stage.threshold_met(best_val)
            ) if stage.stage_kind != "sweep" else None,
            # Measured results for a live sweep, so the caller can report the
            # measurement rather than just the fact that shots were queued.
            "sweep_results": sweep_results,
            # What was measured at every point, for the non-sweep stages. Sweeps
            # carry theirs in `sweep_results`.
            "shot_rows": None if stage.stage_kind == "sweep" else iter_rows,
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


def wait_for_shot_signal(
    data_root: Path,
    since_ts: float,
    timeout: float = 300.0,
    poll: float = 2.0,
    first_file_grace: float = 45.0,
):
    """Wait for one shot newer than `since_ts` that lyse has analysed.

    Returns (signal, diagnosis). The signal is None unless a shot file appeared
    *and* carries at least one analysis result -- a shot that ran but was not
    analysed has measured nothing that can be read.

    This is what an optimize iteration needs and what `collect_sweep_results` is
    the many-shot version of. Kept separate because the caller here needs the
    ShotSignal itself, not a row of metrics, and because the sweep path is the
    one that works and should not be disturbed.
    """
    from superradiant_assistant.interfaces.hdf5_reader import list_shots, read_shot
    from superradiant_assistant.orchestrator.agents import suppress_heartbeat

    start = time.time()
    deadline = start + timeout
    saw_file = False

    with suppress_heartbeat():
        while time.time() < deadline:
            newest, newest_ts = None, since_ts
            for p in list_shots(data_root):
                try:
                    ts = p.stat().st_mtime
                except OSError:
                    continue
                if ts < since_ts:
                    continue
                saw_file = True
                if ts >= newest_ts:
                    newest, newest_ts = p, ts
            if newest is not None:
                try:
                    sig = read_shot(newest)
                except Exception:
                    sig = None
                if sig is not None and sig.available_metrics():
                    return sig, "ok"
            # A sequence that fails to compile aborts within seconds, so if no
            # file at all has appeared by now it is not going to.
            if not saw_file and time.time() - start > first_file_grace:
                return None, "no_shot_files"
            time.sleep(poll)

    return None, ("shot_ran_but_no_results" if saw_file else "no_shot_files")


def collect_sweep_results(
    data_root: Path,
    since_ts: float,
    expected: int,
    sweep_param: str,
    timeout: float = 300.0,
    poll: float = 3.0,
    first_file_grace: float = 30.0,
    stall_grace: float = 90.0,
) -> Dict[str, Any]:
    """Wait for queued shots to finish, then read what lyse saved for them.

    `run_sweep` returns the moment runmanager engages, so at that point no shot
    has run and no analysis exists. Without this step the agent reports "queued
    N shots" and never learns anything about the measurement it just asked for.

    Waits for `expected` shot files newer than `since_ts` that carry at least
    one analysis result, then returns their metrics. Returns whatever it has
    when `timeout` expires — a partial answer beats blocking forever.
    """
    from superradiant_assistant.interfaces.hdf5_reader import list_shots, read_shot
    from superradiant_assistant.orchestrator.agents import suppress_heartbeat

    start = time.time()
    deadline = start + timeout
    seen: Dict[str, Dict[str, Any]] = {}
    files_appeared: set = set()
    last_progress = start

    # The spinner and this progress bar both own "the current line", and the
    # spinner redraws eight times a second, so it wins every race and the bar is
    # never seen. During this wait the bar is the better indicator anyway: it
    # counts shots actually analysed, not seconds elapsed.
    with suppress_heartbeat():
        while time.time() < deadline:
            before = (len(files_appeared), len(seen))
            for p in list_shots(data_root):
                try:
                    if p.stat().st_mtime < since_ts:
                        continue
                    # Track the file separately from its results: a shot that
                    # exists but carries no results means something very
                    # different from a shot that never got compiled at all.
                    files_appeared.add(str(p))
                    if str(p) in seen:
                        continue
                    sig = read_shot(p)
                except Exception:
                    continue
                metrics = sig.available_metrics()
                if not metrics:
                    continue  # shot exists but lyse has not analysed it yet
                x = sig.all_globals.get(sweep_param)
                # The path travels with the row so the report can find the
                # figures the lyse routine saved next to the shot.
                seen[str(p)] = {"shot_id": sig.shot_id, "shot_path": str(p),
                                sweep_param: x, **metrics}

            now = time.time()
            if (len(files_appeared), len(seen)) != before:
                last_progress = now

            # Redrawn every poll, not only when the count moves: the first shot
            # can take a minute to appear, and a blank screen for that minute is
            # indistinguishable from a hang. It is one line, rewritten in place.
            _sweep_bar(len(seen), expected, len(files_appeared), now - start)

            if len(seen) >= expected:
                break

            # Fail fast instead of burning the whole timeout. A compilation
            # error aborts within seconds, so if nothing has appeared by now it
            # is not going to. Waiting the full timeout to say so is the same as
            # not saying it: the operator sees a frozen line and has to guess.
            if not files_appeared and now - start > first_file_grace:
                _sweep_msg(f"no shot files after {first_file_grace:.0f}s — giving "
                           f"up early, the sequence almost certainly failed to "
                           f"compile")
                break
            if now - last_progress > stall_grace:
                _sweep_msg(f"nothing new for {stall_grace:.0f}s — stopping early")
                break

            time.sleep(poll)

        # Inside the suppression: printed the moment the wait ends, before the
        # spinner is allowed to take the line back.
        _sweep_msg(f"{len(seen)}/{expected} shots analysed")

    rows = sorted(seen.values(), key=lambda r: str(r.get("shot_id")))
    if len(rows) >= expected:
        diagnosis = "ok"
    elif not files_appeared:
        diagnosis = "no_shot_files"
    elif not rows:
        diagnosis = "shots_ran_but_no_results"
    else:
        diagnosis = "partial"

    return {
        "collected": len(rows),
        "expected": expected,
        "shot_files_seen": len(files_appeared),
        "diagnosis": diagnosis,
        "complete": len(rows) >= expected,
        "rows": rows,
    }


# What each diagnosis means and what to do about it. The model needs the cause,
# not just the absence of data — told only "no results", it went and filled the
# gap with unrelated historical shots.
_DIAGNOSIS_HELP = {
    "no_shot_files": (
        "NO SHOT FILES WERE CREATED AT ALL. The shots were accepted by runmanager "
        "but never compiled. Read the traceback in runmanager's output pane and "
        "tell the operator what it says. Common causes, in order:\n"
        "  * `KeyError: 'labscriptlib'` in double_import_denier — the compiler "
        "subprocess is poisoned by an EARLIER failed compile, not by this "
        "sequence. Ask the operator to press 'Restart subprocess' in runmanager; "
        "the code is probably fine.\n"
        "  * UnicodeDecodeError with the 'gbk' codec — some imported file is not "
        "pure ASCII. labscript copies every imported script into the shot file "
        "with a bare open().read(), which uses the system codepage.\n"
        "  * A genuine syntax error or a global the sequence reads but which is "
        "not defined.\n"
        "There is no data from this run; do not substitute earlier shots for it."
    ),
    "shots_ran_but_no_results": (
        "SHOT FILES EXIST BUT CARRY NO ANALYSIS RESULTS. The shots ran; the "
        "analysis did not produce anything. Either lyse is not running, the "
        "analysis routine is not added in lyse, or it raised an exception (a "
        "common cause: reading a scope channel that was not captured). Check "
        "lyse's output, fix the analysis, and re-run. There is no data from this "
        "run; do not substitute earlier shots for it."
    ),
    "partial": (
        "ONLY SOME SHOTS WERE ANALYSED. Report the count honestly and treat the "
        "series as incomplete."
    ),
}


def summarise_sweep_results(res: Dict[str, Any], sweep_param: str) -> str:
    """One-line-per-shot rendering of collected results, for model and operator."""
    rows = res.get("rows") or []
    diagnosis = res.get("diagnosis", "partial")
    detail = (res.get("failure_detail") or "").strip()
    if not rows:
        body = (f"{res.get('shot_files_seen', 0)} shot file(s) appeared, "
                f"0 analysed.\n{_DIAGNOSIS_HELP.get(diagnosis, '')}")
        if detail:
            body += ("\n\nTHE ACTUAL ERRORS (read from the labscript suite's own "
                     "logs and from compiling the sequence). Quote these to the "
                     "operator instead of listing possible causes:\n" + detail)
        return body

    metric_keys = [k for k in rows[0]
                   if k not in ("shot_id", "shot_path", sweep_param)]
    lines = []
    for r in rows:
        vals = "  ".join(
            f"{k}={r[k]:.4g}" if isinstance(r.get(k), float) else f"{k}={r.get(k)}"
            for k in metric_keys
        )
        lines.append(f"  {r.get(sweep_param)}: {vals}")

    head = (f"{res['collected']}/{res['expected']} shots analysed"
            + ("" if res.get("complete") else " (INCOMPLETE)"))
    tail = "" if res.get("complete") else "\n" + _DIAGNOSIS_HELP.get(diagnosis, "")
    return head + "\n" + "\n".join(lines) + tail


def _real_failure_detail(executor, since_ts: float, diagnosis: str) -> str:
    """The actual errors behind a failed sweep, from the logs and a compile check.

    Guessing was the previous behaviour and it cost hours: the agent reported
    "lyse may not be running" while BLACS.log held the real cause verbatim.
    """
    parts = []

    # Device-side failures BLACS logged while the shots should have been running.
    try:
        from superradiant_assistant.interfaces.lab_logs import describe_failures
        logged = describe_failures(since_ts)
        if logged:
            parts.append(logged)
    except Exception as e:
        parts.append(f"(could not read suite logs: {e})")

    # Compile errors exist only in runmanager's GUI pane, so reproduce them.
    if diagnosis == "no_shot_files":
        rm = getattr(executor, "runmanager", None)
        if rm is not None and hasattr(rm, "compile_check"):
            try:
                res = rm.compile_check()
                if res.get("compiles") is False and res.get("error"):
                    parts.append("--- sequence does not compile ---\n"
                                 + str(res["error"]).rstrip())
                elif res.get("compiles") is True:
                    parts.append(
                        "--- the sequence compiles fine on its own ---\n"
                        "So the failure is runmanager's compiler subprocess, not this "
                        "sequence. Ask the operator to close and reopen runmanager."
                    )
            except Exception as e:
                parts.append(f"(compile check failed to run: {e})")

    return "\n".join(parts)


def _sweep_msg(text: str) -> None:
    """A `[sweep]` line, printed after closing any live progress line.

    The progress bar owns its line and is redrawn in place; anything printed
    without clearing it first lands on top of a half-drawn bar.
    """
    from superradiant_assistant import splash as S
    S.end_status_line()
    print(f"  {S.tag('sweep')} {S.c(S.TEXT)}{text}{S.RESET}", flush=True)


def _sweep_bar(done: int, total: int, files: int, elapsed: float) -> None:
    """The live sweep progress line: one bar, rewritten in place.

    Counts shots *analysed*, which is the thing being waited on. Shot files are
    reported alongside because the two diverge in the informative case: files
    appearing with no results means lyse is not running the routine.
    """
    from superradiant_assistant import splash as S
    cells = 24
    filled = int(cells * done / max(1, total))
    bar = (f"{S.c(S.ACCENT)}{'█' * filled}"
           f"{S.c(S.DIM)}{'░' * (cells - filled)}{S.RESET}")
    S.status_line(f"{S.tag('sweep')} {bar} "
                  f"{S.c(S.TEXT)}{done}/{total} analysed{S.RESET}"
                  f"{S.c(S.DIM)}   {files} shot file(s)   {elapsed:.0f}s"
                  f"{S.RESET}")


def _print_stage_banner(stage, max_iters: int, goal, live: bool) -> None:
    """The stage header, framed like everything else the operator reads.

    It used to be a block of `=` signs, which in a scrolling transcript is
    indistinguishable from the confirmation prompt and the plan.
    """
    from superradiant_assistant import splash as S

    rows = [
        f"{S.c(S.GREEN if live else S.AMBER)}"
        f"{'LIVE — shots queue on hardware' if live else 'OFFLINE REPLAY — nothing reaches hardware'}"
        f"{S.RESET}",
        f"{S.c(S.TEXT)}{stage.description[:160]}{S.RESET}",
        "",
    ]

    def row(label, value):
        return f"{S.c(S.DIM)}{label:<10}{S.c(S.TEXT)}{value}{S.RESET}"

    if stage.stage_kind == "sweep":
        rows.append(row("sweep", stage.sweep_param))
        rows.append(row("plot", stage.plot_ratio or "(none)"))
        if stage.sweep_range_mhz and stage.sweep_step_mhz:
            rows.append(row("range", f"{stage.sweep_range_mhz*1000:.2f} kHz   "
                                     f"step {stage.sweep_step_mhz*1000:.4f} kHz"))
    else:
        rows.append(row("target", f"{stage.target_metric} "
                                  f"{stage.threshold_op} {stage.threshold}"))
        rows.append(row("goal mode",
                        "on" if goal.goal_mode else "off (single shot)"))
    rows.append(row("points", str(max_iters)))

    print()
    print(S.framed(f"{stage.stage_kind} · {stage.name}", "\n".join(rows),
                   colour=S.ACCENT))
    print()


def _is_live(executor) -> bool:
    """True when this executor actually queues shots on hardware.

    The offline replay executor has no runmanager; the live one does. Every
    user-facing "queued" claim is gated on this.
    """
    return getattr(executor, "runmanager", None) is not None


def _load_stage_sequence(executor, stage) -> Optional[str]:
    """Load `stage.sequence_file` into runmanager. Returns an error string, or None.

    A no-op when the executor has no runmanager (offline replay), or when the
    stage names no sequence.
    """
    rm = getattr(executor, "runmanager", None)
    if rm is None or not stage.sequence_file:
        return None

    from superradiant_assistant.safety import (
        validate_sequence_file, sequence_files_match,
    )
    try:
        vetted = validate_sequence_file(stage.sequence_file)
        current = rm.get_labscript_file()
        if sequence_files_match(current, vetted):
            print(f"  {_S.tag('sequence')} already loaded: {Path(vetted).name}")
            return None
        rm.set_labscript_file(vetted)
        actual = rm.get_labscript_file()
        if not sequence_files_match(actual, vetted):
            return (f"runmanager still reports {actual!r} after being told to load "
                    f"{vetted!r} — refusing to queue shots against the wrong sequence")
        print(f"  {_S.tag('sequence')} loaded {Path(vetted).name} (was {Path(current).name if current else 'nothing'})")
        return None
    except Exception as e:
        return f"could not load {stage.sequence_file}: {type(e).__name__}: {e}"


def _plot_sweep(stage, history: List[ShotSignal]) -> None:
    """Generate a plot after a sweep stage completes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  {_S.tag('sweep')} matplotlib not available, skipping plot")
        return

    xs, ys = [], []
    for sig in history:
        # Use requested_globals first (the value the coder asked for — always a clean float).
        # Fall back to all_globals (historical shot's stored value, may be an expression/array).
        x = sig.requested_globals.get(stage.sweep_param)
        if x is None:
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
        print(f"  {_S.tag('sweep')} No valid data points to plot for sweep_param={stage.sweep_param}")
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
    print(f"  {_S.tag('sweep')} Plot saved → {out_path.resolve()}")


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
