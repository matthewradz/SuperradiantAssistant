# Tool Schema — the unified interface for the labscript agent

This is the design document for wrapping labscript as a Claude Agent tool layer. Every field corresponds to a real signature in the existing code. Nothing here was designed in the abstract.

**This document is the input to every step that follows**: the `tools` whitelist for each subagent, the `matcher` for each hook, the conditions that decide goal mode, and the content boundaries of SKILL.md all derive from it.

> **Read this first**: this is the **original design record**, not an inventory of the current state. The implementation has grown from the 8 tools listed here to 35, and the team has changed from `main agent / planner / coder / answer` to `lead / planner / coder / advisor`. The role names and the permission matrix in §5 have been reconciled against the implementation.
> **The parameters, metrics, and sequence files enumerated in §1 have not been.** Those came from the `config.json` of one particular apparatus (the ybclock 171-Yb cavity), which this checkout no longer ships. Read them only as an illustration of the principle "enumerations are generated from configuration." Do not read them as a description of your apparatus.
>
> The running system is always the authority: for parameters, look at your own `config.json`, and for tool permissions, type `/team`.

---

## 0. Three design principles

**Principle 1: the model picks tools, the code picks parameters.**
The model can decide "run an optimization, target Neta_2 > 700." It **cannot** decide "set z_bias to -0.103 on this shot." That is computed by `HillClimbOptimizer`, because it has to be reproducible. Compare figure (a): Execute is the grey non-AI step, and `run_loop` belongs to the grey region.

**Principle 2: enumerations come from `config.json`, not from the prompt.**
Every legal value for a parameter name, sequence file, or analysis script is read out of `config.json` at startup and turned into an enum. The model cannot propose a parameter name that is not on the whitelist, because the schema does not contain one.

**Principle 3: a schema constraint is not a security boundary.**
The model can still put an out-of-range number in `value`. **The real enforcement lives in the validation inside the tool function body** (checked against the min/max in `config.json`) and in the PreToolUse hook. The schema makes illegal input difficult, not impossible.

---

## 1. Where the enumerations come from (generated from config.json at startup)

### 1.1 Tunable parameters (both `min` and `max` non-null, 6 total)

The search space for `run_optimization`. Produced by the filter at `loop.py:35-47`: a parameter becomes a `ParamSpec` only if both `min` and `max` are present.

| Parameter | min | max | Type | Physical meaning |
|---|---|---|---|---|
| `z_bias_field_loading` | -0.110 | -0.090 | float | Z bias field during loading |
| `y_bias_field_loading` | 0.0 | 0.5 | float | Y bias field during loading |
| `green_mot_frequency` | 47.5 | 49.5 | float | Green MOT handoff frequency (MHz) |
| `lattice_loading_frequency` | 48.0 | 50.0 | float | Green lattice handoff frequency (MHz) |
| `calibration_precession_time` | 0.1 | 20.0 | float | Ramsey wait time (ms) |
| `rf_larmor_frequency` | 10000 | 11000 | float | Larmor frequency (Hz) |

### 1.2 Parameters that cannot be tuned automatically (`min`/`max` null, 2 total)

**These two do not enter the `run_optimization` search space.** They can only be set explicitly, by `run_sweep` or `set_runmanager_global`.

| Parameter | Type | Why it has no range |
|---|---|---|
| `TD_loading` | bool | Boolean, an interval means nothing |
| `clock_pi_resonance_frequency_list` | float[] | A list with one value per shot, used only for sweeps |

### 1.3 Metrics (the `ShotSignal` fields in `signals.py`)

`Neta_1` `Neta_2` `Neta_3` `Neta_4` `Neta_5` `chi_square_2` `r_sq_2`

### 1.4 Sequence files (`config.json` -> `sequences`)

| File | use_case |
|---|---|
| `sequences/assistant_sequences/calibrate_larmor_frequency_clean.py` | calibrate |
| `sequences/assistant_sequences/clock_transition_RABI.py` | resonance |

### 1.5 Analysis scripts (`config.json` -> `analysis_scripts`)

| File | use_case |
|---|---|
| `analysis/scripts/meta/improved_cost_clean.py` | resonance, optimize |
| `analysis/scripts/meta/calibrate_larmor_frequency_clean.py` | calibrate |

---

## 2. The eight tools

Risk levels:
- 🟢 **Read-only** — no side effects, no hook needed
- 🟡 **Computation** — writes files (images), but does not touch hardware
- 🔴 **Changes external state** — queues experiments, changes hardware parameters, or edits source. **Must pass a PreToolUse confirmation gate and be audited.**

---

### 🟢 T1. `search_lab_knowledge`

Retrieves lab documents and sequence code. This is the Search Agent in figure (b), currently a keyword implementation.

```python
{
  "query": str,                            # natural-language query
  "role": "planner" | "coder" | "advisor", # biases which document kinds are preferred
  "top_k": int,                            # 1-10, default 6
}
```

| Item | Detail |
|---|---|
| Wraps | `search_for_role(knowledge, query, role, top_k)` plus `augment_with_function_excerpts(docs, query)` |
| Validation | `top_k` is clamped to [1, 10], so one call cannot blow out the context |
| Returns | title / kind / summary for each document, plus body excerpts of matching functions |
| signal | `"found {n} docs: {titles}"` |
| Available to | lead, planner, coder, advisor |

> **Current limitation**: pure keyword scoring (title/summary/tags weighted x3). None of the vectorize, cosine similarity, or Remove-for-deduplication steps from figure (b) exist. That is sufficient for three handwritten documents. Revisit it once there are more.

---

### 🟢 T2. `load_skill`

Loads the SOP for one experimental procedure on demand. This is Knowledge -> Documents in figure (a), the skill loading taught in step06.

```python
{
  "skill_name": <enum, generated at startup by scanning skills/*/SKILL.md>,
}
```

| Item | Detail |
|---|---|
| Wraps | A hand-written `SkillLoader` (modeled on step12 lines 53-95). **It does not depend on the SDK's skill discovery mechanism.** |
| Validation | `skill_name` must appear in the list scanned at startup |
| Returns | The body of SKILL.md, wrapped in `<skill name="...">` |
| signal | `"loaded skill: {name}"` |
| Available to | lead, planner, coder, advisor |

---

### 🟢 T3. `read_shot_results`

Reads a summary of HDF5 experimental data.

```python
{
  "limit": int,              # 1-200, default 20, counting back from the most recent
  "metrics": [<metric enum>], # which metrics to return, default all
  "include_globals": bool,    # attach a parameter snapshot, default False
}
```

| Item | Detail |
|---|---|
| Wraps | `list_shots(root)` plus `read_shot(path)` |
| **Path source** | **Fixed to `CONFIG.historical_data_root`. A path from the model is not accepted.** |
| Validation | `limit` is clamped. With `include_globals=True`, only whitelisted parameter names are returned |
| Returns | A table of shot_id plus each metric value |
| signal | `"read {n} shots, best {metric}={value}"` |
| Available to | lead, planner, coder, advisor |

> ⚠️ **Never let the model pass a folder path.** That is the entry point for path traversal. The data root is set by `config.json`, and the tool works only inside it.

---

### 🟡 T4. `analyze_results`

Runs a fit and produces a physical conclusion.

```python
{
  "analysis_type": "resonance" | "calibration",
  "sweep_param": <tunable parameter enum | null>,  # required for resonance
  "plot_ratio": str | null,                        # e.g. "Neta_5/Neta_4", required for resonance
}
```

| Item | Detail |
|---|---|
| Wraps | `analyze_resonance_sweep(data_root, sweep_param, plot_ratio)` / `analyze_larmor_calibration(data_root)` |
| Algorithm | `scipy.optimize.curve_fit` (Levenberg-Marquardt) fitting a Lorentzian dip or Ramsey fringes |
| Validation | When `analysis_type=="resonance"`, both `sweep_param` and `plot_ratio` must be non-null. `plot_ratio` must have the form `A/B` with A and B both legal metric names |
| Side effect | Writes one png (`plot_path`) |
| Returns | Fit parameters, whether a feature was found, and the plot path |
| signal | resonance: `"resonance at {f:.6f} MHz, linewidth {lw:.0f} Hz, min ratio {r:.3f}"`<br>calibration: `"correction {c:+.2f} Hz, fit freq {f:.1f} Hz"` |
| Available to | lead, planner, coder, advisor |

---

### 🟢/🔴 T5. `get_runmanager_globals`

Reads the current hardware parameters.

```python
{
  "names": [<all-parameters enum>] | null,  # null = all
}
```

| Item | Detail |
|---|---|
| Wraps | `RunmanagerInterface().get_globals()` |
| Precondition | **The labscript GUI must be running** (this goes through a conda bridge subprocess). In offline mode it should return a clear error rather than crash |
| Validation | The return value is filtered, exposing only parameters on the `config.json` whitelist |
| Risk | 🟢 Read-only, but it spawns a subprocess. **Do not expose this tool in offline mode.** |
| signal | `"read {n} globals"` |
| Available to | lead, planner, coder, advisor |

---

### 🔴 T6. `run_optimization`

Runs one parameter optimization loop. **This is the only entry point to `run_loop`.**

```python
{
  "target_metric": <metric enum>,
  "threshold": float,
  "threshold_op": ">" | ">=" | "<" | "<=" | "==",
  "max_iterations": int,                    # 1-50
  "sequence_file": <sequence enum>,
  "signal_spec": str,                       # figure (d): Plan states what to report
  "reason": str,                            # required, for the audit log
}
```

| Item | Detail |
|---|---|
| Wraps | Builds a `Stage(task_type="optimize", stage_kind="optimize", ...)` plus a `Goal`, then calls `run_loop(...)` |
| Who picks the values | **`HillClimbOptimizer`.** The search space is the 6 parameters in §1.1. The model takes no part in choosing values |
| Validation | `max_iterations` <= min(50, shots remaining in the corpus). `threshold` must be finite. `sequence_file` must be on the whitelist |
| Risk | In live mode this queues real experiments to BLACS, so it needs a **PreToolUse confirmation gate and an audit record** |
| Returns | The `run_loop` summary plus a signal formatted per `signal_spec` |
| signal | e.g. `"target_met at iter 4 | best Neta_2=716.7 | 4 shots used"` |
| Available to | lead, coder |

**Key design point**: `signal_spec` is where figure (d) becomes real. The Plan stage decides what it wants to see, the tool guarantees it is produced, the PostToolUse hook writes it into history, and the next Plan reads it back. This closes the gap where `signal_description` is currently generated and then discarded.

---

### 🔴 T7. `run_sweep`

Sweeps one parameter, queueing the whole set to BLACS at once.

```python
{
  "sweep_param": <all-parameters enum>,
  "mode": "centered" | "explicit",
  # mode == "centered" (frequency sweep, centered on runmanager's current value)
  "range_mhz": float | null,
  "step_mhz": float | null,
  # mode == "explicit" (explicit endpoints, any units)
  "start": float | null,
  "end": float | null,
  "n_points": int,                    # 2-101
  "plot_ratio": str | null,
  "sequence_file": <sequence enum>,
  "signal_spec": str,
  "reason": str,
}
```

| Item | Detail |
|---|---|
| Wraps | Builds a `Stage(stage_kind="sweep", ...)`, and `run_loop` then selects `DeterministicSweepCoder` internally |
| Mechanism | Writes an `np.linspace(start, end, n)` expression into runmanager in one go, and **runmanager queues n shots by itself** (`sweep_coder.py:80-81`). The loop runs exactly one iteration and exits (docstring line 7) |
| Side effect | Adds 0.1 to `delta_duration` as the unique identifier for this sweep group (`sweep_coder.py:83-91`) |
| Validation | `mode=="centered"` requires `range_mhz` and `step_mhz` both non-null with `step_mhz > 0`. `mode=="explicit"` requires `start != end`. The resulting point count must be >= 2. If `sweep_param` ends in `_list`, the base parameter (the name without `_list`) must be readable from runmanager |
| Risk | 🔴 **Queues n real experiments in one action.** The PreToolUse gate must show start, end, and n for confirmation, and the action must be audited |
| signal | `"queued {n} shots: {param} from {start} to {end}"` |
| Available to | lead, coder |

---

### 🔴🔴 T8. `set_runmanager_global`

**The highest-risk tool.** It writes a value into a real hardware parameter.

```python
{
  "name": <all-parameters enum>,
  "value": float | bool,
  "reason": str,          # required, written to the audit log
}
```

| Item | Detail |
|---|---|
| Wraps | `RunmanagerInterface().set_globals({name: value})` |
| **Validation (in code, not bypassable)** | 1. `name` must appear in the `globals` block of `config.json`<br>2. If the parameter has a `min`/`max`, `value` must fall inside the closed interval<br>3. If `min`/`max` are null (`TD_loading`, any `..._list`), the write is **refused by default** unless the parameter is on a separate explicit allow list<br>4. The type must match (`TD_loading` accepts only a bool) |
| PreToolUse hook | **Always requires human confirmation**, in every permission_mode. The prompt must show the parameter name, the current value, the new value, the permitted interval, and the reason |
| PostToolUse hook | **Must be audited**: timestamp, parameter, old value, new value, reason, session_id |
| Risk | 🔴🔴 Directly changes the state of the physical apparatus |
| signal | `"{name}: {old} -> {new} Hz"` |
| Available to | **lead only** (not planner, coder, or advisor) |

> This tool replaces the block at `run_optimization.py` lines 338-343: a bare `input()` confirmation, no audit, and exceptions only printed. That is the single place in the project most in need of a hook.

---

## 3. Validation summary

| Tool | Enforced in code | PreToolUse | Audit |
|---|---|---|---|
| T1 `search_lab_knowledge` | top_k clamped | — | — |
| T2 `load_skill` | name on the list | — | — |
| T3 `read_shot_results` | path fixed, limit clamped | — | — |
| T4 `analyze_results` | required field combinations, plot_ratio format | — | ✓ |
| T5 `get_runmanager_globals` | return value filtered to whitelist | — | ✓ |
| T6 `run_optimization` | iteration cap, threshold finite, sequence whitelisted | ✓ confirm | ✓ |
| T7 `run_sweep` | n >= 2, step > 0, base parameter exists | ✓ confirm (lists n shots) | ✓ |
| T8 `set_runmanager_global` | whitelist + interval + type + refuse-on-null | ✓ **mandatory confirm** | ✓ **required** |

---

## 4. Signal format convention

One shape throughout: `"<conclusion> | <key numbers> | <cost>"`

```
target_met at iter 4 | best Neta_2=716.7 | 4 shots used
resonance at 100.014 MHz | linewidth 152 Hz | min ratio 0.243
correction +2.34 Hz | fit freq 102.3 Hz | 21 shots
queued 81 shots | clock_pi_..._list 99.0->101.0 | delta=1.6
no dip found | min ratio 0.981 | widen the sweep
```

Rules:
1. **One line, at most 120 characters.** It gets injected into the Plan context over and over
2. **It must contain numbers.** "Looks good" is not a signal
3. **Failures need a signal too.** `"executor_error: No more historical shots"` is more useful than an empty string
4. It is produced by the **tool function**, not by the model. The model expresses what it wants to see only through `signal_spec`

---

## 5. Tool x agent permission matrix

This section records the original 8 tools. The implementation has since grown to **35**, allocated as lead 24 / coder 20 / advisor 17 / planner 12. **The running registry is the authority**: type `/team` in a session, or call `build_registry(...).names_for("<role>")`. The table below is a record of design intent only.

| Tool | lead | planner | coder | advisor |
|---|---|---|---|---|
| T1 search_lab_knowledge | ✓ | ✓ | ✓ | ✓ |
| T2 load_skill | ✓ | ✓ | ✓ | ✓ |
| T3 read_shot_results | ✓ | ✓ | ✓ | ✓ |
| T4 analyze_results | ✓ | ✓ | ✓ | ✓ |
| T5 get_runmanager_globals | ✓ | ✓ | ✓ | ✓ |
| T6 run_optimization | ✓ | — | ✓ | — |
| T7 run_sweep | ✓ | — | ✓ | — |
| T8 set_runmanager_global | ✓ | — | — | — |

**`advisor` gets no tool that can move hardware and no tool that writes persistent state.** This is a **hard constraint, not a recommendation**: `ToolSpec.allowed_agents` is a whitelist, and `registry.dispatch()` refuses against it before the handler is ever called, so even a model that has been talked into calling one **physically cannot** reach the experiment along that path. This is what makes the "answer the query before running the command" behavior discussed earlier safe.

Concretely, `advisor` has none of:
`run_sweep`, `run_optimization`, `engage_shot`, `set_runmanager_global`, `load_sequence`, `set_lyse_routines`, `remember`, or any of the creative-mode write tools (`write_shot`, `write_analysis`, `write_report`, `propose_global`).
`tests/test_lab_tools.py::TestPermissionMatrix` guards each one.

The three read-only tools (T3/T4/T5) were later opened to every role: diagnosis needs to be able to read data, and they change nothing. The original table marked them `—` for some roles, which does not match the implementation.

---

## 6. Open questions

1. **Compliance**: go through the official Anthropic API, or stay on MIT Parley? (blocking)
2. **Offline behavior of T5/T8**: when labscript is not running, should these raise or return a mock? Raising is recommended, since a silent mock makes people believe a write succeeded.
3. **Writing `clock_pi_resonance_frequency_list`**: it is a `..._list` parameter, so T8 refuses it by default. Should it be reset to a scalar automatically after a sweep? (`loop.py:222-231` does this today. Keep it?)
4. **Corpus exhaustion**: `OfflineReplayExecutor` samples without replacement and raises once its 60 shots are used. Should the `max_iterations` cap in T6 read the remaining count dynamically?
