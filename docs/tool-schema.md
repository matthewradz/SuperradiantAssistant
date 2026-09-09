# Tool Schema — labscript agent 的统一接口

这是把 labscript 包装成 Claude Agent 工具层的设计文档。每个字段都对应现有代码里的真实签名，没有凭空设计的东西。

**这份文档是后续所有步骤的输入**：subagent 的 `tools` 白名单、hook 的 `matcher`、goal 模式的判定条件、SKILL.md 的内容边界，全部从这里派生。

---

## 0. 三条设计原则

**原则 1：模型选工具，代码定参数。**
模型能决定"跑一次优化，目标 Neta_2 > 700"，但**不能**决定"这一炮 z_bias 设 -0.103"。后者由 `HillClimbOptimizer` 算，因为它必须可复现。对照图 (a)：Execute 是灰色的 non-AI 步骤，`run_loop` 属于灰色区。

**原则 2：枚举来自 `config.json`，不来自提示词。**
所有参数名、序列文件、分析脚本的合法取值都从 `config.json` 在启动时读出来生成 enum。模型不可能提出一个不在白名单里的参数名——因为 schema 里没有。

**原则 3：schema 约束 ≠ 安全边界。**
模型仍然可以往 `value` 里填一个超范围的数。**真正的强制在工具函数体内的校验**（对着 `config.json` 的 min/max）和 PreToolUse hook。schema 只是让非法输入变得困难，不是让它变得不可能。

---

## 1. 枚举来源（启动时从 config.json 生成）

### 1.1 可调参数（`min`/`max` 都非 null，共 6 个）

`run_optimization` 的搜索空间。来自 `loop.py:35-47` 的筛选逻辑：`min`/`max` 都存在才会变成 `ParamSpec`。

| 参数名 | min | max | 类型 | 物理含义 |
|---|---|---|---|---|
| `z_bias_field_loading` | -0.110 | -0.090 | float | 装载时 Z 向偏置场 |
| `y_bias_field_loading` | 0.0 | 0.5 | float | 装载时 Y 向偏置场 |
| `green_mot_frequency` | 47.5 | 49.5 | float | 绿光 MOT 交接频率 (MHz) |
| `lattice_loading_frequency` | 48.0 | 50.0 | float | 绿光晶格交接频率 (MHz) |
| `calibration_precession_time` | 0.1 | 20.0 | float | Ramsey 等待时间 (ms) |
| `rf_larmor_frequency` | 10000 | 11000 | float | Larmor 频率 (Hz) |

### 1.2 不可自动调节的参数（`min`/`max` 为 null，共 2 个）

**这两个不进 `run_optimization` 的搜索空间**，只能被 `run_sweep` 或 `set_runmanager_global` 显式指定。

| 参数名 | 类型 | 为什么没有范围 |
|---|---|---|
| `TD_loading` | bool | 布尔量，无区间概念 |
| `clock_pi_resonance_frequency_list` | float[] | 每炮一个值的列表，扫描专用 |

### 1.3 指标（来自 `signals.py` 的 `ShotSignal` 字段）

`Neta_1` `Neta_2` `Neta_3` `Neta_4` `Neta_5` `chi_square_2` `r_sq_2`

### 1.4 序列文件（`config.json` → `sequences`）

| 文件 | use_case |
|---|---|
| `sequences/assistant_sequences/calibrate_larmor_frequency_clean.py` | calibrate |
| `sequences/assistant_sequences/clock_transition_RABI.py` | resonance |

### 1.5 分析脚本（`config.json` → `analysis_scripts`）

| 文件 | use_case |
|---|---|
| `analysis/scripts/meta/improved_cost_clean.py` | resonance, optimize |
| `analysis/scripts/meta/calibrate_larmor_frequency_clean.py` | calibrate |

---

## 2. 七个工具

风险等级说明：
- 🟢 **只读** — 无副作用，不需要 hook
- 🟡 **计算** — 会写文件（图片），但不碰硬件
- 🔴 **改变外部状态** — 排队实验 / 改硬件参数 / 改源码，**必须走 PreToolUse 确认门 + 审计**

---

### 🟢 T1. `search_lab_knowledge`

检索实验室文档和序列代码。对应图 (b) 的 Search Agent（当前是关键词版）。

```python
{
  "query": str,                          # 自然语言查询
  "role": "planner" | "coder" | "answer", # 影响文档类型偏好
  "top_k": int,                          # 1-10，默认 6
}
```

| 项 | 内容 |
|---|---|
| 包装 | `search_for_role(knowledge, query, role, top_k)` + `augment_with_function_excerpts(docs, query)` |
| 校验 | `top_k` 钳位到 [1, 10]，防止一次塞爆上下文 |
| 返回 | 每篇文档的 title / kind / 摘要 + 命中函数的正文片段 |
| signal | `"found {n} docs: {titles}"` |
| 谁能用 | planner, coder, answer |

> **当前局限**：纯关键词打分（title/summary/tags 权重 ×3）。图 (b) 的 vectorize + cosine similarity + Remove 去重都没有。3 篇手写文档下够用，文档变多后再升级。

---

### 🟢 T2. `load_skill`

按需加载某个实验流程的 SOP。对应图 (a) 的 Knowledge → Documents，也就是 step06 学的技能加载。

```python
{
  "skill_name": <enum，启动时扫 skills/*/SKILL.md 生成>,
}
```

| 项 | 内容 |
|---|---|
| 包装 | 自写 `SkillLoader`（仿 step12 第53-95行），**不依赖 SDK 的技能发现机制** |
| 校验 | `skill_name` 必须在启动时扫出的名单里 |
| 返回 | `<skill name="...">` 包裹的 SKILL.md 正文 |
| signal | `"loaded skill: {name}"` |
| 谁能用 | planner, coder, answer |

---

### 🟢 T3. `read_shot_results`

读 HDF5 实验数据摘要。

```python
{
  "limit": int,                  # 1-200，默认 20，从最新往回读
  "metrics": [<指标 enum>],       # 要哪些指标，默认全部
  "include_globals": bool,       # 是否附带参数快照，默认 False
}
```

| 项 | 内容 |
|---|---|
| 包装 | `list_shots(root)` + `read_shot(path)` |
| **路径来源** | **固定为 `CONFIG.historical_data_root`，不接受模型传入路径** |
| 校验 | `limit` 钳位；`include_globals=True` 时只返回白名单内的参数名 |
| 返回 | 表格：shot_id + 各指标值 |
| signal | `"read {n} shots, best {metric}={value}"` |
| 谁能用 | planner, answer |

> ⚠️ **不要让模型传 folder 路径**。这是路径穿越的入口。数据根目录由 `config.json` 定，工具只在其内部工作。

---

### 🟡 T4. `analyze_results`

跑拟合分析并产出物理结论。

```python
{
  "analysis_type": "resonance" | "calibration",
  "sweep_param": <可调参数 enum | null>,    # resonance 必填
  "plot_ratio": str | null,                # 如 "Neta_5/Neta_4"，resonance 必填
}
```

| 项 | 内容 |
|---|---|
| 包装 | `analyze_resonance_sweep(data_root, sweep_param, plot_ratio)` / `analyze_larmor_calibration(data_root)` |
| 算法 | `scipy.optimize.curve_fit`（Levenberg–Marquardt）拟合洛伦兹凹陷 / Ramsey 条纹 |
| 校验 | `analysis_type=="resonance"` 时 `sweep_param` 和 `plot_ratio` 必须都非空；`plot_ratio` 必须形如 `A/B` 且 A、B 都是合法指标名 |
| 副作用 | 写一张 png（`plot_path`） |
| 返回 | 拟合参数 + 是否找到 + plot 路径 |
| signal | resonance: `"resonance at {f:.6f} MHz, linewidth {lw:.0f} Hz, min ratio {r:.3f}"`<br>calibration: `"correction {c:+.2f} Hz, fit freq {f:.1f} Hz"` |
| 谁能用 | planner, coder |

---

### 🟢/🔴 T5. `get_runmanager_globals`

读当前硬件参数。

```python
{
  "names": [<全部参数 enum>] | null,   # null = 全部
}
```

| 项 | 内容 |
|---|---|
| 包装 | `RunmanagerInterface().get_globals()` |
| 前置 | **需要 labscript GUI 在运行**（走 conda bridge 子进程）；离线模式下应返回明确错误而非崩溃 |
| 校验 | 返回值过滤，只暴露 `config.json` 白名单内的参数 |
| 风险 | 🟢 只读，但会起子进程。**不要在离线模式下暴露这个工具** |
| signal | `"read {n} globals"` |
| 谁能用 | planner, coder |

---

### 🔴 T6. `run_optimization`

跑一次参数优化循环。**这是 `run_loop` 的唯一入口。**

```python
{
  "target_metric": <指标 enum>,
  "threshold": float,
  "threshold_op": ">" | ">=" | "<" | "<=" | "==",
  "max_iterations": int,                    # 1-50
  "sequence_file": <序列 enum>,
  "signal_spec": str,                       # ← 图 (d)：Plan 指定要报告什么
  "reason": str,                            # ← 审计用，必填
}
```

| 项 | 内容 |
|---|---|
| 包装 | 构造 `Stage(task_type="optimize", stage_kind="optimize", ...)` + `Goal` → `run_loop(...)` |
| 参数由谁定 | **`HillClimbOptimizer`**，搜索空间是 §1.1 那 6 个参数。模型不参与选值 |
| 校验 | `max_iterations` ≤ min(50, 语料库剩余 shot 数)；`threshold` 必须是有限数；`sequence_file` 在白名单内 |
| 风险 | live 模式下会往 BLACS 排队真实实验 → **PreToolUse 确认门 + 审计** |
| 返回 | `run_loop` 的 summary + 按 `signal_spec` 格式化的 signal |
| signal | 例：`"target_met at iter 4 | best Neta_2=716.7 | 4 shots used"` |
| 谁能用 | coder |

**关键设计**：`signal_spec` 是图 (d) 的落地点。Plan 阶段决定"我要看到什么"，工具保证产出它，PostToolUse hook 把它写进 history，下一轮 Plan 读到它。这补上了现在 `signal_description` 生成后被丢弃的缺口。

---

### 🔴 T7. `run_sweep`

扫描一个参数并一次性排队到 BLACS。

```python
{
  "sweep_param": <全部参数 enum>,
  "mode": "centered" | "explicit",
  # mode == "centered"（频率扫描，以 runmanager 当前值为中心）
  "range_mhz": float | null,
  "step_mhz": float | null,
  # mode == "explicit"（显式起止，任意单位）
  "start": float | null,
  "end": float | null,
  "n_points": int,                    # 2-101
  "plot_ratio": str | null,
  "sequence_file": <序列 enum>,
  "signal_spec": str,
  "reason": str,
}
```

| 项 | 内容 |
|---|---|
| 包装 | 构造 `Stage(stage_kind="sweep", ...)` → `run_loop` 内部选用 `DeterministicSweepCoder` |
| 机制 | 生成 `np.linspace(start, end, n)` 表达式一次性写入 runmanager，**它自动排 n 炮**（`sweep_coder.py:80-81`）。循环只跑 1 轮就退出（docstring 第7行） |
| 副作用 | 会把 `delta_duration` 加 0.1 作为本组扫描的唯一标识（`sweep_coder.py:83-91`） |
| 校验 | `mode=="centered"` 需要 `range_mhz` 和 `step_mhz` 都非空且 `step_mhz > 0`；`mode=="explicit"` 需要 `start != end`；算出的点数必须 ≥ 2；`sweep_param` 若以 `_list` 结尾，基准参数（去掉 `_list`）必须能从 runmanager 读到 |
| 风险 | 🔴 **一次排 n 炮真实实验** → PreToolUse 确认门必须列出 start/end/n 让人确认 + 审计 |
| signal | `"queued {n} shots: {param} from {start} to {end}"` |
| 谁能用 | coder |

---

### 🔴🔴 T8. `set_runmanager_global`

**风险最高的一个**。往真实硬件参数里写值。

```python
{
  "name": <全部参数 enum>,
  "value": float | bool,
  "reason": str,          # 必填，写进审计日志
}
```

| 项 | 内容 |
|---|---|
| 包装 | `RunmanagerInterface().set_globals({name: value})` |
| **校验（代码层，不可绕过）** | 1. `name` 必须在 `config.json` 的 `globals` 里<br>2. 若该参数有 `min`/`max`，`value` 必须落在闭区间内<br>3. 若 `min`/`max` 为 null（`TD_loading` / `..._list`），**默认拒绝**，除非在单独的显式允许清单里<br>4. 类型必须匹配（`TD_loading` 只接受 bool） |
| PreToolUse hook | **总是要人工确认**，无论什么 permission_mode。确认提示必须显示：参数名、当前值、新值、允许区间、reason |
| PostToolUse hook | **必须审计**：时间戳、参数、旧值、新值、reason、session_id |
| 风险 | 🔴🔴 直接改变物理装置状态 |
| signal | `"{name}: {old} -> {new} Hz"` |
| 谁能用 | **只有主 agent**（planner/coder/answer 全都不给） |

> 这个工具替代的是现在 `run_optimization.py` 第338-343行那段——裸 `input()` 确认、无审计、异常只打印。那是全项目最该补 hook 的地方。

---

## 3. 校验总表

| 工具 | 代码层强制校验 | PreToolUse | 审计 |
|---|---|---|---|
| T1 `search_lab_knowledge` | top_k 钳位 | — | — |
| T2 `load_skill` | 名单内 | — | — |
| T3 `read_shot_results` | 路径固定、limit 钳位 | — | — |
| T4 `analyze_results` | 必填字段组合、plot_ratio 格式 | — | ✓ |
| T5 `get_runmanager_globals` | 返回值白名单过滤 | — | ✓ |
| T6 `run_optimization` | 迭代上限、阈值有限、序列白名单 | ✓ 确认 | ✓ |
| T7 `run_sweep` | 点数 ≥2、step>0、基准参数存在 | ✓ 确认（列出 n 炮） | ✓ |
| T8 `set_runmanager_global` | 名单 + 区间 + 类型 + null 拒绝 | ✓ **强制确认** | ✓ **必须** |

---

## 4. Signal 格式约定

统一形状：`"<结论> | <关键数字> | <代价>"`

```
target_met at iter 4 | best Neta_2=716.7 | 4 shots used
resonance at 100.014 MHz | linewidth 152 Hz | min ratio 0.243
correction +2.34 Hz | fit freq 102.3 Hz | 21 shots
queued 81 shots | clock_pi_..._list 99.0→101.0 | delta=1.6
no dip found | min ratio 0.981 | widen the sweep
```

规则：
1. **一行，不超过 120 字符** — 它会被反复注入 Plan 的上下文
2. **必须含数字** — "效果不错"不是 signal
3. **失败也要有 signal** — `"executor_error: No more historical shots"` 比空字符串有用
4. 由**工具函数**生成，不由模型生成 — 模型只通过 `signal_spec` 表达"我想看什么"

---

## 5. 工具 × Agent 权限矩阵

| 工具 | 主 agent | planner | coder | answer |
|---|---|---|---|---|
| T1 search_lab_knowledge | ✓ | ✓ | ✓ | ✓ |
| T2 load_skill | ✓ | ✓ | ✓ | ✓ |
| T3 read_shot_results | ✓ | ✓ | — | ✓ |
| T4 analyze_results | ✓ | ✓ | ✓ | — |
| T5 get_runmanager_globals | ✓ | ✓ | ✓ | — |
| T6 run_optimization | ✓ | — | ✓ | — |
| T7 run_sweep | ✓ | — | ✓ | — |
| T8 set_runmanager_global | ✓ | — | — | — |

`answer` 拿不到任何会动硬件的工具——这是**硬约束**（SDK 的 `AgentDefinition.tools` 是白名单，不是建议）。这直接实现了之前讨论的"先答 query 再跑 command"的安全性：回答问题的那条路径**物理上无法**触发实验。

---

## 6. 待确认的问题

1. **合规**：走 Anthropic 官方 API 还是留在 MIT Parley？（阻塞项）
2. **T5/T8 的离线行为**：labscript 没开时应该报错还是返回 mock？建议报错——静默 mock 会让人以为改成功了。
3. **`clock_pi_resonance_frequency_list` 的写入**：它是 `..._list` 后缀参数，T8 默认拒绝。扫描后要不要自动重置回标量？（`loop.py:222-231` 现在会做这件事，要保留吗）
4. **语料库耗尽**：`OfflineReplayExecutor` 不放回抽样，60 炮用完就报错。T6 的 `max_iterations` 上限要不要动态读剩余数量？
