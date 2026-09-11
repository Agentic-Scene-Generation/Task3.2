# Fast Memory 评价与验收（v9 / v7）

本实现观察原生流程，不改动 main 的 Designer/Critic 算法、评分、阶段顺序、失败策略和物理处理。**记忆送达、出现相关动作、目标达到、配对增益是四件不同的事。** 不应把任何单项当作因果证明。

## 1. 每次运行的独立结果

现有生成/reuse launcher 的 metrics 收集保持不变，新增以下输出：

| 相对本次 OUTPUT_ROOT 的路径 | 内容 |
|---|---|
| `metrics/run_metrics.json`、`run_metrics.md` | 全部 assigned cases 的运行、质量、记忆与成本摘要 |
| `assigned_cases.csv` | 启动 worker 前固定完整任务清单；即使某个 batch 从未启动，也保留为未完成 |
| `metrics/scene_metrics.csv`、`scene_metrics.jsonl` | 每个 case 的指标，失败/未完成不删除 |
| `metrics/memory_usage.json` | 逐来源、逐阶段的检索/接受/实际 payload/相关动作/目标检查证据 |
| `metrics/attempt_costs.json` | 最后 attempt 和所有 `failed_attempts` 的独立成本、失败分类、缺失项 |
| `<PAIR_ID>_metrics/paired_metrics.json`、`paired_metrics.md` | OFF/ON 配对结果与严格门禁 |

每个 scene 原有 `scene_expert/memory_activity.json` 增加 `decision_usage`、`post_scene_state`；同一阶段再次执行时，旧条目保存在 `stage_attempt_history`，不被最后一次覆盖。

### 如何理解 Memory 证据

- `accepted`：通过 Planner 与确定性适配检查；不是已送达。
- `payload_observed`：**完整 accepted item** 出现在实际 Designer 调用审计 payload 的 prompt 中；不是仅在 context metadata 或模型输出中出现。
- `delivered`：上述请求还存在无异常的调用完成记录。失败调用保留 payload/错误，但不能冒充模型已完成使用；审计缺失不等于确定没送达。
- `action_observed=true`：同一调用存在成功的相关变更工具结果，call ID 与实际对象 ID 对上。创建新物体还要在真实 post-stage inventory 中确认该新 ID/角色。它只证明相关操作发生，**不证明整套建议被完整执行**。
- `target_verified`：复用 main 的逐 constraint 证据，核对当前约束内容 hash；整阶段 pass、模型自述或一般文字不能替代它。`false` 是已验证目标失败，`null` 是未知。
- `causal_benefit` 始终不由单条记录自动填写。工具参数/result hash、payload 路径/hash 与 native 检查证据留在记录中，便于复核。

Skill utility 只有实际送达 + 相关动作 + 精确目标结果时才记录对应观察；基础设施失败不自动处罚全部召回记忆。冻结 bank 下仍不写 Writer、utility、晋升或 public events。旧库不迁移、不重建。

`first_critic_pass_round=null` 是有意的：当前持久化证据不足以恢复所有原生 Critic 回合的逐目标检查序列。本轮不侵入 main 补造该序列，也不从最终成功推断“首轮通过”。可读完整请求/工具记录与阶段目标结果；如果后续需要真正的首次通过率，应先补齐原生逐回合证据再统计。

### 如何理解成本

- `all_attempt_time_sec`：从首个 worker 写下 running 状态到最后一个 attempt 终态的 case 时间跨度，**包括失败重试和 retry gaps**，但不含首个 worker 启动前的排队。
- `attempt_service_time_sec`：各 attempt 已测时间之和，与时间跨度分开。
- `trace_time_sec` / `final_attempt_trace_time_sec`：最后 attempt 的 wrapper trace 时间，只作辅助，不能冒充总成本。
- attempt 序号缺失/重复、时间不全或重叠时，完整成本为 `null`，同时提供 `observed_attempt_time_lower_bound_sec` 与原因。
- LLM role、渲染/工具等已有计时分别列出。Planner 可能嵌套 Designer/Critic，不能把这些耗时相加当 wall time；嵌套 token 也不冒充去重后的全调用 token。缺少 usage/排队/TTFT 时明确未知。
- OFF 原有 GlobalPlanner 成本不是 Fast Memory 新增成本。ON/OFF 的全部 case 成本差才是评价输入；本轮增加的观测与审计开销也自然计入。

## 2. 最简 reuse ACP

使用已同步的新代码和现有 `tmp/acp/` 文件。不开启预算或额外修复，不新增 Git 检查。

沿用 ACP 已有的模型、并发、共享检查点与 Memory 路径设置，不需要另填 GPU 型号或模型服务版本标签。新终端中直接运行：

```bash
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

PAIR_ID=full_memory_pair_sceneeval100_hard_qwen38_v5 ARM_ORDER=off_on \
bash tmp/acp/acp_qwen38_full_memory_pair_off_reuse.sh

# OFF 结束后；如果换了 ACP 作业，仍使用完全相同的 PAIR_ID。
PAIR_ID=full_memory_pair_sceneeval100_hard_qwen38_v5 ARM_ORDER=off_on \
bash tmp/acp/acp_qwen38_full_memory_pair_on_reuse.sh
```

先核对现有 pair `.env` 中的 SOURCE_RUN_ID、SOURCE_MEMORY_DIR、CASE_SET、SCENE_SELECTION。OFF 建库快照期间停止源库 Writer。不要复用旧 PAIR_ID 或把旧 OFF 与新 ON 拼起来。以上两条是两个独立 shell 命令；不要用 `&&` 将 ON 是否执行绑定到 OFF 的退出码。OFF 任务失败但已经正常生成 metrics 时，仍应保留该 arm 结果。

若 ON 以失败退出，旧 tmp wrapper 可能尚未运行配对汇总；可单独执行：

```bash
PAIR_ID=full_memory_pair_sceneeval100_hard_qwen38_v5
python -m scenesmith.scene_expert.paired_metrics \
  --baseline "outputs/critic_probe/${PAIR_ID}_memory_off" \
  --treatment "outputs/critic_probe/${PAIR_ID}_memory_on" \
  --output-dir "outputs/critic_probe/${PAIR_ID}_metrics"
```

也可加载相同 pair 配置后，使用受版本管理的 `scripts/run_sceneexpert_full_memory_pair.sh`，`PAIR_ACTION=both` 串行完成两臂并汇总：任一 arm 失败不会跳过另一个 arm；最终仍返回非零退出码，不吞掉失败。

第二轮使用 **新 PAIR_ID + `ARM_ORDER=on_off` + 同一冻结 bank + 同一 COMPILED_INPUTS_DIR + 同一 shared base/场景清单**。因为旧 ON wrapper 要求 OFF metrics 存在，反向顺序应使用中央 runner 的 `PAIR_ACTION=both`；预先指定已有 FROZEN_MEMORY_ROOT/DIR。新 PAIR_METRICS_DIR、PAIR_CONTRACT_ROOT 也要对应新 ID，不能遗留上一轮的派生目录。两臂不要争用同一 GPU/服务并行运行来声称加速。

## 3. 配对门禁与判断

`comparison_ready` 会检查 case/prompt、逐 case compiler/intent/shared-base、真实首个 Designer checkpoint、source bundle、control signature、Python/依赖版本、冻结 bank 和开关隔离。重复 case、旧版“准备好提示”证据不能通过门禁。

还要求 `assignment_inventory_complete=true`。旧运行只有已启动 batch 的清单时，不能证明“全部分配任务均已纳入”；仍可查看描述性结果，但不允许用成功子集声称全任务收益。

`speed_comparison_ready` 要求自动记录的模型/配置/源码/软件身份与冻结输入一致、全部 attempts 成本完整；**不再要求** `SCENEEXPERT_EVAL_RESOURCE_CLASS` 或 `SCENEEXPERT_EVAL_SERVICE_DEPLOYMENT`。新 trace 不读取这两个环境变量，也不引入 GPU 探测、额外模型请求或运行前检查。

硬件资源、后端部署与是否存在竞争负载由运行侧维持一致。报告明确标注 `hardware_equivalence_verified=false`，这是“未做硬件认证”，不是“硬件不一致”，不会阻止速度统计。旧结果的手填标签若存在明确矛盾，仍会提示并阻止速度结论；缺少标签记为 `runtime_resource_match=null`，不再视为不匹配。配置相同不代表资源条件已经自动核实，速度增益应在同卡、同模型服务、同资源分配、无其他竞争任务的前提下解释。

看 `all_assigned_mean_time_delta_sec`、`all_assigned_total_time_delta_sec` 和完成/失败转移；共同完成子集上的 Critic 分差只能作为辅助。失败分类仍沿用 main，不把基础设施异常强行计为 Memory 负例。不更改 Critic 分数、不删除不利 case。

`comparison_ready=true` 仅意味着可比较，**不是“Memory 已证明有效”**。完成两轮平衡重复，按事先固定的质量/成本目标判断：收益足够则保留；只在某类问题有效则按类开启；完整且公平的实验仍无实用收益时，允许默认关闭跨任务检索，不阻塞其他工作。

## 4. 固定检查点诊断与预登记入口

`python -m scenesmith.scene_expert.checkpoint_evaluation --help` 提供 `register`、`verify`、`assess`。它负责固定输入与验收，不另建工具执行循环，不自动调用付费模型，不启动 DPO。

先从已有、与训练记忆来源独立的 checkpoint 中选 6–10 个问题。JSON spec 每个 `problems` 条目需要：

- `case_id`、稳定 `task_id`（本项目 prompt SHA ID）、`stage`；
- `inputs`：`checkpoint`、`task_spec`、`intent_contract`、`model_config`、`tool_config` 的实际文件路径。model_config 使用记录权重版本/服务设置的小型配置文件，不要让每个 scene 重读巨大权重文件；
- `expected_identity`：已有冻结输入和 checkpoint 的 `shared_base_fingerprint`、`task_spec_fingerprint`、`intent_contract_fingerprint`、`decision_state_fingerprint`；
- 可选 `target_constraint_ids`、`reviewed_memory_sources`。后者仅登记经过人工核查的独立训练任务来源，不自动生成或注入“人工正确答案”；
- 顶层 `acceptance` 显式填写 `max_mean_quality_loss` 与 `min_cost_reduction_fraction`，范围 [0,1]。没有替用户设定必然成功的默认阈值。

```bash
python -m scenesmith.scene_expert.checkpoint_evaluation register \
  --spec tmp/memory_checkpoint_spec.json --output tmp/memory_checkpoint_plan.json
python -m scenesmith.scene_expert.checkpoint_evaluation verify \
  --manifest tmp/memory_checkpoint_plan.json
export SCENEEXPERT_EVAL_CHECKPOINT_PLAN="$PWD/tmp/memory_checkpoint_plan.json"
```

注册只创建新文件，不覆盖已有登记。接着通过原生 resume 路径运行既定场景；单阶段诊断沿用 `experiment.pipeline.start_stage=<stage>`、`experiment.pipeline.stop_stage=<stage>`、`experiment.pipeline.resume_from_path=<原生实验目录>`，并保持 Writer 关闭和输入一致。完整 Full reuse 则仍通过原有 ACP。未提供可运行的真实检查点时，入口不会捏造问题或自动运行空实验。

两轮配对之后：

```bash
python -m scenesmith.scene_expert.checkpoint_evaluation assess \
  --manifest tmp/memory_checkpoint_plan.json \
  --comparison outputs/critic_probe/PAIR_A_metrics/paired_metrics.json \
  --comparison outputs/critic_probe/PAIR_B_metrics/paired_metrics.json \
  --output tmp/memory_checkpoint_acceptance.json
```

验收检查登记内容/文件 hash、实际 case 和输入指纹、运行开始前的登记时间、不同 run ID、两种平衡 arm 顺序及全部 case 的成本/质量。缺少证据则 `inconclusive`；达到工程阈值也不等于总体显著性或持续学习证明。本地登记时间不是第三方时间戳认证。

人工审定来源的登记用于检查覆盖率、适用性与泄漏风险；本轮**没有**实现第三臂自动改写生成提示的 launcher。若做人工上下文诊断，应单独审定并保持同一 checkpoint/任务/工具，不把不同上下文 ON/OFF 输出直接当 DPO chosen/rejected。

旧运行可重新生成描述性 metrics，但缺失的实际请求、检查点或时间不能追溯补造。当前代码需在 Drake/Blender/Qwen 服务器完成真实冻结对照，不能仅凭 CPU 回归通过就宣布 Memory 有正向收益。
