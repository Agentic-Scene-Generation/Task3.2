# Qwen3.8 Slow Memory：021 训练审计与下一步

审计日期：2026-10-09。审计对象为服务器上已结束的 `qwen38_dpo_pilot_021`，训练产物记录的完成时间为 2026-09-29。以下结论来自实际文件、训练日志和内容哈希；ACP 网页状态仅作辅助信息。

**后续更新（2026-10-09）：** 022 和原生协议对照 023 均完成 22 对评估，验证结果为 2/7；底层差异为 eval-only 路径缺少 `Trainer.train()` 应用的 Liger 模型算子初始化。补齐后 025 已恢复原生验证 loss 0.7104305625 和 3/7 accuracy，与 021 一致。诊断执行完成、历史复现与模型发布质量分别报告，发布门禁保持不变。最新原因分析与可运行命令见 [022 诊断修复报告](sceneexpert_dpo_022_diagnostic_repair.md)。不要再次使用已有的 022 RUN_ID。

## 结论

021 完成了完整 DPO pilot，训练、验证、adapter 保存和轻量打包均成功。可以进入后续诊断和数据设计验证。目前验证指标没有支持偏好泛化改善的证据，adapter 暂不进入默认生产推理或论文效果主表。

| 核验项 | 实际结果 | 判断 |
| --- | --- | --- |
| 监督进程与完成检查 | supervisor `completed`，子进程退出 0；本次 execution_id 匹配；completion_check 无错误 | 真实完成 |
| 优化与恢复产物 | 8/8 optimizer steps，2 epochs；保留 checkpoint-6/7/8 的 optimizer、scheduler、RNG 与完整标记 | 完整 |
| 最终 adapter | 318,843,864 bytes；独立读取 512 个 tensor 均有限、256 个 LoRA B 非零 | 已保存、已更新 |
| 数据 | train 15 对、validation 7 对、test 0；四个数据文件哈希与训练快照一致 | 本轮训练没有数据漂移 |
| 平均训练 loss | 0.679068 | 优化过程运行正常 |
| 验证 DPO loss | 0.710431；同一 policy/reference 的 sigmoid 理论基线为 ln(2) ≈ 0.693147 | 未取得验证改善 |
| 验证 relative reward accuracy | 3/7 = 42.86% | 未通过当前 55% 离线门槛 |
| 验证平均 reward margin | -0.033524 | 平均偏好变化方向不理想 |
| 发布门禁 | `pilot_only`，`offline_validation_passed=false`，`promotable=false` | 符合 pilot 隔离设计 |
| 完成耗时 | 训练 66.24 min，最终验证 2.08 min，supervisor 全程 73.11 min | 020 的约 45 min 截断没有在本轮重现 |
| GPU/主机内存 | 峰值已分配 CUDA memory 54.75 GiB；进程 VmHWM 52.19 GiB；本轮记录无 cgroup OOM/oom_kill | 本轮完成没有 OOM 证据 |
| 轻量结果包 | 15,757,401 bytes，44 个归档文件哈希通过；训练最终标记存在 | 可以在本地复核，完整权重留在服务器 |

`eval_rewards/accuracies` 统计的是 chosen 相对 rejected 的 **policy/reference 对数概率变化**是否更大。它不是场景生成成功率，也不是直接的模型选择题正确率。7 对验证数据不足以证明模型整体退化；同样不能用于宣称方法有效。

审计摘要和实际 adapter/package SHA-256 保存在 `docs/reports/evidence/pilot021_audit_evidence.json`。原始 021 轻量包已下载至本地 `tmp/pilot021_audit/`；包内缺少独立 `tmp/acp_logs/021` 的提示不影响本次复核，因为完整 train.log、supervisor 和最终状态均保存在训练目录并已入包。

## 目前的底层限制

1. **监督目标与评判粒度不同。** 22 对样本监督第一轮 assistant 输出，偏好来自各自完整执行后的 raw 场景评分。逐条查看 44 个 completion，工具均属于 `observe_scene`、`designer_todo_manager` 或 `list_available_assets`；没有资产生成或摆放动作作为目标。标签包含后续随机采样与执行差异，第一轮待办文字对最终质量的信用分配较间接。真实执行证据仍然有效，但不能自动证明第一轮动作是差异的主要原因。
2. **样本量很小。** 15 个训练任务、7 个验证任务、8 个优化步骤。训练日志第二个 epoch 的 loss 下降，验证 margin 为负，与小数据拟合/监督噪声相容；目前无法仅凭汇总指标确定唯一原因。
3. **计算主要花在上下文。** 模板审计记录最长 prompt 为 19,190 tokens，最长 completion 103 tokens。完整上下文必须保留，completion-only 投影已经降低 vocabulary loss 的内存开销，但仍需计算完整 context。继续堆积同类首轮观察样本不一定能高效增强布局能力。
4. **效果证据尚缺场景对照。** 021 没有 held-out SceneEval 的 base/adapter 成对执行结果，也没有验证生产 GGUF 推理链路的 adapter 兼容性。现有 Harness/Fast Memory 的收益应独立报告。

没有发现本轮训练中断、权重未保存、数据哈希失配或模板前缀漂移。当前优先解决采样与监督粒度，而不是继续修改 Critic/Repair 或开展无目标的超参扫描。保留已验证的执行、配对与证据完整性要求。

## 本轮交付的诊断入口

`scripts/diagnose_sceneexpert_dpo.py` 与 `tmp/acp/acp_qwen38_dpo_diagnose.sh`：

- 只读已完成 pilot；先检查 supervisor、execution_id、最终产物和数据快照。
- 使用同一 safetensors 基座、QLoRA adapter、完整 context、closed-thinking 模板与 completion-only sigmoid objective。
- 分别为 train/validation 的每一对记录 policy/reference log probabilities、completion token 数、relative reward margin 与 DPO loss；保留 token 平均值供长度影响分析。
- 检查完整验证集聚合是否复现 021；partial smoke 单独标记，不替代完整验证。
- 记录每个训练目标的工具作用范围，不重写偏好标签、不调用场景工具、不进行 optimizer 更新。
- 单卡运行，保存逐对进度；失败时保留诊断日志并生成带哈希与遗漏记录的轻量结果包。

该入口采用项目固定的 TRL 1.13 评估接口。完整验证集的 loss 复现容差为 0.002，accuracy 必须一致；这项检查验证诊断方法，不用它筛选新训练数据。

本轮验证：本地 43 passed / 1 Windows signal-test skipped；CCI 同组 44 passed。真实数据 preflight 确认 22 对、44 个观察/规划目标及 adapter SHA-256。CCI 另运行 `qwen38_dpo_policy_smoke_022b`，固定选择第一条 train pair，诊断与打包均退出 0，9 个归档文件通过完整性校验；总耗时 **305.28 s（5.09 min）**，峰值 CUDA 已分配内存 **28.45 GiB**。该单对的 relative margin 为 +0.61793，DPO loss 0.43117，只说明加载后的诊断接口可运行，不能代替 22 对结果或验证集门禁。原始诊断记录保存在 `docs/reports/evidence/pilot021_policy_smoke_evidence.json`。

## 下一步：022 一次完整逐对诊断

先执行下列命令，申请 **1×H100 80GB，CPU memory 128 GiB，最长 2 h**。代码已同步后再提交，命令本身不包含 Git 操作。不重复启动 021 或使用已有 RUN_ID 覆盖结果。

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

CUDA_VISIBLE_DEVICES=0 \
RUN_ID=qwen38_dpo_policy_audit_022 \
TRAIN_RUN_ID=qwen38_dpo_pilot_021 \
PACKAGE_MAX_FILE_MIB=8 PACKAGE_MAX_TOTAL_MIB=32 \
bash tmp/acp/acp_qwen38_dpo_diagnose.sh
```

预期：22 对评分（15 train / 7 validation）、`policy_diagnostics.json.completed=true`、`full_dataset_evaluated=true`、source_validation_reproduction 通过，diagnostic/package exit 均为 0。验证集应复现约 0.71043 / 3-of-7；复现负面指标仍属于诊断成功。

**匹配的轻量打包命令**（主入口也会自动打包；仅在需要重新打包或补齐最后的 workflow 标记时执行，原始结果保留）：

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_dpo_policy_audit_022
PACKAGE_PATH="$PWD/tmp/results/slow_memory/${RUN_ID}_review_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
RUN_ID="$RUN_ID" PACKAGE_PATH="$PACKAGE_PATH" \
PYTHON_BIN="$PWD/.venv_dpo/bin/python" \
PACKAGE_MAX_FILE_MIB=8 PACKAGE_MAX_TOTAL_MIB=32 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
"$PWD/.venv_dpo/bin/python" scripts/package_sceneexpert_results.py --verify "$PACKAGE_PATH"
```

下载 `.tar.gz` 与对应 `.sha256`。归档保留 `outputs/slow_memory/...` 和 `tmp/acp_logs/...` 的项目相对组织结构；上限分别为单文件 8 MiB、总计 32 MiB，包含日志、配置、逐对分数、进度、完成检查与哈希清单。无须下载基座、adapter、optimizer 或重放资产。

| 环节 | 预计耗时 | 人工/无人值守 |
| --- | --- | --- |
| 同步核验、提交参数检查 | 5–10 min | 人工 |
| 单卡加载与 22 对评分 | 10–30 min；冷编译/共享存储拥塞可接近 60 min | 无人值守 |
| 汇总、复现检查、工具范围审计 | 5–15 min | 人工 |
| 打包与校验 | 1–3 min | 无人值守 |

依据：021 的 7 对最终评估为 124.73 s，22 对纯评分约 6.5 min；本轮含模型加载的单对 smoke 为 5.09 min。上述区间另外预留 tokenization、不同长度下的首次编译与 AFS IO，时间不是保证。总体人工约 10–25 min。

022 后的决策只做一次：若 train 明显改善而 validation 不改善，优先扩大并改进直接动作监督；若 train 也没有改善，再核查有效步数和训练目标，而非马上扩大训练。下一项采集研发采用 **共享观察后、资产/布局决策前分叉**，先验收 2–4 组相同上下文、独立执行与评分；该新采集入口目前尚未交付，不应把现有首步 ACP 参数改名后当成直接动作采集。

完成新采集入口验收后，再冻结 SceneEval task split、Fast Memory 快照和评价协议，批采、训练、开展 held-out base/adapter 场景对照。大批采集和场景对照的运行 ACP 与轻量打包脚本应随对应实现交付；沿用 021 adapter 作为探索性对照，不把失败的离线门禁改成论文正结果。
