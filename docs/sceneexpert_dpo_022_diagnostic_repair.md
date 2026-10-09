# 022 诊断退出原因与评估协议修复

审计日期：2026-10-09。原始训练为 `qwen38_dpo_pilot_021`，原始诊断为 `qwen38_dpo_policy_audit_022`。本报告依据 CCI 上实际日志、逐对评分与退出标记。

## 022 实际执行情况

022 完成全部 22 对评估（15 train、7 validation），耗时 718.87 秒，约 11.98 分钟；没有启动异常、Traceback 或 OOM 记录。轻量包包含 9 个文件，14,398 bytes，校验成功。`diagnostic_exit_code=2`，`package_exit_code=0`。

退出发生在最后的历史验证指标复现检查：

| 指标 | 021 最终验证 | 022 重新加载后诊断 |
| --- | ---: | ---: |
| 验证 loss | 0.71043056 | 0.71188280 |
| 验证 relative reward accuracy | 3/7 | 2/7 |
| 验证平均 reward margin | -0.03352398 | -0.03624854 |
| 训练集 relative reward accuracy | 未保存完整逐对终态 | 13/15 |

loss 差 0.00145223 在原容差 0.002 内，但旧入口要求离散 accuracy 完全相同，一条符号差异即使全程成功也会返回 2。因此，ACP 的失败状态不能直接解释为启动失败。再次使用已存在的 RUN_ID 会被不可覆盖保护立即拒绝；没有额外启动日志时，不能据此推断用户是否发生了第二次启动。

## 本轮修复

旧诊断直接遍历 dataloader 调用 `prediction_step`，绕过原生 `Trainer.evaluate()` 的模型准备与评估生命周期，还改动了 gradient checkpointing/offload 配置、将两种数据拆分拼接成一个 dataset。单对 smoke 无法验证完整验证集复现。023 的完整对照已经证明：统一原生协议后，全部 22 对得分与 022 一致；这项实现改进不是历史 accuracy 差异的原因。

本轮改为重建原训练配置与 train/validation 数据集，经 `DPOTrainer.evaluate()` 执行完整评估。逐对观察器只读取每步新产生的指标，不清空或修改 Trainer 的累计指标；每个拆分的逐对聚合必须与原生汇总一致。模板、完整 context、adapter、禁用 adapter 的 reference、completion-only sigmoid objective 均沿用训练协议。

新增 `evaluation_protocol.json`、`native_evaluation_metrics.json`，保存原生评估入口、参数 dtype、源代码内容指纹与原生指标。最终摘要分别记录执行是否完整、完整数据是否覆盖、历史数值是否复现以及明确失败原因。重用 RUN_ID 与诊断非零退出均输出具体路径与原因。

**底层数值协议差异：模型算子补丁缺失。** 在服务器实际安装的 Transformers 5.15 中，`Trainer.train()` 会调用 `transformers.integrations.liger.apply_liger_kernel(model, args.liger_kernel_config)`，而单独的 `evaluate()` 不调用这个初始化。TRL 的 fused DPO loss 与模型内部 RMSNorm/MLP 的 Liger 实现是两项独立设置；仅有 `use_liger_kernel=true` 和 fused loss 不足以复现经过训练的模型计算路径。源基座的架构配置为 `Qwen3_5ForConditionalGeneration`，支持该模型补丁。

诊断重新加载 adapter 后没有重放 train 入口的模型补丁，导致与 021 最终评估使用不同算子；near-zero reward margin 使小数值差异能改变离散 accuracy。新增 `prepare_training_kernels_for_evaluation()` 使用依赖自身的同一 API 和配置，记录补丁前后的 forward 模块清单，并在评分前初始化。025 的原生验证已精确恢复 021 的 loss 与 accuracy，确认该原因。

**执行状态设计也需要修正。** 旧工作流把“历史统计完全一致”作为“成功完成测量”的必要条件。当前只保存了 021 的聚合统计，无法定位历史一票差异对应的具体样本。负面结果或历史复现警告应成为可分析的诊断产出，不能让运行人员反复重启相同实验以获得一个成功状态。

因此，执行成功要求真实输入与 adapter 哈希一致、全部选定样本评分完毕、指标有限、sigmoid loss 正确、逐对聚合与原生评估一致。历史复现保持原容差和 `passed=false`，作为 `completed_with_warnings` 明确保存；它不再将完整诊断变成执行失败。数据/权重变化、非有限数值、目标不一致或缺失样本仍会失败。没有放宽训练发布质量门禁，`promotable=false` 保持不变。

后续单卡、batch=1 的 furniture pilot 训练，在同一次最终原生评估中保存 `validation_pair_scores.jsonl`，记录逐对原始得分，并将文件 SHA-256 绑定到训练 manifest；重新加载审计拒绝被改动的记录。这使后续差异能定位到具体样本，无须重新训练来找证据。

没有修改 Critic、Designer、Repair 或已有训练权重，也没有通过替换 preference 标签改善统计。

原生协议改进提交 `9a34df7` 已推送并以离线 bundle 同步 CCI；本地 45 passed、1 Windows signal-test skipped，CCI 46 passed。增加完成状态分离、训练逐对记录与防篡改回归后，本地定向测试 46 passed、1 Windows signal-test skipped。用户的 PPT 与演讲稿本地改动保留。

## CCI 验证

`qwen38_dpo_policy_audit_023` 已在单卡 H100 80GB 完成：22/22，738.33 秒，12.31 分钟，峰值已分配 CUDA memory 30.43 GiB。原生与逐对聚合一致，验证 loss 0.7118828 / accuracy 2/7，训练 loss 0.6375559 / accuracy 13/15，均与 022 一致。旧历史复现门禁仍使这个中间验证返回 2，打包成功。保留该结果作为 A/B 证据。

`qwen38_dpo_policy_audit_024` 验收了状态分离：22/22，730.74 秒，12.18 分钟，diagnostic/package exit 均为 0。历史 accuracy 不一致保持 `passed=false`，状态为 `completed_with_warnings`；尚未包含模型算子补丁修复，保留为对照。

模型算子修复已由新 RUN_ID `qwen38_dpo_policy_audit_025` 完成全部验收。原始 021/022/023/024 均保留，不通过重写旧退出状态伪造完成。

025 验证集已经完成：原生 `eval_loss=0.7104305624961853` 与 021 完全相同，`eval_rewards/accuracies=3/7`；mean reward margin 为 -0.0335239384，与 021 的差仅约 4.1e-8。算子清单确认：新增 1 个模型 forward、64 个 SwiGLU、129 个 RMSNorm Liger forward。对照记录中 `dpo_b447b04c52a6881507322491a1b9295b` 的 margin 从未初始化模型算子的 -0.00871468 变成 +0.00506973，解释了一票符号差异。原始 021 没有逐对文件，所以该定位来自 024 与 025 的控制对照。

| 025 最终验收项 | 实际结果 |
| --- | --- |
| 全部配对评估 | 22/22，15 train / 7 validation |
| 完成状态与历史复现 | `completed=true`、`full_dataset_evaluated=true`、`source_validation_reproduction.passed=true`、warnings 空 |
| 逐对聚合 vs 历史验证 | loss 差 1.70299e-8，accuracy 差 0；原生 loss 与历史完全相同 |
| Train / validation accuracy | 13/15 = 86.67% / 3/7 = 42.86% |
| Train / validation mean DPO loss | 0.63863559 / 0.71043058（原生 validation 0.71043056） |
| 评估 / 入口总耗时 | 704.26 s = 11.74 min / 710.76 s = 11.85 min |
| 峰值已分配 CUDA memory | 30.43 GiB |
| 诊断 / 打包退出码 | 0 / 0 |
| 最终轻量包 | 46,049 bytes，12 个文件，0 遗漏；服务器与本地逐文件校验通过 |
| 已验证方法效果 | `scene_effectiveness_measured=false`、`promotable=false`；仍为 pilot |

模型算子修复的定向测试：本地 47 passed / 1 Windows signal-test skipped，CCI 48 passed。所有原始数据与 adapter SHA-256 在四次诊断中一致；022 和 023 的 22 对得分逐项完全相同。完整对照记录、依赖源码身份、符号变化与轻量包 SHA-256 保存在 [审计证据](reports/evidence/policy_audit_022_025_evidence.json)。

服务器最终包为 `tmp/results/slow_memory/qwen38_dpo_policy_audit_025_review_final.tar.gz`，配套 `.sha256` 已下载至本地 `tmp/dpo_audit_022_fix/reviews/`。包中保留原项目结构，并补入最后的 workflow 退出标记。基座、adapter 与训练数据留在原服务器目录。

诊断通过与模型有效是两个结论。即使入口完成评估并通过复现检查，当前第一轮观察/待办目标、小验证集与缺少场景级 base/adapter 对照的限制仍然存在。

## 可复核 ACP 与轻量打包

本轮的完整验证由研发直接在 CCI 执行，无需重复提交。只有需要在独立 ACP 运行环境复核时，使用新的 026 编号；已有 022–025 目录必须保留。申请 1×H100 80GB、CPU memory 128 GiB、最长运行时间 1 h。命令不包含 Git 操作。

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

CUDA_VISIBLE_DEVICES=0 \
RUN_ID=qwen38_dpo_policy_audit_026 \
TRAIN_RUN_ID=qwen38_dpo_pilot_021 \
PACKAGE_MAX_FILE_MIB=8 PACKAGE_MAX_TOTAL_MIB=32 \
bash tmp/acp/acp_qwen38_dpo_diagnose.sh
```

入口会自动打包。若需要在运行结束后补充最终 workflow 标记并重新打包，使用同一 RUN_ID：

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_dpo_policy_audit_026
PACKAGE_PATH="$PWD/tmp/results/slow_memory/${RUN_ID}_review_$(date -u +%Y%m%dT%H%M%SZ)_final.tar.gz"
RUN_ID="$RUN_ID" PACKAGE_PATH="$PACKAGE_PATH" \
PYTHON_BIN="$PWD/.venv_dpo/bin/python" \
PACKAGE_MAX_FILE_MIB=8 PACKAGE_MAX_TOTAL_MIB=32 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
"$PWD/.venv_dpo/bin/python" scripts/package_sceneexpert_results.py --verify "$PACKAGE_PATH"
```

保留 `outputs/slow_memory/<RUN_ID>/` 和 `tmp/acp_logs/<RUN_ID>/` 的项目相对结构，包含诊断日志、配置、模板审计、模型算子清单、逐对指标、原生聚合与退出状态。单文件上限 8 MiB、总计 32 MiB，遗漏与截断及文件哈希保存在包 manifest。下载 `.tar.gz` 和配套 `.sha256` 即可；基座、adapter、optimizer、数据库、网格与 replay assets 留在服务器。轻量包不替代训练数据与完整 replay。

| 环节 | 预估时长 | 人工/无人值守 |
| --- | --- | --- |
| 参数及已同步代码核验 | 2–5 min | 人工 |
| 单卡加载、tokenization、22 对评估 | 10–25 min；冷编译/AFS 拥塞可能更久 | 无人值守 |
| 完成、逐对聚合与历史复现核验 | 3–5 min | 人工 |
| 自动打包与哈希验证 | 10–60 s，手动补包 1–3 min | 无人值守 |

时长依据：022 11.98 min，023 12.31 min，024 12.18 min，025 11.74 min（入口含打包 11.85 min）；人工约 5–10 min。算子初始化的冷编译与 AFS IO 存在不确定性，估计不是时限保证。只读复核使用一个 GPU；后续超过 1 h 的采集/训练应由 ACP 提交。

## 后续研发重点

最终以 025 的 train 13/15、validation 3/7 为准，44 个 completion 全是观察/规划目标。修复入口与精确复现原始训练指标均已完成；这些结果仍不能当作 SceneEval 效果提升证据。

后续优先实现共享观察完成后的实际资产/布局决策分叉，使成对候选拥有相同决策上下文与冻结记忆、独立执行和 raw 评分，再训练与所评结果更直接相关的动作。当前初始调用入口没有该分叉实现；本文的诊断 ACP 不启动这种采集，也不建议用改名的旧初始调用批量产出更多观察目标。

该采集入口的实现、恢复与隔离验收、以及对应 ACP/轻量打包应作为下一项研发交付，随后才安排大批量采集和 held-out 场景对照。当前不重复启动 021 训练或无目标的超参扫描。
