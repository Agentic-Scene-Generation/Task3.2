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

诊断重新加载 adapter 后没有重放 train 入口的模型补丁，导致与 021 最终评估使用不同算子；near-zero reward margin 使小数值差异能改变离散 accuracy。新增 `prepare_training_kernels_for_evaluation()` 使用依赖自身的同一 API 和配置，记录补丁前后的 forward 模块清单，并在评分前初始化。该原因的完整 GPU 对照以 025 的复现结果验收。

**执行状态设计也需要修正。** 旧工作流把“历史统计完全一致”作为“成功完成测量”的必要条件。当前只保存了 021 的聚合统计，无法定位历史一票差异对应的具体样本。负面结果或历史复现警告应成为可分析的诊断产出，不能让运行人员反复重启相同实验以获得一个成功状态。

因此，执行成功要求真实输入与 adapter 哈希一致、全部选定样本评分完毕、指标有限、sigmoid loss 正确、逐对聚合与原生评估一致。历史复现保持原容差和 `passed=false`，作为 `completed_with_warnings` 明确保存；它不再将完整诊断变成执行失败。数据/权重变化、非有限数值、目标不一致或缺失样本仍会失败。没有放宽训练发布质量门禁，`promotable=false` 保持不变。

后续单卡、batch=1 的 furniture pilot 训练，在同一次最终原生评估中保存 `validation_pair_scores.jsonl`，记录逐对原始得分，并将文件 SHA-256 绑定到训练 manifest；重新加载审计拒绝被改动的记录。这使后续差异能定位到具体样本，无须重新训练来找证据。

没有修改 Critic、Designer、Repair 或已有训练权重，也没有通过替换 preference 标签改善统计。

原生协议改进提交 `9a34df7` 已推送并以离线 bundle 同步 CCI；本地 45 passed、1 Windows signal-test skipped，CCI 46 passed。增加完成状态分离、训练逐对记录与防篡改回归后，本地定向测试 46 passed、1 Windows signal-test skipped。用户的 PPT 与演讲稿本地改动保留。

## CCI 验证

`qwen38_dpo_policy_audit_023` 已在单卡 H100 80GB 完成：22/22，738.33 秒，12.31 分钟，峰值已分配 CUDA memory 30.43 GiB。原生与逐对聚合一致，验证 loss 0.7118828 / accuracy 2/7，训练 loss 0.6375559 / accuracy 13/15，均与 022 一致。旧历史复现门禁仍使这个中间验证返回 2，打包成功。保留该结果作为 A/B 证据。

`qwen38_dpo_policy_audit_024` 验收了状态分离：22/22，730.74 秒，12.18 分钟，diagnostic/package exit 均为 0。历史 accuracy 不一致保持 `passed=false`，状态为 `completed_with_warnings`；尚未包含模型算子补丁修复，保留为对照。

模型算子修复使用新 RUN_ID `qwen38_dpo_policy_audit_025` 进行完整验收，结果在本轮结束前补充。原始 021/022/023/024 均保留，不通过重写旧退出状态伪造完成。

诊断通过与模型有效是两个结论。即使入口完成评估并通过复现检查，当前第一轮观察/待办目标、小验证集与缺少场景级 base/adapter 对照的限制仍然存在。
