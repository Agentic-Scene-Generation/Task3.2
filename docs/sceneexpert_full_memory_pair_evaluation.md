# Full 模式 Fast Memory 配对评测

该流程用于回答一个严格限定的问题：在相同 Full 模式、相同模型、相同
prompt、相同 shared base 和相同冻结 Memory bank 下，仅启用 Fast Memory
检索与注入，是否改变场景生成的完成率、速度和质量。

## 前置条件

- shared base 已完整生成，并可由 `tmp/acp/acp_qwen38_full_reuse.sh` 复用；
- Memory bank 已预先建立，且包含 `manifest.json`、三个长期记忆 JSONL 和
  `events.jsonl`；
- 评测期间不得由其他任务修改该 Memory bank；
- 两臂使用相同的模型服务参数、case 选择和 stage policy。

## 最简命令

```bash
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

PAIR_ID=full_memory_sceneeval100_hard_v1 \
SOURCE_RUN_ID=<生成_shared_base_的_run_id> \
FROZEN_MEMORY_DIR=/mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2/outputs/scene_expert_memory/<bank_name> \
CASE_SET=sceneeval100 \
DIFFICULTY_SELECTION=hard \
SCENE_SELECTION=all \
ARM_ORDER=off_on \
bash scripts/run_sceneexpert_full_memory_pair.sh
```

启动器依次运行：

1. `memory_off`：Full 模式、Slow Memory 轨迹采集开启、Fast Memory 检索关闭、MemoryWriter 关闭；
2. `memory_on`：除 Fast Memory 检索开启外，其他生成条件与第一臂相同；
3. 对两个 `run_metrics.json` 进行严格的逐 case 配对比较。

## 结果位置

- OFF 单臂：`outputs/critic_probe/<PAIR_ID>_memory_off/metrics/`
- ON 单臂：`outputs/critic_probe/<PAIR_ID>_memory_on/metrics/`
- 配对报告：`outputs/critic_probe/<PAIR_ID>_metrics/paired_metrics.json`
- 可读摘要：`outputs/critic_probe/<PAIR_ID>_metrics/paired_metrics.md`
- 逐场景差异：`outputs/critic_probe/<PAIR_ID>_metrics/paired_scene_metrics.csv`

只有 `comparison_ready=true` 时，速度、完成率和结果差异才具备本次单变量
实验的归因资格。该门禁要求逐 case 的 prompt/shared-base 指纹一致、非处理
变量签名一致、源代码包一致、模型一致、Memory bank ID/revision/内容指纹
一致，且两臂运行前后 bank 均未变化。

`quality_delta_ready=true` 进一步表示两臂全部场景均有完整 critic 和 trace
证据。报告同时给出完成/退化转移、耗时、critic 分数、required coverage、
hard constraint 和 relation satisfaction 的逐 case delta，以及速度、critic
和空间关系指标的双侧精确符号检验。单次实验即使通过身份门禁，也应根据
非零配对样本数和符号检验结果判断证据强度；建议使用不同 `PAIR_ID` 重复
运行，以评估模型采样波动，并在重复实验之间交替使用 `ARM_ORDER=off_on`
与 `ARM_ORDER=on_off`，降低固定执行顺序对耗时结果的偏置。
