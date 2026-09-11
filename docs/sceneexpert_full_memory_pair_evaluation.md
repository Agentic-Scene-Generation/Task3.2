# Full 模式 Fast Memory 配对评测

该流程用于回答一个严格限定的问题：在相同 Full 模式、相同模型、相同
prompt、相同 shared base 和相同冻结 Memory bank 下，仅启用 Fast Memory
检索与注入，是否改变场景生成的完成率、速度和质量。

## 前置条件

- shared base 已完整生成，并可由 `tmp/acp/acp_qwen38_full_reuse.sh` 复用；
- Memory bank 已预先建立，且包含 `manifest.json`、三个长期记忆 JSONL 和
  `events.jsonl`；
- 评测期间不得由其他任务修改该 Memory bank；运行时场景级审计继续写入各自
  输出目录，不再回写冻结库的 `events.jsonl`；
- 两臂使用相同的模型服务参数、case 选择和 stage policy。

## 最简命令

```bash
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

PAIR_ID=full_memory_sceneeval100_hard_v1 \
SOURCE_RUN_ID=<生成_shared_base_的_run_id> \
FROZEN_MEMORY_ROOT=/mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2/outputs/scene_expert_memory/<frozen_snapshot_root> \
FROZEN_MEMORY_DIR=/mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2/outputs/scene_expert_memory/<frozen_snapshot_root>/ablation_4c \
CASE_SET=sceneeval100 \
DIFFICULTY_SELECTION=hard \
SCENE_SELECTION=all \
ARM_ORDER=off_on \
bash scripts/run_sceneexpert_full_memory_pair.sh
```

`SCENEEXPERT_MEMORY_DIR` 是 Hydra 的 Memory 根目录；Full 模式会在其后自动
追加 `ablation_4c`。因此 `FROZEN_MEMORY_ROOT` 必须是父目录，而
`FROZEN_MEMORY_DIR` 必须是实际包含 `manifest.json` 与 JSONL 的
`<root>/ablation_4c`。启动器会验证这两个目录的关系，避免重复追加或漏掉
实验 namespace。

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

冻结快照的实验身份由 `manifest.json`、`success_cases.jsonl`、
`failure_cases.jsonl` 和 `skills.jsonl` 共同定义。这四个文件会直接决定检索
候选与注入内容。`events.jsonl` 仅为不可检索的历史审计日志：启动器仍要求
它存在，但兼容旧快照时不会把其旧哈希漂移误判为检索内容污染。更新后的
评测运行不会再向冻结库追加新事件。
