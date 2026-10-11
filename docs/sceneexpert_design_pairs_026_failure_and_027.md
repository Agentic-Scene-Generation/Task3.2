# 026 复核与 027 历史恢复修复

## 已确认结果

ACP `pt-psdm7z0s` 在 2026-10-11 20:31:30（北京时间）显示失败；实际 invocation 为 `20261011T115027Z_61`，执行约 40 min。两条主场景生成完成、均 degraded、missing=0，生成 launcher exit=0。两组配对均不完整，strict 与 verified_relative_v1 导出都是 0 对，因此 campaign / workflow collection exit=2。打包 exit=0。运行完成、配对完整性、偏好产量是三个不同结论。

- `group_000`（SceneEval 273 / playroom）：A 的 generate_assets 请求无法在 HSSD 找到描述的软坐垫，SDK 将异常返回为未处理工具错误；在 canonical evidence 检查时被隔离，B 未运行。保留该组及错误证据，本轮不改变其质量或完整性门禁。
- `group_001`（SceneEval 360 / office）：A 已独立执行和评分；B 恢复场景与工具后、发送模型请求之前，被 agents 0.6.4 拒绝。完整原生历史是 list，生产入口同时传 SQLite session，却没有 `RunConfig.session_input_callback`。SDK 明确要求声明 list 与 session 的合并规则。这是执行契约错误，不能靠降低偏好分数门槛解决。
- 主场景、资产与记忆的 B 前后隔离哈希一致，但 B 未执行，因此不是一对有效偏好，也不支持模型效果结论。

## 根因修复

修复仅位于 SceneExpert `paired_runtime.py`：first_mutation shadow 的 RunConfig 显式配置历史合并回调，要求 SDK 合并时 session 仍为空，并原样返回完整冻结 input。保留原生 session 持久化、模型 provider、输入过滤器、剩余调用预算和发送前 HTTP 一致性检查。其他 initial 边界继续使用原来的配置。拒绝已有 session 内容和不完整字符串输入，避免上下文被重复拼接或替换成摘要。

回归测试使用实际 agents SDK、SQLiteSession 和生产 RunConfig 构造函数，验证空 session 下完整前缀只注入一次；污染 session 在模型和工具执行前拒绝；字符串 initial 边界不改变。此前测试的 B 未传 session，漏掉了生产 SDK 的这项前置契约。

监控路径已纠正：实际运行数据在 `outputs/slow_memory/<RUN_ID>`，控制日志与 workflow 退出记录在 `tmp/acp_logs/<RUN_ID>/<invocation>`，`tmp/results/slow_memory` 保存 review 压缩包。该目录下没有同名结果文件夹不意味着任务尚未执行。此前仅检查打包目录得到的运行判断已更正。

## 026 最终复核包

服务器已生成并校验 `tmp/results/slow_memory/qwen38_design_pairs_026_review_final.tar.gz`，17,415,407 bytes（约 16.6 MiB），包含 390 个文件、省略 1766 个文件；单文件上限 16 MiB，总选入上限 128 MiB。保留项目相对目录、失败组、运行配置、审计及最终退出记录；包中记录省略和校验和。原始服务器数据不变，包不能替代完整 replay。

## 下一步 ACP（供运行端提交，本次监控不自动新建任务）

保持上一任务的单卡资源、镜像和 AFS 配置；用新 RUN_ID 重新冻结和执行两个任务，避免修改 026 的旧 snapshot 或拼接不同版本的候选。更改运行代码后不能将旧 A 与新 B 强行配对。

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

CUDA_VISIBLE_DEVICES=0 \
RUN_ID=qwen38_design_pairs_027 \
MAX_TASKS=2 CHUNK_SIZE=2 ACP_PARALLELISM=1 \
PACKAGE_MAX_FILE_MIB=16 PACKAGE_MAX_TOTAL_MIB=128 \
bash tmp/acp/acp_qwen38_design_pairs.sh
```

入口会自动打包。真正结束后补充最终包，并检查完整性：

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_design_pairs_027 \
PACKAGE_PATH="$PWD/tmp/results/slow_memory/qwen38_design_pairs_027_review_final.tar.gz" \
PACKAGE_MAX_FILE_MIB=16 PACKAGE_MAX_TOTAL_MIB=128 \
bash tmp/acp/pack_qwen38_initial_pairs.sh

.venv/bin/python scripts/package_sceneexpert_results.py \
  --verify tmp/results/slow_memory/qwen38_design_pairs_027_review_final.tar.gz
```

027 验收要求：office 的 B 真实执行、A/B 首个请求相同，完整性与隔离审计通过；分别报告两组的完整配对数、direct-action contrast 数及未通过原因。有效偏好产量不保证为正。取得有效样本再扩大同一冻结 campaign，随后安排包含图像上下文的容量探针、DPO pilot 和预留 test 对照。

预计准备/配置 5–10 min（人工）；单卡执行 60–150 min（无人值守、排队另计）；审计 5–15 min（部分人工）；打包 1–5 min（无人值守）。人工约 10–20 min，总计约 1–3 h 加排队。依据 026 两条主场景、一次失败的 B 初始化约 40 min；修复后 B 要真正生成、执行工具和评分，因此增加余量。AFS IO、任务难度和闲时资源回收可能延长时长。实际采集继续由 ACP 执行，不在 CCI 自行启动预计超过一小时的 GPU 作业。
