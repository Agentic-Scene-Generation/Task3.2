# 026：共享观察后的实际设计动作采集

## 本轮交付记录（2026-10-11）

采集代码及测试提交为 `1463d15578a7ad0ba8003e4d1701f70b89543393`，本地、origin 与 CCI 的 HEAD 一致；1123 个已跟踪文件按提交内容核对通过，CCI 工作区和 index 干净。本地已有的汇报 PPT 与演讲稿修改保留，未纳入本轮提交。随后仅补充本交付记录，不改变运行源码。

本地 81 项可移植测试通过；CCI 使用生产 `.venv` 的 agents 0.6.4 / httpx 0.28.1 运行 109 项测试，全部通过（51.37 s）。启动脚本语法检查和 SDK preflight 通过，SceneEval train/validation/test 划分与 019 一致。生产模型采集尚需 ACP 运行验证，单元测试不能代替真实配对验收。

已于 19:00:23（北京时间）提交 ACP：显示名称 `task_3_2_design_pairs_026`，任务 ID `pt-psdm7z0s`。19:01:33 页面状态为“启动中”，资源、镜像、AFS 与保存的命令已核验；这不是实验完成或偏好产出的证据。请勿再次提交同一任务。本对话已安排每 30 min 检查一次，完成、失败或需要处理时更新。结构化记录见 `docs/reports/evidence/direct_design_pairs_026_delivery.json`。

025 已恢复 021 的原生评估：validation loss 0.7104305625、accuracy 3/7，22 对全部完成。当前限制是监督目标：021 的 44 个 completion 都是首轮观察/待办等操作，train 13/15 与 validation 3/7 不能佐证场景泛化收益。接下来验证直接设计动作的采集入口，不重复政策诊断。

## 实现与适用范围

- 专用入口 `tmp/acp/acp_qwen38_design_pairs.sh`，默认只处理两个 SceneEval task（train/validation 各一个），一张 GPU、单场景并发，furniture 后停止。沿用 019 的 128/24/32、seed=42 划分，test 不参与采集；实际启动前检查划分一致。
- Canonical A 正常运行。共享前缀只支持 observe_scene、get_current_scene_state、list_available_assets 和 designer_todo_manager。在 A 首个自然设计响应返回、工具尚未执行时保存完整原生 SDK 输入、实际 HTTP 请求、场景、图片、随机状态、渲染状态、TODO 状态及剩余调用预算。B 在私有场景恢复状态，直接从该输入采样一次并执行。首个请求与 A 不一致时在发送前拒绝；正常模式行为保持原样。
- 没有强制工具或重复抽样。采样边界依赖 A 首次选择设计动作，因此这是条件采集，不能宣称无选择偏差。未知前缀和无法恢复的状态明确分流，保留错误证据。
- 两侧 raw rollout 独立评分，继续使用现有 strict 与 verified_relative_v1 两套标签。relative margin 保持 0.03；完整性、同上下文、physics、工具执行和隔离门禁保留。
- `first_turn` completion 的两侧都必须含真实设计工具，观察/待办目标被过滤并计数。训练只监督 assistant 动作，后续执行结果提供 rollout 级偏好；这不等同于该单步动作的即时因果奖励。候选 B 首步继续观察的情况不重采，也不拼接不同上下文。
- 使用独立 campaign 和协议文件，禁止将旧初始采集批次重新解释为新边界数据。冻结记忆；专用对比采集关闭 Writer，常规场景流程的 Writer 不变。采集不启动训练。

## 提交 ACP

配置参考用户提供的 pt-491ruzly：computing_cluster_01e，1×N6lS-80GB、22 vCPU、230 GiB、1 replica，开发机镜像 `ccr-zhicheng-02/scene_vlm:lwz-workspace-20260611125259`，AFS afs-p3-01e 的 `/` 挂到 `/mnt/afs`，最大重启次数 0。闲时资源可能被回收，排队及回收不计入预计执行时间；如平台提供时限设置，建议至少 6 h。

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

CUDA_VISIBLE_DEVICES=0 \
RUN_ID=qwen38_design_pairs_026 \
MAX_TASKS=2 CHUNK_SIZE=2 ACP_PARALLELISM=1 \
PACKAGE_MAX_FILE_MIB=16 PACKAGE_MAX_TOTAL_MIB=128 \
bash tmp/acp/acp_qwen38_design_pairs.sh
```

入口会自动打包并校验。结束后另做最终轻量复核包（包含最后的 workflow 退出记录）：

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_design_pairs_026 \
PACKAGE_PATH="$PWD/tmp/results/slow_memory/qwen38_design_pairs_026_review_final.tar.gz" \
PACKAGE_MAX_FILE_MIB=16 PACKAGE_MAX_TOTAL_MIB=128 \
bash tmp/acp/pack_qwen38_initial_pairs.sh

.venv/bin/python scripts/package_sceneexpert_results.py \
  --verify tmp/results/slow_memory/qwen38_design_pairs_026_review_final.tar.gz
```

下载最终 `.tar.gz` 及 `.sha256` 即可。包保留原项目相对结构、配置、状态、日志、配对报告和限量媒体，记录遗漏、截断及校验和；模型、mesh、SQLite 和 replay 副本保留在服务器。轻量包不能代替完整 replay 数据或完整训练集。

## 验收与后续判断

验收分开看三项：两个 task 的 A/B 均完整且隔离；首个真实决策请求完全相同，TODO 与剩余预算恢复；direct-action 数据产出与过滤原因明确。2 组可执行不保证 2 对偏好：至少取得一对有效 direct-action contrast 后扩大同一冻结 campaign；零产出先分析过滤分母，不以名义 exit=0 或修改标签强行通过。

如果新上下文含图像，后续训练应启用现有 multimodal 路径并先做完整上下文容量探针，不能把图像删除后沿用 021 的 text-only 配置。小批验证后才安排扩大采集、DPO pilot 与预留 test 的 base/adapter 场景对照，场景级增益尚未测量。

## 耗时估计

| 阶段 | 时间范围 | 人工/无人值守 |
|---|---:|---|
| 源码核对、任务配置、提交 | 5–10 min | 人工；本轮由研发处理 |
| 两个 SceneEval task，共四条剩余 rollout | 45–150 min | 无人值守；不含排队 |
| 审计、检查直接目标与媒体完整性 | 5–15 min | 部分人工 |
| 最终打包与校验 | 1–5 min | 无人值守 |

依据：019 在并发 2 下 72 次尝试约 16 h 41 min，平均每次约 13.9 min 墙钟；新边界、两个 task 的不同难度、模型加载、AFS IO 和闲时抢占会改变用时，因此上限放宽。预计可能超过 1 h，本轮只在 CCI 做 CPU/SDK 验证，实际采集由 ACP 执行。总计约 1–3 h 加排队，人工约 10–20 min；不是时长保证。
