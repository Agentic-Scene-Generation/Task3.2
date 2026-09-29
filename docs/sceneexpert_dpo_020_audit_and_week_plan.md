# 020 中断审计与未来 7 天计划

核验时间：2026-09-29 20:29（北京时间）。原始服务器结果和 019 数据保持不变。

## 020 实际结果

020 未完成。`train.log` 最后记录 `4/8 [33:56<32:03, 480.82s/it]`；从预检产物到退出文件约 45 分 53 秒。输出目录只有配置、预检、模板审计、日志和退出文件，没有任何 checkpoint、adapter、训练/验证指标或 `training_manifest.json`。因此无法恢复已执行的 4 步，也没有可评估的训练后模型。

`training_exit_code=0` 与 `package_exit_code=0` 并不能说明训练完成。9,137 字节的 review 包包含 7 个有效校验文件，证明传输完整，不证明训练有效。服务器上没有该训练进程。用户提供的 ACP 尾部记录也没有异常栈或明确的调度结束原因。

### 已确认的根因与尚未确定的外因

- **中断保障缺陷：**`save_steps=8` 等于本轮总步数，前半程没有持久 checkpoint。
- **完成判定缺陷：**Bash EXIT trap 仅信任进程状态，没有要求本轮完整训练 manifest、验证指标和 adapter。中断/外部退出可能留下错误的成功标记。
- **诊断缺陷：**终端进度条和缓冲日志不能可靠保存中间训练指标；未记录训练进程的 RSS、cgroup 内存上限、OOM 事件和终止信号。
- **停止触发源未知：**现有证据不能区分平台终止、时间限制、CPU 内存限制或其他外因。不能把 softmax 优化提示或 tokenizer token 对齐警告当作致命报错，也不能把“约 45 分钟”当作已证实的超时设置。

## 本轮实现

1. initial 训练 profile 每个优化器步骤保存完整 optimizer、scheduler、RNG、trainer state 和 adapter，保留最近 3 个 checkpoint。完整保存后写独立 marker；恢复前检查 marker 和模型/数据/训练配置身份。
2. 每次 log/save 持久化进度和指标，禁用训练进度条，使用无缓冲输出。独立 supervisor 每 30 秒记录心跳、进程 RSS 和可访问的 cgroup 内存用量/上限/OOM 计数。
3. Bash 将 TERM/INT/HUP 转交 supervisor，并保留非零退出；supervisor 记录实际子进程退出码和信号。外部 SIGKILL 无法被任何用户态 trap 捕获，此时缺少终态记录仍按未完成处理。
4. 退出 0 必须同时满足：本次 invocation 的 manifest、完整优化器计划、有限损失、非零 LoRA 更新、要求的验证指标，以及非空 adapter。缺失证据返回 4，不再误报成功。
5. 完成训练后先保存 adapter，再做一次最终验证。initial profile 取消末步自动验证与手工验证的重复执行；评估失败仍保留已训练产物。
6. 打包在缺少训练 manifest 时也能识别为未完成训练，并优先保留中间状态与诊断文件。

这次修复不修改模型目标、数据阈值、Qwen 上下文长度或 Designer/Critic 逻辑。保留本地用户修改的汇报 PPT，研发提交不包含该文件。

## 立即执行：021 重跑现有小样本 pilot

020 没有可用 checkpoint，021 必须从原始基座开始。不要重复使用 020 的 RUN_ID。提交 ACP，申请 **1×H100 80GB**，CPU 内存至少 **128 GiB，若可选优先 256 GiB**，任务最长运行时间设为 **4 小时**。这里的主机内存是保守分配建议，不是已证实的 020 OOM 结论。Shell 环境变量不能替代 ACP 页面上的资源和时限设置。

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

CUDA_VISIBLE_DEVICES=0 \
CAMPAIGN_ID=qwen38_sceneeval_dpo_019 \
RUN_ID=qwen38_dpo_pilot_021 \
PROFILE=furniture_initial_pilot \
BASE_MODEL=/mnt/afs/task3_2/share_model/Qwen/Qwen3.8-27B \
PACKAGE_MAX_FILE_MIB=32 PACKAGE_MAX_TOTAL_MIB=128 \
bash tmp/acp/acp_qwen38_dpo_train.sh
```

入口自动打包并校验。若平台强制终止或需要事后重新打包，用以下脚本；每次创建新的包，不修改服务器原始结果：

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
RUN_ID=qwen38_dpo_pilot_021
PACKAGE="tmp/results/slow_memory/${RUN_ID}_review_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
.venv_dpo/bin/python scripts/package_sceneexpert_results.py \
  --project-root "$PWD" --run-id "$RUN_ID" --output "$PACKAGE" \
  --max-file-mib 32 --max-total-mib 128
.venv_dpo/bin/python scripts/package_sceneexpert_results.py --verify "$PACKAGE"
printf 'Download: %s\nChecksum: %s.sha256\n' "$PACKAGE" "$PACKAGE"
```

包保留项目相对路径、配置、日志、进度、资源采样、指标和 checkpoint 完成标记。大模型、adapter 权重和 optimizer/RNG 二进制保留服务器，省略项、截断和校验和在 manifest 中。review 包不能用于恢复训练。

**通过条件：**`training_supervisor.json` 为 `completed`，真实子进程退出 0，8/8 优化器步骤完成，manifest 与本轮 execution ID 一致，loss 有限，有非零 LoRA 更新，7 个验证样本的指标和 adapter 均存在；workflow 训练/打包退出均为 0。验证准确率是否改善单独解释，低于阈值不等于程序失败，也不能将 7 个样本的结果当作强泛化结论。pilot 始终不自动部署。

如果再次中断，先读取 supervisor、资源日志及最近完整 checkpoint。使用通过 marker 和训练身份检查的 checkpoint 恢复，避免盲目从头重复。尚无最终结果时不启动场景效果评估。

| 阶段 | 估计时长 | 人工/自动 |
| --- | --- | --- |
| 检查 ACP 资源、时限并提交 | 3–5 分钟 | 人工 |
| 训练、逐步保存和最终验证 | 1.5–2.5 小时 | 无人值守，1 GPU |
| 结果审计 | 10–20 分钟 | 人工/脚本 |
| 轻量打包及校验 | 1–5 分钟 | 自动 |

依据：020 前 4 步约 34 分钟，前 3 步每步约 9 分钟；完整序列容量探针为两步约 321 秒。步数的梯度累积和数据长度不同，所以容量探针不能直接按两步时间外推正式训练。增加 checkpoint 会增加共享存储开销。人工约 15–25 分钟，以上是区间估计。

## 7 天安排与决策点

以下是有依赖的工作顺序，立即可运行的只有上面的 021。后续训练/采集任务在实现和前置验收完成后提供独立 ACP 与打包命令，不将尚未实现的执行入口写成可运行命令。

| 时间 | 目标与产物 | 预算及决策点 |
| --- | --- | --- |
| 第 1 天 | 完成 021，核验 train/validation 指标、参数更新与保存；检查 adapter 推理兼容性 | 训练约 2 小时，审计/适配约半天。最多一次有明确原因的运行修复，不开展无目标的超参扫描 |
| 第 2 天 | 提高采集信号直接性：在共享观察后、实际资产/布局决策前分叉，保持相同上下文；先验收 2–4 个组 | 开发与验收约 0.5–1 天。未通过不宣称直接动作数据已具备，也不跨上下文拼接历史轨迹 |
| 第 2–3 天 | 验收通过后批采 60–100 个 SceneEval 决策组，目标新增 30–50 对直接动作偏好 | 预计 20–35 GPU 小时，依据 019 的约 16h41m 批次，实际受阶段长度和资产失败率影响。数量是目标，不是保证；冻结 train/val/test |
| 第 4 天 | 对新增数据训练；以固定验证集选择一次候选，接入实验推理入口 | 预计 4–10 GPU 小时，须按新数据长度复估。当前 22 对首步数据仅作为 pilot/对照，避免把更多首步采集误当作更强布局监督 |
| 第 5–6 天 | 相同任务、Fast Memory 快照、解码和评判下比较基座与 adapter；优先 20 个 held-out 任务的成对场景 | 暂留 30–50 GPU 小时，先以少量完整场景测时校正。报告完成率、硬约束、物理违规、成对胜负与成本；资源不足时缩小样本并明确统计局限 |
| 第 7 天 | 汇总结果、失败归因、可视化案例与论文表格 | 约 0.5–1 天人工；冻结方法与数据，不为得到正结果修改 held-out 标签或继续调参 |

单卡全天可用的一周预算应优先保障“训练产物可复现 + 场景级受控对照”。若第 2 天内直接动作采集仍不能完成验收，则冻结现有方案，只对已经可用的候选模型完成受控评估，把 Slow Memory 结论限定为探索性结果。已有 Harness/Fast Memory 结果作为各自模块的证据，不把它们冒充为 DPO 收益。
