# 0929 进展汇报

`0929进展汇报.pptx` 为 4 页可编辑汇报，覆盖 2026-09-11 至 2026-09-29，沿用用户提供的 `0911进展汇报.pptx` 内容页版式。

1. Slow Memory 采集、评判与训练适配。
2. 014/015 办公室候选 A/B 对比。
3. 019 SceneEval 批量采集与 strict/relative 首步偏好产出。
4. 019e/019g 最长样本容量验证，以及 020 pilot 后续工作。

文字、表格和柱状图可在 PowerPoint 中编辑。数据来源和解释写在每页演讲者备注中。对应完整报告及可执行实验、轻量打包命令见 [项目报告](../sceneexpert_progress_20260911_20260929.md)，原始证据索引见 [证据 JSON](../evidence/sceneexpert_progress_20260911_20260929.json)。

## 图片来源

服务器项目下：

```text
outputs/slow_memory/qwen38_initial_pairs_new3_014/runs/paired_initial/group_000/A/raw_scene/room_office/scene_renders/furniture/renders_002/0_top.png
outputs/slow_memory/qwen38_initial_pairs_new3_014/runs/paired_initial/group_000/B/raw_scene/room_office/scene_renders/furniture/renders_002/0_top.png
```

本地原图保存在 `../assets/20260929/`，未经内容修改：

| 文件 | SHA256 |
| --- | --- |
| office_A_process_top.png | b7a7b3feb04f7e16ef853b11538086001df6d46690f06b45921f61ebad97acba |
| office_B_process_top.png | 81c946b48fd99bc7c9bc1a717ceb7a832cacbd15fbe216b8fa4eec5dd074ce1e |

两张图为独立候选的执行过程渲染，标签来自候选原始状态与独立评分证据。它们不代表 DPO 训练前后。relative 的 22 对包含 strict 的 11 对。两步容量验证不代表模型学习或场景质量已提升。

## 020 状态补充

与正文报告更早的截点相比，2026-09-29 19:55（北京时间）远程核验发现 `outputs/slow_memory/qwen38_dpo_pilot_020/` 已生成配置、预检、模板审计和训练日志。日志最近记录为 `2/8 [18:42<56:05, 560.97s/it]`，尚无最终 metrics / manifest。PPT 因此记录“已启动，待完整结果审计”，不据此认定运行完成或进程仍活跃。不要重复提交同名 020 任务。

## 文件与验证

- PPTX：4 页，2 个原生表格、1 个原生柱状图及嵌入的数据工作簿、2 张原图。
- 通过包结构、页面尺寸、字体、表格/图表检查，并重新导入最终文件逐页渲染检查。
- 未在桌面 PowerPoint 内执行验证。
- PPTX SHA256：`99cde892a16badaae337b82cf23d327deeb13080c61fd749d7aaa5efa4ef8b02`。
