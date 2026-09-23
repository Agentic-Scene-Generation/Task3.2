# 资产库正面复判正式发布（2026-09-23）

本版已把 5,026 件原先 `uncertain` / `unusable` 资产的透视图 Luna 复判写回正式 `canonical_front`。发布快照共 36,789 条记录：HSSD 10,963、3D-Future 14,823、Others 5,564、生成记录 5,439（含 2 件迁移验证样品）。所有记录都有 normalized asset-local Y-up 坐标系中的水平正面方向；没有重定向原始 mesh。

复判原始状态：`unique` 3,071、`multiple_valid` 316、`uncertain` 1,225、`no_unique` 401、`unusable` 13。正式发布后，其中 3,334 件标为语义正面，53 件保留 Luna 的低置信度视觉方向但不标为已确认语义正面，1,639 件沿用明确标源的非语义 fallback。未进入本轮复判的资产保留既有正式方向。`multiple_valid` 只选择一个可用方向，同时保留原始多方向证据。

生成资产在此快照中有 4,138 条源图/提示词风格，1,301 条无明确一级风格；提示词风格计入调度库存，不表示生成的 3D 外观已经独立验收。非生成库明确一级风格 29,911 件，另有 1,439 件低优先级风格 fallback。生成资产其他标注仍是候选：3,548 件有网格实测与规则先验，1,851 件只有类别先验，34 件待归一化，4 件待追加实测，2 件为迁移验证样品。不能把候选字段的存在当作关系、尺度、净空或物理语义已经验收。

消费入口：`data/CURRENT_ANNOTATION_RELEASE.json`（Task3.2 为 `scenesmith/scenebenchmark_critic/asset_annotation_data/CURRENT_ANNOTATION_RELEASE.json`）。`FRONT_V3_RELEASE_20260923.json` 给出各文件 SHA-256 与逐状态计数；原始 `front_visual_candidate_v3`、上一版 `canonical_front_v3_previous` 保留在资产记录中。共享端版本化证据在 `/data/task3_2/share_data/scenesmith/all_assets_style_v2_delivery_20260909/front_luna_v3_20260921/full_run/`，共享静态快照在同级 `front_v3_release_20260923/`。

生成线仍运行，新回传记录由定时增量同步追加，默认带明确来源的正面 fallback；这些后续新增记录不属于本版 36,789 条冻结快照。风格×类型空缺生产和非正面候选补齐继续进行；现有 embedding 未因本次正面发布自动重算。
