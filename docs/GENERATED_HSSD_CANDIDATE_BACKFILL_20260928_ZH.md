# 生成资产 HSSD 候选字段回填快照（2026-09-28）

本次冻结生成资产正式索引 **5,511** 件；共享全库合计 **36,861** 件。正式索引中已有原正面与风格记录；此次只把候选层已有实际值、且可追溯来源的 HSSD 同构字段回填到缺失处，不覆盖既有正面或已存在值。所有生成资产仍为 `annotation_complete=false`、`default_scene_route=false`，**字段存在不等于语义验收完成**。

回填后的字段覆盖：类别与 DOF 各 5,475/5,475，非空环境参照及 relation 各 5,475/5,475；真实归一化几何引用 3,624，物理／质量候选各 3,624/3,624；官方 HSSD Stage2 原始掩码引用 2,266。原始掩码审计 issue 0 仅说明文件、身份、形状一致，不代表 mask 语义、尺度和坐标链通过。未测净空、operation space、空白名单及 null 值没有伪造填充。

路径：`generated_annotations.json.gz` 是正式读取入口；`generated_hssd_parity_candidates.json.gz` 为隔离候选快照；`generated_hssd_candidate_summary.json` 与 `generated_hssd_stage2_raw_audit.json` 是其证据摘要；`GENERATED_HSSD_BACKFILL_RELEASE_20260928.json` 含文件 SHA-256。Task3.2 仓库中这些文件位于 `scenesmith/scenebenchmark_critic/asset_annotation_data/`；独立仓库中位于 `data/`。实际 mesh、掩码 NPZ 和 labels.json 未打包进 Git，引用指向共享目录 `/data/task3_2/share_data/scenesmith/` 与 `/data/task3_2/generated_assets_hunyuan/`，部署时须有该共享挂载。

共享目录中的 `asset_library/data/generated_annotations.json.gz` 仍会随生成资产滚动更新；这两个 Git 仓库发布的是本时间点的冻结快照，不会自动跟随后续新增资产。64 Key 正面复判的剩余候选尚未覆盖正式正面；本次未重算 embedding，也未发布新的独立语义质量结论。
