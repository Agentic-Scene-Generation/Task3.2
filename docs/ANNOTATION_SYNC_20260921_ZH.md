# 标注更新与读取（2026-09-21 UTC）

本次只同步已经存在的标注成果，不新做判定，不停止后台任务，不重算 embedding。

## 本次新增

- 原 31,350 个非生成资产的完整已有标注和最新规范风格继续保留。
- 冻结时已完成的 **34,820 个 Luna 正面候选**随库交付，覆盖 HSSD、3D-FUTURE、Others 以及历史生成资产。179 个未完成 UID 单独列出。
- 更新生成资产的已有标注（包括近期 Flare 成品），准确数量以 `ANNOTATION_RELEASE_20260921.json` 为准。
- 同步正面原始可观察判断、A–G 视图映射、源坐标转换矩阵、证据文件路径/哈希、提示词版本和异常修复来源。

**完成一次判定不等于确定了唯一正面，也不等于通过人工/消费者坐标验收。**

新结果在 `front_visual_candidate` 字段，原 `canonical_front` 完全不覆盖。`unique`、`multiple_valid`、`no_unique`、`uncertain`、`unusable` 原样保留。生成资产仍保持 `annotation_complete=false`，Flare 不经过 Luna；其 `generation_target_unverified` 是生成目标标签，不冒充视觉验证标签。本次没有补齐尚缺的 relation、mask 等，也没有更新 embedding 向量。

## 三处路径

1. 本地完整消费入口：`/data/task3_2/share_data/scenesmith/all_assets_style_v2_delivery_20260909/asset_library/data/`。
2. 本地冻结发布：`/data/task3_2/share_data/scenesmith/all_assets_style_v2_delivery_20260909/annotation_release_20260921/`。
3. 独立仓库 `K-Chronofox/hssd-annotations` 的 `main`：`data/`。
4. Task3.2 的 `dev_yz`：`scenesmith/scenebenchmark_critic/asset_annotation_data/`。

仓库与冻结发布的数据文件逐项 SHA256 一致。本地主入口的三份 lookup 保留共享几何/mask 路径，风格 overlay 保持原始压缩字节及运行中的发布哈希，因此这些文件不要求与 Git 的 portable 版本整文件相同；新增候选字段逐 UID 一致。本地主入口清单的 `sha256` 对应本地实际文件，`portable_snapshot_sha256` 对应冻结/Git 副本。Git 不上传 mesh、渲染图、API 请求日志或凭据。所有实时证据引用均应在共享目录消费，不需要读取私人工作目录。

## 如何读取

最新版本从数据目录下 `CURRENT_ANNOTATION_RELEASE.json` 查找；旧日期清单仅保留历史用途，不用于校验本次已更新的数据文件。历史 HSSD 源索引和尺度参考也已复制到共享 `asset_library/data/annotation_provenance/`，哈希列于本次清单；原索引内部的旧相对路径仅作审计，实际输入记录使用 lookup 的 `input_record_path`。

既有 `canonical_front` 语义不变，新候选为可选字段：

```python
# Task3.2
from scenesmith.scenebenchmark_critic.current_asset_annotations import get_current_asset_annotation
record = get_current_asset_annotation("hssd:<asset_id>")
candidate = record.get("front_visual_candidate")
if candidate is not None:
    status = candidate["result"]["front_status"]
    source_axis = candidate["front_axis_source_local"]
    # 不可不经消费者坐标检查就替换 canonical_front。
```

独立库/共享库通过 `AssetLibrary(..., sources="all").require(uid)` 读取非生成记录中的新候选字段。注意使用 `require`，不是固定字段投影的 `require_unified`；后者保持原统一契约，不返回此次新增扩展字段。生成记录通过 `generated_annotations.json.gz` 或 Task3.2 的上述入口读取。

也可直接读取以下 gzip JSON（均为以 UID 为键的映射）：

- `front_visual_candidates_v2.json.gz`：全部完成候选；`snapshot_source_sha256` 可追溯原始 result。
- `front_visual_evidence_v2.json.gz`：全部对应的视图映射与证据哈希。
- `front_visual_pending_v2.json`：冻结时未完成的 UID；后台后来完成不会自动改写本次快照。
- `front_visual_prompt_v2.json`：本次协议与实际提示词。
- `ANNOTATION_RELEASE_20260921.json`：来源、状态计数、生成资产数、SHA256 和限制。

后续补跑继续写实时目录，下一次发布才纳入新的完成结果。本次清单不是实时监测器，也不能把旧 `COMPLETE.json` 当作 100% 成功证明。远端推送结果以项目交付记录中的实际 commit SHA 为准。
