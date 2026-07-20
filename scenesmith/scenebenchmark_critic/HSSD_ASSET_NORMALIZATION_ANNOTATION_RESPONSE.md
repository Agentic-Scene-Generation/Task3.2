# HSSD 资产规范化标注交付回执

本文回应 `asset_normalization_annotations.md` 提出的 HSSD front、material、
mass、friction、quality 和功能标注接入要求。实现位于
`xy3xy3/scenesmith:dev_yz_from_hrk`，该分支直接从
`Agentic-Scene-Generation/Task3.2:dev_hrk@b26591b` 建立，没有把旧 `yz`
分支混入同学的基线。

## 1. 交付结论

| 要求 | 状态 | 当前实现 |
| --- | --- | --- |
| 以 HSSD 唯一 ID 查表 | 完成 | gzip lookup 顶层 key 为无 `hssd:` 前缀的完整 ID |
| 保留 asset-local canonical front | 完成 | 复用 `canonical_front`，转换发生在消费者内，不写回 world `+Y` |
| 标准 material key | 完成 | 10,963 条均为基线 `materials.yaml` 中的 key |
| `mass_kg` 与范围 | 完成 | 10,963 条均有正数估计和包围估计值的不确定区间 |
| 独立 friction | 完成 | `MeshPhysicsAnalysis.friction_coefficient` 可选；标注优先、材质回退 |
| 独立 quality | 完成 | 10,963 条均有评分、可接受性、证据、失败/警告标签 |
| 单次统一覆盖 | 完成 | `_override_hssd_asset_annotations()` 一次读取并逐字段校验 |
| 非法/缺失字段降级 | 完成 | 每个字段独立回退，不因一个坏值丢弃整条记录 |
| SDF 消费摩擦系数 | 完成 | `generate_drake_sdf()` 优先使用标注值 |
| quality 不污染物理量 | 完成 | 当前只 warning，不静默过滤检索结果或修改 mass/friction |
| 功能/净空等既有字段保留 | 完成 | enrichment 只覆盖 `asset_physics`、`asset_quality` |
| 全量审计 | 完成 | ID、公式、区间、渲染目录和代表图重复筛查均有机器报告 |

## 2. 数据覆盖与语义

当前 lookup：

```text
scenesmith/scenebenchmark_critic/asset_annotation_data/
  hssd_annotation_lookup.json.gz
```

覆盖 10,963 个 HSSD ID。每条记录新增：

```json
{
  "asset_physics": {
    "schema_version": "hssd_asset_physics@1.0",
    "material": "wood",
    "material_confidence": 0.64,
    "mass_kg": 12.0,
    "mass_range_kg": [6.6, 21.6],
    "friction_coefficient": 0.4,
    "confidence": 0.64,
    "estimation_basis": {},
    "provenance": {},
    "audit": {"status": "passed_rule_checks", "flags": []}
  },
  "asset_quality": {
    "schema_version": "hssd_asset_quality@1.0",
    "overall_score": 0.9,
    "is_acceptable": true,
    "single_object": true,
    "complete_geometry": true,
    "stable_on_support": true,
    "watertight": null,
    "confidence": 0.72,
    "failure_tags": [],
    "warning_tags": ["watertight_not_measured"],
    "evidence": {},
    "provenance": {}
  }
}
```

这里的 material 和 mass 是用于仿真的先验，不冒充逐资产称重或材质实验。
`watertight=null` 是有意保留的未知值：渲染图和 geometry path 存在不能证明
拓扑封闭，不能按要求示例随意写成 `false` 或 `true`。

已有字段未被改名或删除，包括：

```text
canonical_front
scenebenchmark_fd_sa
scenebenchmark_functional_hints
environment_anchors
support_regions
interaction_clearance
operation_space
clearance_intrusion_whitelist_refs
post_replacement
self_emission
```

## 3. 生成和审计方法

Material 使用完整词/短语规则，不使用裸子串。这样避免：

- `tablet computer` 被 `table` 规则识别为桌子；
- `toiletry` 被 `toilet` 规则识别为马桶；
- `table runner`、`desk calendar`、`pet bed` 继承家具重量；
- `bottle opener` 继承玻璃瓶材质。

Mass 先取 469 个 HSSD 类别的参考值。7,998 个有有效 bbox 的资产，再按同类别
bbox 中位体积做有界尺度修正；2,965 个无尺寸资产保留更宽区间和较低置信度。
特殊尺寸级别只在 bbox 能明确证明尺度不同的情况下使用，例如一个 0.36 m
摩托模型按微缩模型处理，而全尺寸摩托仍使用全尺寸先验。

全量审计结果：

```text
standalone / SceneSmith ID count       10,963 / 10,963
standalone-only / SceneSmith-only      0 / 0
physics/quality payload mismatch       0
formula mismatch                       0
invalid mass interval / dimensions     0 / 0
annotated ID without render directory  0
representative render images hashed    10,963
unresolved cross-category duplicates   0
passed_rule_checks                     7,219
bounded_estimate                       3,744
```

机器报告：

```text
asset_annotation_data/ASSET_PHYSICS_QUALITY_AUDIT.json
asset_annotation_data/ASSET_PHYSICS_CATEGORY_AUDIT.csv
```

`bounded_estimate` 不是错配，表示缺尺寸、尺度修正触边界或材质置信度较低，调用方
可以据此决定是否人工复核。

## 4. SceneSmith 消费边界

实现文件：

```text
scenesmith/agent_utils/asset_manager.py
scenesmith/agent_utils/mesh_physics_analyzer.py
scenesmith/agent_utils/sdf_generator.py
```

规范化流程中，HSSD 分析完成后、`canonicalize_mesh()` 之前调用统一 override。
它会分别验证 front、material、mass、mass range 和 friction；单个字段非法时只
回退该字段。Quality 保持 advisory：不可接受资产会写 warning，但不会在当前
版本中静默降低 retrieval recall。

配置键为：

```yaml
asset_manager:
  hssd_asset_annotations:
    source: scenebenchmark_critic
    annotation_lookup_path: null
```

家具、墙面、manipuland 和 ceiling 配置均已接入。旧 `hssd_front_axis` 仍作为
兼容配置接受。

## 5. 复现与验证

仅将独立标注中的 physics/quality 更新到同学 lookup：

```bash
python scripts/enrich_hssd_asset_physics_quality.py \
  --source /path/to/hssd-annotations/data/hssd_annotation_lookup.json.gz \
  --target scenesmith/scenebenchmark_critic/asset_annotation_data/hssd_annotation_lookup.json.gz
```

该脚本只复制两个新增 family，不会覆盖同学分支的 FD/SA 或既有净空数据。

相关测试：

```text
tests/unit/test_asset_manager.py
tests/unit/test_hssd_asset_physics_annotations.py
tests/unit/test_sdf_generator.py
```

独立 API 当前测试为 32 passed。当前机器缺少 `bpy` 和 `pydrake`，因此本次
data-only 刷新没有重新跑完整 SceneSmith pytest；直接加载真实 bundled lookup
并读取 front/material/mass/friction/quality 的烟雾测试通过，消费端代码未在本次
审计刷新中修改。

## 6. 明确未作的声明

- 没有把 canonical front 改成 SceneSmith 场景中的 world `+Y`。
- 没有声称 10,963 个资产经过逐个称重或材质实验。
- 没有把 render bundle 存在当成 watertight 证明。
- 没有让 quality 静默删除资产。
- 没有用新增字段覆盖或删除同学的功能、关系和净空标注。

