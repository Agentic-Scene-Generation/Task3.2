# 全量可用正面发布与后续复判

## 本次发布的含义

按用户授权，每一件已登记资产都提供非空、单位长度的水平正面方向，包括 uncertain、unusable、no_unique、未完成 Luna 及新生成资产。方向可用于摆放，不代表所有资产都有经确认的语义正面，也不证明 unusable 资产的 mesh 已修复。

读取 `canonical_front.canonical_orientation_axis` 或 `canonical_front.asset_local_front_axis`，两者一致。坐标为 normalized asset-local Y-up，方向位于 XZ 平面；不是 Blender 世界坐标，也不是原始 Z-up OBJ 坐标。Others 的 Z-up 输入已按其既有几何规范转换；不更改 mesh 或其 upright。SceneSmith 的实例变换仍须由消费者应用。

选择顺序：通过一致性检查且置信度足够、符合既有水平坐标约定的 Luna unique；Luna multiple_valid 中确定性选择一个有效方向；否则保留原有有效水平轴，若缺失则使用 +Z。未判断且原本已有语义方向的保留并明确来源。Luna 判断为 uncertain / unusable / no_unique 时，保留方向但不沿用旧的语义真实性声明。

## 必须保留的区别

- `canonical_orientation_source`：Luna 单次判断、多个有效面中选一面、保留旧方向或默认 +Z。
- `canonical_orientation_is_semantic_front`：是否有语义依据。false 只表示摆放约定；不是语义判断成功。
- `luna_front_status`、`luna_consistency_flags`：保留原始判断的不确定状态。
- `is_strict_front=false`：未声称人工真值或独立语义验收。
- `canonical_front_previous`：发布前的完整旧字段；原始 `front_visual_candidate` 不改写。
- `annotation_complete` 不因补齐正面而改为 true；其他缺失标注仍如实保留。

此版替代旧发布“正面候选不覆盖正式方向”的消费策略，但不伪造或改写原始 Luna 记录里的 historical `formal_annotation_overwritten=false`。实际生效方向以 canonical_front 和当前发布指针为准。

## 位置与验收

共享消费数据：`/data/task3_2/share_data/scenesmith/all_assets_style_v2_delivery_20260909/asset_library/data/`。

共享快照与验证：`/data/task3_2/share_data/scenesmith/all_assets_style_v2_delivery_20260909/front_required_20260921/`。

独立仓库：`data/`。Task3.2 的 dev_yz：`scenesmith/scenebenchmark_critic/asset_annotation_data/`。三处均读取 `CURRENT_ANNOTATION_RELEASE.json` → `FRONT_REQUIRED_20260921.json`。manifest 含实际快照总量、来源统计及校验和；旧 dated manifests 仅描述旧快照。

`canonical_front_required.json.gz` 是覆盖全部来源的 UID→正面映射，亦可直接读取生成资产；非生成 AssetLibrary.require / require_unified 与 Task3.2 get_current_asset_annotation 返回同一 canonical_front。

发布验证检查全量非空、非零、单位长度、水平轴，三处校验和、来源坐标转换及实际消费 API。不重算 embedding，不修改风格 overlay，不停止生成或原有后台任务。备份在 `/mnt/aoss2/codex_backup/front_required_20260921/`。

这是快照库存覆盖，不宣称自动包含未来生成的新资产。

本次实际覆盖 36,665 件：HSSD 10,963、3DFuture 14,823、Others 5,564、生成 5,315。Luna unique 13,924、multiple_valid 选一面 820、保留旧方向作为 fallback 18,030、未完成 Luna 而保留原语义方向 108、新补 +Z fallback 3,783。全量方向、三处文件与源坐标转换、实际 require/require_unified 和 Task 数据读取接口通过；nobody 用户能读全量 36,665 条方向。

测试边界：新增解析规则 5 项通过；历史 HSSD front 审计测试改为检查 canonical_front_previous，并新增当前方向全覆盖检查，相关 4 项通过。独立库首次完整测试 138 通过 / 5 失败，其中 4 项为上述旧快照断言；另 1 项 `test_record_completion_has_no_pending_families` 因旧构建器读取 style@2.0 中不存在的 annotation_status，发布前备份亦缺该旧字段，不属于本次正面变更。Task3.2 完整测试被环境缺少 bpy 阻挡；局部 open-mesh 7 项通过，2 条既有 pytest 配置警告。未宣称完整测试全绿。

## 发布之后的复判

先发布，再新增结构信息更丰富的多角度透视图。复判把“正面识别可靠性”与“上下方向可靠性”分开，不再仅因 up_confidence 低就抹去已识别正面；仍检查轴和视图映射，不直接把斜视相机方向当作资产正面。先用柜子、床、办公椅及无唯一正面的对照样例验证，未验证的新结果不覆盖本次可用方向。
