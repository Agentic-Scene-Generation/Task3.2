# 现有标注同步（2026-09-14）

本次是现有结果发布，不是新一轮补标或语义正面全量复验。

## 内容与边界

- 非生成库31,350个：HSSD 10,963、3D-FUTURE 14,823、小库5,564。
- 最新确认风格29,911个，尚无明确风格1,439个；保留有效判断次数与五次上限记录。
- HSSD语义正面6,902/fallback4,061；3D-FUTURE语义正面11,060/fallback3,763；小库语义正面3,396/fallback2,168。语义标记可能来自源约定、类别或几何规则，不等于全部逐资产视觉验证。
- 生成库3,649个，仅发布已有尺度/输入图风格标注及几何/渲染准备证据。front、DOF、relation、功能mask、净空等尚未完成，记录始终保持annotation_complete=false，未混入非生成完整库。
- 本次没有重算embedding。已有向量不代表新的风格描述已编码。

## 三处交付

1. 独立仓库 `K-Chronofox/hssd-annotations`：`data/`下三份lookup已内嵌最新asset_style；配置加载`asset_style_annotations_v2.shared.json.gz`。生成已有标注单列`data/generated_annotations.json.gz`。
2. `Agentic-Scene-Generation/Task3.2` 的 `dev_yz`：先保留原dev_yz历史合并`dev_hrk_week37`（5db84280e665dd091328ce7027f46fc384faa385），再同步到`scenesmith/scenebenchmark_critic/asset_annotation_data/`。不是改dev_yz_0910。
3. 共享发布快照：`/data/task3_2/share_data/scenesmith/all_assets_style_v2_delivery_20260909/annotation_release_20260914/`。各副本用`ANNOTATION_RELEASE_20260914.json`校验数据哈希。

远程是否推送成功以本次交付回执及远端SHA核验为准；本说明不单独充当推送成功证据。

## 本机完整消费入口

现有主入口不变：

`/data/task3_2/share_data/scenesmith/all_assets_style_v2_delivery_20260909/asset_library/data/`

从共享`asset_library`导入`AssetLibrary`，使用`sources="all"`和`require_unified(uid, expand=True)`读取完整已有标注及mask引用。本机消费使用该data_root（其几何/完整依赖已改为共享绑定），**不要用仓库旧config里的私人目录绑定**。

仓库保留原portable几何/affordance相对引用；Git不上传mesh、NPZ大包或API请求日志。换机器需要重新绑定数据根，不应误以为Git clone包含所有大文件。共享发布快照用于版本冻结，不替换当前共享API的路径配置。

Task3.2读取四类已有记录：

```python
from scenesmith.scenebenchmark_critic.current_asset_annotations import get_current_asset_annotation
record = get_current_asset_annotation("3dfuture:00030133-9c36-412d-bb48-6123ea2da899")
front = record["canonical_front"]
# HSSD可用hssd:<id>或裸ID；Others使用others:<dataset>:<id>；生成使用generated:<asset_id>。
```

原HSSD消费类继续读取原文件位置，已有接口保留。新多来源读取器不把未完成生成物伪装成完整record，也不自动把它们加入场景生成。
