# 完整共享资产库：补迁与验收

用户要求完整交付，同学不得依赖私人工作区。前次embedding迁移只覆盖模型、图片和向量，不能视为完整标注交付。本次以原库31,350条注册记录为范围，包括未进入30,830条embedding集合的资产；不存在的源文件不伪造，单列SOURCE_GAPS。

## 共享入口

目标库根：`/data/task3_2/share_data/scenesmith/all_assets_style_v2_delivery_20260909/asset_library/`。

包括完整lookup、正面/功能方向、精确关系、原自由度编码、环境锚点、净空及伙伴白名单引用、物理/材质/质量、自发光、可动性、affordance记录/点云mask、操作空间、风格overlay，以及消费API、schema、构建脚本和文档。原始来源/置信度/方法标记保留，不把规则估计或canonical fallback包装成真实测量。

HSSD完整affordance文件从实际unified层补取，不能使用filtered仓库的少量样本冒充全库；其他来源聚合mask一起迁移。3D-FUTURE已有可访问的共享完整mask，直接绑定共享路径，不重复复制大体积文件。611条可动替换记录使用251个去重PartNet-Mobility目录，连同mobility.urdf及引用网格完整复制。embedding静态输入mesh与可动替换realization保持各自来源，不互相冒充。

已有30,830条共享模型和向量复用；其他有几何但缺图的注册资产也补迁几何。几何缩放、坐标轴、texture引用和点云引用与实际共享文件一起绑定。原始标注lookup另外保留在annotation_migration/original_lookups，历史provenance文本可能记录旧来源路径，但不能成为当前API读取必需路径。

## 使用方式（等待验收完成后）

```bash
export PYTHONPATH=/data/task3_2/share_data/scenesmith/all_assets_style_v2_delivery_20260909/asset_library
python - <<'PY'
from hssd_asset_library import AssetLibrary
root='/data/task3_2/share_data/scenesmith/all_assets_style_v2_delivery_20260909/asset_library/data'
library=AssetLibrary(data_root=root, sources='all')
asset=library.require('3dfuture:1460b4ef-2317-347a-82d9-ae24ea0083be', expand=True)
print(asset['geometry_resolved'])
print(asset['affordance_full'])
print(asset['asset_style'])
PY
```

不要求访问/data/250010098/codex_communication。不要求取得API密钥，不复制任何凭据。`asset_style`采用本次embedding已交付的风格快照，保留needs_review等状态；完整迁移不意味着每个标签都已经确认。

## 成功门槛

当前进度：交付根 `annotation_migration/STATUS.json`。**只有stage=complete才可声称全部消费验收通过。** 复制成功不等于可读，API有字段不等于文件存在。

一次性收尾程序用UID65534调用全部31,350条 `require(..., expand=True)`，核查关键字段、共享模型/URDF、mask读取、操作空间及风格overlay。结果 `NON_OWNER_FULL_API_AUDIT.json`。源缺失记入 `SOURCE_GAPS.json`；任何未解决缺口保留audit_gaps_require_review，不放出无条件完成标记。

按用户明确要求，本次交付子树最终权限为777。其他已有数据目录不扩大权限；未停止Luna、生成、渲染或CCI服务。
