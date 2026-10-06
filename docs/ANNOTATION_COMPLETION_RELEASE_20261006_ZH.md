# 全库标注补全发布（2026-10-06）

本次发布已有补标成果，不启动新标注、不改变正面或风格分类、不把规则候选升级为独立验收。

## 资产与入口

全库真实资产 **37,186** 件：HSSD 10,963、3DFuture 14,823、Others 5,564、生成 5,836。生成原始索引另含 2 件 gcheck 测试项，不计入库存。

共享根目录：`/data/task3_2/share_data/scenesmith/all_assets_style_v2_delivery_20260909`。正式消费目录为 `asset_library/data/`。全库合并视图为 `formal_annotation_completion_20261004/all_asset_formal_annotations.json.gz`；它约 120MB，超过普通 Git 单文件上限，故保留共享原文件，Git 使用完整四来源 lookup 加风格 overlay，并提供相同证据引用。

此次冻结快照：`annotation_release_20261006/`；校验清单：`ANNOTATION_COMPLETION_RELEASE_20261006.json`。独立仓库的数据目录为 `data/`；Task3.2/dev_yz 为 `scenesmith/scenebenchmark_critic/asset_annotation_data/`。`CURRENT_ANNOTATION_RELEASE.json` 保留现有 front release 并增加本次补全清单。

## 回填内容与实际状态

已包含 10月4日的风格五轮履历、unknown终态来源、类别补回、实际几何/材质扫描及协议修复，以及10月6日的真实交互掩码、公开读取引用修复、尺度候选GLB、物理先验、净空与操作空间候选、掩码视觉审计记录。

| 生成库项目 | 已有正式字段/记录 | 尚缺 |
|---|---:|---:|
| affordance掩码引用 | 5,445 | 391 |
| 尺度、物理、净空、操作空间候选 | 4,262 | 1,574 |
| 掩码视觉审计记录 | 5,018 | 818 |

尺度缺口为比例冲突442件、无法确定尺度1,132件。首批补跑已结束，部分提案和视觉请求耗尽最多3次尝试。cron监督仍在，但不等于剩余失败项仍有有效推进。全部生成资产仍 `annotation_complete=false`，独立完整语义验收通过数为0；视觉unsupported/uncertain不自动放行。已有字段也可能需修复。

类别估计尺度不是实测尺寸；物理参数不是实测质量；静态mesh不证明真实物体没有关节；稀疏mask点投影无精确遮挡过滤。正面相关的候选空间仍可能需要人工/视觉确认。非生成HSSD另有611件关节资产操作空间待补，8,934件静态资产缺该引用属不适用。

风格当前明确34,568、unknown/fallback2,618。unknown低检索优先级，保留最多五轮实际履历；生成目标风格按用户授权计入库存，不强制Luna再判。

同步修复消费入口：Task3.2读取最新风格overlay而非lookup内的旧风格；独立库将生成资产裸ID绑定为标准UID，并同步共享source_registry/config。生成库仍不纳入默认已验收all范围，需显式选择generated或registered；没有放宽验收门。

## embedding

最新活跃入口：`/data/task3_2/share_data/scenesmith/ACTIVE_INDEX.json`；37,186件，10月3日批次，10月4日本地发布。本次同步其索引指针及核验信息，**没有重新计算向量**。新增/补回34件类别文本尚未刷新；此次新米制候选mesh路径没有自动改写已有embedding路径。原mesh保留，现有路径仍可访问。

两个Git仓库同时新增 `data/embedding_release_20261003/portable/`（records、19片实际向量、manifest），及 `data/ACTIVE_EMBEDDING_INDEX.json`。这是37,186件的真实可移植向量包，不只是路径说明；19片逐文件SHA、形状、有限值和UID去重核验通过。旧embedding包保留，不删除。

## 证据与权限

Git内 `completion_20261006/` 收录本次字段计数、尺度测量、Stage2最终报告、消费者抽验和逐资产字段缺口清单。完整NPZ/GLB/渲染及逐资产审核证据在共享根的 `generated_completion_20261006/` 与 `formal_annotation_completion_20261004/`。每条新正式引用均指向共享路径，不依赖私人目录。新增交付目录777、文件666；凭据不外发。

本次冻结文件SHA逐项一致。三处发布回执见项目feedback的 `ANNOTATION_PUBLICATION_20261006_ZH.md`；历史文档中的旧计数不作为当前状态。
