# 风格、主题色与材质正式覆盖发布（2026-11-12）

## 正式入口

正式消费目录：/data/task3_2/share_data/scenesmith/formal_current/annotations/

本轮覆盖 **38,497件真实资产**：HSSD 10,963、3D-FUTURE 14,823、Others 5,564、生成7,147。
2件gcheck测试项保持原状。共享正式目录的四来源lookup均内嵌新标签。
Git中的3D-FUTURE采用完整基础lookup与appearance overlay组合存储，
由正式读取器合并成相同的完整记录，避免单文件超过GitHub的100MB限制。

风格overlay沿用兼容文件名 asset_style_annotations_v2.shared.json.gz，
其内容现为 **asset_style@3.0**，包含所有四来源。当前词表为
style_ontology_user_photo9_v1.json，config.json 的 style_ontology 指向该文件。
旧Bonn词表与历史快照保留供解释旧记录使用。

## 九类风格、多标签及判定来源

| 一级 | 二级初始库 |
| --- | --- |
| 现代 | 简约、黑白灰、极简、奶油、轻奢 |
| 中式 | 传统中式、新中式、宋代中式 |
| 欧式 | 简欧、北欧、田园、中古 |
| 美式 | 乡村田园、复古、简约、轻奢、中古 |
| 日式 | 传统、原木、侘寂 |
| 法式 | 轻法、奶油、田园、复古 |
| 地中海 | 蓝白系、黄褐系 |
| 东南亚 | 传统、南洋复古 |
| 混搭 | 暂无固定二级 |

共有9个一级、22个共享二级概念、28条父子边。风格是多标签参考：
asset_style.styles 与 style_references 保存完整一级/二级配对，style_label_ids 提供一级ID。
同一“奶油”等二级词可出现在不同一级下，不能丢失父子配对。

已有标签的Luna语义转换与输出模型多视图观察分别标记 existing_label_mapping、
output_visual_observation。文字已有充分依据的标签保持source-reference态。
两种依据不同则保留各自来源，并写review flag。

本轮视觉处理完成 38,385件；
未完成状态为 {"failed": 8, "evidence_wait": 104}。
无鲜明风格、图像证据不足、词表未覆盖、图像不可用分别记录。
unknown_style是状态fallback，其检索优先级为1；有风格参考时为0。
旧风格判定次数保留，新九类体系视觉判定次数单列；
本轮最多一次有效视觉判别，未知不意味着新词表下已经判满五轮。

本轮正式风格状态：
{"source_reference_only": 7275, "visual_reference_available": 20459, "no_distinctive_style": 9039, "outside_schema": 752, "insufficient_evidence": 564, "unusable": 363, "needs_visual": 45}

## 主题色

theme_color记录颜色ID、主色/辅色/点缀色、作用部件与视图依据；
theme_color_label_ids提供轻量标签。采用14种基础颜色加gold/silver外观词，
不填写未经测量的RGB、HEX或面积占比。透明、多色和证据不足各自独立记录。

{"observed": 32287, "multicolor": 3580, "insufficient_evidence": 465, "unusable": 2030, "not_processed": 112, "transparent": 23}

无法获取可靠主题色时，状态及具体原因保留，轻量标签使用unknown_color。

## 材质

asset_material是语义材质与表面外观记录，material_label_ids提供轻量标签。
具有可靠源材质的资产保留源参考；缺参考的资产使用输出模型渲染补录。
视觉观察标记visual_inferred，并保留appearance、part和view_ids；
木纹外观、底材组成、真实制造材料分别表述。

{"observed": 16220, "source_reference_retained": 14354, "insufficient_evidence": 6810, "unusable": 1004, "not_processed": 109}

没有可靠语义材质时使用unknown_material并保留原因与原材质参考。
asset_physics中的material、质量、摩擦、密度等原物理参数全部保持不变。
现有AssetLibrary.search(material=...)仍按原物理材质字段过滤；
新的语义材质从material_label_ids及asset_material读取，两种语义不混用。

## 消费方式

AssetLibrary.get/require返回三组正式字段；
get_unified/require_unified也包含theme_color、asset_material及轻量标签。
search(style_level1=..., style_level2=...)使用当前配置词表，
支持新九类与共享二级；search返回摘要同时带风格、主题色和材质ID。
旧asset_style@2.0消费兼容仍保留，v3采用独立验证器hssd_asset_library.style_v3。

Task3.2的get_current_asset_annotation从四来源lookup读取主题色和材质，
非生成风格继续读取authoritative overlay，生成风格从已更新lookup读取。
重新创建读取器或清除既有缓存后加载新数据；本次没有重启任何使用者服务。

## 核验、追溯与发布范围

发布清单：STYLE9_PALETTE_MATERIAL_RELEASE_20261112.json。
完整逐资产外观标注、冻结队列和未完成清单位于：
/data/task3_2/share_data/scenesmith/style9_palette_material_release_20261112/

四来源lookup序列化前后逐资产比较：
Front、实际mesh、尺度、物理参数、净空、relation、自由度、affordance等非本轮字段保持不变。
旧风格存于asset_style_previous_20261112；覆盖前正式文件存于previous_formal/。
每条结果保留模型、输入fingerprint、结果SHA、提示词SHA和真实渲染证据。
为避免重复大段证据导致仓库文件超过100MB，风格记录保存主证据引用，
主题色和材质的provenance通过record_json_pointer引用同一资产的asset_style.provenance。
完整逐视图与mesh/贴图hash保存在result_path中，由result_sha256绑定；标签及观察依据没有删减。
2件早期视觉观察先于原风格字段恢复，
已重新核对同一UID、材质任务标志、实际mesh及资源hash后复用，
同时保留新旧输入fingerprint、mapping group和显式来源标记。

共享交付位置为上述正式目录；独立仓库使用data/；
Task3.2/dev_yz使用scenesmith/scenebenchmark_critic/asset_annotation_data/。
两仓库同步四来源完整标注、风格与外观overlay、当前词表、配置、发布清单及本文档；
其中3D-FUTURE基础lookup保留发布前的非外观字段，所有新正式字段都在
asset_appearance_overlay_20261112.json.gz中，config.json的appearance_overlay指向该文件。
AssetLibrary与Task3.2读取器自动合并，无需使用者手动拼接。
共享内嵌版本与Git组合版本逐资产验证逻辑结果一致，3D-FUTURE基础文件字节本身不同。
大文件JSONL、实际mesh、完整渲染证据保持共享路径引用。
新增交付目录777、文件666，公开标注不含私人目录依赖或凭据。

## embedding状态

本轮仅覆盖发布标注，**没有重新计算或切换embedding**。
现有向量仍对应旧风格文本，当前标注指针已标记embedding_style9_metadata_pending。
新标签已能通过正式标注读取；基于新标签的向量检索需要后续刷新。

## 归档状态

/mnt/aoss2挂载返回Transport endpoint is not connected，本次未向桶写备份。覆盖前文件完整保留在本轮发布目录previous_formal/，本次未重挂载或重启容器。
