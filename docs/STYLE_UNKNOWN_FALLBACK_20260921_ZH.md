# 未识别风格：显式标签与检索降级

用户要求：不能留空，也不能随便赋一个真实风格；降低检索优先级，不篡改置信度。

共享非生成库31,350条已添加非空 `style_label_ids`：其中29,911条保留真实风格ID，1,439条为 `["unknown_style"]`（中文“风格未识别”），HSSD906、3DFuture5、Others528。新增 `style_is_fallback` 和 `style_fallback` 保存原因；5轮用尽与仍待判别分开。原始styles、confidence、判定次数、证据、style_status不改。未知不等于风格中性，也不是第18种Bonn风格；17类统计和稀缺生成调度仍只读真实styles，不把unknown当稀缺类。

消费时展示 `style_label_ids`，不要再把只存真实本体路径的 `styles=[]` 当成没有任何标签。后续有真实风格证据时，读取器自动移除fallback并恢复正常优先级。

## 检索

- `style_retrieval_priority_tier=0`：明确风格；`=1`：unknown fallback。数字越小越优先。
- 共享AssetLibrary查询在截取limit前将fallback排后；不删除资产。明确请求某种真实风格的严格筛选不把unknown冒充匹配。
- 共享 `scripts/query_all_asset_embeddings.py` 先召回5倍候选（扩展上限1000），候选池内按tier稳定重排，再取top-k；同tier保持向量原排序，原score不变。它不是全索引全局重排。
- 此次策略覆盖非生成库；向量返回若不在该overlay（例如部分生成资产），标记 `style_priority_policy_applied=false`，不凭缺映射误认没有真实风格，不改其原有优先级。
- 未重算embedding，未修改索引向量。初次共享更新后，本次dev_yz提交同步了标签overlay、独立排序helper和查询脚本。共享新进程/脚本会读取新规则；已缓存旧代码或旧数据的常驻服务需重载，不能宣称其他人自定义检索器已自动生效。

数据：`asset_library/data/asset_style_annotations_v2.shared.json.gz`。
变更记录、哈希和备份：`asset_library/data/STYLE_FALLBACK_POLICY_20260921.json`。
实现：`asset_library/hssd_asset_library/style_fallback.py`、`store.py`、`scripts/query_all_asset_embeddings.py`。
生成和Luna正面复判未被重启或修改。

## 验证范围

新增行为测试先失败后通过；fallback/风格schema定向测试12通过。测试覆盖非空标签、不改置信度、limit前降级、保留unknown可检索、严格真实风格筛选、候选池稳定重排及后续新证据解除fallback。向量请求脚本完成编译检查；未新增付费embedding请求，未声称在线服务端到端已重载。

共享库完整测试155通过、5失败；不能称全绿。失败为本轮未修改的front/affordance数据与旧快照断言不一致：`test_wardrobe_fronts_are_asset_verified`、`test_canonical_front_verified_present`、`test_every_asset_has_canonical_orientation_axis`、`test_semantic_direction_release_contract`、`test_affordance_absent_degrades`。本轮未为让这些断言通过而改回旧front或移除真实affordance。
