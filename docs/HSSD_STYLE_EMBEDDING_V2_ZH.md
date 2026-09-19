# HSSD 新风格规范的 embedding 接入与迁移

> 历史记录：下文“未运行/服务不可达”为 2026-09-08 准备阶段状态，不是当前状态。2026-09-09 已完成包含 HSSD 的 30,830 条全来源 GPU 向量；当前路径、验证和交付以 [全库交付说明](ALL_ASSET_EMBEDDING_V2_ZH.md) 为准。

更新日期：2026-09-08。范围：上次同学要求的“将已确认风格加入 Qwen3-VL 多模态 embedding 文本”，不是另建 CLIP 风格分类器，也不是训练 embedding 模型。

## 本次完成与边界

已更新隔离副本的索引接入、严格校验与真实标签输入快照。没有改正式向量库，没有重新计算全库向量，也没有证明检索指标提升。CPU 当前缺少 `zvec`，默认 `http://127.0.0.1:8014/props` 不可达；没有因此占用 g 区生成卡或停止任何后台任务。

需要更正此前“上次只做旧48类”的说法：2026-08-31原型已接受 `asset_style@2.0` 一级/二级路径，但缺少严格本体校验，只以合成标签和确定性假向量服务验证了接线。这次使用真实双轮裁决结果，旧48类不能混入当前一级标签。

## 风格规范

本体 `bonn_furniture_styles_hierarchical@1.0`，一级17类：

`asian, beach, contemporary, craftsman, eclectic, farmhouse, industrial, mediterranean, midcentury, modern, rustic, scandinavian, southwestern, traditional, transitional, tropical, victorian`。

默认一个一级标签；原标注契约允许有依据的最多三个路径，不凭 embedding 相似度添加新标签。

二级只保留数据集原生标签，不能由视觉模型猜测。目前11类及唯一父类：

| 一级 | 允许的原生二级 |
|---|---|
| asian | japanese, korean, southeast_asian, new_chinese, ming_qing, chinoiserie |
| contemporary | light_luxury |
| farmhouse | american_country |
| modern | minimalist |
| traditional | neoclassical, european_classic |

二级还需匹配本体登记的 `source_dataset` 和 `source_label`。当前这些二级来源为 3D-FUTURE；它们是通用适配器支持范围，不代表 HSSD 本次真实快照拥有这些原生标签。

## 进入 embedding 的数据

只接受 `source_metadata` 或 `visual_reviewed`。校验 schema、本体、confidence、证据类型、一级/二级父子关系、视觉证据ID及 agreed 裁决元数据；不合法的“已确认”记录直接报错，不悄悄降级或忽略。

`needs_review`、`unlabeled`、第三轮单次候选、旧 embedding 排名、ABO 源图片候选均不转换为正式风格。未确认不等于 modern，也不等于 eclectic。没有确认风格的资产仍可按原图像和元数据编码，只是不附加风格句子。

支持40位十六进制 HSSD ID、`hssd:` 前缀，以及263个原库 `xxxx…` 格式ID；前缀别名若出现不同已确认路径则报错，不覆盖其中一条。

注入位置不变：同一份 `build_asset_content()` 进入 multi-image 和 single-view prompt，也保存在 Zvec `content` 字段。例如真实资产 `0001fb06b075a743e6289236cf049df3ad5dfa9c` 在本快照中加入：

```text
Verified styles: industrial.
```

合法原生二级的格式示例（仅格式，不是上述HSSD资产标签）：`Verified styles: traditional > european classic.`

不是给原向量末尾拼 one-hot；修改模型输入文本需要重新调用同一个 embedding 模型。维度、向量空间、图像预处理和查询侧模型必须一致。不要把新旧向量在同一个集合中无标记混用。

## 本次真实快照

目录：`/data/250010098/codex_communication/3_scenesmith_asset_relations/feedback/experiments/hssd-style-embedding-v2-20260908/full_ids/`。

- `hssd_confirmed_styles_v2.json.gz`：10,963条HSSD记录；6,347条有确认风格，4,616条无可注入的确认风格。
- `style_text_inputs.jsonl`：6,347条资产ID、风格文本和路径。
- `manifest.json`：输入路径/哈希、本体哈希、计数、限制说明。

以原 source-first overlay 为底，复用原项目 `merge_visual_reviews()`，只在明确待视觉复核的记录上接入清洁的两次裁决。source_metadata 不被覆盖。读取 results 和4个 results_multikey shard；不读取 round3 单轮结果。发现293个跨输入文件记录内容不一致的ID（并非都表示风格标签不同），保守排除这些裁决，不擅自选择一个版本；因此这份输入快照不能等同于门户采用其他去重策略的计数。

上层目录的初次快照只含10,700个十六进制ID，是发现ID兼容问题时的中间产物；消费时必须使用 `full_ids/`，不能使用上层同名文件。它们保留作修复证据。

快照不是新的 canonical 标签发布，也不是三维模型质量验收。原始图片、裁决日志和正式 overlay 未改。

## 代码与重建

仓库根：`/data/250010098/codex_communication/3_scenesmith_asset_relations/feedback/repos/Task3.2-style-zvec-test`。

- `scripts/style_embedding_v2.py`：无模型依赖的输入校验与ID连接。
- `scripts/style_v2_contract.py`、`configurations/style_ontology_bonn_v1.json`：从现有资产库原样复制的契约与本体；本体 SHA256 为 `0b7692f0ca5d6881c19252425fb68343d2fb5a022674678b6de6ac960ad77d94`。
- `scripts/prepare_style_embedding_snapshot.py`：只读现有标注，写入新的隔离快照；输出目录必须不存在，避免覆盖历史快照。
- `scripts/index_rendered_hssd_assets_zvec.py`：既有 `--style-annotations` 参数现在使用严格适配器；硬保护正式 `/data/task3_2/share_data/scenesmith/hssd_zvec_collection` 路径。

准备新快照：

```bash
cd /data/250010098/codex_communication/3_scenesmith_asset_relations/feedback/repos/Task3.2-style-zvec-test
python scripts/prepare_style_embedding_snapshot.py \
  --annotation-root /data/250010098/codex_communication/3_scenesmith_asset_relations/feedback/repos/hssd-annotations-filtered \
  --reviews-root /data/250010098/codex_communication/_shared/style_v2_luna_full \
  --output /data/250010098/hssd_style_embedding_next_snapshot
```

下面是依赖和真实 Qwen3-VL-Embedding 服务恢复后的执行入口，本轮**未执行**。先确认输出集合不存在，再创建独立集合；禁止使用原正式库，禁止向合成 smoke 向量集合继续写入。

```bash
python scripts/index_rendered_hssd_assets_zvec.py \
  --render-root /data/task3_2/share_data/scenesmith/hssd_rendered_assets \
  --style-annotations /data/250010098/codex_communication/3_scenesmith_asset_relations/feedback/experiments/hssd-style-embedding-v2-20260908/full_ids/hssd_confirmed_styles_v2.json.gz \
  --collection-path /data/task3_2/share_data/scenesmith/hssd_zvec_collection_style_v2_20260908 \
  --base-url http://127.0.0.1:8014 \
  --embedding-dimension 2048
```

此命令沿用脚本默认元数据路径；正式重算时还需按实际数据位置传 `--preprocessed-path` 和 `--hssd-root`，否则名称/WordNet/mesh路径可能缺失。已有渲染不一定覆盖10,963项，最终要报告实际发现数、缺图数、成功向量数和失败数，不能用标签数代替向量数。

## 验证与下一步

17项定向测试通过：新本体、二级来源、视觉二级禁用、候选排除、别名冲突、xxxx ID、文本构造、真实本地HTTP请求边界。HTTP测试返回的是明确标识的假向量，仅验证请求中的风格和图像标记；未安装假 zvec，也未写任何向量集合。pytest有两条现存asyncio配置警告，不影响这组测试。

```bash
python -m pytest --confcutdir=tests/unit \
  tests/unit/test_style_embedding_v2_contract.py \
  tests/unit/test_index_rendered_hssd_assets_zvec_style.py -q
```

使用 `--confcutdir` 仅绕过无关的全仓Blender/Drake初始化，这组测试不声称验证渲染。下一步是在可用真实embedding运行时先做少量真实图像的新集合写入/重开验证，再重算全库；最后用固定查询集比较无风格/新规范风格的 Recall@K、nDCG 和类别相关性，才能判断检索收益。
