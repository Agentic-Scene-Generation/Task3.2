# 全资产库新风格 embedding：阶段交付说明

> 完整本体标注补迁见 [完整共享资产库](FULL_ASSET_LIBRARY_ZH.md)。模型/向量迁移完成不代表完整标注已交付，最终以 annotation_migration/STATUS.json 与普通账号全库API验收为准。

> 2026-09-10 共享访问修复：原向量库权限已开放只读；3D-FUTURE已有共享原始数据，其他来源正后台迁移。请先读 [共享权限与资产路径修复](SHARED_ACCESS_FIX_ZH.md)，以交付根 ACTIVE_INDEX.json 和迁移 STATUS.json 为当前入口，勿把尚未完成的新索引当作可用库。

更新：2026-09-09。本文件是当前状态入口；[HSSD 规范说明](HSSD_STYLE_EMBEDDING_V2_ZH.md)保留早期准备阶段历史。

## 完成情况

已用现有 Qwen3-VL-Embedding-2B-Q8_0 在 e-2 四张 H100 上完成 **30,830 条真实、2048 维图文向量**。不是训练模型，也不是把风格 one-hot 拼到旧向量末尾；新规范风格文本与已有图像一起重新编码。全量只读重开核验：所有向量有限且单位化、全部字段匹配清单、完成记录无重复且覆盖全部 ready 输入。

| 来源 | 已完成向量 |
|---|---:|
| HSSD | 10,843 |
| 3D-FUTURE | 14,823 |
| ABO 原有三维模型 | 1,113 |
| GSO | 1,030 |
| Objaverse | 2,490 |
| Poly Haven | 395 |
| Smithsonian | 57 |
| YCB | 79 |
| 合计 | 30,830 |

输入快照共 31,903 条：另外 102 条缺几何、418 条缺图片、553 条 Hunyuan 候选缺实际生成 mesh 渲染。这 1,073 条未生成向量，不能用输入照片替代生成模型渲染。已有库注册资产为 31,350 条，生成候选另列；未生成的 ABO 二维队列不属于此索引。

## 本地交付位置

- **向量集合**：`/data/task3_2/share_data/scenesmith/all_assets_style_v2_gpu_compact_20260909`，约 628 MiB（本次 du 统计）。同级 `.input.json` 为输入/模型签名，`.done.jsonl` 为完成及图片哈希记录。
- **交付包及本文**：`/data/task3_2/share_data/scenesmith/all_assets_style_v2_delivery_20260909/`；本文位于 `docs/ALL_ASSET_EMBEDDING_V2_ZH.md`。
- 交付包 `queue/` 为完整输入和统计，`styles/` 为冻结风格 overlay 和来源统计，`evidence/` 为 GPU 一致性、全量索引验证和查询 smoke 证据，`ACTIVE_INDEX.json` 指向当前集合。
- 独立运行时：`/data/250010098/asset_embedding_service_20260909/runtime/bin/python`。运行维护入口在同根 `README.md`、`gpu_tick.sh`；索引已完成，`GPU_COMPLETE` 阻止重复全量下发。
- GitHub 已交付：`Agentic-Scene-Generation/Task3.2` 的 `dev_yz_0910`，从 `dev_hrk_week37` 的 `6039165a65772b041d92b56a592559968818150b` 新建，使用用户确认的 `K-Chronofox` 身份推送成功。此前默认 SSH 误选 Shphd3 导致拒绝，现已解决；未修改其他任务的全局认证配置。只提交代码、测试、规范和小型核验证据，不上传模型、图片、向量库或凭据。本地仓库为 `/data/250010098/codex_communication/3_scenesmith_asset_relations/feedback/repos/Task3.2-dev-yz-0910`。

集合中的 mesh/image 仍引用原有本机绝对路径；交付包不是独立可搬迁的完整资产压缩包。搬到其他主机需挂载或显式重映射原始资产目录。没有覆盖旧正式 `hssd_zvec_collection`，也没有改场景生成默认配置。

## 风格及图像契约

采用 17 个一级风格及 11 个仅限来源原生标签的二级，详见 HSSD 规范说明和 `configurations/style_ontology_bonn_v1.json`。只注入严格通过校验的 `source_metadata`、`visual_reviewed`；不注入旧 48 类 CLIP 排名、needs_review、round3 单轮候选。不对无标签资产默认补 modern。

本快照 21,835 条资产有确认风格，9,515 条没有；这是注册库标签统计，不等于已完成向量中的风格数量。清洁双轮裁决与来源标签按 source-first 合并，1,444 个内容冲突的重复裁决保守排除，不覆盖 canonical 发布数据。

优先使用已有真实 mesh 多视图；3D-FUTURE 可使用明确标记 `dataset_product_preview` 的原生 `image.jpg`，不是统一 mesh 渲染。不会使用纹理色卡。完整命名空间 UID 保存在 `asset_id`，Zvec 文档 ID 为其 SHA256；例如 `3dfuture:<uuid>`、`others:abo:<id>`。不要把来源当成 object_groups，也不要把 SHA 当原资产 ID。

## 模型与复现边界

模型 ID：`Qwen3-VL-Embedding-2B-Q8_0_llamacpp_0b5be7e4_cuda12_sm90`；llama.cpp 原生 `/embeddings`，last pooling、L2 normalize=2、ctx=16384、batch/ubatch=2048、每 GPU 一个副本。已验证副本入口为本机 SSH 转发 `127.0.0.1:8020` 至 `8023`，端口依赖远端容器和现有维护脚本，不能假定永久可达。

四 GPU 副本对三种真实来源输入的余弦一致性约为 1；CPU/GPU 最低余弦 0.997116，未通过预设 0.999 混用门限。因此最终集合全部为 GPU 向量，旧 CPU 集合独立保留，不能与它无标记混用。查询也应使用同一 GPU 后端；传入 model-id 仅校验声明与索引签名，不是远端模型密码学认证。

输入 JSONL SHA256：`8983ccd27ea00e8816fbdef5591df08ab83a9e218a1b3c34c8b68f1136567263`。

## 直接查询

从本分支仓库根执行，返回完整 UID、mesh 路径、图片与内容：

```bash
/data/250010098/asset_embedding_service_20260909/runtime/bin/python \
  scripts/query_all_asset_embeddings.py \
  --collection-path /data/task3_2/share_data/scenesmith/all_assets_style_v2_gpu_compact_20260909 \
  --base-url http://127.0.0.1:8020 \
  --model-id Qwen3-VL-Embedding-2B-Q8_0_llamacpp_0b5be7e4_cuda12_sm90 \
  --text 'A Scandinavian wooden chair for a dining room' --top-k 3
```

旧 `HssdZvecSearcher` 依赖 HSSD 的 ID/WordNet 和 lookup，**不能只换 collection_path 就宣称完成全来源场景接入**。本次交付全来源离线索引和独立查询入口；场景主链的跨库资产加载适配仍是后续工作。

## 重算、续跑与测试

`prepare_style_embedding_snapshot.py --all-sources` 从外部已有标注项目（`--annotation-root`）读取 canonical 合并器，不要求将原始标注日志上传本仓库；`prepare_all_asset_embeddings.py` 构建所有来源统一输入；`index_all_asset_embeddings.py` 消费 ready 项。参数见各脚本 `--help`。消费已有集合无需重新准备快照或重算。

索引以四个有界请求并行、单写者、每 256 条 flush，成功持久化后才原子更新 ledger。续跑校验输入签名、已完成图片哈希；输入/模型改变需新集合，不静默混写。崩溃落在 flush 与 ledger 之间时同 ID upsert 可重放。禁止删除或覆盖旧正式集合。

已验证环境：Python 3.10、zvec 0.7.0、numpy 2.2.6、pytest 9.1.1，另需 Pillow、tqdm。week37 基线移植后以下 27 项聚焦测试通过：

```bash
python -m pytest --confcutdir=tests/unit \
  tests/unit/test_all_asset_embeddings.py \
  tests/unit/test_style_embedding_v2_contract.py \
  tests/unit/test_index_rendered_hssd_assets_zvec_style.py -q
```

有两条现存 asyncio 配置警告；`--confcutdir` 避开无关 Blender/Drake 初始化，不代表全仓场景生成测试通过。小型证据见 `docs/evidence/embedding_v2_20260909/`。全量重开检验和真实文本查询只证明数据与接口可用，尚未做固定查询集 Recall@K/nDCG 对照，不能宣称检索质量提升。
