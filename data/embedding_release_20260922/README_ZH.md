# Embedding 正式发布（2026-09-22）

本次发布实际 34,969 条、2048 维 float32 向量，不是仅提供索引路径。包含非生成及已具备输入条件的生成资产；并不代表全部资产均已具备 embedding。输入采用冻结快照，不纳入尚未完成的 Front 复判。

## 本地使用

正式入口：`/data/task3_2/share_data/scenesmith/ACTIVE_INDEX.json`。
当前 collection：`/data/task3_2/share_data/scenesmith/embedding_release_20260922/restored_index`。
资产路径均使用共享目录，不能把本机绝对路径理解为 Git 下载后自动存在的 mesh。

查询时使用仓库 `scripts/query_all_asset_embeddings.py`，传入上述 `--collection-path`、`--base-url http://127.0.0.1:8020`、`--model-id Qwen3-VL-Embedding-2B-Q8_0_llamacpp_0b5be7e4_cuda12_sm90` 和 `--text`。服务地址是本机 GPU 转发入口，其他机器须自行配置同模型/后端服务；不能混用 CPU 或不同模型向量。

## Git 交付与重建

两个仓库均发布 `data/embedding_release_20260922/portable/`，其中包含真实向量 `.npy` 分片、元数据 `records.jsonl.gz`、校验清单 `manifest.json`。单个分片约 16 MiB，无须 Git LFS。

在装有 numpy、zvec 的 Python 环境运行：

```bash
python scripts/export_embedding_release.py verify data/embedding_release_20260922/portable
python scripts/export_embedding_release.py restore data/embedding_release_20260922/portable /your/shared/new_index
```

目标目录必须尚不存在。重建同时生成 `.input.json` 模型签名；查询须显式传入自己的 collection 路径。资产本体仍从共享资产库获取，向量包不包含 mesh/纹理本体。

## 验收与范围

- 重建后 34,969 条向量与字段逐条一致，9 次 ANN 探测通过，见 `REBUILD_VERIFIED.json`。
- 实际 GPU 文本查询 `wooden dining chair` 返回 Dining Chair；非所有者 nobody 可打开 34,969 条记录。
- Task3.2 dev_yz 包含当前最新 HRK Week 分支 dev_hrk_week37 的 e65b5c9（Front 清单对齐修正）。
- 旧 collection `all_assets_style_v2_gpu_shared_20260910` 保留用于回退；未删除旧数据，未替换在途 Front 复判。
- 本次不发布尚在开发的人工审计网页。
