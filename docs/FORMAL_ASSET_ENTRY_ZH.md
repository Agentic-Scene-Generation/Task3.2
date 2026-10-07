# 正式资产与 embedding 固定入口

更新日期：2026-10-07。用于场景生成接入，不需要迁移或复制全量资产。

## 直接替换运行命令

推荐：在 `cd .../Task3.2` 后、启动场景 Python 任务前执行：

```bash
source /data/task3_2/share_data/scenesmith/formal_current/activate.sh
```

这一行会一次性固定正式版本，并启用最新正式标注读取器。删除原运行命令中指向旧库的两个路径赋值，让任务继承已导出的环境变量；如果需要保留这两个赋值，则写成：

```bash
HSSD_ZVEC_COLLECTION_PATH="$FORMAL_ASSET_RELEASE/collection" \
HSSD_ALL_ASSETS_MANIFEST_PATH="$FORMAL_ASSET_RELEASE/embedding_inputs.jsonl" \
```

其余启动参数不因本次入口调整而改变。`HSSD_EMBEDDING_BASE_URL=http://127.0.0.1:8014` 保持原样，使用同一 Qwen3-VL-Embedding-2B-Q8_0 模型（2048 维）。本次没有重启已有模型服务、场景任务或生产任务；仅临时启动本项目 GPU embedding 服务补算，验收后释放。

## 各路径是什么

| 固定入口 | 用途 |
| --- | --- |
| `/data/task3_2/share_data/scenesmith/formal_current/collection` | 正式 Zvec 向量库，消费者应只读打开 |
| `/data/task3_2/share_data/scenesmith/formal_current/embedding_inputs.jsonl` | 与该索引配套的冻结清单；当前检索代码从中读取 unit scale |
| `/data/task3_2/share_data/scenesmith/formal_current/asset_library` | 正式资产库入口；不是所有 mesh 集中存放的单一文件夹 |
| `/data/task3_2/share_data/scenesmith/formal_current/annotations` | 正式资产标注目录 |
| `/data/task3_2/share_data/scenesmith/formal_current/RELEASE.json` | 当前入口目标、校验记录与已知限制 |
| `/data/task3_2/share_data/scenesmith/formal_current/activate.sh` | 固定版本并启用正式标注的推荐入口 |

mesh 的真实路径仍由索引及资产记录提供，本次没有修改内部路径，也没有搬迁或删除资产。

## 当前正式版本与边界

固定入口当前指向 `formal_releases/retrieval_20261007_completed`，初次版本及旧索引仍保留：

- 向量库：`embedding_release_20261007/restored_index`，2026-10-07 正式启用，37,186 件资产。
- 配套清单：`all_assets_style_v2_delivery_20260909/embedding_refresh_20261007/embedding_inputs.jsonl`。
- 资产与标注：`all_assets_style_v2_delivery_20260909/asset_library`；正式 Front 修订记录为其 `data/FRONT_V6_REVISION_20261006.json`。
- 已补算 34 件类别变化向量，37,152 条原向量精确复用；本轮已知类别更新缺口为 0。风格没有被本轮重新判别，全部正式风格标签与清单一致。
- Front 为 2026-10-06 正式覆盖后的结果。Front 更新属于独立标注，不要求重算全部向量；推荐的 `activate.sh` 已提供实际消费端接入，不再仅仅更换检索路径。
- 资产库和标注目标仍是现有 live 目录，并非本次创建的不可变快照；`RELEASE.json` 中的校验是发布时证据。

## 使用者的代码如何接入标注

使用者的 HRK checkout 属于另一账号（UID10692），没有写权限。本次未 chmod/chown、未切换其分支、未改写其文件。共享 `formal_consumer_20261007` 提供显式 opt-in 的兼容读取器，无须先修改该 checkout。

`activate.sh` 设置 `HSSD_ASSET_ANNOTATION_ROOT`、显式启用标志及 `PYTHONPATH`。新 Python 进程通过 `sitecustomize.py` 启用正式 reader，合并非生成资产的正式风格 overlay，同时修复旧 HSSD getter 对 Others/generated 的读取路由。旧 hint builder 的列表/字典格式差异只在临时输入中适配，原始正式标注不修改。

只替换两个检索路径、但不执行 `source`，旧 checkout 仍可能读旧标注。该接入要求正常 Python 启动（不能使用禁用 site/PYTHONPATH 的 `-S`/`-I`）。已运行的进程不热刷新；新任务启动前执行即可。正式标注中的不确定性、fallback 和未验收状态完整保留，不把字段存在冒充语义验收。

## 以后如何更新和回退

1. 完成新的 embedding 和配套清单，保留原版本，不在已打开的数据库目录原地覆盖。
2. 发布者统一调用 `feedback/publish_active_embedding_index_20261007.py --index-pointer <已验收候选指针.json> --release <新版本目录名>`。该钩子已接入原 `finish_embedding_delta_20261003.py` 成功路径，正常发布会同步更新固定入口，不需 AI 或 cron 轮询。
3. 钩子检查 passed proof、计数、manifest 与索引签名，再核验只读 mmap 打开、索引完整性、唯一 UID 和全部模型路径；通过后原子切换 `formal_current`，更新旧式 `ACTIVE_INDEX.json`，保存 `PUBLICATION.json`。支持先 stage、实际消费端测试通过后激活；已有版本不覆盖。
4. 新任务使用固定路径即可。正在运行的任务不要热切换数据库：已打开的索引/标注缓存不会自动刷新。长实验应在启动前解析 `formal_current` 的真实版本目录，并以该版本目录同时配置两项路径，避免恰好在两次打开之间发生版本切换。
5. 回退时由发布者把 `formal_current` 原子切回保留的旧版本入口，并同步其配套旧式指针。旧数据库和旧资产路径不删除。

直接手改 `ACTIVE_INDEX.json` 的其它程序不会自动触发该钩子；未来发布应使用统一入口。该工作没有调整生成规则、容器监测器或既有后台任务。本轮向量实体发布在共享目录；本交接文档同步到 Task3.2 的 `dev_yz` 分支，未重复上传大型向量包到 Git。

## 空间处理

验收时共享 `/data` 曾满（80 TB，可用 0），导致验证 GLB 写失败，不能报告为成功。仅将本次新生成的 portable 包校验后迁到 `/mnt/aoss2/codex_backup/formal_entry_closeout_20261007/embedding_release_20261007/portable`，21 个文件、308,557,177 字节全部保留，并保持原 portable 路径的符号链接。原有资产和旧版本没有删除。

**场景检索必需的正式索引与 manifest 仍在 `/data`，没有要求使用者的场景机器挂载 `/mnt/aoss2`。** portable 备份/重建包和验证 GLB 位于中转桶；只有读取这些辅助产物才需要该挂载。共享盘仍接近满，未来大批次重建需要先确认空间，不能把本次归档说成全局容量问题已解决。

使用者原场景命令设置了 `CRITIC_PROBE_MIN_FREE_GIB=64`。文档发布前状态检查 `/data` 可用约158 GiB，已高于64 GiB门槛；共享容量会波动，启动时须重新检查。这只解除当时的空间限制，不等于完整场景任务已运行或全部依赖均已验收。本次没有擅自删除原有资产/其它项目数据，也没有降低安全门槛；空间恢复不归因于本次文档发布。

## 查询当前版本

验收通过：37,186 条向量、完整性 1.0，UID 全唯一、模型路径缺失0；全量正式 Front/style 与实际 reader 一致；5 类来源的 legacy/AssetManager 路由一致；在使用者 checkout/venv 中真实文本 embedding 查询5次，返回10个实际模型，GLB 导出回读通过。8项定向测试通过（含格式回归重跑），无特权 `nobody` 实际读取入口/清单/标注/索引/兼容代码25个文件成功。权限检查不冒充该用户的 Zvec 环境测试。

没有运行完整 LLM 场景规划/critic 多阶段实验，也没有重新认证全库渲染或语义质量。本轮端到端范围是实际资产消费链，不是新的 SceneEval 实验。

另外使用者原命令的 `http://127.0.0.1:8014` 已实际验证一次文本检索并返回可加载模型（7363顶点）；仅作为查询向量，没有混入CPU计算的存储向量。证据 `embedding_refresh_20261007/BASE_URL_8014_VERIFIED.json`。

证据：`all_assets_style_v2_delivery_20260909/embedding_refresh_20261007/CONSUMER_VERIFIED.json`、`GPU_PARITY.json`、`UNIT_TESTS.xml`、`PUBLICATION_GUARD_TESTS.xml`、`PARTNER_SCHEMA_REGRESSION.xml`；固定入口 `PUBLICATION.json`；新索引 `REBUILD_VERIFIED.json`。

```bash
readlink -f /data/task3_2/share_data/scenesmith/formal_current
cat /data/task3_2/share_data/scenesmith/formal_current/RELEASE.json
```

发布者维护发布脚本；使用者不需要读取发布者的私人目录。索引构建使用已有 embedding runtime 的兼容 Zvec；实际消费验证使用使用者 Task3.2 venv，不改其依赖。

## Task3.2 分支同步核验（2026-10-07）

本次直接 fetch 并查询远端分支，不依赖本地旧 tracking ref：

- `dev_hrk_week37` 最新提交：`e65b5c9ff43471bb548f5a4532e1c4820fb6795e`。
- 本次发布前 `dev_yz`：`06f237ac4ecd271aa5f7dc6d1ce6007e053b0dfb`。
- `git merge-base --is-ancestor <HRK提交> <YZ提交>` 成功；YZ 已包含 HRK 的所有提交，并有6个资产交付提交。因此无需重复 merge/cherry-pick，也没有丢弃 YZ 的标注成果。

本次只新增本交接文档，不修改使用者 checkout、原有标注或后台任务。Git 中历史 embedding 文档/包描述其当时版本；当前共享正式消费路径以本文件和 `formal_current/RELEASE.json` 为准。完整向量实体没有在本次文档发布中复制到 Git。
