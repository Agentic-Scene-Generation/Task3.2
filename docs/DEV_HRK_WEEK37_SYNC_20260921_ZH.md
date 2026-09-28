# dev_hrk_week37 同步记录（2026-09-21）

用户要求将尚未采用的最新 dev_hrk_week37 更新合并到当前 dev_yz。合并前 dev_yz 为 `ed1be60318c664d1cb146a7fffb1e72f59d1ee1e`，最新上游为 `7b3def3ef3ef0dedb186bca42abed0d28990f87c`；此前同步过的上游提交为 `5db84280e665dd091328ce7027f46fc384faa385`。

## 完整引入的8个提交

| 提交 | 更新 |
|---|---|
| 8945a07 | 3DFuture 正面标注接入资产管理器和 critic，新增读取/坐标测试 |
| 9d5551f | 床边家具碰撞修复逻辑及相关测试 |
| 41391d0 | 候选资产图像缓存、从多库 manifest 读取渲染图 |
| 5861195 | 并行评测脚本修复 |
| a57d5df | 并行评测脚本及 SceneExpert 运行契约测试更新 |
| 7403572 | 运行脚本收尾报错修复 |
| 2eeaf93 | intent contract v8：混合类别组、数量对齐及相应 FD/SA 支持 |
| 7b3def3 | 新增6个风格场景案例、评测脚本支持及最终视图可选 Cycles 引擎 |

上游净变化为20文件、1527行增加/152行删除。完整 merge，无冲突、无 cherry-pick 筛选、无 reset/checkout/clean 或 force push。只同步 Task3.2 代码分支，不把其场景管线复制到独立资产库，也不启动上游新增实验脚本。

## 合并时发现并修复的兼容问题

3DFuture记录通过新入口读取后，通用 `build_scenebenchmark_annotation` 仍填入历史默认 `hssd_annotations`。上游 adapter 使用 setdefault，无法改正这个值，导致新增的来源断言失败。对真实3DFuture记录和实际函数复现后，将来源字段由实际读取分支明确赋值，追加 classification_source 回归断言；未改方向向量或资产标注文件。

## 验证与限制

- 合并前后36,665件正面发布文件及全量front overlay的SHA256一致；没有覆盖刚发布的 Luna/fallback 方向、风格、embedding。
- 新旧命名空间实际数据读取及纯函数隔离检查通过：3DFuture asset-local `[0,0,1]` 正确转换为 SceneSmith `-Y`，来源为 `3dfuture_annotations`。脚本 `scripts/verify_front_merge_smoke.py` 先复现失败，再通过。
- 该脚本直接运行真实数据读取器和从实际源码AST编译的函数，不替换函数逻辑；它不是完整SceneSmith导入/端到端测试。
- 上游变更18个Python文件语法检查通过，`run_parallel_critic_on.sh` 的 `bash -n` 通过，6条风格CSV记录可解析且ID唯一。
- `tests/unit/test_hssd_open_mesh_annotations.py --confcutdir=tests/unit`：7 passed，2条既有pytest配置警告。
- 完整 `python -m pytest -q` 在 conftest 导入阶段被本机缺少 `bpy` 阻挡；当前主机Python为3.10，项目要求3.11。不声称完整回归或真实场景生成测试通过。未安装重型环境或修改后台容器。

上游更新仅进入代码仓库；正在运行的资产生成、渲染、Luna、监测器没有重启或重配。
