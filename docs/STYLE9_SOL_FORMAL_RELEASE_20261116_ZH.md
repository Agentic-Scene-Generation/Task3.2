# 风格、主题色与材质正式发布（2026-11-16）

## 当前正式状态

正式资产共38,497件，有明确风格参考34,753件，显式unknown_style 3,744件。
本次首轮补出6,429件，追加复核补出590件，共7,019件。
采用九类一级风格、多标签体系；可只保留可靠一级，未知项有具体原因和较低检索优先级。

首轮同时补齐所选资产的主题色及材质观察；追加复核仅更新风格，保留已完成的主题色、材质、Front及其他标注。
累计次数、上一版标签、父轮与追加复核的结果路径和哈希均保留。追加复核读取过父轮意见，属于上下文复核。

## 使用入口

- 共享正式标注：`/data/task3_2/share_data/scenesmith/formal_current/annotations/`。
- Task3.2：`dev_yz`，数据目录`scenesmith/scenebenchmark_critic/asset_annotation_data/`。
- 独立仓库：`K-Chronofox/hssd-annotations`的`main`，数据目录`data/`。
- 配置`config.json`指向本次权威外观覆盖层`asset_appearance_overlay_20261116.json.gz`；应通过随库读取器获取合并后的记录。
- 3D-FUTURE在Git中保留完整紧凑基表，通过覆盖层读取本次外观字段；已逐件验证与共享正式记录一致。
- Embedding仍使用原已发布索引，本次没有重算向量或切换路径；新标签检索索引需后续更新。
- 资产生成保持暂停。

## Task3.2正式分支同步

发布前将最新远端`main`（`d2d50f7e37967f85335f2696ad31174d12ca4705`）合并到`dev_yz`。
当前远端已无`dev_hrk_week37`，其历史改动已进入`main`。合并冲突仅涉及旧标注与外观读取器：
保留`dev_yz`较新的标注、Front修正和外观覆盖层读取功能；其余正式分支变更由Git合并。

## 未完成项及边界

首轮54项请求/处理失败、1项证据失败，继续保留之前正式标签与原始失败记录。
追加复核33项证据问题单列，未强填明确风格；其余仍unknown的结果保留无鲜明风格、词表外或证据不足等原因。
本次是标注发布，不代表Embedding或生成任务更新。

## 统计与可追溯报告

下表按资产计数；多标签资产可计入多个风格，风格列和可大于资产总数。

| 来源 | 总资产 | 有明确风格 | unknown_style | 追加复核新增明确风格 |
|---|---:|---:|---:|---:|
| hssd_annotation_lookup.json.gz | 10963 | 9828 | 1135 | 365 |
| 3dfuture/3dfuture_annotation_lookup.json.gz | 14823 | 14793 | 30 | 11 |
| external/others_annotation_lookup.json.gz | 5564 | 4235 | 1329 | 72 |
| generated_annotations.json.gz | 7147 | 5897 | 1250 | 142 |

### 一级风格统计

| 标签 | 资产数 |
|---|---:|
| american | 1849 |
| chinese | 928 |
| eclectic | 738 |
| european | 5137 |
| french | 372 |
| japanese | 1978 |
| mediterranean | 371 |
| modern | 23813 |
| southeast_asian | 1063 |
| unknown_style | 3744 |

正式机器报告：`STYLE9_SOL_FORMAL_RELEASE_20261116.json`。逐项异常及两仓库远端回执保存在`/data/task3_2/share_data/scenesmith/style_sol_formal_release_20261116`。
