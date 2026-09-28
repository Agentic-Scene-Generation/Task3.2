# 资产库当前状态与正面复判说明

更新时间：2026-09-21 UTC。分支：Task3.2 `dev_yz`。

## 最重要的结论

**当前正式标签继续使用；正在运行的 Luna 正面复判只写独立候选目录，不影响当前标签。只有本轮复判全部完成、结果检查通过之后，才统一发布并覆盖对应的正式正面标签。不会边复判边覆盖。**

本次提交同步的是已经完成的正面覆盖补齐与风格 fallback，不是新一轮透视复判结果。没有更改模型坐标、重算 embedding 或重启生成/标注任务。

## 当前发布快照

| 资产来源 | 数量 |
|---|---:|
| HSSD | 10,963 |
| 3DFuture | 14,823 |
| Others 非生成小库 | 5,564 |
| 生成资产 | 5,364 |
| 合计 | 36,714 |

这是已验收的共享导出快照，不是不断增长的生产目录实时总量。相对之前 Git 的36,665件版本，新增49件生成资产的正面记录及对应生成注释。后台继续生成，不把随后尚未导出的资产计入此表。

所有36,714件都已有可用的非空水平正面方向，检查了有限值、单位长度、Y-up坐标、两个方向字段一致以及来源标识。**非空不代表全部有确定的真实语义正面**：uncertain、无唯一正面等保留明确标注来源的 fallback；不会把fallback伪装为高置信度识别。

数据入口：`scenesmith/scenebenchmark_critic/asset_annotation_data/CURRENT_ANNOTATION_RELEASE.json`。新清单的哈希针对本仓库实际文件计算，旧的日期快照清单保留作历史，不用于验证本次更新文件。

## 正在做的 Luna 正面复判

任务共5,026件，针对此前 uncertain/unusable：Others725、HSSD1,606、3DFuture2,043、历史生成资产652，先处理Others。Flare稀缺风格生成资产继续遵循用户约定，不送Luna。

新方案为每件资产渲染10张真实mesh透视图：六方向图加四张斜上方结构图。提示词提供相反相机的几何配对，帮助区分背面和底面；不提供旧正面答案。正面判断与up判断分开，不能因up不确定就直接抹掉明确的正面，也不会自动旋转mesh。

报告取样时：已完成78/5,026件，均来自Others，HTTP错误0。此数字只是取样进度，不是完成声明或准确率。实际进度以共享目录 `front_luna_v3_20260921/full_run/status.json` 为准；模型回答、图像哈希、几何指纹和坐标链逐件保留在该目录的assets子目录。

初次启动曾因独立渲染目录漏带policy依赖导致305件渲染失败，当时API请求0；修复后同队列重跑，保留原始错误记录，没有跳过失败资产来凑完成率。

### 发布隔离与完成门槛

1. 当前worker不写正式canonical_front，候选的 `published_canonical_overwritten=false`。
2. 不能凭进程结束、COMPLETE文件或部分HTTP200认定全量完成；需要5,026件逐项对账，渲染失败、请求失败、返回不完整等均须处理清楚。
3. 全量结束后，检查视图/几何/提示词哈希、方向合法性、消费者坐标链和代表样例，再统一生成新版本。仍无明确正面的资产保留有来源的fallback，方向不能变空。
4. 正式覆盖时保留此前标签与版本，统一更新清单及共享/仓库发布，不把部分新结果混进旧标签。

## 无明确风格的处理

非生成31,350件全部有非空 `style_label_ids`。其中1,439件为 `unknown_style`（风格未识别）：HSSD906、3DFuture5、Others528。它不是第18种Bonn风格，也不是“风格中性”。

原styles证据、confidence、判定次数和状态不变。原始 `styles=[]` 表示没有真实风格本体路径，展示标签请读取 `style_label_ids`。检索采用独立优先级：有明确风格tier0、unknown tier1；不篡改相似度分数，不删除未知风格资产。严格指定真实风格时不能把unknown冒充匹配。

本仓库 `scripts/query_all_asset_embeddings.py` 已加入候选池内降级：召回5倍候选（扩展上限1000），同tier保持原排序，再取top-k。不在非生成overlay内的ID不会因为缺映射被误判为未知。自定义检索器或已缓存旧数据的常驻服务需要接入/重载；不是所有外部服务自动生效。详见 [风格fallback规则](STYLE_UNKNOWN_FALLBACK_20260921_ZH.md)。

## 其他标注：尚不能宣称全部高质量完成

- 非生成31,350件有DOF、关系、环境参照、净空、affordance引用、物理、质量、自发光等主要字段；物理参数多为估计，不是实测。
- 生成5,364件还缺完整DOF、关系、净空、affordance、物理/质量等正式字段。模型加载/材质/校验和通过不等于完整标注通过。
- Others掩码由类别和坐标阈值规则产生，并非原HSSD官方模型分割；5,501件基于真实mesh采样，63件是bbox代理，必须区别对待。
- 1,822件旧/新存储正面方向不同，需要核对相关功能方向和净空坐标链；未擅自旋转旧净空。
- HSSD缺独立operation_space_ref不直接等于缺失：5,996件有内联非关节净空，799件有内联关节扫掠，2,750件未声明需要净空。
- 88,289个唯一模型/affordance/operation引用路径本机检查无缺失/不可读/私人路径问题；不是逐个语义验收，也不等于所有用户ACL均已验证。

## 使用位置与验证范围

共享数据根：`/data/task3_2/share_data/scenesmith/all_assets_style_v2_delivery_20260909/asset_library/data/`。
共享审计：`/data/task3_2/share_data/scenesmith/all_assets_style_v2_delivery_20260909/annotation_audit_20260921/`。
生成模型：`/data/task3_2/generated_assets_hunyuan/`；大模型本体不塞入Git。

本分支已包含上一轮同步的dev_hrk_week37（7b3def3）更新，合并基线63c8017；本次不修改SceneExpert研究路线。

共享fallback/schema定向测试12通过，完整共享库测试155通过、5项旧front/affordance快照断言失败，详见fallback文档；不宣称全绿。Task3.2完整运行环境仍依赖bpy/Drake，此次数据及检索同步不代表全场景生成端到端验收。检索降级不要求重算embedding；本次没有新增模型调用或训练。
