# 共享访问修复（2026-09-10）

原问题：Zvec集合目录/文件为700/600，其他账号不可读；结果中的asset_path/image_paths仍有私有工作区绝对路径。不是同学的模型调用错误。

## 已立即修复

现有 `all_assets_style_v2_gpu_compact_20260909` 及交付目录已补充共享读/目录遍历权限，同级input.json和done.jsonl可读。非属主UID65534已真实只读打开30,830条索引。无需开放777写权限，也未开放私有工作区。

消费时必须只读打开：

```python
import zvec
c = zvec.open('/data/task3_2/share_data/scenesmith/all_assets_style_v2_gpu_compact_20260909',
              option=zvec.CollectionOption(read_only=True))
```

3D-FUTURE原始资产已有共享副本：`/data/task3_2/share_data/scenesmith/3DFuture/raw/3D-FUTURE-model/<UUID>/`。截图所示1460b4ef-2317-347a-82d9-ae24ea0083be的raw_model.obj、model.mtl、texture.png、image.jpg与私有源逐文件SHA256一致，可直接使用共享副本。暂时从旧索引取出的该来源路径需改为此共享前缀。

## 全来源共享路径版（后台交付）

只复制当前30,830条索引所需模型、相关OBJ/GLTF依赖和图片；3D-FUTURE复用既有共享副本并逐文件核验，不复制62GB整库第二份。不复制API密钥、日志或私有项目代码。新资源根：`/data/task3_2/share_data/scenesmith/all_assets_style_v2_assets_20260910/`。

新向量集合目标 `all_assets_style_v2_gpu_shared_20260910`。原向量与非路径字段逐条保持一致，只改asset_path和image_paths，不重新调用embedding服务。旧库保留。未就绪/缺图记录不伪造共享资产。

以资源根 `STATUS.json` 为实时状态：只有 `stage=complete` 才算交付成功。复制、向量一致性验证、非属主30,830条模型/图片读取全部通过后，一次性收尾进程才更新交付根 `ACTIVE_INDEX.json`；失败不切换当前索引。不要使用尚未完成的新库。程序可续跑，校验过的资产有receipts；失败报告不会被当作成功。

目录权限问题已修复，但全来源路径迁移进行中时，不能声称旧索引所有资产路径都已可用。完整文件校验可能耗时；后续查看STATUS和ACTIVE_INDEX，无需AI轮询。
