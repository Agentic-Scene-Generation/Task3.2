# Bailian Qwen-Image-Edit 测试结果

## 测试配置
- **模型**: qwen-image-edit (百炼平台)
- **API endpoint**: 默认 (dashscope)
- **输入图像**: 768×768 空房间俯视图
- **Prompt**: 2738 字符 (完整 scenesmith furniture context prompt)

## 性能指标
| 指标 | 百炼 qwen-image-edit | OKCodex (gpt-image-1.5) |
|---|---|---|
| **响应时间** | **17.1 秒** | 35-55 秒 |
| **成功率** | 1/1 (100%) | 2/3 最近测试 (okcodex 有临时抖动) |
| **输出格式** | OSS URL | b64_json |

## 质量评估

### ✅ 优点
- **速度快**: 17 秒，比 okcodex 快 2-3 倍
- **稳定**: 首次调用即成功，无 502/timeout

### ⚠️ 潜在问题
- **尺寸变化**: 输入 768×768 → 输出 1024×1024
  - 违反 prompt 约束 "Preserve canvas size, image orientation"
  - 需要验证房间结构（墙/门/窗）是否保持精确对齐
- **需要视觉验证**:
  - 是否保持俯视角（未变透视图）
  - 门窗是否被家具遮挡/移位
  - 家具数量/类别是否匹配要求
  - 布局合理性（沙发朝向、间距）

## 文件
- `empty_room_top_view.png` — 输入 (768×768, 233 KB)
- `bailian_qwen_output.png` — 输出 (1024×1024, 629 KB)
- `prompt.txt` — 完整 prompt

## 下一步
1. 视觉对比输入/输出，检查约束遵守情况
2. 如果尺寸变化导致结构错位，需要在 scenesmith 集成时添加 resize 逻辑
3. 用 scenesmith 的 VLM quality gate 打分，与 okcodex 结果对比
