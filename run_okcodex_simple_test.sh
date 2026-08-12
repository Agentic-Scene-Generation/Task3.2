#!/bin/bash
set -e

# 使用正确的 Python 环境
PYTHON_BIN="/mnt/afs/visitor33/scenesmith-qwen/.venv/bin/python"

# OKCodex API configuration
export OKCODEX_API_KEY="sk-2859167011902c268ec87a71178a31b5a481d61722b30ef2bcb836eb23a68cad"
export OKCODEX_BASE_URL="https://api.okcodex.cn"
export OKCODEX_IMAGE_MODEL="gpt-image-1.5"

# GroundingDINO service
export GROUNDING_DINO_BASE_URL="http://127.0.0.1:18030"

# vLLM服务地址
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://127.0.0.1:18010/v1"

# Qwen模型ID
export QWEN_MODEL_ID="qwen3_5_32b_sft_merged_v8"

# HSSD数据路径
HSSD_DATA_DIR="/mnt/afs-p3/task3_2/share_data/hsm/hssd-models"

# 输出目录
OUTPUT_DIR="/mnt/afs/visitor33/Task3.2/outputs_okcodex_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "===== OKCodex Context Image Test ====="
echo "Output directory: $OUTPUT_DIR"
echo "Backend: okcodex"
echo "GroundingDINO: $GROUNDING_DINO_BASE_URL"
echo "Python: $PYTHON_BIN"
echo ""

cd /mnt/afs/visitor33/Task3.2

# 直接运行Python主程序
"$PYTHON_BIN" main.py \
  experiment=ablation_3_qwen3_harness \
  furniture_agent=okcodex_furniture_agent \
  paths.hssd_data_dir="$HSSD_DATA_DIR" \
  paths.output_dir="$OUTPUT_DIR" \
  articulated_retrieval_server.host=null \
  objaverse_retrieval_server.host=null \
  base_experiment.prompts='["A bedroom with a bed, nightstand, and wardrobe."]' \
  base_experiment.num_workers=1 \
  pipeline.stop_stage=furniture

echo ""
echo "===== Test Complete ====="
echo "Check output in: $OUTPUT_DIR"
