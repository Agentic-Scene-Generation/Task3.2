#!/bin/bash
# =============================================================================
# Qwen3.5-35B-A3B 部署脚本
# 适用配置：2x A100/H100 80GB，从 ModelScope 下载
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_DIR/models/Qwen3.5-35B-A3B"
MODEL_ID="Qwen/Qwen3.5-35B-A3B"
VLLM_PORT=8000

echo "=========================================="
echo " Qwen3.5-35B-A3B 部署脚本"
echo "=========================================="
echo "项目目录: $PROJECT_DIR"
echo "模型目录: $MODEL_DIR"
echo ""

# --------------------------------------------------------------------------
# 步骤 1：安装依赖
# --------------------------------------------------------------------------
echo "[1/4] 安装依赖 (modelscope, vllm)..."
# pip install modelscope vllm  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip show vllm
echo "  ✓ 依赖安装完成"

# --------------------------------------------------------------------------
# 步骤 2：下载模型
# --------------------------------------------------------------------------
if [ -d "$MODEL_DIR" ] && [ "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "[2/4] 模型目录已存在，跳过下载: $MODEL_DIR"
else
    echo "[2/4] 从 ModelScope 下载模型 (约 72GB，请耐心等待)..."
    mkdir -p "$MODEL_DIR"
    modelscope download \
        --model "$MODEL_ID" \
        --local_dir "$MODEL_DIR"
    echo "  ✓ 模型下载完成"
fi

# --------------------------------------------------------------------------
# 步骤 3：设置环境变量
# --------------------------------------------------------------------------
echo "[3/4] 配置环境变量..."
cat > "$PROJECT_DIR/.env" << EOF
# SceneSmith + Qwen3.5-35B-A3B 环境变量
# 将 OpenAI 客户端指向本地 vLLM 服务
export OPENAI_API_KEY="not-needed"
export OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1"
EOF
echo "  ✓ 环境变量已写入 $PROJECT_DIR/.env"
echo "  运行 'source .env' 使其生效"

# --------------------------------------------------------------------------
# 步骤 4：启动 vLLM 服务
# --------------------------------------------------------------------------
echo "[4/4] 启动 vLLM 服务..."
echo "  模型: $MODEL_DIR"
echo "  端口: $VLLM_PORT"
echo "  Tensor Parallel: 2"
echo ""
echo "  服务启动后按 Ctrl+C 停止"
echo "=========================================="

# 激活环境变量
source "$PROJECT_DIR/.env"

vllm serve "$MODEL_DIR" \
    --served-model-name "$MODEL_ID" \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --port "$VLLM_PORT" \
    --max-model-len 262144  \
    --gpu-memory-utilization 0.90
