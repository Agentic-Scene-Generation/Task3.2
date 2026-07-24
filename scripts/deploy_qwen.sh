#!/bin/bash
# =============================================================================
# Qwen 部署脚本
# 适用配置：多卡 A100/H100/L40S，从 ModelScope 下载
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${SCENEEXPERT_ENV_FILE:-$PROJECT_DIR/.env}"

source_env_file() {
    local env_path="$1"
    local tmp_env
    tmp_env="$(mktemp)"
    sed 's/\r$//' "$env_path" > "$tmp_env"
    # shellcheck disable=SC1090
    source "$tmp_env"
    rm -f "$tmp_env"
}

if [ -f "$ENV_FILE" ]; then
    source_env_file "$ENV_FILE"
    echo "已加载配置文件: $ENV_FILE"
fi
MODEL_ID="${SCENEEXPERT_MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
MODELS_DIR="${SCENEEXPERT_MODELS_DIR:-$PROJECT_DIR/models}"
MODEL_DIR="${SCENEEXPERT_MODEL_DIR:-$MODELS_DIR/${MODEL_ID##*/}}"
VLLM_PORT="${SCENEEXPERT_VLLM_PORT:-8000}"
TENSOR_PARALLEL_SIZE="${SCENEEXPERT_TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${SCENEEXPERT_MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${SCENEEXPERT_GPU_MEMORY_UTILIZATION:-0.90}"
ENABLE_AUTO_TOOL_CHOICE="${SCENEEXPERT_ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${SCENEEXPERT_TOOL_CALL_PARSER:-qwen3_xml}"
REASONING_PARSER="${SCENEEXPERT_REASONING_PARSER:-qwen3}"
GDN_PREFILL_BACKEND="${SCENEEXPERT_GDN_PREFILL_BACKEND:-triton}"
ENABLE_PREFIX_CACHING="${SCENEEXPERT_VLLM_ENABLE_PREFIX_CACHING:-1}"
PIP_INDEX_URL="${SCENEEXPERT_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "=========================================="
echo " Qwen vLLM 部署脚本"
echo "=========================================="
echo "项目目录: $PROJECT_DIR"
echo "模型 ID: $MODEL_ID"
echo "模型目录: $MODEL_DIR"
echo ""

# --------------------------------------------------------------------------
# 步骤 1：安装依赖
# --------------------------------------------------------------------------
echo "[1/4] 安装依赖 (modelscope, vllm)..."
if python -m pip show modelscope vllm >/dev/null 2>&1; then
    echo "  ✓ modelscope/vllm 已安装"
elif [ "${SCENEEXPERT_INSTALL_RUNTIME_DEPS:-0}" = "1" ]; then
    python -m pip install modelscope vllm -i "$PIP_INDEX_URL"
else
    echo "  错误：缺少 modelscope 或 vllm"
    echo "  在线/内网镜像安装：SCENEEXPERT_INSTALL_RUNTIME_DEPS=1 bash scripts/deploy_qwen.sh"
    echo "  离线安装：python -m pip install --no-index --find-links /path/to/wheels modelscope vllm"
    exit 1
fi
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
mkdir -p "$(dirname "$ENV_FILE")"
cat > "$ENV_FILE" << EOF
# SceneExpert + Qwen/vLLM 环境变量
# 将 OpenAI 客户端指向本地 vLLM 服务
export SCENEEXPERT_MODEL_ID="$MODEL_ID"
export SCENEEXPERT_MODELS_DIR="$MODELS_DIR"
export SCENEEXPERT_MODEL_DIR="$MODEL_DIR"
export SCENEEXPERT_VLLM_PORT="$VLLM_PORT"
export SCENEEXPERT_ENABLE_AUTO_TOOL_CHOICE="$ENABLE_AUTO_TOOL_CHOICE"
export SCENEEXPERT_TOOL_CALL_PARSER="$TOOL_CALL_PARSER"
export SCENEEXPERT_REASONING_PARSER="$REASONING_PARSER"
export SCENEEXPERT_GDN_PREFILL_BACKEND="$GDN_PREFILL_BACKEND"
export SCENEEXPERT_VLLM_ENABLE_PREFIX_CACHING="$ENABLE_PREFIX_CACHING"
export OPENAI_API_KEY="not-needed"
export OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1"
EOF
echo "  ✓ 环境变量已写入 $ENV_FILE"
echo "  运行 'source $ENV_FILE' 使其生效"

# --------------------------------------------------------------------------
# 步骤 4：启动 vLLM 服务
# --------------------------------------------------------------------------
echo "[4/4] 启动 vLLM 服务..."
echo "  模型: $MODEL_DIR"
echo "  端口: $VLLM_PORT"
echo "  Tensor Parallel: $TENSOR_PARALLEL_SIZE"
echo ""
echo "  服务启动后按 Ctrl+C 停止"
echo "=========================================="

# 激活环境变量
source_env_file "$ENV_FILE"

VLLM_ARGS=(
    vllm serve "$MODEL_DIR"
    --served-model-name "$MODEL_ID"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --port "$VLLM_PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)
if [ "$ENABLE_AUTO_TOOL_CHOICE" = "1" ]; then
    VLLM_ARGS+=(--enable-auto-tool-choice)
    if [ -n "$TOOL_CALL_PARSER" ]; then
        VLLM_ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")
    fi
fi
if [ -n "$REASONING_PARSER" ]; then
    VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi
if [ -n "$GDN_PREFILL_BACKEND" ]; then
    VLLM_ARGS+=(--gdn-prefill-backend "$GDN_PREFILL_BACKEND")
fi
if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
    VLLM_ARGS+=(--enable-prefix-caching)
fi
"${VLLM_ARGS[@]}"
