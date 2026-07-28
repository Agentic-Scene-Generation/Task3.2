#!/bin/bash
# =============================================================================
# 仅启动 vLLM 服务（模型已下载的情况下快速启动）
# 用法：bash scripts/start_vllm.sh
# =============================================================================

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
VLLM_HEALTH_URL="${SCENEEXPERT_VLLM_HEALTH_URL:-http://localhost:${VLLM_PORT}/health}"
VLLM_WAIT_TIMEOUT_SECONDS="${SCENEEXPERT_VLLM_WAIT_TIMEOUT_SECONDS:-1800}"
VLLM_ENGINE_READY_TIMEOUT_S="${SCENEEXPERT_VLLM_ENGINE_READY_TIMEOUT_S:-$VLLM_WAIT_TIMEOUT_SECONDS}"
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
TENSOR_PARALLEL_SIZE="${SCENEEXPERT_TENSOR_PARALLEL_SIZE:-$NUM_GPUS}"
MAX_MODEL_LEN="${SCENEEXPERT_MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${SCENEEXPERT_GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${SCENEEXPERT_VLLM_DTYPE:-auto}"
QUANTIZATION="${SCENEEXPERT_VLLM_QUANTIZATION:-}"
CPU_OFFLOAD_GB="${SCENEEXPERT_VLLM_CPU_OFFLOAD_GB:-0}"
KV_CACHE_DTYPE="${SCENEEXPERT_VLLM_KV_CACHE_DTYPE:-auto}"
SAFETENSORS_LOAD_STRATEGY="${SCENEEXPERT_VLLM_SAFETENSORS_LOAD_STRATEGY:-}"
USE_DEEP_GEMM="${SCENEEXPERT_VLLM_USE_DEEP_GEMM:-0}"
MOE_USE_DEEP_GEMM="${SCENEEXPERT_VLLM_MOE_USE_DEEP_GEMM:-0}"
DEEP_GEMM_WARMUP="${SCENEEXPERT_VLLM_DEEP_GEMM_WARMUP:-skip}"
ENFORCE_EAGER="${SCENEEXPERT_VLLM_ENFORCE_EAGER:-0}"
ENABLE_AUTO_TOOL_CHOICE="${SCENEEXPERT_ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${SCENEEXPERT_TOOL_CALL_PARSER:-qwen3_xml}"
REASONING_PARSER="${SCENEEXPERT_REASONING_PARSER:-qwen3}"
VLLM_LOG="${SCENEEXPERT_VLLM_LOG:-$PROJECT_DIR/vllm_server.log}"

if [ "${SCENEEXPERT_SKIP_PYTHON_PREFLIGHT:-0}" != "1" ]; then
    echo "运行 runtime dependency compatibility preflight..."
    PYTHONDONTWRITEBYTECODE=1 python "$PROJECT_DIR/scripts/check_runtime_compatibility.py"
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "错误：模型目录不存在: $MODEL_DIR"
    echo "请先运行 bash scripts/deploy_qwen.sh 下载模型"
    exit 1
fi

if [ "$TENSOR_PARALLEL_SIZE" -gt "$NUM_GPUS" ]; then
    echo "错误：tensor_parallel_size=$TENSOR_PARALLEL_SIZE 大于当前可见 GPU 数量 $NUM_GPUS。"
    echo "请修改 .env 中的 SCENEEXPERT_TENSOR_PARALLEL_SIZE，或申请更多 GPU。"
    exit 1
fi

if [[ "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] && [ "$TENSOR_PARALLEL_SIZE" -eq 1 ] && [ "$MAX_MODEL_LEN" -gt 65536 ]; then
    echo "警告：当前只有单卡 tensor parallel，但 SCENEEXPERT_MAX_MODEL_LEN=$MAX_MODEL_LEN。"
    echo "35B MoE 单卡长上下文很容易显存不足，建议先设为 32768 或 65536。"
fi
NO_CPU_OFFLOAD=0
if [ -z "$CPU_OFFLOAD_GB" ] || [ "$CPU_OFFLOAD_GB" = "0" ] || [ "$CPU_OFFLOAD_GB" = "0.0" ]; then
    NO_CPU_OFFLOAD=1
fi
if [ "$TENSOR_PARALLEL_SIZE" -eq 1 ] && [[ "$MODEL_ID" == *"35B"* ]] && [ -z "$QUANTIZATION" ] && [ "$NO_CPU_OFFLOAD" = "1" ]; then
    echo "警告：当前是单卡未量化 35B 模型，且未开启 CPU offload。"
    echo "最直接的修复是设置 SCENEEXPERT_VLLM_CPU_OFFLOAD_GB=20；否则很可能继续在 FusedMoE 权重加载阶段失败。"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU 状态:"
    GPU_STATUS="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader 2>&1 || true)"
    echo "$GPU_STATUS"
    if echo "$GPU_STATUS" | grep -q "Insufficient Permissions"; then
        echo "提示：nvidia-smi 无权显示显存信息，忽略该提示并继续启动 vLLM。"
    fi
fi

VLLM_LAUNCH_MODE=""
if ! command -v vllm >/dev/null 2>&1; then
    if python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('vllm.entrypoints.openai.api_server') else 1)" >/dev/null 2>&1; then
        VLLM_LAUNCH_MODE="python-module"
        echo "未找到 vllm 命令，改用当前 Python 的 vLLM 模块入口启动。"
    else
        echo "错误：找不到 vLLM 命令，当前 Python 也无法导入 vLLM。"
        echo "  当前 Python: $(python -c 'import sys; print(sys.executable)')"
        echo "  安装示例：python -m pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple"
        echo "  离线安装：python -m pip install --no-index --find-links /path/to/wheelhouse vllm"
        echo "  如果已有其他 OpenAI-compatible 本地服务，请设置 SCENEEXPERT_START_VLLM=0 并使用 run_experiment.sh 连接它。"
        exit 1
    fi
else
    VLLM_LAUNCH_MODE="cli"
fi

echo "启动 vLLM 服务..."
echo "  模型: $MODEL_DIR"
echo "  served model: $MODEL_ID"
echo "  端口: $VLLM_PORT"
echo "  可见 GPU 数量: $NUM_GPUS"
echo "  Tensor Parallel: $TENSOR_PARALLEL_SIZE"
echo "  max model len: $MAX_MODEL_LEN"
echo "  dtype: $DTYPE"
echo "  quantization: ${QUANTIZATION:-none}"
echo "  CPU offload: ${CPU_OFFLOAD_GB} GiB/GPU"
echo "  KV cache dtype: $KV_CACHE_DTYPE"
echo "  safetensors load strategy: ${SAFETENSORS_LOAD_STRATEGY:-default}"
echo "  health timeout: ${VLLM_WAIT_TIMEOUT_SECONDS}s"
echo "  engine ready timeout: ${VLLM_ENGINE_READY_TIMEOUT_S}s"
echo "  DeepGEMM: use=${USE_DEEP_GEMM}, moe_use=${MOE_USE_DEEP_GEMM}, warmup=${DEEP_GEMM_WARMUP}"
echo "  enforce eager: ${ENFORCE_EAGER}"
echo "  启动方式: $VLLM_LAUNCH_MODE"

# 设置环境变量
export SCENEEXPERT_MODEL_ID="$MODEL_ID"
export OPENAI_API_KEY="not-needed"
export OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ENGINE_READY_TIMEOUT_S="$VLLM_ENGINE_READY_TIMEOUT_S"
export VLLM_USE_DEEP_GEMM="$USE_DEEP_GEMM"
export VLLM_MOE_USE_DEEP_GEMM="$MOE_USE_DEEP_GEMM"
export VLLM_DEEP_GEMM_WARMUP="$DEEP_GEMM_WARMUP"

# 后台启动，日志写入文件
if [ "$VLLM_LAUNCH_MODE" = "cli" ]; then
    VLLM_ARGS=(
        vllm serve "$MODEL_DIR"
        --served-model-name "$MODEL_ID"
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
        --port "$VLLM_PORT"
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --dtype "$DTYPE"
        --kv-cache-dtype "$KV_CACHE_DTYPE"
    )
else
    VLLM_ARGS=(
        python -m vllm.entrypoints.openai.api_server
        --model "$MODEL_DIR"
        --served-model-name "$MODEL_ID"
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
        --port "$VLLM_PORT"
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --dtype "$DTYPE"
        --kv-cache-dtype "$KV_CACHE_DTYPE"
    )
fi
if [ -n "$QUANTIZATION" ]; then
    VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi
if [ "$CPU_OFFLOAD_GB" != "0" ] && [ -n "$CPU_OFFLOAD_GB" ]; then
    VLLM_ARGS+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
fi
if [ -n "$SAFETENSORS_LOAD_STRATEGY" ]; then
    VLLM_ARGS+=(--safetensors-load-strategy "$SAFETENSORS_LOAD_STRATEGY")
fi
if [ "$ENFORCE_EAGER" = "1" ]; then
    VLLM_ARGS+=(--enforce-eager)
fi
if [ "$ENABLE_AUTO_TOOL_CHOICE" = "1" ]; then
    VLLM_ARGS+=(--enable-auto-tool-choice)
    if [ -n "$TOOL_CALL_PARSER" ]; then
        VLLM_ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")
    fi
fi
if [ -n "$REASONING_PARSER" ]; then
    VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi
"${VLLM_ARGS[@]}" > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM 进程 PID: $VLLM_PID，日志: $VLLM_LOG"

check_vllm_health() {
    python - "$VLLM_HEALTH_URL" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=2) as response:
        raise SystemExit(0 if 200 <= response.status < 300 else 1)
except Exception:
    raise SystemExit(1)
PY
}

print_vllm_log_tail() {
    if [ -f "$VLLM_LOG" ]; then
        echo "========== vLLM LOG TAIL ($VLLM_LOG) =========="
        tail -n 120 "$VLLM_LOG" || true
        echo "========== END vLLM LOG TAIL =========="
    fi
}

# 等待服务就绪。大模型在 FUSE/AFS 上首次加载和编译可能很慢。
echo "等待 vLLM 服务就绪..."
VLLM_WAIT_STEPS=$(( (VLLM_WAIT_TIMEOUT_SECONDS + 4) / 5 ))
if [ "$VLLM_WAIT_STEPS" -lt 1 ]; then
    VLLM_WAIT_STEPS=1
fi
for i in $(seq 1 "$VLLM_WAIT_STEPS"); do
    if check_vllm_health; then
        echo "vLLM 服务已就绪 (等待了 $((i * 5)) 秒)"
        exit 0
    fi
    # 检查进程是否意外退出
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "错误：vLLM 进程意外退出，查看日志: $VLLM_LOG"
        print_vllm_log_tail
        exit 1
    fi
    if [ $((i % 12)) -eq 0 ]; then
        ELAPSED=$((i * 5))
        echo "[$(date)] 等待 vLLM 就绪: ${ELAPSED}/${VLLM_WAIT_TIMEOUT_SECONDS}s (PID=$VLLM_PID)"
        if [ -f "$VLLM_LOG" ]; then
            LAST_VLLM_LINE="$(tail -n 1 "$VLLM_LOG" 2>/dev/null || true)"
            if [ -n "$LAST_VLLM_LINE" ]; then
                echo "[$(date)] vLLM 最新日志: $LAST_VLLM_LINE"
            fi
        fi
    fi
    sleep 5
done

echo "错误：vLLM 服务在 ${VLLM_WAIT_TIMEOUT_SECONDS}s 内未就绪，查看日志: $VLLM_LOG"
print_vllm_log_tail
exit 1
