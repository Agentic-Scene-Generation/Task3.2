#!/bin/bash
# =============================================================================
# 集群单任务提交脚本：在同一个 job 内启动 vLLM + 运行实验
# 用法：
#   本地：bash scripts/run_experiment.sh ablation_2_qwen3_naive
#   SLURM：sbatch scripts/run_experiment.sh ablation_2_qwen3_naive
# =============================================================================
#SBATCH --job-name=scenesmith
#SBATCH --gpus=2
#SBATCH --output=outputs/slurm_%j.log

set -euo pipefail

EXPERIMENT=${1:-"ablation_2_qwen3_naive"}
shift || true

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
    echo "  已加载配置文件: $ENV_FILE"
fi
MODEL_ID="${SCENEEXPERT_MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
MODELS_DIR="${SCENEEXPERT_MODELS_DIR:-$PROJECT_DIR/models}"
MODEL_DIR="${SCENEEXPERT_MODEL_DIR:-$MODELS_DIR/${MODEL_ID##*/}}"
VLLM_PORT="${SCENEEXPERT_VLLM_PORT:-8000}"
VLLM_LOG="${SCENEEXPERT_VLLM_LOG:-$PROJECT_DIR/vllm_server.log}"
VLLM_HEALTH_URL="${SCENEEXPERT_VLLM_HEALTH_URL:-http://localhost:${VLLM_PORT}/health}"
VLLM_WAIT_TIMEOUT_SECONDS="${SCENEEXPERT_VLLM_WAIT_TIMEOUT_SECONDS:-1800}"
VLLM_ENGINE_READY_TIMEOUT_S="${SCENEEXPERT_VLLM_ENGINE_READY_TIMEOUT_S:-$VLLM_WAIT_TIMEOUT_SECONDS}"
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
START_VLLM="${SCENEEXPERT_START_VLLM:-1}"
DISABLE_ARTICULATED="${SCENEEXPERT_DISABLE_ARTICULATED:-0}"
DISABLE_MATERIALS="${SCENEEXPERT_DISABLE_MATERIALS:-0}"
CONVEX_READY_TIMEOUT="${SCENEEXPERT_CONVEX_READY_TIMEOUT:-}"
CONVEX_MAX_OMP_THREADS="${SCENEEXPERT_CONVEX_MAX_OMP_THREADS:-}"
RUN_NAME="${SCENEEXPERT_RUN_NAME:-$EXPERIMENT}"
DATA_DIR="${SCENEEXPERT_DATA_DIR:-$PROJECT_DIR/data}"
HSSD_DATA_DIR="${SCENEEXPERT_HSSD_DATA_DIR:-$DATA_DIR}"
OUTPUT_DIR="${SCENEEXPERT_OUTPUT_DIR:-$PROJECT_DIR/outputs}"
OPENCLIP_DIR="${SCENEEXPERT_OPENCLIP_DIR:-$PROJECT_DIR/data/openclip}"
OPENCLIP_CHECKPOINT="${SCENEEXPERT_OPENCLIP_CHECKPOINT:-$OPENCLIP_DIR/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin}"
OPENCLIP_CHECKPOINT_FILE="$OPENCLIP_CHECKPOINT"
if [ -d "$OPENCLIP_CHECKPOINT_FILE" ] || [[ "$OPENCLIP_CHECKPOINT_FILE" != *.bin ]]; then
    OPENCLIP_CHECKPOINT_FILE="$OPENCLIP_CHECKPOINT_FILE/open_clip_pytorch_model.bin"
fi
REQUIRE_LOCAL_OPENCLIP="${SCENEEXPERT_REQUIRE_LOCAL_OPENCLIP:-0}"
ACTIVATE_VENV="${SCENEEXPERT_ACTIVATE_VENV:-1}"
VENV_PATH="${SCENEEXPERT_VENV_PATH:-$PROJECT_DIR/.venv}"

cd "$PROJECT_DIR"

# ── 1. 激活环境 ──────────────────────────────────────────────
if [ "$ACTIVATE_VENV" = "1" ]; then
    if [ ! -f "$VENV_PATH/bin/activate" ]; then
        echo "错误：找不到虚拟环境激活脚本: $VENV_PATH/bin/activate"
        echo "如果你已经在 Conda/其他 venv 中，请设置 SCENEEXPERT_ACTIVATE_VENV=0。"
        exit 1
    fi
    source "$VENV_PATH/bin/activate"
    echo "  已激活虚拟环境: $VENV_PATH"
else
    echo "  跳过脚本内虚拟环境激活，使用当前 Python: $(python -c 'import sys; print(sys.executable)')"
fi

# Fail before the expensive vLLM startup when a repository edit contains a
# Python syntax error.
if [ "${SCENEEXPERT_SKIP_PYTHON_PREFLIGHT:-0}" != "1" ]; then
    echo "  Running Python syntax preflight..."
    python -m compileall -q main.py scenesmith
fi

NUMPY_VERSION="$(python -c 'import numpy as np; print(np.__version__)' 2>/dev/null || echo missing)"
if [[ "$NUMPY_VERSION" == 2.* ]] && [ "${SCENEEXPERT_ALLOW_NUMPY2:-0}" != "1" ]; then
    echo "错误：当前 NumPy 版本为 $NUMPY_VERSION，但 bpy/Blender 扩展需要 NumPy 1.x ABI。"
    echo "请先在当前虚拟环境中执行："
    echo "  python -m pip install 'numpy>=1.26,<2.0'"
    echo "如果你确认当前 bpy wheel 已兼容 NumPy 2.x，可设置 SCENEEXPERT_ALLOW_NUMPY2=1 跳过该检查。"
    exit 1
fi

# Import the same core chain used by main.py before allocating GPU memory.
if [ "${SCENEEXPERT_SKIP_PYTHON_PREFLIGHT:-0}" != "1" ]; then
    echo "  Running Python import preflight..."
    PYTHONDONTWRITEBYTECODE=1 python -c \
        "from scenesmith.experiments import build_experiment; from scenesmith.agent_utils.stage_working_memory import StageWorkingMemory; print('  Python preflight passed')"
fi

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
TENSOR_PARALLEL_SIZE="${SCENEEXPERT_TENSOR_PARALLEL_SIZE:-$NUM_GPUS}"
echo "  可见 GPU 数量: $NUM_GPUS (tensor_parallel_size=$TENSOR_PARALLEL_SIZE)"
echo "  模型 ID: $MODEL_ID"
echo "  模型目录: $MODEL_DIR"
echo "  可写数据目录: $DATA_DIR"
echo "  HSSD/HSM 只读目录: $HSSD_DATA_DIR"
echo "  OpenCLIP 目录: $OPENCLIP_DIR"
echo "  OpenCLIP checkpoint: $OPENCLIP_CHECKPOINT_FILE"
if [ -n "$CONVEX_READY_TIMEOUT" ]; then
    echo "  ConvexDecompositionServer ready timeout: ${CONVEX_READY_TIMEOUT}s"
fi
if [ -n "$CONVEX_MAX_OMP_THREADS" ]; then
    echo "  ConvexDecompositionServer max OMP threads: $CONVEX_MAX_OMP_THREADS"
fi
echo "  输出目录: $OUTPUT_DIR"
echo "  禁用 articulated retrieval: $DISABLE_ARTICULATED"
echo "  禁用 materials retrieval: $DISABLE_MATERIALS"

if [ "$REQUIRE_LOCAL_OPENCLIP" = "1" ] && [ ! -f "$OPENCLIP_CHECKPOINT_FILE" ]; then
    echo "错误：缺少 HSSD 检索所需的本地 OpenCLIP 权重。"
    echo "  期望文件: $OPENCLIP_CHECKPOINT_FILE"
    echo "  请把 DFN5B-CLIP-ViT-H-14-378 的 open_clip_pytorch_model.bin 放到该路径，"
    echo "  或在 .env 中修改 SCENEEXPERT_OPENCLIP_CHECKPOINT。"
    echo "  临时允许在线/HF 缓存回退可设 SCENEEXPERT_REQUIRE_LOCAL_OPENCLIP=0，但离线集群不建议这样做。"
    exit 1
fi

EXTRA_HYDRA_OVERRIDES=()
if [ -n "$CONVEX_READY_TIMEOUT" ]; then
    EXTRA_HYDRA_OVERRIDES+=(
        "furniture_agent.collision_geometry.server_ready_timeout=$CONVEX_READY_TIMEOUT"
        "manipuland_agent.collision_geometry.server_ready_timeout=$CONVEX_READY_TIMEOUT"
        "wall_agent.collision_geometry.server_ready_timeout=$CONVEX_READY_TIMEOUT"
        "ceiling_agent.collision_geometry.server_ready_timeout=$CONVEX_READY_TIMEOUT"
    )
fi

if [ -n "$CONVEX_MAX_OMP_THREADS" ]; then
    EXTRA_HYDRA_OVERRIDES+=(
        "furniture_agent.collision_geometry.max_omp_threads=$CONVEX_MAX_OMP_THREADS"
        "manipuland_agent.collision_geometry.max_omp_threads=$CONVEX_MAX_OMP_THREADS"
        "wall_agent.collision_geometry.max_omp_threads=$CONVEX_MAX_OMP_THREADS"
        "ceiling_agent.collision_geometry.max_omp_threads=$CONVEX_MAX_OMP_THREADS"
    )
fi

if [ "$DISABLE_ARTICULATED" = "1" ]; then
    EXTRA_HYDRA_OVERRIDES+=(
        "furniture_agent.asset_manager.router.strategies.articulated.enabled=false"
        "manipuland_agent.asset_manager.router.strategies.articulated.enabled=false"
        "wall_agent.asset_manager.router.strategies.articulated.enabled=false"
        "ceiling_agent.asset_manager.router.strategies.articulated.enabled=false"
        "furniture_agent.asset_manager.articulated.sources.partnet_mobility.enabled=false"
        "furniture_agent.asset_manager.articulated.sources.artvip.enabled=false"
        "manipuland_agent.asset_manager.articulated.sources.partnet_mobility.enabled=false"
        "manipuland_agent.asset_manager.articulated.sources.artvip.enabled=false"
        "wall_agent.asset_manager.articulated.sources.partnet_mobility.enabled=false"
        "wall_agent.asset_manager.articulated.sources.artvip.enabled=false"
        "ceiling_agent.asset_manager.articulated.sources.partnet_mobility.enabled=false"
        "ceiling_agent.asset_manager.articulated.sources.artvip.enabled=false"
    )
    echo "  将通过 Hydra 覆盖关闭 articulated 策略；补齐 artvip_sdf/partnet_mobility_sdf 后可设 SCENEEXPERT_DISABLE_ARTICULATED=0。"
fi

if [ "$DISABLE_MATERIALS" = "1" ]; then
    EXTRA_HYDRA_OVERRIDES+=(
        "floor_plan_agent.materials.use_retrieval_server=false"
        "furniture_agent.asset_manager.router.strategies.thin_covering.enabled=false"
        "furniture_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false"
        "manipuland_agent.asset_manager.router.strategies.thin_covering.enabled=false"
        "manipuland_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false"
        "wall_agent.asset_manager.router.strategies.thin_covering.enabled=false"
        "wall_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false"
        "ceiling_agent.asset_manager.router.strategies.thin_covering.enabled=false"
        "ceiling_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false"
    )
    echo "  将通过 Hydra 覆盖关闭 materials retrieval/thin_covering；补齐 materials/embeddings 后可设 SCENEEXPERT_DISABLE_MATERIALS=0。"
fi

if [ "$START_VLLM" = "1" ]; then
    VLLM_LAUNCH_MODE=""
    if ! command -v vllm >/dev/null 2>&1; then
        if python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('vllm.entrypoints.openai.api_server') else 1)" >/dev/null 2>&1; then
            VLLM_LAUNCH_MODE="python-module"
            echo "  未找到 vllm 命令，改用当前 Python 的 vLLM 模块入口启动。"
        else
            echo "错误：找不到 vLLM 命令，当前 Python 也无法导入 vLLM。"
            echo "  当前 Python: $(python -c 'import sys; print(sys.executable)')"
            echo "  安装示例：python -m pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple"
            echo "  离线安装：python -m pip install --no-index --find-links /path/to/wheelhouse vllm"
            echo "  如果已有其他 OpenAI-compatible 本地服务，请设置："
            echo "    export SCENEEXPERT_START_VLLM=0"
            echo "    export OPENAI_BASE_URL=http://host:port/v1"
            exit 1
        fi
    else
        VLLM_LAUNCH_MODE="cli"
    fi
    if [ ! -d "$MODEL_DIR" ]; then
        echo "错误：模型目录不存在: $MODEL_DIR"
        echo "请确认 .env 中的 SCENEEXPERT_MODEL_DIR 指向已下载好的本地模型目录。"
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
        elif [ "${SCENEEXPERT_GPU_PREFLIGHT_CHECK:-1}" = "1" ]; then
            GPU_PREFLIGHT_FAILED=0
            while IFS=',' read -r gpu_idx gpu_name total_mib used_mib free_mib; do
                total_num="$(echo "$total_mib" | tr -dc '0-9')"
                used_num="$(echo "$used_mib" | tr -dc '0-9')"
                free_num="$(echo "$free_mib" | tr -dc '0-9')"
                if [ -z "$total_num" ] || [ -z "$free_num" ]; then
                    continue
                fi
                required_num="$(python -c "import math; print(math.ceil(float('$total_num') * float('$GPU_MEMORY_UTILIZATION')))" 2>/dev/null || echo "")"
                if [ -n "$required_num" ] && [ "$free_num" -lt "$required_num" ]; then
                    echo "错误：GPU $gpu_idx 启动前空闲显存不足。"
                    echo "  GPU: $gpu_name"
                    echo "  total=${total_num}MiB used=${used_num:-unknown}MiB free=${free_num}MiB required≈${required_num}MiB (gpu_memory_utilization=$GPU_MEMORY_UTILIZATION)"
                    GPU_PREFLIGHT_FAILED=1
                fi
            done <<< "$GPU_STATUS"
            if [ "$GPU_PREFLIGHT_FAILED" = "1" ]; then
                echo "vLLM 会在 worker 初始化阶段失败。请换用空闲 ACP GPU、清理残留进程，或确认脚本没有覆盖调度器分配的 CUDA_VISIBLE_DEVICES。"
                echo "确需跳过该检查可设置 SCENEEXPERT_GPU_PREFLIGHT_CHECK=0，但不建议在正式实验中这样做。"
                exit 1
            fi
        fi
    fi
fi

export SCENEEXPERT_MODEL_ID="$MODEL_ID"
export SCENEEXPERT_DATA_DIR="$DATA_DIR"
export SCENEEXPERT_HSSD_DATA_DIR="$HSSD_DATA_DIR"
export SCENEEXPERT_OPENCLIP_DIR="$OPENCLIP_DIR"
export SCENEEXPERT_REQUIRE_LOCAL_OPENCLIP="$REQUIRE_LOCAL_OPENCLIP"
if [ -n "${SCENEEXPERT_OPENCLIP_CHECKPOINT:-}" ] || [ -f "$OPENCLIP_CHECKPOINT_FILE" ] || [ "$REQUIRE_LOCAL_OPENCLIP" = "1" ]; then
    export SCENEEXPERT_OPENCLIP_CHECKPOINT="$OPENCLIP_CHECKPOINT_FILE"
else
    unset SCENEEXPERT_OPENCLIP_CHECKPOINT
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed}"
if [ "$START_VLLM" = "1" ]; then
    export OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    export VLLM_ENGINE_READY_TIMEOUT_S="$VLLM_ENGINE_READY_TIMEOUT_S"
    export VLLM_USE_DEEP_GEMM="$USE_DEEP_GEMM"
    export VLLM_MOE_USE_DEEP_GEMM="$MOE_USE_DEEP_GEMM"
    export VLLM_DEEP_GEMM_WARMUP="$DEEP_GEMM_WARMUP"
else
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:${VLLM_PORT}/v1}"
fi

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

# ── 2. 后台启动 vLLM ─────────────────────────────────────────
VLLM_PID=""
if [ "$START_VLLM" = "1" ]; then
    echo "[$(date)] 启动 vLLM..."
    echo "  模型目录: $MODEL_DIR"
    echo "  served model: $MODEL_ID"
    echo "  端口: $VLLM_PORT"
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
    echo "[$(date)] vLLM PID=$VLLM_PID，等待就绪..."
else
    echo "[$(date)] 跳过启动 vLLM，使用已有服务: $OPENAI_BASE_URL"
fi

cleanup() {
    if [ -n "$VLLM_PID" ]; then
        echo "[$(date)] 关闭 vLLM (PID=$VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                wait "$VLLM_PID" 2>/dev/null || true
                return
            fi
            sleep 1
        done
        echo "[$(date)] vLLM 未在 30 秒内退出，强制结束 (PID=$VLLM_PID)"
        kill -9 "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── 3. 等待 /health ──────────────────────────────────────────
READY=0
VLLM_WAIT_STEPS=$(( (VLLM_WAIT_TIMEOUT_SECONDS + 4) / 5 ))
if [ "$VLLM_WAIT_STEPS" -lt 1 ]; then
    VLLM_WAIT_STEPS=1
fi
for i in $(seq 1 "$VLLM_WAIT_STEPS"); do
    if check_vllm_health; then
        echo "[$(date)] vLLM 就绪 ($((i * 5))s)"
        READY=1
        break
    fi
    if [ -n "$VLLM_PID" ] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(date)] 错误：vLLM 进程退出，见 $VLLM_LOG"
        print_vllm_log_tail
        exit 1
    fi
    if [ $((i % 12)) -eq 0 ]; then
        ELAPSED=$((i * 5))
        echo "[$(date)] 等待 vLLM 就绪: ${ELAPSED}/${VLLM_WAIT_TIMEOUT_SECONDS}s (PID=${VLLM_PID:-external})"
        if [ -f "$VLLM_LOG" ]; then
            LAST_VLLM_LINE="$(tail -n 1 "$VLLM_LOG" 2>/dev/null || true)"
            if [ -n "$LAST_VLLM_LINE" ]; then
                echo "[$(date)] vLLM 最新日志: $LAST_VLLM_LINE"
            fi
        fi
    fi
    sleep 5
done
if [ "$READY" != "1" ]; then
    echo "[$(date)] 错误：vLLM 服务在 ${VLLM_WAIT_TIMEOUT_SECONDS}s 内未就绪，见 $VLLM_LOG"
    print_vllm_log_tail
    exit 1
fi

# ── 4. 运行实验 ──────────────────────────────────────────────
echo "[$(date)] 运行实验: $EXPERIMENT"
python main.py experiment="$EXPERIMENT" +name="$RUN_NAME" "${EXTRA_HYDRA_OVERRIDES[@]}" "$@"
# python main.py +name=branch_2 \
#   experiment.pipeline.start_stage=ceiling_mounted \
#   experiment.pipeline.resume_from_path=outputs/2026-05-29/10-07-02
