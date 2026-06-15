#!/bin/bash
# =============================================================================
# ACP job entry for SceneExpert.
#
# Usage inside an ACP task:
#   cd /mnt/afs/task3_2/L202500276_lwz/projects/SceneExpert && bash scripts/acp_sceneexpert.sh
#
# Prefer editing the TODO block below for ACP-only parameters. Keep .env for
# base machine settings such as model/data/output paths and vLLM port.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Project path and optional .env loading.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="${SCENEEXPERT_PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
BASE_ENV_FILE="${SCENEEXPERT_BASE_ENV_FILE:-$PROJECT_DIR/.env}"

source_env_file() {
    local env_path="$1"
    local tmp_env
    tmp_env="$(mktemp)"
    sed 's/\r$//' "$env_path" > "$tmp_env"
    # shellcheck disable=SC1090
    source "$tmp_env"
    rm -f "$tmp_env"
}

copy_lf_file() {
    local src="$1"
    local dst="$2"
    sed 's/\r$//' "$src" > "$dst"
}

if [ -f "$BASE_ENV_FILE" ]; then
    source_env_file "$BASE_ENV_FILE"
fi

# ---------------------------------------------------------------------------
# 2. TODO: ACP job configuration.
#    Edit this block for each ACP multi-GPU submission.
#    Do not duplicate these ACP-only values in .env; this script is the source
#    of truth for the generated job-specific env overrides.
# ---------------------------------------------------------------------------

# TODO: Choose experiment. Recommended values:
#   ablation_2_qwen3_naive            baseline Qwen3 without SceneExpert
#   ablation_3_qwen3_harness          SceneExpert harness without memory
#   ablation_4_qwen3_harness_memory   legacy SceneExpert memory MVP
#   ablation_4a_qwen3_lexical_memory  lexical memory ablation, no vector index
#   ablation_4b_qwen3_vector_memory   BGE-M3 vector memory, requires index
#   ablation_4c_qwen3_hybrid_memory   recommended hybrid memory, requires index
#   ablation_5_qwen3_full             full/LoRA model, only after LoRA merge exists
ACP_EXPERIMENT="ablation_2_qwen3_naive"

# TODO: Match ACP requested GPU count. Leave ACP_CUDA_VISIBLE_DEVICES empty on
# scheduler-managed ACP jobs so the platform-provided CUDA_VISIBLE_DEVICES is
# preserved. Fill it only for manual debugging on a known-clean node.
ACP_GPUS=2
ACP_CUDA_VISIBLE_DEVICES=""

# TODO: 2xH100: 65536 is the stable default. For a faster smoke test, use 32768.
# 4xH100: try 131072 first.
ACP_MAX_MODEL_LEN=65536
ACP_GPU_MEMORY_UTILIZATION=0.90

# TODO: Multi-GPU should normally keep this at 0. If 2 GPUs still fail while
# loading the model, try 10. Single-GPU fallback may need 20.
ACP_CPU_OFFLOAD_GB=0

# TODO: Large MoE models on AFS/FUSE can need more than 30 minutes for first
# load + torch.compile. 7200 seconds is conservative for ACP jobs.
ACP_VLLM_WAIT_TIMEOUT_SECONDS=7200
ACP_VLLM_ENGINE_READY_TIMEOUT_S=7200

# TODO: The cluster model path is usually on AFS/FUSE. Prefetching safetensors
# avoids very slow shard-by-shard lazy reads when enough host RAM is available.
ACP_SAFETENSORS_LOAD_STRATEGY="prefetch"

# TODO: The current offline vLLM environment does not provide a compatible
# DeepGEMM backend. Disable it to avoid FP8 DeepGEMM warmup startup failures.
ACP_VLLM_USE_DEEP_GEMM=0
ACP_VLLM_MOE_USE_DEEP_GEMM=0
ACP_VLLM_DEEP_GEMM_WARMUP="skip"

# TODO: Keep this at 1 in the current offline vLLM 0.22.x environment. Logs
# showed DeepGEMM warmup still ran with the DeepGEMM env flags disabled when
# enforce_eager=False, so this hard-bypasses compile/cudagraph warmup.
ACP_VLLM_ENFORCE_EAGER=1

# TODO: Keep this at 1 unless artvip_sdf or partnet_mobility_sdf has been
# prepared under writable SCENEEXPERT_DATA_DIR. The fast HSSD-only reproduction should
# not start the articulated retrieval server.
ACP_DISABLE_ARTICULATED=0

# TODO: Keep this at 1 unless materials/ and materials/embeddings/ have been
# prepared under writable SCENEEXPERT_DATA_DIR. The fast HSSD-only reproduction should
# not start the materials retrieval server.
ACP_DISABLE_MATERIALS=0

# TODO: ACP nodes can expose 100+ CPU cores and be slow to spawn native
# geometry libraries from AFS/FUSE. 180s + 32 OMP threads avoids false startup
# failures without changing the collision geometry algorithm.
ACP_CONVEX_READY_TIMEOUT=180
ACP_CONVEX_MAX_OMP_THREADS=32

# TODO: Leave empty to use SCENEEXPERT_OUTPUT_DIR from .env.
ACP_OUTPUT_DIR=""

# TODO: Keep memory experiments sequential by default.
ACP_HYDRA_OVERRIDES="experiment.num_workers=1"

EXPERIMENT="${1:-$ACP_EXPERIMENT}"
shift || true

RUN_STAMP="$(date +'%Y%m%d_%H%M%S')"
RUN_NAME="${SCENEEXPERT_RUN_NAME:-acp_${EXPERIMENT}_${RUN_STAMP}}"
LOG_DIR="${SCENEEXPERT_ACP_LOG_DIR:-$PROJECT_DIR/tmp/acp_logs/$RUN_NAME}"
LOG_FILE="$LOG_DIR/console.log"
JOB_ENV_FILE="$LOG_DIR/sceneexpert_acp.env"

# ---------------------------------------------------------------------------
# 3. Single-node multi-GPU runtime environment.
# ---------------------------------------------------------------------------
if [ -n "$ACP_CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$ACP_CUDA_VISIBLE_DEVICES"
fi
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo,eth0,bond0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# Offline cluster defaults. The model and datasets should already be local.
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ---------------------------------------------------------------------------
# 4. Initialize workspace and generate per-job env file.
# ---------------------------------------------------------------------------
echo "========== INIT SCENEEXPERT ACP JOB =========="
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR"

if [ -f "$BASE_ENV_FILE" ]; then
    copy_lf_file "$BASE_ENV_FILE" "$JOB_ENV_FILE"
else
    echo "# Generated SceneExpert ACP env" > "$JOB_ENV_FILE"
fi

cat >> "$JOB_ENV_FILE" <<EOF

# --- ACP multi-GPU overrides generated at $RUN_STAMP ---
export SCENEEXPERT_RUN_NAME="$RUN_NAME"
export SCENEEXPERT_START_VLLM=1
export SCENEEXPERT_TENSOR_PARALLEL_SIZE=$ACP_GPUS
export SCENEEXPERT_MAX_MODEL_LEN=$ACP_MAX_MODEL_LEN
export SCENEEXPERT_GPU_MEMORY_UTILIZATION=$ACP_GPU_MEMORY_UTILIZATION
export SCENEEXPERT_VLLM_CPU_OFFLOAD_GB=$ACP_CPU_OFFLOAD_GB
export SCENEEXPERT_VLLM_WAIT_TIMEOUT_SECONDS=$ACP_VLLM_WAIT_TIMEOUT_SECONDS
export SCENEEXPERT_VLLM_ENGINE_READY_TIMEOUT_S=$ACP_VLLM_ENGINE_READY_TIMEOUT_S
export SCENEEXPERT_VLLM_SAFETENSORS_LOAD_STRATEGY="$ACP_SAFETENSORS_LOAD_STRATEGY"
export SCENEEXPERT_VLLM_USE_DEEP_GEMM=$ACP_VLLM_USE_DEEP_GEMM
export SCENEEXPERT_VLLM_MOE_USE_DEEP_GEMM=$ACP_VLLM_MOE_USE_DEEP_GEMM
export SCENEEXPERT_VLLM_DEEP_GEMM_WARMUP="$ACP_VLLM_DEEP_GEMM_WARMUP"
export SCENEEXPERT_VLLM_ENFORCE_EAGER=$ACP_VLLM_ENFORCE_EAGER
export SCENEEXPERT_DISABLE_ARTICULATED=$ACP_DISABLE_ARTICULATED
export SCENEEXPERT_DISABLE_MATERIALS=$ACP_DISABLE_MATERIALS
export SCENEEXPERT_CONVEX_READY_TIMEOUT=$ACP_CONVEX_READY_TIMEOUT
export SCENEEXPERT_CONVEX_MAX_OMP_THREADS=$ACP_CONVEX_MAX_OMP_THREADS
export SCENEEXPERT_VLLM_LOG="$LOG_DIR/vllm_server.log"
EOF

if [ -n "$ACP_OUTPUT_DIR" ]; then
    cat >> "$JOB_ENV_FILE" <<EOF
export SCENEEXPERT_OUTPUT_DIR="$ACP_OUTPUT_DIR"
export SCENEEXPERT_MEMORY_DIR="\${SCENEEXPERT_OUTPUT_DIR}/scene_expert_memory"
EOF
fi

export SCENEEXPERT_ENV_FILE="$JOB_ENV_FILE"

echo "Project: $PROJECT_DIR"
echo "Experiment: $EXPERIMENT"
echo "Run name: $RUN_NAME"
echo "Log dir: $LOG_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<scheduler-default>}"
echo "Tensor parallel size: $ACP_GPUS"
echo "Max model len: $ACP_MAX_MODEL_LEN"
echo "CPU offload GB/GPU: $ACP_CPU_OFFLOAD_GB"
echo "vLLM wait timeout: ${ACP_VLLM_WAIT_TIMEOUT_SECONDS}s"
echo "vLLM engine ready timeout: ${ACP_VLLM_ENGINE_READY_TIMEOUT_S}s"
echo "safetensors load strategy: ${ACP_SAFETENSORS_LOAD_STRATEGY:-default}"
echo "DeepGEMM: use=${ACP_VLLM_USE_DEEP_GEMM}, moe_use=${ACP_VLLM_MOE_USE_DEEP_GEMM}, warmup=${ACP_VLLM_DEEP_GEMM_WARMUP}"
echo "enforce eager: ${ACP_VLLM_ENFORCE_EAGER}"
echo "disable articulated retrieval: ${ACP_DISABLE_ARTICULATED}"
echo "disable materials retrieval: ${ACP_DISABLE_MATERIALS}"
echo "convex ready timeout: ${ACP_CONVEX_READY_TIMEOUT}s"
echo "convex max OMP threads: ${ACP_CONVEX_MAX_OMP_THREADS}"
echo "Env file: $SCENEEXPERT_ENV_FILE"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "========== GPU STATUS =========="
    nvidia-smi || true
fi

# ---------------------------------------------------------------------------
# 5. Run SceneExpert. Keep memory-mode runs conservative by default.
# ---------------------------------------------------------------------------
echo "========== START SCENEEXPERT =========="
set +e
# shellcheck disable=SC2086
bash scripts/run_experiment.sh "$EXPERIMENT" $ACP_HYDRA_OVERRIDES "$@" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo "========== SCENEEXPERT FINISHED =========="
echo "EXIT_CODE=$EXIT_CODE"
echo "Console log: $LOG_FILE"
echo "vLLM log: $LOG_DIR/vllm_server.log"
echo "Job env: $JOB_ENV_FILE"

exit "$EXIT_CODE"
