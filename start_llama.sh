#!/usr/bin/env bash
set -euo pipefail

# Qwen3.6-27B-MTP-GGUF launcher for the ACP CUDA 12.4 / Hopper (sm_90) image.
# The bundled llama-server was built from ggml-org/llama.cpp commit c5a4a0b.
#
# ACP service job (default, stays attached to the allocation):
#   bash start_llama.sh
#
# Interactive background launch:
#   LLAMA_RUN_MODE=background bash start_llama.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLAMA_WRAPPER_DIR="${LLAMA_WRAPPER_DIR:-/mnt/afs-p3/task3_2/share_scripts/llama.cpp}"
LLAMA_WRAPPER="${LLAMA_WRAPPER:-${LLAMA_WRAPPER_DIR}/run_qwen36_27b_llama_cpp.sh}"
LLAMA_SERVER="${LLAMA_SERVER:-${SCRIPT_DIR}/bin/llama-server-cuda12-sm90}"

MTP_MODEL_DIR="${MTP_MODEL_DIR:-/mnt/afs-p3/task3_2/share_model/unsloth/Qwen3.6-27B-MTP-GGUF}"
MTP_MODEL="${MTP_MODEL:-${MTP_MODEL_DIR}/Qwen3.6-27B-UD-Q8_K_XL.gguf}"
MMPROJ="${MMPROJ:-/mnt/afs-p3/task3_2/share_model/unsloth/Qwen3.6-27B-GGUF/mmproj-F16.gguf}"

HOST="${LLAMA_HOST:-0.0.0.0}"
PORT="${LLAMA_PORT:-8002}"
CUDA_DEVICES="${LLAMA_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}"
RUN_MODE="${LLAMA_RUN_MODE:-foreground}"
LOG_FILE="${LLAMA_LOG_FILE:-${SCRIPT_DIR}/logs/llama_qwen36_27b_mtp.log}"
PID_FILE="${LLAMA_PID_FILE:-${SCRIPT_DIR}/logs/llama_qwen36_27b_mtp.pid}"

require_file() {
    local description="$1"
    local path="$2"
    if [ ! -f "$path" ]; then
        echo "[ERROR] ${description} not found: $path" >&2
        exit 1
    fi
}

require_file "llama.cpp wrapper" "$LLAMA_WRAPPER"
require_file "Qwen3.6-27B MTP GGUF" "$MTP_MODEL"
require_file "Qwen3.6 vision projector" "$MMPROJ"
require_file "CUDA 12 llama-server" "$LLAMA_SERVER"

if [ ! -x "$LLAMA_WRAPPER" ]; then
    echo "[ERROR] llama.cpp wrapper is not executable: $LLAMA_WRAPPER" >&2
    exit 1
fi
if [ ! -x "$LLAMA_SERVER" ]; then
    echo "[ERROR] llama-server is not executable: $LLAMA_SERVER" >&2
    exit 1
fi

MISSING_LIBS="$(ldd "$LLAMA_SERVER" 2>/dev/null | awk '/not found/ { print }')"
if [ -n "$MISSING_LIBS" ]; then
    echo "[ERROR] llama-server has missing runtime libraries:" >&2
    echo "$MISSING_LIBS" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[ERROR] no NVIDIA GPU is visible; run this script inside a GPU ACP task" >&2
    exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
cd "$LLAMA_WRAPPER_DIR" || exit 1

SERVER_COMMAND=(
    env
    "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
    "LLAMA_SERVER=${LLAMA_SERVER}"
    "HOST=${HOST}"
    "PORT=${PORT}"
    "CTX_SIZE=${LLAMA_CTX_SIZE:-294912}"
    "PARALLEL=${LLAMA_PARALLEL:-3}"
    "N_GPU_LAYERS=${LLAMA_N_GPU_LAYERS:-999}"
    "CACHE_TYPE_K=${LLAMA_CACHE_TYPE_K:-q8_0}"
    "CACHE_TYPE_V=${LLAMA_CACHE_TYPE_V:-q8_0}"
    "VISION=${LLAMA_VISION:-true}"
    "THREADS=${LLAMA_THREADS:-96}"
    "THREADS_HTTP=${LLAMA_THREADS_HTTP:-32}"
    "BATCH_SIZE=${LLAMA_BATCH_SIZE:-1024}"
    "UBATCH_SIZE=${LLAMA_UBATCH_SIZE:-256}"
    "MTP=true"
    "SPEC_DRAFT_N_MAX=${LLAMA_SPEC_DRAFT_N_MAX:-2}"
    "THINKING=${LLAMA_THINKING:-true}"
    "REASONING=${LLAMA_REASONING:-auto}"
    "REASONING_PRESERVE=${LLAMA_REASONING_PRESERVE:-true}"
    "TEMP=${LLAMA_TEMP:-1.0}"
    "TOP_P=${LLAMA_TOP_P:-0.95}"
    "TOP_K=${LLAMA_TOP_K:-20}"
    "MIN_P=${LLAMA_MIN_P:-0.00}"
    "PRESENCE_PENALTY=${LLAMA_PRESENCE_PENALTY:-0.0}"
    "REPEAT_PENALTY=${LLAMA_REPEAT_PENALTY:-1.0}"
    "MTP_MODEL_DIR=${MTP_MODEL_DIR}"
    "MTP_MODEL=${MTP_MODEL}"
    "MMPROJ=${MMPROJ}"
    "ALIAS=${LLAMA_ALIAS:-unsloth/Qwen3.6-27B-GGUF}"
    "$LLAMA_WRAPPER"
    --no-kv-unified
    --cache-prompt
    --cache-ram "${LLAMA_CACHE_RAM_MIB:-32768}"
    --cache-idle-slots
    --ctx-checkpoints "${LLAMA_CTX_CHECKPOINTS:-64}"
    --cache-reuse "${LLAMA_CACHE_REUSE:-256}"
    --slot-prompt-similarity "${LLAMA_SLOT_PROMPT_SIMILARITY:-0.5}"
)

echo "[INFO] llama-server: $LLAMA_SERVER"
echo "[INFO] model:        $MTP_MODEL"
echo "[INFO] mmproj:       $MMPROJ"
echo "[INFO] CUDA devices: $CUDA_DEVICES"
echo "[INFO] endpoint:     http://127.0.0.1:${PORT}/v1"
echo "[INFO] log:          $LOG_FILE"
echo "[INFO] run mode:     $RUN_MODE"

case "$RUN_MODE" in
    foreground)
        echo "[INFO] starting in foreground; stop the ACP task to stop llama-server"
        exec "${SERVER_COMMAND[@]}" >> "$LOG_FILE" 2>&1
        ;;
    background)
        nohup "${SERVER_COMMAND[@]}" >> "$LOG_FILE" 2>&1 &
        SERVER_PID=$!
        printf '%s\n' "$SERVER_PID" > "$PID_FILE"
        echo "[INFO] llama-server started in background: PID=$SERVER_PID"
        echo "[INFO] wait for readiness with:"
        echo "       EXPECTED_MODEL=unsloth/Qwen3.6-27B-GGUF PORT=$PORT bash ${LLAMA_WRAPPER_DIR}/check_llama_cpp.sh"
        ;;
    *)
        echo "[ERROR] LLAMA_RUN_MODE must be foreground or background, got: $RUN_MODE" >&2
        exit 2
        ;;
esac
