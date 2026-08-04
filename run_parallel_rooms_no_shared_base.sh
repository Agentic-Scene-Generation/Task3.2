#!/usr/bin/env bash
# Start the embedding service and Qwen3.6-27B llama-server once, then generate
# different scenes concurrently without generating or loading a shared base.
#
# Default:
#   - 6 concurrent scene processes
#   - 8 different cases from scripts/run_parallel_critic_on.sh
#   - each scene runs independently from floor_plan through manipuland
#
# Usage:
#   bash run_parallel_rooms_no_shared_base.sh
#
# Common overrides:
#   SCENE_CONCURRENCY=4 MAX_CASES=8 bash run_parallel_rooms_no_shared_base.sh
#   APT_UBUNTU_MIRROR=https://mirrors.aliyun.com/ubuntu bash run_parallel_rooms_no_shared_base.sh
#   APT_UBUNTU_MIRROR= bash run_parallel_rooms_no_shared_base.sh  # use system sources

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_ROOT="${USER_ROOT:-$(dirname "$PROJECT_ROOT")}"
SHARED_ROOT="/mnt/afs-p3/task3_2"

SCENE_CONCURRENCY="${SCENE_CONCURRENCY:-6}"
MAX_CASES="${MAX_CASES:-8}"
FLOOR_PLAN_MODE="${FLOOR_PLAN_MODE:-room}"
EMBEDDING_PORT="${EMBEDDING_PORT:-8014}"
LLM_PORT="${LLM_PORT:-8002}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-7200}"
APT_UPDATE_TIMEOUT="${APT_UPDATE_TIMEOUT:-300}"
APT_INSTALL_TIMEOUT="${APT_INSTALL_TIMEOUT:-900}"
# Use "-" rather than ":-" so an explicitly empty value disables the mirror.
APT_UBUNTU_MIRROR="${APT_UBUNTU_MIRROR-https://mirrors.tuna.tsinghua.edu.cn/ubuntu}"
EXPECTED_MODEL="${EXPECTED_MODEL:-unsloth/Qwen3.6-27B-GGUF}"
RUN_ID="${RUN_ID:-parallel_rooms_no_shared_base_$(date +%Y%m%d_%H%M%S)}"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
        PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
    elif [ -x "$USER_ROOT/scenesmith-qwen/.venv/bin/python" ]; then
        PYTHON_BIN="$USER_ROOT/scenesmith-qwen/.venv/bin/python"
    else
        PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
    fi
fi
EMBEDDING_SERVER="${EMBEDDING_SERVER:-$PROJECT_ROOT/bin/llama-server-cuda12-sm90}"
EMBEDDING_MODEL_DIR="${EMBEDDING_MODEL_DIR:-$SHARED_ROOT/share_model/Qwen/Qwen3-VL-Embedding}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-$EMBEDDING_MODEL_DIR/Qwen3-VL-Embedding-2B-Q8_0.gguf}"
EMBEDDING_MMPROJ="${EMBEDDING_MMPROJ:-$EMBEDDING_MODEL_DIR/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf}"
LLAMA_LAUNCHER="${LLAMA_LAUNCHER:-$PROJECT_ROOT/start_llama.sh}"
LLM_CHECKER="${LLM_CHECKER:-$SHARED_ROOT/share_scripts/llama.cpp/check_llama_cpp.sh}"
CRITIC_RUNNER="${CRITIC_RUNNER:-$PROJECT_ROOT/scripts/run_parallel_critic_on.sh}"

MTP_MODEL_DIR="${MTP_MODEL_DIR:-$SHARED_ROOT/share_model/unsloth/Qwen3.6-27B-MTP-GGUF}"
MTP_MODEL="${MTP_MODEL:-$MTP_MODEL_DIR/Qwen3.6-27B-UD-Q8_K_XL.gguf}"
MMPROJ="${MMPROJ:-$SHARED_ROOT/share_model/unsloth/Qwen3.6-27B-GGUF/mmproj-F16.gguf}"

HSSD_DATA_PATH="${HSSD_DATA_PATH:-$SHARED_ROOT/share_data/hsm/hssd-models}"
HSSD_PREPROCESSED_PATH="${HSSD_PREPROCESSED_PATH:-$SHARED_ROOT/share_data/hsm/preprocessed}"
HSSD_RENDERED_ASSETS_DIR="${HSSD_RENDERED_ASSETS_DIR:-$SHARED_ROOT/share_data/scenesmith/hssd_rendered_assets}"
HSSD_ZVEC_SOURCE_PATH="${HSSD_ZVEC_SOURCE_PATH:-$SHARED_ROOT/share_data/scenesmith/hssd_zvec_collection}"
# Zvec opens its index files with writable mappings even in read-only mode.
# /mnt/afs-p3 is mounted read-only, so use a persistent writable local copy.
HSSD_ZVEC_COLLECTION_PATH="${HSSD_ZVEC_COLLECTION_PATH:-$USER_ROOT/.cache/scenesmith/hssd_zvec_collection}"

EMBEDDING_LOG="${EMBEDDING_LOG:-$PROJECT_ROOT/logs/llama_embedding.log}"
LLM_LOG="${LLM_LOG:-$PROJECT_ROOT/logs/llama_qwen36_27b_mtp.log}"
LLM_LAUNCH_LOG="${LLM_LAUNCH_LOG:-$PROJECT_ROOT/logs/llama_qwen36_27b_launcher.log}"
SYSTEM_DEPS_LOG="${SYSTEM_DEPS_LOG:-$PROJECT_ROOT/logs/system_deps_parallel_rooms.log}"

SYSTEM_PACKAGES=(
    libgl1
    libgl1-mesa-dri
    libglib2.0-0
    libgomp1
    libx11-6
    libxrender1
    libsm6
    libice6
    libxext6
    libxi6
    libxxf86vm1
    libxfixes3
    libxkbcommon0
    libegl1
    libegl-mesa0
    libgles2
    libegl-dev
)

require_positive_integer() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] $name must be a positive integer, got: $value" >&2
        exit 2
    fi
}

require_file() {
    local description="$1"
    local path="$2"
    if [ ! -f "$path" ]; then
        echo "[ERROR] $description not found: $path" >&2
        exit 1
    fi
}

require_directory() {
    local description="$1"
    local path="$2"
    if [ ! -d "$path" ]; then
        echo "[ERROR] $description not found: $path" >&2
        exit 1
    fi
}

ensure_local_zvec_collection() {
    if [ -d "$HSSD_ZVEC_COLLECTION_PATH" ]; then
        echo "[OK] writable HSSD Zvec collection is available: $HSSD_ZVEC_COLLECTION_PATH"
        return
    fi

    require_directory "shared HSSD Zvec collection" "$HSSD_ZVEC_SOURCE_PATH"
    local target_parent
    local staging_path
    target_parent="$(dirname "$HSSD_ZVEC_COLLECTION_PATH")"
    mkdir -p "$target_parent"
    staging_path="$(mktemp -d "${HSSD_ZVEC_COLLECTION_PATH}.tmp.XXXXXX")"

    echo "[INFO] staging HSSD Zvec collection on writable storage"
    echo "[INFO] source: $HSSD_ZVEC_SOURCE_PATH"
    echo "[INFO] target: $HSSD_ZVEC_COLLECTION_PATH"
    if ! cp -a --no-preserve=ownership \
        "$HSSD_ZVEC_SOURCE_PATH/." "$staging_path/"; then
        echo "[ERROR] failed to stage the HSSD Zvec collection" >&2
        echo "        incomplete staging directory: $staging_path" >&2
        exit 1
    fi
    mv "$staging_path" "$HSSD_ZVEC_COLLECTION_PATH"
    echo "[OK] HSSD Zvec collection staged"
}

system_deps_installed() {
    command -v dpkg-query >/dev/null 2>&1 || return 1
    local package status
    for package in "${SYSTEM_PACKAGES[@]}"; do
        status="$(dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null || true)"
        [ "$status" = "installed" ] || return 1
    done
}

ensure_system_deps() {
    local apt_sources_file=""
    local -a apt_source_options=()

    if system_deps_installed; then
        echo "[OK] Task3.2 system graphics dependencies are already installed"
        return
    fi
    if [ "${SCENEEXPERT_INSTALL_SYSTEM_DEPS:-auto}" = "0" ]; then
        echo "[ERROR] Task3.2 graphics dependencies are missing and installation is disabled" >&2
        exit 1
    fi
    if [ "$(id -u)" -ne 0 ] || ! command -v apt-get >/dev/null 2>&1; then
        echo "[ERROR] Task3.2 graphics dependencies are missing" >&2
        echo "        Run in the root ACP image or install: ${SYSTEM_PACKAGES[*]}" >&2
        exit 1
    fi

    echo "[INFO] Installing Task3.2 system graphics dependencies"
    echo "[INFO] installation log: $SYSTEM_DEPS_LOG"
    echo "[INFO] apt update timeout: ${APT_UPDATE_TIMEOUT}s; install timeout: ${APT_INSTALL_TIMEOUT}s"
    if [ -n "$APT_UBUNTU_MIRROR" ]; then
        if [ ! -f /etc/apt/sources.list ]; then
            echo "[ERROR] APT_UBUNTU_MIRROR was set, but /etc/apt/sources.list does not exist" >&2
            exit 1
        fi
        apt_sources_file="$(mktemp /tmp/task32-apt-sources.XXXXXX.list)"
        sed \
            -e "s|http://archive.ubuntu.com/ubuntu|${APT_UBUNTU_MIRROR%/}|g" \
            -e "s|https://archive.ubuntu.com/ubuntu|${APT_UBUNTU_MIRROR%/}|g" \
            -e "s|http://security.ubuntu.com/ubuntu|${APT_UBUNTU_MIRROR%/}|g" \
            -e "s|https://security.ubuntu.com/ubuntu|${APT_UBUNTU_MIRROR%/}|g" \
            /etc/apt/sources.list > "$apt_sources_file"
        apt_source_options=(
            -o "Dir::Etc::sourcelist=$apt_sources_file"
            -o "Dir::Etc::sourceparts=-"
        )
        echo "[INFO] apt mirror: ${APT_UBUNTU_MIRROR%/} (temporary; system sources unchanged)"
    else
        echo "[INFO] apt mirror: disabled; using system sources"
    fi

    if ! timeout --foreground "${APT_UPDATE_TIMEOUT}s" \
        env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        -u ALL_PROXY -u all_proxy \
        apt-get \
        "${apt_source_options[@]}" \
        -o Acquire::Retries=2 \
        -o Acquire::http::Timeout=30 \
        -o Acquire::https::Timeout=30 \
        -o DPkg::Lock::Timeout=60 \
        update -q > "$SYSTEM_DEPS_LOG" 2>&1; then
        echo "[ERROR] apt-get update failed or exceeded ${APT_UPDATE_TIMEOUT}s; last log lines:" >&2
        tail -n 40 "$SYSTEM_DEPS_LOG" >&2 || true
        [ -z "$apt_sources_file" ] || rm -f "$apt_sources_file"
        exit 1
    fi
    if ! timeout --foreground "${APT_INSTALL_TIMEOUT}s" \
        env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        -u ALL_PROXY -u all_proxy DEBIAN_FRONTEND=noninteractive \
        apt-get \
        "${apt_source_options[@]}" \
        -o Acquire::Retries=2 \
        -o Acquire::http::Timeout=30 \
        -o Acquire::https::Timeout=30 \
        -o DPkg::Lock::Timeout=60 \
        install -y --no-install-recommends \
        "${SYSTEM_PACKAGES[@]}" >> "$SYSTEM_DEPS_LOG" 2>&1; then
        echo "[ERROR] graphics dependency installation failed or exceeded ${APT_INSTALL_TIMEOUT}s; last log lines:" >&2
        tail -n 60 "$SYSTEM_DEPS_LOG" >&2 || true
        [ -z "$apt_sources_file" ] || rm -f "$apt_sources_file"
        exit 1
    fi
    [ -z "$apt_sources_file" ] || rm -f "$apt_sources_file"
    echo "[OK] Task3.2 system graphics dependencies installed"
}

require_positive_integer SCENE_CONCURRENCY "$SCENE_CONCURRENCY"
require_positive_integer MAX_CASES "$MAX_CASES"
require_positive_integer EMBEDDING_PORT "$EMBEDDING_PORT"
require_positive_integer LLM_PORT "$LLM_PORT"
require_positive_integer WAIT_TIMEOUT "$WAIT_TIMEOUT"
require_positive_integer APT_UPDATE_TIMEOUT "$APT_UPDATE_TIMEOUT"
require_positive_integer APT_INSTALL_TIMEOUT "$APT_INSTALL_TIMEOUT"
case "$FLOOR_PLAN_MODE" in
    room|house|polygon) ;;
    *)
        echo "[ERROR] FLOOR_PLAN_MODE must be room, house, or polygon" >&2
        exit 2
        ;;
esac

require_file "embedding llama-server" "$EMBEDDING_SERVER"
require_file "embedding model" "$EMBEDDING_MODEL"
require_file "embedding vision projector" "$EMBEDDING_MMPROJ"
require_file "Task3.2 llama launcher" "$LLAMA_LAUNCHER"
require_file "llama.cpp readiness checker" "$LLM_CHECKER"
require_file "parallel scene runner" "$CRITIC_RUNNER"
require_file "Qwen MTP model" "$MTP_MODEL"
require_file "Qwen vision projector" "$MMPROJ"
require_directory "HSSD model directory" "$HSSD_DATA_PATH"
require_directory "HSSD preprocessed directory" "$HSSD_PREPROCESSED_PATH"
require_directory "HSSD rendered-assets directory" "$HSSD_RENDERED_ASSETS_DIR"
ensure_local_zvec_collection
require_directory "HSSD Zvec collection" "$HSSD_ZVEC_COLLECTION_PATH"

if [ ! -x "$EMBEDDING_SERVER" ]; then
    echo "[ERROR] embedding llama-server is not executable: $EMBEDDING_SERVER" >&2
    exit 1
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "[ERROR] Task3.2 Python is not executable: $PYTHON_BIN" >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "[ERROR] curl is required for service readiness checks" >&2
    exit 1
fi

mkdir -p "$(dirname "$EMBEDDING_LOG")" "$(dirname "$LLM_LOG")" \
    "$(dirname "$LLM_LAUNCH_LOG")" "$(dirname "$SYSTEM_DEPS_LOG")"

ensure_system_deps

# The fallback venv uses the SceneSmith Conda interpreter. Native extensions
# such as sqlite3 and ICU must resolve libstdc++ from that same base prefix.
PYTHON_BASE_PREFIX="$("$PYTHON_BIN" -c 'import sys; print(sys.base_prefix)')"
SCENE_RUNTIME_LIB_DIR="${SCENEEXPERT_SCENE_RUNTIME_LIB_DIR:-$PYTHON_BASE_PREFIX/lib}"
if [ -d "$SCENE_RUNTIME_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${SCENE_RUNTIME_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export PATH="$(dirname "$PYTHON_BIN"):${PYTHON_BASE_PREFIX}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LIDRA_SKIP_INIT=1
unset LIBGL_DRIVERS_PATH

if ! "$PYTHON_BIN" -c \
    'import bpy; import sqlite3; import zvec; from pydrake.all import Quaternion' \
    >/dev/null 2>&1; then
    echo "[ERROR] Task3.2 native import preflight failed" >&2
    echo "        Python: $PYTHON_BIN" >&2
    "$PYTHON_BIN" -c \
        'import bpy; import sqlite3; import zvec; from pydrake.all import Quaternion' \
        >&2 || true
    exit 1
fi
echo "[OK] Task3.2 native import preflight passed"

EMBEDDING_PID=""
LLM_PID=""
cleanup_started=false

cleanup() {
    local exit_code=$?
    if [ "$cleanup_started" = "true" ]; then
        return
    fi
    cleanup_started=true
    trap - EXIT INT TERM HUP

    for pid in "$LLM_PID" "$EMBEDDING_PID"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for pid in "$LLM_PID" "$EMBEDDING_PID"; do
        if [ -n "$pid" ]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    exit "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

wait_for_health() {
    local name="$1"
    local port="$2"
    local pid="$3"
    local deadline=$((SECONDS + WAIT_TIMEOUT))

    echo "[INFO] waiting for $name on port $port"
    while [ "$SECONDS" -lt "$deadline" ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[ERROR] $name exited before becoming ready" >&2
            return 1
        fi
        if curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            echo "[OK] $name is ready"
            return 0
        fi
        sleep 5
    done

    echo "[ERROR] timed out waiting for $name after ${WAIT_TIMEOUT}s" >&2
    return 1
}

cd "$PROJECT_ROOT"

echo "[1/3] Starting embedding service"
nohup env \
    LD_LIBRARY_PATH="$(dirname "$EMBEDDING_SERVER")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    LLAMA_MEDIA_MARKER='<__media__>' \
    "$EMBEDDING_SERVER" \
    --model "$EMBEDDING_MODEL" \
    --mmproj "$EMBEDDING_MMPROJ" \
    --alias Qwen3-VL-Embedding-2B-Q8_0 \
    --host 0.0.0.0 \
    --port "$EMBEDDING_PORT" \
    --embedding \
    --pooling last \
    --embd-normalize 2 \
    --ctx-size $((2048 * SCENE_CONCURRENCY)) \
    --n-gpu-layers auto \
    --parallel "$SCENE_CONCURRENCY" \
    --batch-size 4096 \
    --ubatch-size 4096 \
    --threads-http "$SCENE_CONCURRENCY" \
    --no-warmup \
    > "$EMBEDDING_LOG" 2>&1 &
EMBEDDING_PID=$!

echo "[2/3] Starting Qwen3.6-27B through Task3.2/start_llama.sh"
nohup env \
    LLAMA_RUN_MODE=foreground \
    LLAMA_WRAPPER_DIR="$SHARED_ROOT/share_scripts/llama.cpp" \
    LLAMA_PORT="$LLM_PORT" \
    LLAMA_CTX_SIZE=$((65536 * SCENE_CONCURRENCY)) \
    LLAMA_PARALLEL="$SCENE_CONCURRENCY" \
    LLAMA_N_GPU_LAYERS=999 \
    LLAMA_CACHE_TYPE_K=q8_0 \
    LLAMA_CACHE_TYPE_V=q8_0 \
    LLAMA_VISION=true \
    LLAMA_THREADS=16 \
    LLAMA_THREADS_HTTP="$SCENE_CONCURRENCY" \
    LLAMA_BATCH_SIZE=1024 \
    LLAMA_UBATCH_SIZE=256 \
    LLAMA_SPEC_DRAFT_N_MAX=2 \
    LLAMA_THINKING=true \
    LLAMA_REASONING=auto \
    LLAMA_REASONING_PRESERVE=true \
    LLAMA_TEMP=1.0 \
    LLAMA_TOP_P=0.95 \
    LLAMA_TOP_K=20 \
    LLAMA_MIN_P=0.00 \
    LLAMA_PRESENCE_PENALTY=0.0 \
    LLAMA_REPEAT_PENALTY=1.0 \
    LLAMA_CACHE_RAM_MIB=65536 \
    LLAMA_ALIAS="$EXPECTED_MODEL" \
    LLAMA_LOG_FILE="$LLM_LOG" \
    MTP_MODEL_DIR="$MTP_MODEL_DIR" \
    MTP_MODEL="$MTP_MODEL" \
    MMPROJ="$MMPROJ" \
    bash "$LLAMA_LAUNCHER" \
    > "$LLM_LAUNCH_LOG" 2>&1 &
LLM_PID=$!

wait_for_health "embedding service" "$EMBEDDING_PORT" "$EMBEDDING_PID"

HOST=127.0.0.1 \
PORT="$LLM_PORT" \
WAIT_TIMEOUT="$WAIT_TIMEOUT" \
EXPECTED_MODEL="$EXPECTED_MODEL" \
bash "$LLM_CHECKER"

if ! kill -0 "$LLM_PID" 2>/dev/null; then
    echo "[ERROR] Qwen llama-server exited after its readiness check" >&2
    tail -n 80 "$LLM_LOG" >&2 || true
    exit 1
fi

echo "[3/3] Generating $MAX_CASES independent scenes with concurrency=$SCENE_CONCURRENCY"
echo "[INFO] run id: $RUN_ID"
echo "[INFO] shared base: disabled"
echo "[INFO] floor plan mode: $FLOOR_PLAN_MODE"

cd "$PROJECT_ROOT"

env \
    PYTHON_BIN="$PYTHON_BIN" \
    SCENEEXPERT_EXPERIMENT="${SCENEEXPERT_EXPERIMENT:-ablation_3_qwen3_harness}" \
    MODEL_NAME="$EXPECTED_MODEL" \
    OPENAI_API_KEY="${OPENAI_API_KEY:-sk-123}" \
    OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:${LLM_PORT}/v1}" \
    OPENAI_USE_RESPONSES=false \
    HF_HOME="${HF_HOME:-$USER_ROOT/checkpoints/hf_cache}" \
    HSSD_RETRIEVAL_BACKEND=embedding \
    HSSD_DATA_PATH="$HSSD_DATA_PATH" \
    HSSD_PREPROCESSED_PATH="$HSSD_PREPROCESSED_PATH" \
    HSSD_RENDERED_ASSET_CHOICE=true \
    HSSD_RENDERED_ASSET_CHOICE_TOP_N=4 \
    HSSD_RENDERED_ASSETS_DIR="$HSSD_RENDERED_ASSETS_DIR" \
    HSSD_ZVEC_COLLECTION_PATH="$HSSD_ZVEC_COLLECTION_PATH" \
    HSSD_EMBEDDING_BASE_URL="http://127.0.0.1:${EMBEDDING_PORT}" \
    SCENEEXPERT_DISABLE_ARTICULATED=1 \
    SCENEEXPERT_DISABLE_MATERIALS=1 \
    SCENEEXPERT_DISABLE_BWRAP=1 \
    CRITIC_PROBE_PARALLEL=true \
    CRITIC_PROBE_INNER_PARALLELISM="$SCENE_CONCURRENCY" \
    CRITIC_PROBE_MAX_SAFE_INNER_PARALLELISM="${CRITIC_PROBE_MAX_SAFE_INNER_PARALLELISM:-4}" \
    CRITIC_PROBE_ALLOW_UNSAFE_PARALLELISM=true \
    SCENEEXPERT_CONVEX_MAX_OMP_THREADS="${SCENEEXPERT_CONVEX_MAX_OMP_THREADS:-2}" \
    CRITIC_PROBE_PORT_BASE="${CRITIC_PROBE_PORT_BASE:-13000}" \
    CRITIC_PROBE_PORT_BLOCK_SIZE="${CRITIC_PROBE_PORT_BLOCK_SIZE:-400}" \
    SCENE_BATCH_SIZE=1 \
    SCENE_WORKERS_PER_PROCESS=1 \
    GENERATE_SHARED_BASE=false \
    BRANCH_FROM_SHARED_BASE=false \
    MAX_CASES="$MAX_CASES" \
    FLOOR_PLAN_MODE="$FLOOR_PLAN_MODE" \
    FURNITURE_PLACEMENT_ORDER_ENABLED="${FURNITURE_PLACEMENT_ORDER_ENABLED:-}" \
    CASE_FILTER="${CASE_FILTER:-}" \
    FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS="${FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS:-true}" \
    PIPELINE_STOP_STAGE="${PIPELINE_STOP_STAGE:-manipuland}" \
    CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE=true \
    CRITIC_PROBE_RENDER_FINAL_VIEWS="${CRITIC_PROBE_RENDER_FINAL_VIEWS:-true}" \
    RUN_ID="$RUN_ID" \
    bash "$CRITIC_RUNNER"

echo "[OK] Parallel scene generation finished"
echo "[OK] Output: $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
