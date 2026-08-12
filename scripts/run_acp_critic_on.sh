#!/usr/bin/env bash
# Run the colleague-aligned ACP critic-on stack for either stable case set.
# It owns the embedding service, Qwen llama.cpp server, and critic runner.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"
RUNNER="$SCRIPT_DIR/run_parallel_critic_on.sh"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_acp_critic_on.sh <new3|legacy8|old8> <create|reuse> [shared-base-root] [runner options...]

Modes:
  new3 create                 Generate a three-scene floor-plan shared base,
                              then branch critic-on from it.
  new3 reuse <shared-base>    Reuse a three-scene floor-plan shared base.
  legacy8 create              Generate an eight-scene floor-plan shared base,
                              then branch critic-on from it.
  legacy8 reuse <shared-base> Reuse an eight-scene floor-plan shared base.

The optional trailing options are passed to run_parallel_critic_on.sh, for
example: --scenes bedroom or --scenes default_classroom.

Set DRY_RUN=true to print the resolved critic command without starting the
embedding or llama.cpp services. The shared-base root must include /shared_base.
EOF
}

CASE_SET="${1:-}"
MODE="${2:-}"
if [ -z "$CASE_SET" ] || [ -z "$MODE" ]; then
    usage >&2
    exit 2
fi
shift 2

case "$CASE_SET" in
    new3) MAX_CASES_DEFAULT=3 ;;
    legacy8|old8)
        CASE_SET=legacy8
        MAX_CASES_DEFAULT=8
        ;;
    *)
        echo "ERROR: case set must be new3 or legacy8, got '$CASE_SET'" >&2
        exit 2
        ;;
esac
case "$MODE" in
    create|reuse) ;;
    *)
        echo "ERROR: mode must be create or reuse, got '$MODE'" >&2
        exit 2
        ;;
esac

SHARED_BASE_ROOT=""
if [ "$MODE" = "reuse" ]; then
    if [ "$#" -eq 0 ] || [ -z "${1:-}" ]; then
        echo "ERROR: reuse mode requires <shared-base-root>" >&2
        exit 2
    fi
    SHARED_BASE_ROOT="$1"
    shift
    if [[ "$SHARED_BASE_ROOT" != */shared_base ]]; then
        echo "ERROR: shared-base root must end in /shared_base: $SHARED_BASE_ROOT" >&2
        exit 2
    fi
    if [ ! -d "$SHARED_BASE_ROOT" ]; then
        echo "ERROR: shared-base root does not exist: $SHARED_BASE_ROOT" >&2
        exit 2
    fi
fi

if [ ! -f "$RUNNER" ]; then
    echo "ERROR: critic-on runner is missing: $RUNNER" >&2
    exit 1
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-shared_base_${CASE_SET}_${MODE}_${RUN_STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/critic_probe/$RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python interpreter is not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [ "$MODE" = "create" ]; then
    GENERATE_SHARED_BASE=true
    BRANCH_FROM_SHARED_BASE=true
else
    GENERATE_SHARED_BASE=false
    BRANCH_FROM_SHARED_BASE=true
fi

DRY_RUN="${DRY_RUN:-false}"
case "${DRY_RUN,,}" in
    1|true|yes|y|on) DRY_RUN=true ;;
    0|false|no|n|off|'') DRY_RUN=false ;;
    *)
        echo "ERROR: DRY_RUN must be true or false, got '$DRY_RUN'" >&2
        exit 2
        ;;
esac

run_critic() {
    env \
        PYTHON_BIN="$PYTHON_BIN" \
        SCENEEXPERT_EXPERIMENT="${SCENEEXPERT_EXPERIMENT:-ablation_3_qwen3_harness}" \
        MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.6-27B-GGUF}" \
        OPENAI_API_KEY="${OPENAI_API_KEY:-sk-123}" \
        OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8002/v1}" \
        OPENAI_USE_RESPONSES=false \
        HF_HOME="${HF_HOME:-/data/task3_2/L202500266_hrk/.cache/huggingface}" \
        HSSD_RETRIEVAL_BACKEND="${HSSD_RETRIEVAL_BACKEND:-embedding}" \
        HSSD_RENDERED_ASSET_CHOICE="${HSSD_RENDERED_ASSET_CHOICE:-true}" \
        HSSD_RENDERED_ASSET_CHOICE_TOP_N="${HSSD_RENDERED_ASSET_CHOICE_TOP_N:-4}" \
        HSSD_RENDERED_ASSETS_DIR="${HSSD_RENDERED_ASSETS_DIR:-/data/task3_2/share_data/scenesmith/hssd_rendered_assets}" \
        HSSD_ZVEC_COLLECTION_PATH="${HSSD_ZVEC_COLLECTION_PATH:-/data/task3_2/share_data/scenesmith/hssd_zvec_collection}" \
        HSSD_EMBEDDING_BASE_URL="${HSSD_EMBEDDING_BASE_URL:-http://127.0.0.1:8014}" \
        SCENEEXPERT_DISABLE_ARTICULATED="${SCENEEXPERT_DISABLE_ARTICULATED:-1}" \
        SCENEEXPERT_DISABLE_MATERIALS="${SCENEEXPERT_DISABLE_MATERIALS:-1}" \
        SCENEEXPERT_DISABLE_BWRAP="${SCENEEXPERT_DISABLE_BWRAP:-1}" \
        CRITIC_PROBE_PARALLEL="${CRITIC_PROBE_PARALLEL:-true}" \
        CRITIC_PROBE_INNER_PARALLELISM="${CRITIC_PROBE_INNER_PARALLELISM:-8}" \
        CRITIC_PROBE_MAX_SAFE_INNER_PARALLELISM="${CRITIC_PROBE_MAX_SAFE_INNER_PARALLELISM:-8}" \
        CRITIC_PROBE_ALLOW_UNSAFE_PARALLELISM="${CRITIC_PROBE_ALLOW_UNSAFE_PARALLELISM:-true}" \
        SCENEEXPERT_CONVEX_MAX_OMP_THREADS="${SCENEEXPERT_CONVEX_MAX_OMP_THREADS:-2}" \
        CRITIC_PROBE_PORT_BASE="${CRITIC_PROBE_PORT_BASE:-13000}" \
        CRITIC_PROBE_PORT_BLOCK_SIZE="${CRITIC_PROBE_PORT_BLOCK_SIZE:-400}" \
        SCENE_BATCH_SIZE="${SCENE_BATCH_SIZE:-1}" \
        SCENE_WORKERS_PER_PROCESS="${SCENE_WORKERS_PER_PROCESS:-1}" \
        GENERATE_SHARED_BASE="$GENERATE_SHARED_BASE" \
        BRANCH_FROM_SHARED_BASE="$BRANCH_FROM_SHARED_BASE" \
        SHARED_BASE_ROOT="$SHARED_BASE_ROOT" \
        SHARED_BASE_STOP_STAGE="${SHARED_BASE_STOP_STAGE:-floor_plan}" \
        MAX_CASES="${MAX_CASES:-$MAX_CASES_DEFAULT}" \
        FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS="${FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS:-true}" \
        PIPELINE_STOP_STAGE="${PIPELINE_STOP_STAGE:-manipuland}" \
        CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE="${CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE:-true}" \
        CRITIC_PROBE_RENDER_FINAL_VIEWS="${CRITIC_PROBE_RENDER_FINAL_VIEWS:-true}" \
        RUN_ID="$RUN_ID" \
        OUTPUT_ROOT="$OUTPUT_ROOT" \
        CASE_SET="$CASE_SET" \
        DRY_RUN="$DRY_RUN" \
        bash "$RUNNER" "$@"
}

if [ "$DRY_RUN" = "true" ]; then
    echo "========== ACP CRITIC-ON DRY RUN =========="
    echo "case set: $CASE_SET"
    echo "mode: $MODE"
    echo "run id: $RUN_ID"
    echo "output root: $OUTPUT_ROOT"
    echo "shared base: ${SHARED_BASE_ROOT:-$OUTPUT_ROOT/shared_base}"
    run_critic "$@"
    exit $?
fi

preflight_critic_config() {
    local preflight_root preflight_log preflight_rc
    preflight_root="$(mktemp -d "${TMPDIR:-/tmp}/critic_on_preflight.XXXXXX")"
    preflight_log="$preflight_root/runner.log"
    if DRY_RUN=true OUTPUT_ROOT="$preflight_root/output" run_critic "$@" > "$preflight_log" 2>&1; then
        rm -rf "$preflight_root"
        return 0
    fi
    preflight_rc=$?
    cat "$preflight_log" >&2
    rm -rf "$preflight_root"
    return "$preflight_rc"
}

echo "Validating critic-on case selection and shared-base mapping..."
if ! preflight_critic_config "$@"; then
    echo "ERROR: critic-on preflight failed; ACP services were not started." >&2
    exit 2
fi

EMBEDDING_RUNNER="${EMBEDDING_RUNNER:-$WORKSPACE_ROOT/run_qwen3_vl_embedding_2b_llama_cpp.sh}"
LLAMA_RUNNER="${LLAMA_RUNNER:-/data/task3_2/share_scripts/llama.cpp/run_qwen36_27b_llama_cpp.sh}"
LLAMA_CHECKER="${LLAMA_CHECKER:-$WORKSPACE_ROOT/check_llama_cpp.sh}"
EMBEDDING_LOG="${EMBEDDING_LOG:-$WORKSPACE_ROOT/llama_embedding.log}"
LLAMA_LOG="${LLAMA_LOG:-$WORKSPACE_ROOT/llama_qwen36_27b_mtp.log}"

for required_path in "$EMBEDDING_RUNNER" "$LLAMA_RUNNER" "$LLAMA_CHECKER"; do
    if [ ! -f "$required_path" ]; then
        echo "ERROR: required ACP service script is missing: $required_path" >&2
        exit 1
    fi
done

require_unused_port() {
    local port="$1" label="$2"
    if command -v ss >/dev/null 2>&1 \
        && ss -H -ltn "sport = :$port" 2>/dev/null \
            | awk '$1 == "LISTEN" { found = 1 } END { exit !found }'; then
        echo "ERROR: $label port $port is already listening; refusing to start a second ACP service stack." >&2
        exit 1
    fi
}

require_unused_port "${LLAMA_PORT:-8002}" "llama.cpp"
require_unused_port "${HSSD_EMBEDDING_PORT:-8014}" "embedding"

EMBEDDING_PID=""
LLAMA_PID=""
cleanup() {
    local pid
    trap - EXIT INT TERM HUP
    for pid in "$LLAMA_PID" "$EMBEDDING_PID"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for pid in "$LLAMA_PID" "$EMBEDDING_PID"; do
        if [ -n "$pid" ]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT INT TERM HUP

cd "$WORKSPACE_ROOT"
clashctl on 2>/dev/null || true

setsid env \
    HOST=0.0.0.0 \
    PORT="${HSSD_EMBEDDING_PORT:-8014}" \
    PARALLEL="${HSSD_EMBEDDING_PARALLEL:-6}" \
    THREADS_HTTP="${HSSD_EMBEDDING_THREADS_HTTP:-6}" \
    N_GPU_LAYERS=auto \
    CTX_SIZE=12288 \
    BATCH_SIZE=4096 \
    UBATCH_SIZE=4096 \
    "$EMBEDDING_RUNNER" > "$EMBEDDING_LOG" 2>&1 &
EMBEDDING_PID=$!

setsid env \
    HOST=0.0.0.0 \
    PORT="${LLAMA_PORT:-8002}" \
    CTX_SIZE="${LLAMA_CTX_SIZE:-524288}" \
    PARALLEL="${LLAMA_PARALLEL:-8}" \
    N_GPU_LAYERS=999 \
    CACHE_TYPE_K=q8_0 \
    CACHE_TYPE_V=q8_0 \
    VISION=true \
    THREADS="${LLAMA_THREADS:-16}" \
    THREADS_HTTP="${LLAMA_THREADS_HTTP:-8}" \
    BATCH_SIZE=1024 \
    UBATCH_SIZE=256 \
    MTP=true \
    SPEC_DRAFT_N_MAX=2 \
    THINKING=true \
    REASONING=auto \
    REASONING_PRESERVE=true \
    TEMP=1.0 \
    TOP_P=0.95 \
    TOP_K=20 \
    MIN_P=0.00 \
    PRESENCE_PENALTY=0.0 \
    REPEAT_PENALTY=1.0 \
    MTP_MODEL_DIR="${MTP_MODEL_DIR:-/data/task3_2/share_model/unsloth/Qwen3.6-27B-MTP-GGUF}" \
    MTP_MODEL="${MTP_MODEL:-/data/task3_2/share_model/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q8_K_XL.gguf}" \
    MMPROJ="${MMPROJ:-/data/task3_2/share_model/unsloth/Qwen3.6-27B-GGUF/mmproj-F16.gguf}" \
    ALIAS="${MODEL_NAME:-unsloth/Qwen3.6-27B-GGUF}" \
    "$LLAMA_RUNNER" \
    --no-kv-unified \
    --cache-prompt \
    --cache-ram 65536 \
    --cache-idle-slots \
    --ctx-checkpoints 64 \
    --slot-prompt-similarity 0.5 > "$LLAMA_LOG" 2>&1 &
LLAMA_PID=$!

PORT="${LLAMA_PORT:-8002}" \
WAIT_TIMEOUT="${WAIT_TIMEOUT:-7200}" \
EXPECTED_MODEL="${MODEL_NAME:-unsloth/Qwen3.6-27B-GGUF}" \
bash "$LLAMA_CHECKER"

cd "$PROJECT_ROOT"
run_critic "$@"
