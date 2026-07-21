#!/usr/bin/env bash
# Own the complete llama.cpp + critic probe lifecycle for one ACP allocation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"

LLAMA_RUNNER="${LLAMA_RUNNER:-$WORKSPACE_ROOT/run_qwen36_35b_a3b_llama_cpp.sh}"
LLAMA_CHECKER="${LLAMA_CHECKER:-$WORKSPACE_ROOT/check_llama_cpp.sh}"
CRITIC_RUNNER="${CRITIC_RUNNER:-$SCRIPT_DIR/run_parallel_critic_on.sh}"
LLAMA_LOG="${LLAMA_LOG:-$WORKSPACE_ROOT/llama_qwen36_35b_a3b_mtp.log}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-7200}"
EXPECTED_MODEL="${EXPECTED_MODEL:-unsloth/Qwen3.6-35B-A3B-GGUF}"
SHUTDOWN_GRACE_SECONDS="${SHUTDOWN_GRACE_SECONDS:-45}"
RUN_ID="${RUN_ID:-critic_on_qwen35_a3b_mtp_$(date +%Y%m%d_%H%M%S)}"

SERVER_PID=""
CHECK_PID=""
PROBE_PID=""
cleanup_started=false

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $1" >&2
        exit 1
    fi
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: required file not found: $1" >&2
        exit 1
    fi
}

process_exited() {
    local pid="$1"
    local state
    if ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    state="$(ps -o stat= -p "$pid" 2>/dev/null | awk '{print $1}' || true)"
    [[ "$state" == Z* ]]
}

process_group_alive() {
    ps -eo pgid=,stat= | awk -v pgid="$1" \
        '$1 == pgid && $2 !~ /^Z/ { found = 1 } END { exit !found }'
}

stop_process_groups() {
    local label="$1"
    shift
    local pids=("$@")
    local pid deadline any_alive

    for pid in "${pids[@]}"; do
        if [ -n "$pid" ] && process_group_alive "$pid"; then
            echo "Stopping $label process group $pid"
            kill -TERM -- "-$pid" 2>/dev/null || true
        fi
    done

    deadline=$((SECONDS + SHUTDOWN_GRACE_SECONDS))
    while [ "$SECONDS" -lt "$deadline" ]; do
        any_alive=false
        for pid in "${pids[@]}"; do
            if [ -n "$pid" ] && process_group_alive "$pid"; then
                any_alive=true
                break
            fi
        done
        if [ "$any_alive" = "false" ]; then
            break
        fi
        sleep 1
    done

    for pid in "${pids[@]}"; do
        if [ -n "$pid" ] && process_group_alive "$pid"; then
            echo "WARNING: force-killing $label process group $pid" >&2
            kill -KILL -- "-$pid" 2>/dev/null || true
        fi
        if [ -n "$pid" ]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
}

cleanup() {
    local exit_code="$1"
    if [ "$cleanup_started" = "true" ]; then
        return
    fi
    cleanup_started=true
    trap - EXIT INT TERM HUP

    # Stop consumers first, then release the model/GPU allocation.
    stop_process_groups "critic/check" "$PROBE_PID" "$CHECK_PID"
    stop_process_groups "llama-server" "$SERVER_PID"
    echo "Job cleanup complete (exit code $exit_code)"
}

on_signal() {
    local exit_code="$1"
    echo "Received termination signal; stopping the complete job" >&2
    exit "$exit_code"
}

trap 'cleanup "$?"' EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM
trap 'on_signal 129' HUP

require_command setsid
require_file "$LLAMA_RUNNER"
require_file "$LLAMA_CHECKER"
require_file "$CRITIC_RUNNER"

for value_name in PORT WAIT_TIMEOUT SHUTDOWN_GRACE_SECONDS; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ]; then
        echo "ERROR: $value_name must be a positive integer, got '$value'" >&2
        exit 2
    fi
done
if [ "$PORT" -gt 65535 ]; then
    echo "ERROR: PORT must not exceed 65535, got '$PORT'" >&2
    exit 2
fi

mkdir -p "$WORKSPACE_ROOT/llama_slot_cache"
cd "$WORKSPACE_ROOT"

echo "Starting llama-server; log: $LLAMA_LOG"
setsid env \
    HOST="$HOST" \
    PORT="$PORT" \
    CTX_SIZE="${CTX_SIZE:-294912}" \
    PARALLEL="${PARALLEL:-3}" \
    N_GPU_LAYERS="${N_GPU_LAYERS:-999}" \
    CACHE_TYPE_K="${CACHE_TYPE_K:-f16}" \
    CACHE_TYPE_V="${CACHE_TYPE_V:-f16}" \
    VISION="${VISION:-true}" \
    THREADS="${THREADS:-96}" \
    THREADS_HTTP="${THREADS_HTTP:-32}" \
    BATCH_SIZE="${BATCH_SIZE:-1024}" \
    UBATCH_SIZE="${UBATCH_SIZE:-256}" \
    MTP="${MTP:-true}" \
    MTP_MODEL_DIR="${MTP_MODEL_DIR:-/data/task3_2/share_model/unsloth/Qwen3.6-35B-A3B-MTP-GGUF}" \
    MTP_MODEL="${MTP_MODEL:-/data/task3_2/share_model/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf}" \
    MMPROJ="${MMPROJ:-/data/task3_2/share_model/unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf}" \
    ALIAS="${ALIAS:-$EXPECTED_MODEL}" \
    "$LLAMA_RUNNER" \
    --no-kv-unified \
    --cache-prompt \
    --cache-ram "${CACHE_RAM:-32768}" \
    --cache-idle-slots \
    --ctx-checkpoints "${CTX_CHECKPOINTS:-64}" \
    --cache-reuse "${CACHE_REUSE:-256}" \
    --slot-prompt-similarity "${SLOT_PROMPT_SIMILARITY:-0.5}" \
    > "$LLAMA_LOG" 2>&1 &
SERVER_PID=$!
echo "llama-server process group: $SERVER_PID"

# Run readiness checking as a managed child so signals remain responsive and a
# server crash during model loading fails immediately instead of waiting 2h.
setsid env \
    HOST=127.0.0.1 \
    PORT="$PORT" \
    WAIT_TIMEOUT="$WAIT_TIMEOUT" \
    EXPECTED_MODEL="$EXPECTED_MODEL" \
    bash "$LLAMA_CHECKER" &
CHECK_PID=$!

while ! process_exited "$CHECK_PID"; do
    if process_exited "$SERVER_PID"; then
        server_rc=0
        wait "$SERVER_PID" || server_rc=$?
        echo "ERROR: llama-server exited before becoming ready (exit code $server_rc)" >&2
        tail -n 80 "$LLAMA_LOG" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

check_rc=0
wait "$CHECK_PID" || check_rc=$?
if [ "$check_rc" -ne 0 ]; then
    echo "ERROR: llama-server readiness check failed (exit code $check_rc)" >&2
    tail -n 80 "$LLAMA_LOG" 2>/dev/null || true
    exit "$check_rc"
fi
if process_exited "$SERVER_PID"; then
    server_rc=0
    wait "$SERVER_PID" || server_rc=$?
    echo "ERROR: llama-server exited immediately after readiness (exit code $server_rc)" >&2
    tail -n 80 "$LLAMA_LOG" 2>/dev/null || true
    exit 1
fi

cd "$PROJECT_ROOT"
echo "Starting critic probe: $RUN_ID"
setsid env \
    PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}" \
    SCENEEXPERT_EXPERIMENT="${SCENEEXPERT_EXPERIMENT:-ablation_3_qwen3_harness}" \
    MODEL_NAME="${MODEL_NAME:-$EXPECTED_MODEL}" \
    OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:${PORT}/v1}" \
    SCENEEXPERT_DISABLE_ARTICULATED="${SCENEEXPERT_DISABLE_ARTICULATED:-1}" \
    SCENEEXPERT_DISABLE_MATERIALS="${SCENEEXPERT_DISABLE_MATERIALS:-1}" \
    SCENEEXPERT_DISABLE_BWRAP="${SCENEEXPERT_DISABLE_BWRAP:-1}" \
    CRITIC_PROBE_PARALLEL="${CRITIC_PROBE_PARALLEL:-true}" \
    CRITIC_PROBE_INNER_PARALLELISM="${CRITIC_PROBE_INNER_PARALLELISM:-3}" \
    MAX_CASES="${MAX_CASES:-3}" \
    PIPELINE_STOP_STAGE="${PIPELINE_STOP_STAGE:-manipuland}" \
    CRITIC_PROBE_PORT_BASE="${CRITIC_PROBE_PORT_BASE:-13000}" \
    RUN_ID="$RUN_ID" \
    bash "$CRITIC_RUNNER" &
PROBE_PID=$!

while ! process_exited "$PROBE_PID"; do
    if process_exited "$SERVER_PID"; then
        server_rc=0
        wait "$SERVER_PID" || server_rc=$?
        echo "ERROR: llama-server crashed during the critic probe (exit code $server_rc)" >&2
        tail -n 80 "$LLAMA_LOG" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

probe_rc=0
wait "$PROBE_PID" || probe_rc=$?
if [ "$probe_rc" -ne 0 ]; then
    echo "ERROR: critic probe failed (exit code $probe_rc)" >&2
    exit "$probe_rc"
fi

echo "Critic job completed successfully: $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
