#!/usr/bin/env bash
# Start one persistent offline GroundingDINO worker on an explicitly selected GPU.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${GROUNDING_DINO_PYTHON:-/mnt/afs/visitor33/miniconda3/envs/vllm/bin/python}"
MODEL_PATH="${GROUNDING_DINO_MODEL_PATH:-/mnt/afs-p3/task3_2/visitor33_ljx/checkpoints/grounding-dino-base}"
HOST="${GROUNDING_DINO_HOST:-127.0.0.1}"
PORT="${GROUNDING_DINO_PORT:-18030}"
LOG_FILE="${GROUNDING_DINO_LOG_FILE:-$PROJECT_ROOT/logs/grounding_dino_server.log}"
PID_FILE="${GROUNDING_DINO_PID_FILE:-$PROJECT_ROOT/logs/grounding_dino_server.pid}"
STARTUP_TIMEOUT="${GROUNDING_DINO_STARTUP_TIMEOUT_SECONDS:-300}"
MODE="${1:---foreground}"
HEALTH_URL="http://${HOST}:${PORT}/health"

pid_running() {
    local pid="$1"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

service_ready() {
    curl -fsS --max-time 2 "$HEALTH_URL" 2>/dev/null | grep -Fq '"ready":true'
}

validate_start() {
    if [ -z "${GROUNDING_DINO_GPU_ID:-}" ]; then
        echo "[ERROR] set GROUNDING_DINO_GPU_ID explicitly; no GPU is selected by default" >&2
        exit 2
    fi
    if [ ! -x "$PYTHON_BIN" ]; then
        echo "[ERROR] GroundingDINO Python is not executable: $PYTHON_BIN" >&2
        exit 1
    fi
    if [ ! -f "$MODEL_PATH/config.json" ]; then
        echo "[ERROR] GroundingDINO model directory is incomplete: $MODEL_PATH" >&2
        exit 1
    fi
    if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
        echo "[ERROR] no NVIDIA GPU is visible; CPU fallback is intentionally disabled" >&2
        exit 1
    fi
}

run_server() {
    mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
    export CUDA_VISIBLE_DEVICES="$GROUNDING_DINO_GPU_ID"
    export GROUNDING_DINO_MODEL_PATH="$MODEL_PATH"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    exec "$PYTHON_BIN" -m uvicorn grounding_dino_server:app \
        --app-dir "$SCRIPT_DIR" --host "$HOST" --port "$PORT" --workers 1
}

wait_ready() {
    local pid="$1"
    local started_at
    started_at="$(date +%s)"
    until service_ready; do
        if ! pid_running "$pid"; then
            echo "[ERROR] GroundingDINO exited before becoming ready" >&2
            tail -n 80 "$LOG_FILE" >&2 || true
            return 1
        fi
        if [ "$(( $(date +%s) - started_at ))" -ge "$STARTUP_TIMEOUT" ]; then
            echo "[ERROR] GroundingDINO readiness timed out" >&2
            tail -n 80 "$LOG_FILE" >&2 || true
            return 1
        fi
        sleep 2
    done
    echo "[READY] GroundingDINO: $HEALTH_URL"
}

case "$MODE" in
    --foreground)
        validate_start
        printf '%s\n' "$$" > "$PID_FILE"
        trap 'rm -f "$PID_FILE"' EXIT
        echo "[INFO] model: $MODEL_PATH"
        echo "[INFO] CUDA device: $GROUNDING_DINO_GPU_ID"
        echo "[INFO] endpoint: http://${HOST}:${PORT}"
        run_server >> "$LOG_FILE" 2>&1
        ;;
    --background)
        validate_start
        if service_ready; then
            echo "[INFO] reusing ready GroundingDINO service: $HEALTH_URL"
            exit 0
        fi
        mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
        rm -f "$PID_FILE"
        if command -v setsid >/dev/null 2>&1; then
            nohup setsid -f "$0" --foreground >/dev/null 2>&1 < /dev/null
        else
            nohup "$0" --foreground >/dev/null 2>&1 < /dev/null &
        fi
        pid=""
        for _ in $(seq 1 100); do
            if [ -s "$PID_FILE" ]; then
                pid="$(tr -d '[:space:]' < "$PID_FILE")"
                if pid_running "$pid"; then
                    break
                fi
            fi
            sleep 0.1
        done
        if [ -z "$pid" ] || ! pid_running "$pid"; then
            echo "[ERROR] detached GroundingDINO process did not start" >&2
            tail -n 80 "$LOG_FILE" >&2 || true
            exit 1
        fi
        wait_ready "$pid"
        ;;
    --status)
        if service_ready; then
            curl -fsS "$HEALTH_URL"
            echo
        else
            echo "[STOPPED] GroundingDINO is not ready on ${HOST}:${PORT}"
            exit 1
        fi
        ;;
    --stop)
        if [ ! -f "$PID_FILE" ]; then
            echo "[INFO] no GroundingDINO PID file: $PID_FILE"
            exit 0
        fi
        pid="$(tr -d '[:space:]' < "$PID_FILE")"
        if pid_running "$pid" && [ -r "/proc/$pid/cmdline" ] && \
            tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq "grounding_dino_server"; then
            kill "$pid"
            echo "[INFO] stopped GroundingDINO PID $pid"
        else
            echo "[WARNING] refusing to stop unrelated or stale PID: $pid" >&2
        fi
        rm -f "$PID_FILE"
        ;;
    *)
        echo "Usage: $0 [--foreground|--background|--status|--stop]" >&2
        exit 2
        ;;
esac
