#!/usr/bin/env bash
#
# Start one persistent Qwen-Image-Edit worker on one GPU.
#
# Usage:
#   bash scripts/start_qwen_image_edit_server.sh --background
#   bash scripts/start_qwen_image_edit_server.sh --foreground
#   bash scripts/start_qwen_image_edit_server.sh --status
#   bash scripts/start_qwen_image_edit_server.sh --stop

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

PYTHON_BIN="${QWEN_IMAGE_EDIT_PYTHON:-/mnt/afs/visitor33/miniconda3/envs/qwen-image/bin/python}"
MODEL_DIR="${QWEN_IMAGE_EDIT_MODEL_DIR:-$PROJECT_ROOT/models/Qwen-Image-Edit}"
HOST="${QWEN_IMAGE_EDIT_HOST:-127.0.0.1}"
PORT="${QWEN_IMAGE_EDIT_PORT:-18020}"
CUDA_DEVICES="${QWEN_IMAGE_EDIT_CUDA_VISIBLE_DEVICES:-1}"
STARTUP_TIMEOUT="${QWEN_IMAGE_EDIT_STARTUP_TIMEOUT_SECONDS:-900}"
LOG_FILE="${QWEN_IMAGE_EDIT_LOG_FILE:-$PROJECT_ROOT/logs/qwen_image_edit_server.log}"
PID_FILE="${QWEN_IMAGE_EDIT_PID_FILE:-$PROJECT_ROOT/logs/qwen_image_edit_server.pid}"

health_url="http://${HOST}:${PORT}/health"
ready_url="http://${HOST}:${PORT}/ready"

pid_is_running() {
    local pid="$1"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || [ -d "/proc/$pid" ]
}

pid_is_our_server() {
    local pid="$1"
    [ -r "/proc/$pid/cmdline" ] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq "qwen_image_edit_server:app"
}

service_is_healthy() {
    local body
    body="$(curl -fsS --max-time 2 "$health_url" 2>/dev/null)" || return 1
    grep -Fq '"service":"qwen-image-edit"' <<< "$body"
}

service_is_ready() {
    local body
    body="$(curl -fsS --max-time 2 "$ready_url" 2>/dev/null)" || return 1
    grep -Fq '"ready":true' <<< "$body"
}

service_load_failed() {
    local body
    body="$(curl -fsS --max-time 2 "$health_url" 2>/dev/null)" || return 1
    grep -Fq '"loading":false' <<< "$body" &&
        ! grep -Fq '"load_error":null' <<< "$body"
}

port_is_open() {
    (exec 3<>"/dev/tcp/${HOST}/${PORT}") 2>/dev/null
}

validate_inputs() {
    if [ ! -x "$PYTHON_BIN" ]; then
        echo "[ERROR] qwen-image Python not executable: $PYTHON_BIN" >&2
        exit 1
    fi
    if [ ! -f "$MODEL_DIR/model_index.json" ]; then
        echo "[ERROR] Qwen-Image-Edit model is incomplete: $MODEL_DIR" >&2
        exit 1
    fi
    if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
        echo "[ERROR] invalid QWEN_IMAGE_EDIT_PORT: $PORT" >&2
        exit 1
    fi
    if ! [[ "$STARTUP_TIMEOUT" =~ ^[0-9]+$ ]] || [ "$STARTUP_TIMEOUT" -lt 1 ]; then
        echo "[ERROR] invalid QWEN_IMAGE_EDIT_STARTUP_TIMEOUT_SECONDS: $STARTUP_TIMEOUT" >&2
        exit 1
    fi
}

show_status() {
    if service_is_ready; then
        echo "[READY] Qwen-Image-Edit: $ready_url"
    elif service_load_failed; then
        echo "[FAILED] Qwen-Image-Edit model loading failed: $health_url"
    elif service_is_healthy; then
        echo "[LOADING] Qwen-Image-Edit: $health_url"
    else
        echo "[STOPPED] Qwen-Image-Edit is not responding on ${HOST}:${PORT}"
    fi

    if [ -f "$PID_FILE" ]; then
        local pid
        pid="$(tr -d '[:space:]' < "$PID_FILE")"
        if pid_is_running "$pid"; then
            echo "[INFO] PID: $pid"
        else
            echo "[INFO] stale PID file: $PID_FILE"
        fi
    fi
    echo "[INFO] log: $LOG_FILE"
}

wait_until_ready() {
    local pid="$1"
    local started_at
    started_at="$(date +%s)"
    while true; do
        if service_is_ready; then
            echo "[READY] Qwen-Image-Edit is ready: $ready_url"
            return 0
        fi
        if service_load_failed; then
            echo "[ERROR] Qwen-Image-Edit model loading failed" >&2
            curl -fsS --max-time 2 "$health_url" >&2 || true
            echo >&2
            tail -n 80 "$LOG_FILE" >&2 || true
            return 1
        fi
        if ! pid_is_running "$pid"; then
            echo "[ERROR] Qwen-Image-Edit exited before becoming ready" >&2
            tail -n 80 "$LOG_FILE" >&2 || true
            return 1
        fi
        if [ "$(( $(date +%s) - started_at ))" -ge "$STARTUP_TIMEOUT" ]; then
            echo "[ERROR] readiness timed out after ${STARTUP_TIMEOUT}s" >&2
            tail -n 80 "$LOG_FILE" >&2 || true
            return 1
        fi
        sleep 2
    done
}

run_foreground() {
    validate_inputs
    if port_is_open && ! service_is_healthy; then
        echo "[ERROR] port ${HOST}:${PORT} is occupied by an unknown process" >&2
        exit 1
    fi
    if service_is_healthy; then
        echo "[INFO] Qwen-Image-Edit is already running: $health_url"
        exit 0
    fi

    mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
    printf '%s\n' "$$" > "$PID_FILE"
    trap 'rm -f "$PID_FILE"' EXIT
    echo "[INFO] model: $MODEL_DIR"
    echo "[INFO] CUDA devices: $CUDA_DEVICES"
    echo "[INFO] endpoint: http://${HOST}:${PORT}/v1"
    echo "[INFO] log: $LOG_FILE"

    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export QWEN_IMAGE_EDIT_MODEL_DIR="$MODEL_DIR"
    exec "$PYTHON_BIN" -m uvicorn qwen_image_edit_server:app \
        --app-dir "$SCRIPT_DIR" \
        --host "$HOST" \
        --port "$PORT" \
        --workers 1
}

start_background() {
    validate_inputs
    if service_is_healthy; then
        echo "[INFO] reusing existing Qwen-Image-Edit service: $health_url"
        show_status
        return 0
    fi
    if port_is_open; then
        echo "[ERROR] port ${HOST}:${PORT} is occupied by an unknown process" >&2
        exit 1
    fi
    if [ -f "$PID_FILE" ]; then
        local old_pid
        old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
        if pid_is_running "$old_pid"; then
            echo "[ERROR] PID file points to a running non-responsive process: $old_pid" >&2
            exit 1
        fi
        rm -f "$PID_FILE"
    fi

    mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
    local server_pid
    if command -v setsid >/dev/null 2>&1; then
        # Detach from the launching terminal/session. -f makes this robust when
        # setsid itself happens to be a process-group leader.
        nohup setsid -f env \
            QWEN_IMAGE_EDIT_PYTHON="$PYTHON_BIN" \
            QWEN_IMAGE_EDIT_MODEL_DIR="$MODEL_DIR" \
            QWEN_IMAGE_EDIT_HOST="$HOST" \
            QWEN_IMAGE_EDIT_PORT="$PORT" \
            QWEN_IMAGE_EDIT_CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
            PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
            "$SCRIPT_PATH" --foreground >> "$LOG_FILE" 2>&1 < /dev/null

        for _ in $(seq 1 100); do
            if [ -s "$PID_FILE" ]; then
                server_pid="$(tr -d '[:space:]' < "$PID_FILE")"
                if pid_is_running "$server_pid"; then
                    break
                fi
            fi
            sleep 0.1
        done
        if [ -z "${server_pid:-}" ] || ! pid_is_running "$server_pid"; then
            echo "[ERROR] detached Qwen-Image-Edit process did not start" >&2
            tail -n 80 "$LOG_FILE" >&2 || true
            exit 1
        fi
    else
        nohup env \
            QWEN_IMAGE_EDIT_PYTHON="$PYTHON_BIN" \
            QWEN_IMAGE_EDIT_MODEL_DIR="$MODEL_DIR" \
            QWEN_IMAGE_EDIT_HOST="$HOST" \
            QWEN_IMAGE_EDIT_PORT="$PORT" \
            QWEN_IMAGE_EDIT_CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
            PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
            "$SCRIPT_PATH" --foreground >> "$LOG_FILE" 2>&1 < /dev/null &
        server_pid=$!
        printf '%s\n' "$server_pid" > "$PID_FILE"
    fi
    echo "[INFO] started Qwen-Image-Edit in background: PID=$server_pid"
    echo "[INFO] loading ~54GB of weights; waiting up to ${STARTUP_TIMEOUT}s"
    wait_until_ready "$server_pid"
}

stop_server() {
    if [ ! -f "$PID_FILE" ]; then
        echo "[INFO] no Qwen-Image-Edit PID file: $PID_FILE"
        return 0
    fi
    local pid
    pid="$(tr -d '[:space:]' < "$PID_FILE")"
    if ! pid_is_running "$pid"; then
        echo "[INFO] removing stale PID file for PID=$pid"
        rm -f "$PID_FILE"
        return 0
    fi
    if ! pid_is_our_server "$pid"; then
        echo "[ERROR] refusing to stop PID=$pid; command does not match Qwen server" >&2
        exit 1
    fi

    kill "$pid"
    local deadline=$(( $(date +%s) + 30 ))
    while pid_is_running "$pid" && [ "$(date +%s)" -lt "$deadline" ]; do
        sleep 1
    done
    if pid_is_running "$pid"; then
        echo "[ERROR] PID=$pid did not stop within 30s; not sending SIGKILL" >&2
        exit 1
    fi
    rm -f "$PID_FILE"
    echo "[INFO] stopped Qwen-Image-Edit PID=$pid"
}

case "${1:---background}" in
    --background|background)
        start_background
        ;;
    --foreground|foreground)
        run_foreground
        ;;
    --status|status)
        show_status
        ;;
    --stop|stop)
        stop_server
        ;;
    *)
        echo "Usage: bash $SCRIPT_PATH [--background|--foreground|--status|--stop]" >&2
        exit 2
        ;;
esac
