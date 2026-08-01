#!/usr/bin/env bash
#
# Download Qwen/Qwen-Image-Edit from ModelScope.
#
# Examples:
#   bash scripts/download_qwen_image_edit.sh --background
#   bash scripts/download_qwen_image_edit.sh --status
#   bash scripts/download_qwen_image_edit.sh --foreground
#
# Optional overrides:
#   QWEN_IMAGE_EDIT_MODEL_ID
#   QWEN_IMAGE_EDIT_MODEL_DIR
#   QWEN_IMAGE_EDIT_LOG_FILE
#   QWEN_IMAGE_EDIT_MODELSCOPE_BIN
#   QWEN_IMAGE_EDIT_BOOTSTRAP_MODELSCOPE=0
#   QWEN_IMAGE_EDIT_PIP_INDEX_URL

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

MODEL_ID="${QWEN_IMAGE_EDIT_MODEL_ID:-Qwen/Qwen-Image-Edit}"
MODEL_DIR="${QWEN_IMAGE_EDIT_MODEL_DIR:-$PROJECT_ROOT/models/Qwen-Image-Edit}"
LOG_FILE="${QWEN_IMAGE_EDIT_LOG_FILE:-$PROJECT_ROOT/logs/download_qwen_image_edit.log}"
PID_FILE="${QWEN_IMAGE_EDIT_PID_FILE:-$PROJECT_ROOT/logs/download_qwen_image_edit.pid}"
LOCK_FILE="${QWEN_IMAGE_EDIT_LOCK_FILE:-$PROJECT_ROOT/logs/download_qwen_image_edit.lock}"
DOWNLOAD_ENV="${QWEN_IMAGE_EDIT_DOWNLOAD_ENV:-$PROJECT_ROOT/.runtime_cache/modelscope-downloader}"
CONDA_MODELSCOPE_BIN="${QWEN_IMAGE_EDIT_MODELSCOPE_BIN:-$PROJECT_ROOT/../miniconda3/envs/download/bin/modelscope}"
BOOTSTRAP_MODELSCOPE="${QWEN_IMAGE_EDIT_BOOTSTRAP_MODELSCOPE:-1}"
PIP_INDEX_URL="${QWEN_IMAGE_EDIT_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

mkdir -p "$(dirname "$LOG_FILE")" "$MODEL_DIR"

pid_is_running() {
    local pid="$1"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || [ -d "/proc/$pid" ]
}

show_status() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid="$(tr -d '[:space:]' < "$PID_FILE")"
        if pid_is_running "$pid"; then
            echo "下载任务正在运行，PID: $pid"
            echo "模型目录: $MODEL_DIR"
            echo "日志文件: $LOG_FILE"
            return 0
        fi
    fi

    echo "下载任务当前未运行。"
    echo "模型目录: $MODEL_DIR"
    echo "日志文件: $LOG_FILE"
    if [ -f "$LOG_FILE" ]; then
        echo
        echo "最近日志:"
        tail -n 20 "$LOG_FILE"
    fi
}

start_background() {
    if [ -f "$PID_FILE" ]; then
        local old_pid
        old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
        if pid_is_running "$old_pid"; then
            echo "下载任务已在运行，PID: $old_pid"
            echo "日志文件: $LOG_FILE"
            return 0
        fi
    fi

    nohup "$SCRIPT_PATH" --foreground >> "$LOG_FILE" 2>&1 < /dev/null &
    local download_pid=$!
    printf '%s\n' "$download_pid" > "$PID_FILE"

    echo "已启动后台下载，PID: $download_pid"
    echo "模型目录: $MODEL_DIR"
    echo "日志文件: $LOG_FILE"
    echo "查看状态: bash $SCRIPT_PATH --status"
}

find_modelscope() {
    if [ -x "$CONDA_MODELSCOPE_BIN" ]; then
        printf '%s\n' "$CONDA_MODELSCOPE_BIN"
        return 0
    fi

    if command -v modelscope >/dev/null 2>&1; then
        command -v modelscope
        return 0
    fi

    if [ -x "$DOWNLOAD_ENV/bin/modelscope" ]; then
        printf '%s\n' "$DOWNLOAD_ENV/bin/modelscope"
        return 0
    fi

    if [ "$BOOTSTRAP_MODELSCOPE" != "1" ]; then
        echo "错误：未找到 ModelScope。请先安装 modelscope，或设置" >&2
        echo "QWEN_IMAGE_EDIT_BOOTSTRAP_MODELSCOPE=1 允许脚本自动安装。" >&2
        return 1
    fi

    echo "未找到 ModelScope，正在创建隔离的下载环境: $DOWNLOAD_ENV" >&2
    python -m venv "$DOWNLOAD_ENV"
    "$DOWNLOAD_ENV/bin/python" -m pip install --upgrade pip \
        --index-url "$PIP_INDEX_URL"
    "$DOWNLOAD_ENV/bin/python" -m pip install modelscope \
        --index-url "$PIP_INDEX_URL"
    printf '%s\n' "$DOWNLOAD_ENV/bin/modelscope"
}

run_download() {
    exec 9> "$LOCK_FILE"
    if ! flock -n 9; then
        echo "另一个 Qwen-Image-Edit 下载任务正在运行。"
        exit 0
    fi

    printf '%s\n' "$$" > "$PID_FILE"
    trap 'rm -f "$PID_FILE"' EXIT

    echo
    echo "============================================================"
    echo "开始时间: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "模型 ID: $MODEL_ID"
    echo "保存目录: $MODEL_DIR"
    echo "============================================================"

    local modelscope_bin
    modelscope_bin="$(find_modelscope)"

    "$modelscope_bin" download \
        --model "$MODEL_ID" \
        --local_dir "$MODEL_DIR"

    echo
    echo "下载完成: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "模型目录: $MODEL_DIR"
}

case "${1:---background}" in
    --background)
        start_background
        ;;
    --foreground)
        run_download
        ;;
    --status)
        show_status
        ;;
    *)
        echo "用法: bash $SCRIPT_PATH [--background|--foreground|--status]" >&2
        exit 2
        ;;
esac
