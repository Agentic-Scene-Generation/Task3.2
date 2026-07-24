#!/usr/bin/env bash
# Run a Task3.2 prompt batch with Qwen3.6-27B-MTP-GGUF through llama.cpp.
# llama-server and every SceneSmith worker share the single GPU assigned by ACP.
# This launcher does not start or validate vLLM.
#
# Usage:
#   bash run_prompt_batch_llama_single_gpu.sh [options] START END \
#       [CSV_PATH] [NAME] [STALL_MIN]
#
# Options:
#   --output-dir DIR
#   --experiment NAME
#   --floor-plan-mode room|house
#   --dry-run
#   -h, --help
#
# Example:
#   bash run_prompt_batch_llama_single_gpu.sh \
#       --output-dir outputs \
#       0 10 data/prompt_gen/prompts_v1.csv qwen36_task32 45
#
# Optional environment overrides:
#   SCENEEXPERT_SINGLE_GPU        GPU index/UUID (normally supplied by ACP)
#   SCENEEXPERT_INSTALL_SYSTEM_DEPS auto (default) or 0
#   SCENEEXPERT_VENV_PATH         SceneSmith virtualenv
#   SCENEEXPERT_DATA_DIR          general Task3.2 data directory
#   SCENEEXPERT_HSSD_DATA_DIR     HSSD/HSM data directory
#   SCENEEXPERT_MATERIALS_DIR     materials directory
#   SCENEEXPERT_OPENCLIP_CHECKPOINT local OpenCLIP .bin path
#   LLAMA_PORT                    default: 8002
#   LLAMA_CTX_SIZE                default: 98304
#   LLAMA_PARALLEL                default: 1
#   LLAMA_ALIAS                   default: qwen36-27b-mtp
#   LLAMA_WAIT_TIMEOUT            default: 7200 seconds

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/mnt/afs/visitor33/miniconda3}"
LEGACY_SCENESMITH_ROOT="${SCENEEXPERT_LEGACY_ROOT:-/mnt/afs/visitor33/scenesmith-qwen}"
START_LLAMA_SCRIPT="${SCENEEXPERT_LLAMA_LAUNCHER:-${WORKDIR}/start_llama.sh}"
CHECK_LLAMA_SCRIPT="${CHECK_LLAMA_SCRIPT:-/mnt/afs-p3/task3_2/share_scripts/llama.cpp/check_llama_cpp.sh}"

usage() {
    sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'
}

OUTPUT_DIR=""
EXPERIMENT="${SCENEEXPERT_BATCH_EXPERIMENT:-indoor_scene_generation}"
FLOOR_PLAN_MODE="${FLOOR_PLAN_MODE:-room}"
DRY_RUN=0
POSITIONAL=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --output-dir|--experiment|--floor-plan-mode)
            [ "$#" -ge 2 ] && [ -n "${2:-}" ] || {
                echo "[ERROR] $1 requires a value" >&2
                exit 2
            }
            case "$1" in
                --output-dir) OUTPUT_DIR="$2" ;;
                --experiment) EXPERIMENT="$2" ;;
                --floor-plan-mode) FLOOR_PLAN_MODE="$2" ;;
            esac
            shift 2
            ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        --experiment=*) EXPERIMENT="${1#*=}"; shift ;;
        --floor-plan-mode=*) FLOOR_PLAN_MODE="${1#*=}"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --)
            shift
            while [ "$#" -gt 0 ]; do
                POSITIONAL+=("$1")
                shift
            done
            ;;
        -*)
            echo "[ERROR] unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

set -- "${POSITIONAL[@]}"
if [ "$#" -lt 2 ] || [ "$#" -gt 5 ]; then
    usage >&2
    exit 2
fi

START="$1"
END="$2"
CSV_PATH="${3:-data/prompt_gen/prompts_v1.csv}"
NAME="${4:-llama_single_gpu_${START}_${END}}"
STALL_MIN="${5:-${SCENEEXPERT_BATCH_STALL_MIN:-30}}"

if ! [[ "$START" =~ ^[0-9]+$ && "$END" =~ ^[0-9]+$ && "$STALL_MIN" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] START, END, and STALL_MIN must be non-negative integers" >&2
    exit 2
fi
if [ "$END" -le "$START" ]; then
    echo "[ERROR] END ($END) must be greater than START ($START)" >&2
    exit 2
fi
if [[ "$FLOOR_PLAN_MODE" != "room" && "$FLOOR_PLAN_MODE" != "house" ]]; then
    echo "[ERROR] --floor-plan-mode must be room or house" >&2
    exit 2
fi

cd "$WORKDIR"
if [[ "$CSV_PATH" != /* ]]; then
    CSV_PATH="${WORKDIR}/${CSV_PATH}"
fi
if [ ! -f "$CSV_PATH" ]; then
    echo "[ERROR] CSV not found: $CSV_PATH" >&2
    exit 1
fi
for required in "$START_LLAMA_SCRIPT" "$CHECK_LLAMA_SCRIPT"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] required file not found: $required" >&2
        exit 1
    fi
done
for command_name in curl setsid; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "[ERROR] required command not found: $command_name" >&2
        exit 1
    fi
done

VENV_PATH="${SCENEEXPERT_VENV_PATH:-}"
if [ -z "$VENV_PATH" ]; then
    if [ -f "${WORKDIR}/.venv/bin/activate" ]; then
        VENV_PATH="${WORKDIR}/.venv"
    else
        VENV_PATH="${LEGACY_SCENESMITH_ROOT}/.venv"
    fi
fi
if [[ "$VENV_PATH" != /* ]]; then
    VENV_PATH="${WORKDIR}/${VENV_PATH}"
fi
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "[ERROR] SceneSmith Python is not executable: $PYTHON_BIN" >&2
    exit 1
fi

# The venv uses the scenesmith Conda interpreter. Its native extensions need
# the matching Conda C++/expat libraries even though the venv is self-contained.
PYTHON_BASE_PREFIX="$("$PYTHON_BIN" -c 'import sys; print(sys.base_prefix)')"
SCENE_RUNTIME_LIB_DIR="${SCENEEXPERT_SCENE_RUNTIME_LIB_DIR:-${PYTHON_BASE_PREFIX}/lib}"
if [ -d "$SCENE_RUNTIME_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${SCENE_RUNTIME_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export PATH="${VENV_PATH}/bin:${PYTHON_BASE_PREFIX}/bin:${PATH}"

SINGLE_GPU="${SCENEEXPERT_SINGLE_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
if [ -z "$SINGLE_GPU" ] || [[ "$SINGLE_GPU" == *,* ]]; then
    echo "[ERROR] expected exactly one GPU, got '$SINGLE_GPU'" >&2
    exit 1
fi

LLAMA_PORT="${LLAMA_PORT:-8002}"
LLAMA_ALIAS="${LLAMA_ALIAS:-qwen36-27b-mtp}"
LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-98304}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
LLAMA_WAIT_TIMEOUT="${LLAMA_WAIT_TIMEOUT:-7200}"
if ! [[ "$LLAMA_PORT" =~ ^[0-9]+$ && "$LLAMA_CTX_SIZE" =~ ^[1-9][0-9]*$ && \
        "$LLAMA_PARALLEL" =~ ^[1-9][0-9]*$ && "$LLAMA_WAIT_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] invalid llama port/context/parallel/wait setting" >&2
    exit 2
fi
if [ "$LLAMA_PORT" -lt 1 ] || [ "$LLAMA_PORT" -gt 65535 ]; then
    echo "[ERROR] invalid LLAMA_PORT: $LLAMA_PORT" >&2
    exit 2
fi

DATA_DIR="${SCENEEXPERT_DATA_DIR:-${LEGACY_SCENESMITH_ROOT}/data}"
HSSD_DATA_DIR="${SCENEEXPERT_HSSD_DATA_DIR:-$DATA_DIR}"
MATERIALS_DIR="${SCENEEXPERT_MATERIALS_DIR:-${LEGACY_SCENESMITH_ROOT}/materials}"
HF_HOME="${HF_HOME:-/mnt/afs/visitor33/checkpoints/hf_cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
OPENCLIP_CHECKPOINT="${SCENEEXPERT_OPENCLIP_CHECKPOINT:-}"
REQUIRE_LOCAL_OPENCLIP="${SCENEEXPERT_REQUIRE_LOCAL_OPENCLIP:-1}"

if [ -n "$OPENCLIP_CHECKPOINT" ] && [ -d "$OPENCLIP_CHECKPOINT" ]; then
    OPENCLIP_CHECKPOINT="${OPENCLIP_CHECKPOINT}/open_clip_pytorch_model.bin"
fi
if [ -z "$OPENCLIP_CHECKPOINT" ]; then
    for candidate in \
        "${HF_HUB_CACHE}"/models--apple--DFN5B-CLIP-ViT-H-14-378/snapshots/*/open_clip_pytorch_model.bin; do
        if [ -f "$candidate" ]; then
            OPENCLIP_CHECKPOINT="$candidate"
            break
        fi
    done
fi
if [ "$REQUIRE_LOCAL_OPENCLIP" = "1" ] && { [ -z "$OPENCLIP_CHECKPOINT" ] || [ ! -r "$OPENCLIP_CHECKPOINT" ]; }; then
    echo "[ERROR] local DFN5B OpenCLIP checkpoint was not found" >&2
    echo "        Set SCENEEXPERT_OPENCLIP_CHECKPOINT to open_clip_pytorch_model.bin" >&2
    exit 1
fi

if [ -n "$OUTPUT_DIR" ]; then
    if [[ "$OUTPUT_DIR" != /* ]]; then
        OUTPUT_DIR="${WORKDIR}/${OUTPUT_DIR}"
    fi
else
    OUTPUT_DIR="${SCENEEXPERT_OUTPUT_DIR:-${WORKDIR}/outputs}"
fi

SAFE_NAME="${NAME//[^[:alnum:]_.-]/_}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${SCENEEXPERT_BATCH_LOG_DIR:-${WORKDIR}/tmp/batch_logs}"
SLICE_ROOT="${SCENEEXPERT_BATCH_SLICE_DIR:-${WORKDIR}/tmp/batch_slices}"
SCENE_LOG_DIR="${LOG_ROOT}/scenes"
BATCH_OUTPUT_DIR="${OUTPUT_DIR}/batches/${SAFE_NAME}_${RUN_STAMP}"
PROGRESS_LOG="${LOG_ROOT}/progress_${SAFE_NAME}_${RUN_STAMP}.log"
LLAMA_LOG="${LLAMA_LOG_FILE:-${LOG_ROOT}/llama_${SAFE_NAME}_${RUN_STAMP}.log}"
LLAMA_PID_FILE="${LLAMA_PID_FILE:-${LOG_ROOT}/llama_${SAFE_NAME}_${RUN_STAMP}.pid}"
SYSTEM_DEPS_LOG="${LOG_ROOT}/system_deps_${SAFE_NAME}_${RUN_STAMP}.log"
RUNTIME_CACHE_DIR="${WORKDIR}/.runtime_cache"
mkdir -p "$LOG_ROOT" "$SCENE_LOG_DIR" "$SLICE_ROOT" "$BATCH_OUTPUT_DIR" \
    "${RUNTIME_CACHE_DIR}/matplotlib"

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

COMPLETE_MARKER="Experiment execution completed in"
WATCHDOG_INTERVAL="${SCENEEXPERT_BATCH_WATCHDOG_INTERVAL:-30}"
STALL_THRESHOLD=$((STALL_MIN * 60))
POST_COMPLETE_GRACE="${SCENEEXPERT_BATCH_POST_COMPLETE_GRACE:-90}"
SIGTERM_GRACE="${SCENEEXPERT_BATCH_SIGTERM_GRACE:-20}"
if ! [[ "$WATCHDOG_INTERVAL" =~ ^[1-9][0-9]*$ && "$POST_COMPLETE_GRACE" =~ ^[1-9][0-9]*$ && \
        "$SIGTERM_GRACE" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] watchdog interval/grace values must be positive integers" >&2
    exit 2
fi

OWN_LLAMA=0
LLAMA_PID=""
ACTIVE_SCENE_PGID=""

cleanup() {
    local rc=$?
    trap - EXIT INT TERM

    if [ -n "$ACTIVE_SCENE_PGID" ]; then
        echo "[INFO] stopping active SceneSmith process group $ACTIVE_SCENE_PGID"
        kill -TERM -- "-${ACTIVE_SCENE_PGID}" 2>/dev/null || true
    fi
    if [ "$OWN_LLAMA" -eq 1 ] && [ -n "$LLAMA_PID" ] && kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "[INFO] stopping llama-server PID=$LLAMA_PID"
        kill -TERM "$LLAMA_PID" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$LLAMA_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$LLAMA_PID" 2>/dev/null; then
            kill -KILL "$LLAMA_PID" 2>/dev/null || true
        fi
    fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

system_deps_installed() {
    command -v dpkg-query >/dev/null 2>&1 || return 1
    local package status
    for package in "${SYSTEM_PACKAGES[@]}"; do
        status="$(dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null || true)"
        [ "$status" = "installed" ] || return 1
    done
}

ensure_system_deps() {
    if system_deps_installed; then
        echo "[1/3] System graphics dependencies already installed"
        return 0
    fi
    if [ "${SCENEEXPERT_INSTALL_SYSTEM_DEPS:-auto}" = "0" ]; then
        echo "[ERROR] SceneSmith graphics dependencies are missing and installation is disabled" >&2
        exit 1
    fi
    if [ "$(id -u)" -ne 0 ] || ! command -v apt-get >/dev/null 2>&1; then
        echo "[ERROR] SceneSmith graphics dependencies are missing; use the root ACP image" >&2
        exit 1
    fi

    echo "[1/3] Installing Task3.2 Mesa/EGL dependencies..."
    echo "      log: $SYSTEM_DEPS_LOG"
    if ! env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        -u ALL_PROXY -u all_proxy \
        apt-get update -qq > "$SYSTEM_DEPS_LOG" 2>&1; then
        echo "[ERROR] apt-get update failed; last log lines:" >&2
        tail -n 40 "$SYSTEM_DEPS_LOG" >&2 || true
        exit 1
    fi
    if ! env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        -u ALL_PROXY -u all_proxy DEBIAN_FRONTEND=noninteractive \
        apt-get install -y "${SYSTEM_PACKAGES[@]}" >> "$SYSTEM_DEPS_LOG" 2>&1; then
        echo "[ERROR] system dependency installation failed; last log lines:" >&2
        tail -n 60 "$SYSTEM_DEPS_LOG" >&2 || true
        exit 1
    fi
    echo "[OK] Task3.2 Mesa/EGL dependencies installed"
}

llama_is_ready() {
    curl -fsS --max-time 5 "http://127.0.0.1:${LLAMA_PORT}/health" >/dev/null 2>&1 &&
        curl -fsS --max-time 5 "http://127.0.0.1:${LLAMA_PORT}/v1/models" 2>/dev/null |
            grep -Fq "$LLAMA_ALIAS"
}

slice_one_row() {
    "$PYTHON_BIN" - "$1" "$2" "$3" <<'PY'
import csv
import sys

source_path, destination_path, wanted = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(source_path, newline="", encoding="utf-8-sig") as source:
    reader = csv.reader(source)
    try:
        header = next(reader)
    except StopIteration:
        raise SystemExit(2)
    if len(header) < 2 or header[0].strip() != "scene_index" or header[1].strip() != "prompt":
        print("CSV header must begin with: scene_index,prompt", file=sys.stderr)
        raise SystemExit(4)
    for row in reader:
        if not row:
            continue
        try:
            scene_index = int(row[0])
        except (ValueError, IndexError):
            continue
        if scene_index == wanted:
            with open(destination_path, "w", newline="", encoding="utf-8") as destination:
                writer = csv.writer(destination)
                writer.writerow(header)
                writer.writerow(row)
            raise SystemExit(0)
raise SystemExit(3)
PY
}

echo "============================================================"
echo " Task3.2 + llama.cpp (single shared GPU)"
echo "  scenes        : [$START, $END)"
echo "  name          : $SAFE_NAME"
echo "  experiment    : $EXPERIMENT"
echo "  floor mode    : $FLOOR_PLAN_MODE"
echo "  ACP GPU       : $SINGLE_GPU (logical GPU 0)"
echo "  model         : $LLAMA_ALIAS"
echo "  context/slots : $LLAMA_CTX_SIZE / $LLAMA_PARALLEL"
echo "  endpoint      : http://127.0.0.1:${LLAMA_PORT}/v1"
echo "  CSV           : $CSV_PATH"
echo "  HSSD data     : $HSSD_DATA_DIR"
echo "  OpenCLIP      : ${OPENCLIP_CHECKPOINT:-<not required>}"
echo "  outputs       : $BATCH_OUTPUT_DIR"
echo "  scene logs    : $SCENE_LOG_DIR"
echo "  progress log  : $PROGRESS_LOG"
echo "============================================================"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY RUN] file and argument validation passed; no install or process was started"
    exit 0
fi

ensure_system_deps

export CUDA_HOME=/usr/local/cuda-12.4
export PATH="/usr/local/cuda-12.4/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="$SINGLE_GPU"
export SCENESMITH_GEOMETRY_GPUS=0
export TORCH_HOME="${TORCH_HOME:-/mnt/afs/visitor33/checkpoints/torch_cache}"
export HF_HOME HF_HUB_CACHE
export XDG_CACHE_HOME="$RUNTIME_CACHE_DIR"
export MPLCONFIGDIR="${RUNTIME_CACHE_DIR}/matplotlib"
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LIDRA_SKIP_INIT=1
unset LIBGL_DRIVERS_PATH

export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-123}"
export OPENAI_BASE_URL="http://127.0.0.1:${LLAMA_PORT}/v1"
export OPENAI_USE_RESPONSES=false
export SCENEEXPERT_MODEL_ID="$LLAMA_ALIAS"
export SCENEEXPERT_DATA_DIR="$DATA_DIR"
export SCENEEXPERT_HSSD_DATA_DIR="$HSSD_DATA_DIR"
export SCENEEXPERT_REQUIRE_LOCAL_OPENCLIP="$REQUIRE_LOCAL_OPENCLIP"
if [ -n "$OPENCLIP_CHECKPOINT" ]; then
    export SCENEEXPERT_OPENCLIP_CHECKPOINT="$OPENCLIP_CHECKPOINT"
fi

if ! "$PYTHON_BIN" -c 'import bpy; import sqlite3; from pydrake.all import Quaternion' >/dev/null 2>&1; then
    echo "[ERROR] Task3.2 native import preflight failed" >&2
    echo "        Python: $PYTHON_BIN" >&2
    "$PYTHON_BIN" -c 'import bpy; import sqlite3; from pydrake.all import Quaternion' >&2 || true
    exit 1
fi
echo "[OK] Task3.2 native import preflight passed"

if llama_is_ready; then
    echo "[OK] reusing llama-server on port $LLAMA_PORT"
else
    echo "[2/3] Starting llama-server..."
    LLAMA_RUN_MODE=background \
    LLAMA_CUDA_VISIBLE_DEVICES="$SINGLE_GPU" \
    LLAMA_PORT="$LLAMA_PORT" \
    LLAMA_ALIAS="$LLAMA_ALIAS" \
    LLAMA_CTX_SIZE="$LLAMA_CTX_SIZE" \
    LLAMA_PARALLEL="$LLAMA_PARALLEL" \
    LLAMA_LOG_FILE="$LLAMA_LOG" \
    LLAMA_PID_FILE="$LLAMA_PID_FILE" \
        bash "$START_LLAMA_SCRIPT"

    if [ ! -s "$LLAMA_PID_FILE" ]; then
        echo "[ERROR] llama launcher did not create PID file: $LLAMA_PID_FILE" >&2
        exit 1
    fi
    LLAMA_PID="$(tr -d '[:space:]' < "$LLAMA_PID_FILE")"
    if ! [[ "$LLAMA_PID" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] invalid llama-server PID: '$LLAMA_PID'" >&2
        exit 1
    fi
    OWN_LLAMA=1

    if ! PORT="$LLAMA_PORT" WAIT_TIMEOUT="$LLAMA_WAIT_TIMEOUT" \
        EXPECTED_MODEL="$LLAMA_ALIAS" bash "$CHECK_LLAMA_SCRIPT"; then
        echo "[ERROR] llama-server did not become ready; last log lines:" >&2
        tail -n 100 "$LLAMA_LOG" >&2 || true
        exit 1
    fi
fi

echo "[3/3] Running Task3.2 scenes one-by-one..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] batch_start name=$SAFE_NAME range=[$START,$END)" > "$PROGRESS_LOG"

ok=0
fail=0
skip=0
stall=0
batch_started="$(date +%s)"

for ((idx=START; idx<END; idx++)); do
    SINGLE_CSV="${SLICE_ROOT}/${SAFE_NAME}_scene_${idx}_${RUN_STAMP}.csv"
    SCENE_LOG="${SCENE_LOG_DIR}/${SAFE_NAME}_scene_${idx}_${RUN_STAMP}.log"
    SCENE_OUTPUT="${BATCH_OUTPUT_DIR}/scene_${idx}/run"

    set +e
    slice_one_row "$CSV_PATH" "$SINGLE_CSV" "$idx"
    slice_rc=$?
    set -e
    if [ "$slice_rc" -ne 0 ]; then
        if [ "$slice_rc" -eq 3 ]; then
            echo "  [SKIP] scene $idx is not present in the CSV"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] scene=$idx skip" >> "$PROGRESS_LOG"
            skip=$((skip + 1))
            continue
        fi
        echo "[ERROR] failed to prepare CSV slice for scene $idx (rc=$slice_rc)" >&2
        exit 1
    fi

    echo ""
    echo "  -- scene $idx --"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] scene=$idx start output=$SCENE_OUTPUT log=$SCENE_LOG" >> "$PROGRESS_LOG"
    : > "$SCENE_LOG"
    scene_started="$(date +%s)"

    COMMAND=(
        "$PYTHON_BIN" main.py
        "experiment=$EXPERIMENT"
        "+name=${SAFE_NAME}_scene_${idx}"
        "hydra.run.dir=$SCENE_OUTPUT"
        "experiment.csv_path=$SINGLE_CSV"
        "experiment.num_workers=1"
        "floor_plan_agent.mode=$FLOOR_PLAN_MODE"
    )
    if [ -d "$MATERIALS_DIR" ]; then
        COMMAND+=(
            "experiment.materials_retrieval_server.data_path=$MATERIALS_DIR"
            "experiment.materials_retrieval_server.embeddings_path=$MATERIALS_DIR/embeddings"
        )
    fi

    setsid env CUDA_VISIBLE_DEVICES="$SINGLE_GPU" "${COMMAND[@]}" >> "$SCENE_LOG" 2>&1 &
    SCENE_PID=$!
    SCENE_PGID=$SCENE_PID
    ACTIVE_SCENE_PGID=$SCENE_PGID
    state=running
    state_started="$(date +%s)"
    last_growth=$state_started
    last_size=0
    stalled=0
    fatal_error=0

    while kill -0 "$SCENE_PID" 2>/dev/null; do
        now="$(date +%s)"
        current_size="$(stat -c %s "$SCENE_LOG" 2>/dev/null || echo 0)"
        if [ "$current_size" -gt "$last_size" ]; then
            last_size=$current_size
            last_growth=$now
        fi

        case "$state" in
            running)
                if grep -qF "$COMPLETE_MARKER" "$SCENE_LOG" 2>/dev/null; then
                    state=complete_marker
                    state_started=$now
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] scene=$idx completion_marker" >> "$PROGRESS_LOG"
                elif grep -Eq 'Error executing job with overrides:|Scene generation failed:|Process crashed \(exitcode=|Unknown prefix:' "$SCENE_LOG" 2>/dev/null; then
                    echo "  [ERROR] scene $idx reported a fatal error; sending TERM"
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] scene=$idx fatal_error" >> "$PROGRESS_LOG"
                    kill -TERM -- "-${SCENE_PGID}" 2>/dev/null || true
                    state=term_sent
                    state_started=$now
                    fatal_error=1
                elif [ $((now - last_growth)) -ge "$STALL_THRESHOLD" ]; then
                    echo "  [WARN] scene $idx stalled; sending TERM"
                    kill -TERM -- "-${SCENE_PGID}" 2>/dev/null || true
                    state=term_sent
                    state_started=$now
                    stalled=1
                fi
                ;;
            complete_marker)
                if [ $((now - last_growth)) -ge "$POST_COMPLETE_GRACE" ]; then
                    kill -INT -- "-${SCENE_PGID}" 2>/dev/null || true
                    state=int_sent
                    state_started=$now
                fi
                ;;
            int_sent)
                if [ $((now - state_started)) -ge "$SIGTERM_GRACE" ]; then
                    kill -TERM -- "-${SCENE_PGID}" 2>/dev/null || true
                    state=term_sent
                    state_started=$now
                fi
                ;;
            term_sent)
                if [ $((now - state_started)) -ge "$SIGTERM_GRACE" ]; then
                    kill -KILL -- "-${SCENE_PGID}" 2>/dev/null || true
                    state=kill_sent
                    state_started=$now
                fi
                ;;
            kill_sent)
                [ $((now - state_started)) -lt 10 ] || break
                ;;
        esac
        sleep "$WATCHDOG_INTERVAL"
    done

    set +e
    wait "$SCENE_PID" 2>/dev/null
    scene_rc=$?
    set -e
    ACTIVE_SCENE_PGID=""
    duration=$(( $(date +%s) - scene_started ))

    if [ "$stalled" -eq 1 ]; then
        stall=$((stall + 1))
        echo "  [STALL] scene $idx after ${duration}s"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] scene=$idx stall rc=$scene_rc duration=$duration" >> "$PROGRESS_LOG"
    elif [ "$fatal_error" -eq 1 ]; then
        fail=$((fail + 1))
        echo "  [FAIL] scene $idx reported a fatal error after ${duration}s; last log lines:"
        tail -n 12 "$SCENE_LOG" | sed 's/^/         /'
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] scene=$idx fail rc=$scene_rc duration=$duration" >> "$PROGRESS_LOG"
    elif grep -qF "$COMPLETE_MARKER" "$SCENE_LOG" 2>/dev/null; then
        ok=$((ok + 1))
        echo "  [OK] scene $idx completed in ${duration}s"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] scene=$idx ok rc=$scene_rc duration=$duration" >> "$PROGRESS_LOG"
    else
        fail=$((fail + 1))
        echo "  [FAIL] scene $idx rc=$scene_rc after ${duration}s; last log lines:"
        tail -n 12 "$SCENE_LOG" | sed 's/^/         /'
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] scene=$idx fail rc=$scene_rc duration=$duration" >> "$PROGRESS_LOG"
    fi
done

elapsed=$(( $(date +%s) - batch_started ))
echo "[$(date '+%Y-%m-%d %H:%M:%S')] batch_done ok=$ok fail=$fail stall=$stall skip=$skip duration=$elapsed" >> "$PROGRESS_LOG"

echo ""
echo "============================================================"
echo " Task3.2 batch finished"
echo "  range        : [$START, $END)"
echo "  ok           : $ok"
echo "  failed       : $fail"
echo "  stalled      : $stall"
echo "  skipped      : $skip"
echo "  elapsed      : ${elapsed}s"
echo "  outputs      : $BATCH_OUTPUT_DIR"
echo "  scene logs   : ${SCENE_LOG_DIR}/${SAFE_NAME}_scene_*_${RUN_STAMP}.log"
echo "  progress log : $PROGRESS_LOG"
echo "============================================================"

if [ "$ok" -eq 0 ] && [ $((fail + stall)) -gt 0 ]; then
    exit 1
fi
