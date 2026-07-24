#!/usr/bin/env bash
# Run a contiguous scene_index range from a prompt CSV through Task3.2.
#
# Each scene gets a fresh main.py process and Hydra output directory. A watchdog
# terminates the whole scene process group if its log stops growing for too long.
# One vLLM server is reused (or started once) for the entire batch.
#
# Usage:
#   bash run_prompt_batch.sh [options] START END [CSV_PATH] [NAME] \
#       [VLLM_GPU_IDS] [SCENE_GPU_IDS] [MODEL_ID] [STALL_MIN]
#
# Options:
#   --output-dir DIR       Parent directory for per-scene Hydra outputs.
#   --model-dir DIR        Local model directory passed to vLLM.
#   --experiment NAME      Hydra experiment config (default: indoor_scene_generation).
#   --floor-plan-mode MODE room or house (default: room).
#   --dry-run              Validate inputs and print resolved settings only.
#   -h, --help             Show this help.
#
# Examples:
#   bash run_prompt_batch.sh 0 1 /path/to/prompts.csv smoke 0,1 2
#   bash run_prompt_batch.sh --model-dir /path/to/Qwen3___6-27B 0 1
#   bash run_prompt_batch.sh --output-dir /path/to/outputs \
#       --experiment ablation_4_qwen3_harness_memory \
#       0 20 /path/to/prompts.csv memory_batch 0,1 2
#
# Runtime configuration is read from .env (or SCENEEXPERT_ENV_FILE), matching
# scripts/run_experiment.sh. Set SCENEEXPERT_START_VLLM=0 to use an already
# running OpenAI-compatible endpoint from OPENAI_BASE_URL.

set -uo pipefail

usage() {
    sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
}

OUTPUT_DIR=""
MODEL_DIR_ARG=""
EXPERIMENT="${SCENEEXPERT_BATCH_EXPERIMENT:-indoor_scene_generation}"
FLOOR_PLAN_MODE="${FLOOR_PLAN_MODE:-room}"
DRY_RUN=0
POSITIONAL=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --output-dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --output-dir)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --output-dir requires a value" >&2
                exit 2
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-dir=*)
            MODEL_DIR_ARG="${1#*=}"
            shift
            ;;
        --model-dir)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --model-dir requires a value" >&2
                exit 2
            fi
            MODEL_DIR_ARG="$2"
            shift 2
            ;;
        --experiment=*)
            EXPERIMENT="${1#*=}"
            shift
            ;;
        --experiment)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --experiment requires a value" >&2
                exit 2
            fi
            EXPERIMENT="$2"
            shift 2
            ;;
        --floor-plan-mode=*)
            FLOOR_PLAN_MODE="${1#*=}"
            shift
            ;;
        --floor-plan-mode)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --floor-plan-mode requires a value" >&2
                exit 2
            fi
            FLOOR_PLAN_MODE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while [ "$#" -gt 0 ]; do
                POSITIONAL+=("$1")
                shift
            done
            ;;
        -* )
            echo "[ERROR] unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL[@]}"
if [ "$#" -lt 2 ]; then
    usage >&2
    exit 2
fi

START="$1"
END="$2"
CSV_PATH="${3:-data/prompt_gen/prompts_v1.csv}"
NAME="${4:-batch_${START}_${END}}"
VLLM_GPU_IDS="${5:-${SCENEEXPERT_BATCH_VLLM_GPU_IDS:-0,1}}"
SCENE_GPU_IDS="${6:-${SCENEEXPERT_BATCH_SCENE_GPU_IDS:-2}}"
MODEL_ID_ARG="${7:-}"
STALL_MIN="${8:-${SCENEEXPERT_BATCH_STALL_MIN:-30}}"

if [ "$#" -gt 8 ]; then
    echo "[ERROR] too many positional arguments" >&2
    usage >&2
    exit 2
fi
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
    echo "        polygon will be enabled in the separate polygon migration." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
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
    echo "[INFO] loaded environment: $ENV_FILE"
fi

MODEL_DIR="${MODEL_DIR_ARG:-${SCENEEXPERT_MODEL_DIR:-}}"
if [ -n "$MODEL_DIR" ] && [[ "$MODEL_DIR" != /* ]]; then
    MODEL_DIR="$PROJECT_DIR/$MODEL_DIR"
fi
MODEL_ID_FROM_DIR=""
if [ -n "$MODEL_DIR" ]; then
    MODEL_ID_FROM_DIR="$(basename "${MODEL_DIR%/}")"
    MODEL_ID_FROM_DIR="${MODEL_ID_FROM_DIR/Qwen3___6/Qwen3.6}"
    MODEL_ID_FROM_DIR="${MODEL_ID_FROM_DIR/Qwen3___5/Qwen3.5}"
fi
MODEL_ID="${MODEL_ID_ARG:-${SCENEEXPERT_MODEL_ID:-${MODEL_ID_FROM_DIR:-Qwen3.6-35B-A3B}}}"
VLLM_PORT="${SCENEEXPERT_VLLM_PORT:-8000}"
VLLM_HEALTH_URL="${SCENEEXPERT_VLLM_HEALTH_URL:-http://localhost:${VLLM_PORT}/health}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:${VLLM_PORT}/v1}"
START_VLLM="${SCENEEXPERT_START_VLLM:-1}"
MAX_MODEL_LEN="${SCENEEXPERT_MAX_MODEL_LEN:-65536}"
ACTIVATE_VENV="${SCENEEXPERT_ACTIVATE_VENV:-1}"
VENV_PATH="${SCENEEXPERT_VENV_PATH:-}"

# This checkout intentionally does not contain a committed virtualenv. On the
# migration host, reuse the environment that already runs scenesmith-qwen so
# `bash run_prompt_batch.sh 0 1` remains a zero-configuration entry point.
if [ -z "$VENV_PATH" ]; then
    if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
        VENV_PATH="$PROJECT_DIR/.venv"
    elif [ -f "$PROJECT_DIR/../scenesmith-qwen/.venv/bin/activate" ]; then
        VENV_PATH="$PROJECT_DIR/../scenesmith-qwen/.venv"
    else
        VENV_PATH="$PROJECT_DIR/.venv"
    fi
fi

if [[ "$VENV_PATH" != /* ]]; then
    VENV_PATH="$PROJECT_DIR/$VENV_PATH"
fi
if [ "$ACTIVATE_VENV" = "1" ]; then
    if [ ! -f "$VENV_PATH/bin/activate" ]; then
        echo "[ERROR] virtual environment not found: $VENV_PATH/bin/activate" >&2
        echo "        Set SCENEEXPERT_ACTIVATE_VENV=0 to use the current environment." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
fi
PYTHON_BIN="$(command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] python is not available after environment setup" >&2
    exit 1
fi

# The reused scenesmith-qwen venv was built from the `scenesmith` Conda
# interpreter. Its standard-library extensions (for example _sqlite3 + ICU)
# require the newer libstdc++ shipped in that base environment. The old qwen
# launcher activated the Conda env and sourced setup_env.sh before every scene;
# reproduce the relevant runtime-library setup without importing its unrelated
# SSH configuration side effects.
PYTHON_BASE_PREFIX="$("$PYTHON_BIN" -c 'import sys; print(sys.base_prefix)')"
SCENE_RUNTIME_LIB_DIR="${SCENEEXPERT_SCENE_RUNTIME_LIB_DIR:-$PYTHON_BASE_PREFIX/lib}"
if [ -f "$SCENE_RUNTIME_LIB_DIR/libstdc++.so.6" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${SCENE_RUNTIME_LIB_DIR}:"*) ;;
        *) export LD_LIBRARY_PATH="${SCENE_RUNTIME_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
    case ":$PATH:" in
        *":${PYTHON_BASE_PREFIX}/bin:"*) ;;
        *) export PATH="${VENV_PATH}/bin:${PYTHON_BASE_PREFIX}/bin:$PATH" ;;
    esac
    echo "[INFO] scene runtime libraries: $SCENE_RUNTIME_LIB_DIR"
fi

VLLM_BIN_DIR="${SCENEEXPERT_VLLM_BIN_DIR:-}"
if [ -z "$VLLM_BIN_DIR" ] && [ -x "/mnt/afs/visitor33/miniconda3/envs/vllm/bin/vllm" ]; then
    VLLM_BIN_DIR="/mnt/afs/visitor33/miniconda3/envs/vllm/bin"
fi

if [ -z "$MODEL_DIR" ]; then
    MODEL_LEAF="${MODEL_ID##*/}"
    MODELSCOPE_MODEL_LEAF="${MODEL_LEAF/Qwen3.6/Qwen3___6}"
    for candidate in \
        "/mnt/afs/visitor33/models/Qwen/$MODELSCOPE_MODEL_LEAF" \
        "/mnt/afs/visitor33/models/$MODEL_LEAF" \
        "$PROJECT_DIR/models/$MODEL_LEAF"; do
        if [ -d "$candidate" ]; then
            MODEL_DIR="$candidate"
            break
        fi
    done
fi
if [ -z "$MODEL_DIR" ] || [ ! -d "$MODEL_DIR" ]; then
    echo "[ERROR] model directory not found: ${MODEL_DIR:-<unresolved>}" >&2
    echo "        Pass --model-dir /absolute/path/to/model" >&2
    exit 1
fi

LEGACY_QWEN_ROOT="$PROJECT_DIR/../scenesmith-qwen"
DATA_DIR="${SCENEEXPERT_DATA_DIR:-}"
HSSD_DATA_DIR="${SCENEEXPERT_HSSD_DATA_DIR:-}"
MATERIALS_DIR="${SCENEEXPERT_MATERIALS_DIR:-}"
if [ -z "$DATA_DIR" ] && [ -d "$LEGACY_QWEN_ROOT/data" ]; then
    DATA_DIR="$LEGACY_QWEN_ROOT/data"
fi
if [ -z "$HSSD_DATA_DIR" ] && [ -d "$LEGACY_QWEN_ROOT/data/hssd-models" ]; then
    HSSD_DATA_DIR="$LEGACY_QWEN_ROOT/data"
fi
if [ -z "$MATERIALS_DIR" ] && [ -d "$LEGACY_QWEN_ROOT/materials/embeddings" ]; then
    MATERIALS_DIR="$LEGACY_QWEN_ROOT/materials"
fi

# The old qwen checkout searches its pre-staged Hugging Face cache for the
# 3.9GB DFN5B OpenCLIP checkpoint. Task3.2 intentionally requires an explicit
# local path, so resolve that cache snapshot here for zero-configuration batch
# runs on the migration host.
export HF_HOME="${HF_HOME:-/mnt/afs/visitor33/checkpoints/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
OPENCLIP_DIR="${SCENEEXPERT_OPENCLIP_DIR:-}"
OPENCLIP_CHECKPOINT="${SCENEEXPERT_OPENCLIP_CHECKPOINT:-}"
REQUIRE_LOCAL_OPENCLIP="${SCENEEXPERT_REQUIRE_LOCAL_OPENCLIP:-1}"

if [ -n "$OPENCLIP_CHECKPOINT" ] && { [ -d "$OPENCLIP_CHECKPOINT" ] || [[ "$OPENCLIP_CHECKPOINT" != *.bin ]]; }; then
    OPENCLIP_CHECKPOINT="$OPENCLIP_CHECKPOINT/open_clip_pytorch_model.bin"
fi
if [ -z "$OPENCLIP_CHECKPOINT" ] && [ -n "$OPENCLIP_DIR" ]; then
    candidate="$OPENCLIP_DIR/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin"
    if [ -f "$candidate" ]; then
        OPENCLIP_CHECKPOINT="$candidate"
    fi
fi
if [ -z "$OPENCLIP_CHECKPOINT" ] && [ -n "$DATA_DIR" ]; then
    candidate="$DATA_DIR/openclip/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin"
    if [ -f "$candidate" ]; then
        OPENCLIP_CHECKPOINT="$candidate"
    fi
fi
if [ -z "$OPENCLIP_CHECKPOINT" ]; then
    for candidate in \
        "$HF_HUB_CACHE"/models--apple--DFN5B-CLIP-ViT-H-14-378/snapshots/*/open_clip_pytorch_model.bin; do
        if [ -f "$candidate" ]; then
            OPENCLIP_CHECKPOINT="$candidate"
            break
        fi
    done
fi

if [ -n "$OPENCLIP_CHECKPOINT" ] && [ -r "$OPENCLIP_CHECKPOINT" ]; then
    echo "[INFO] local OpenCLIP checkpoint: $OPENCLIP_CHECKPOINT"
elif [ "$REQUIRE_LOCAL_OPENCLIP" = "1" ]; then
    echo "[ERROR] local DFN5B OpenCLIP checkpoint was not found or is not readable" >&2
    echo "        Set SCENEEXPERT_OPENCLIP_CHECKPOINT to open_clip_pytorch_model.bin" >&2
    exit 1
else
    OPENCLIP_CHECKPOINT=""
fi

cd "$PROJECT_DIR"
if [[ "$CSV_PATH" != /* ]]; then
    CSV_PATH="$PROJECT_DIR/$CSV_PATH"
fi
if [ ! -f "$CSV_PATH" ]; then
    echo "[ERROR] CSV not found: $CSV_PATH" >&2
    exit 1
fi

if [ -n "$OUTPUT_DIR" ]; then
    if [[ "$OUTPUT_DIR" != /* ]]; then
        OUTPUT_DIR="$PROJECT_DIR/$OUTPUT_DIR"
    fi
else
    OUTPUT_DIR="${SCENEEXPERT_OUTPUT_DIR:-$PROJECT_DIR/outputs}"
fi

LOG_ROOT="${SCENEEXPERT_BATCH_LOG_DIR:-$PROJECT_DIR/tmp/batch_logs}"
SLICE_ROOT="${SCENEEXPERT_BATCH_SLICE_DIR:-$PROJECT_DIR/tmp/batch_slices}"
if [[ "$LOG_ROOT" != /* ]]; then
    LOG_ROOT="$PROJECT_DIR/$LOG_ROOT"
fi
if [[ "$SLICE_ROOT" != /* ]]; then
    SLICE_ROOT="$PROJECT_DIR/$SLICE_ROOT"
fi

TS="$(date +%Y%m%d_%H%M%S)"
SAFE_NAME="${NAME//[^[:alnum:]_.-]/_}"
SCENE_LOG_DIR="$LOG_ROOT/scenes"
PROGRESS_LOG="$LOG_ROOT/progress_${SAFE_NAME}_${TS}.log"
VLLM_LOG="${SCENEEXPERT_VLLM_LOG:-$LOG_ROOT/vllm_${TS}.log}"
BATCH_OUTPUT_DIR="$OUTPUT_DIR/batches/${SAFE_NAME}_${TS}"

STALL_THRESHOLD=$((STALL_MIN * 60))
WATCHDOG_INTERVAL="${SCENEEXPERT_BATCH_WATCHDOG_INTERVAL:-30}"
POST_COMPLETE_GRACE="${SCENEEXPERT_BATCH_POST_COMPLETE_GRACE:-90}"
SIGTERM_GRACE="${SCENEEXPERT_BATCH_SIGTERM_GRACE:-20}"
COMPLETE_MARKER="Experiment execution completed in"
TOTAL=$((END - START))
NUM_VLLM_GPUS="$(awk -F',' '{print NF}' <<< "$VLLM_GPU_IDS")"

if ! [[ "$WATCHDOG_INTERVAL" =~ ^[1-9][0-9]*$ && "$POST_COMPLETE_GRACE" =~ ^[1-9][0-9]*$ && "$SIGTERM_GRACE" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] watchdog interval/grace environment values must be positive integers" >&2
    exit 2
fi

echo "============================================================"
echo " Task3.2 prompt batch [$TS]"
echo "  scene indexes : [$START, $END) (count=$TOTAL)"
echo "  CSV           : $CSV_PATH"
echo "  name          : $NAME"
echo "  experiment    : $EXPERIMENT"
echo "  floor mode    : $FLOOR_PLAN_MODE"
echo "  model         : $MODEL_ID"
echo "  model dir     : ${MODEL_DIR:-<resolve via scripts/start_vllm.sh>}"
echo "  scene python  : $PYTHON_BIN"
echo "  vLLM GPUs     : $VLLM_GPU_IDS (TP=$NUM_VLLM_GPUS)"
echo "  max model len : $MAX_MODEL_LEN"
echo "  scene GPUs    : $SCENE_GPU_IDS"
echo "  vLLM endpoint : $OPENAI_BASE_URL"
echo "  HSSD data     : ${HSSD_DATA_DIR:-<Task3.2 default>}"
echo "  OpenCLIP      : ${OPENCLIP_CHECKPOINT:-<online pretrained tag>}"
echo "  output parent : $OUTPUT_DIR"
echo "  stall timeout : ${STALL_MIN} min"
echo "  logs          : $LOG_ROOT"
echo "============================================================"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] validation passed; no vLLM or scene process was started."
    exit 0
fi

# Match the old scenesmith-qwen launcher on bare containers, but avoid an
# unconditional apt update on hosts that already provide all Blender runtime
# libraries. Checking only libXrender is insufficient: bpy also links directly
# against libX11, libXfixes, libXi, and the other libraries below.
check_blender_system_libraries() {
    "$PYTHON_BIN" - <<'PY'
import ctypes
import sys

libraries = (
    "libGL.so.1",
    "libglib-2.0.so.0",
    "libgomp.so.1",
    "libX11.so.6",
    "libXrender.so.1",
    "libSM.so.6",
    "libICE.so.6",
    "libXext.so.6",
    "libXi.so.6",
    "libXxf86vm.so.1",
    "libXfixes.so.3",
    "libxkbcommon.so.0",
    "libEGL.so.1",
    "libGLESv2.so.2",
)
missing = []
for library in libraries:
    try:
        ctypes.CDLL(library)
    except OSError:
        missing.append(library)
if missing:
    print("[INFO] missing Blender system libraries: " + ", ".join(missing))
    raise SystemExit(1)
PY
}

if ! check_blender_system_libraries; then
    if [ "${SCENEEXPERT_INSTALL_SYSTEM_DEPS:-auto}" = "0" ]; then
        echo "[ERROR] Blender system libraries are missing and system dependency installation is disabled" >&2
        exit 1
    fi
    if [ "$(id -u)" -ne 0 ] || ! command -v apt-get >/dev/null 2>&1; then
        echo "[ERROR] Blender system libraries are missing; install the SceneSmith system packages first" >&2
        exit 1
    fi
    echo "[INFO] installing missing SceneSmith system libraries"
    apt-get update -qq
    apt-get install -y \
        libgl1 libglib2.0-0 libgomp1 libx11-6 libxrender1 libsm6 libice6 libxext6 tmux \
        libxi6 libxxf86vm1 libxfixes3 libxkbcommon0 \
        libegl1 libegl-mesa0 libgles2 libegl-dev
fi

# Fail before the 10+ minute model startup if the scene process's native import
# chain cannot be loaded. Importing bpy alone does not exercise the
# pydrake -> IPython -> sqlite3 path that also depends on the Conda C++ runtime.
if ! "$PYTHON_BIN" -c 'import bpy; import sqlite3; from pydrake.all import Quaternion' >/dev/null 2>&1; then
    echo "[ERROR] scene native import preflight failed after runtime setup" >&2
    echo "        Python: $PYTHON_BIN" >&2
    echo "        Runtime libraries: ${LD_LIBRARY_PATH:-<system default>}" >&2
    "$PYTHON_BIN" -c 'import bpy; import sqlite3; from pydrake.all import Quaternion'
    exit 1
fi
echo "[OK] scene native import preflight passed (bpy, sqlite3, pydrake)"

mkdir -p "$SCENE_LOG_DIR" "$SLICE_ROOT" "$BATCH_OUTPUT_DIR"

export SCENEEXPERT_MODEL_ID="$MODEL_ID"
if [ -n "$MODEL_DIR" ]; then
    export SCENEEXPERT_MODEL_DIR="$MODEL_DIR"
fi
if [ -n "$DATA_DIR" ]; then
    export SCENEEXPERT_DATA_DIR="$DATA_DIR"
fi
if [ -n "$HSSD_DATA_DIR" ]; then
    export SCENEEXPERT_HSSD_DATA_DIR="$HSSD_DATA_DIR"
fi
if [ -n "$OPENCLIP_DIR" ]; then
    export SCENEEXPERT_OPENCLIP_DIR="$OPENCLIP_DIR"
fi
if [ -n "$OPENCLIP_CHECKPOINT" ]; then
    export SCENEEXPERT_OPENCLIP_CHECKPOINT="$OPENCLIP_CHECKPOINT"
fi
export SCENEEXPERT_REQUIRE_LOCAL_OPENCLIP="$REQUIRE_LOCAL_OPENCLIP"
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed}"
export OPENAI_BASE_URL
export SCENEEXPERT_VLLM_LOG="$VLLM_LOG"

check_vllm_health() {
    "$PYTHON_BIN" - "$VLLM_HEALTH_URL" <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
        raise SystemExit(0 if 200 <= response.status < 300 else 1)
except Exception:
    raise SystemExit(1)
PY
}

get_served_model() {
    "$PYTHON_BIN" - "$OPENAI_BASE_URL/models" <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
        payload = json.load(response)
    data = payload.get("data") or []
    print(data[0].get("id", "") if data else "")
except Exception:
    raise SystemExit(1)
PY
}

ensure_vllm() {
    local served
    if check_vllm_health; then
        served="$(get_served_model || true)"
        if [ "$served" != "$MODEL_ID" ]; then
            echo "[ERROR] existing vLLM serves '$served', expected '$MODEL_ID'" >&2
            return 1
        fi
        echo "[OK] reusing vLLM serving '$served'"
        return 0
    fi

    if [ "$START_VLLM" != "1" ]; then
        echo "[ERROR] no healthy vLLM at $VLLM_HEALTH_URL and SCENEEXPERT_START_VLLM=$START_VLLM" >&2
        return 1
    fi

    echo "[INFO] starting vLLM once for the batch"
    CUDA_VISIBLE_DEVICES="$VLLM_GPU_IDS" \
        PATH="${VLLM_BIN_DIR:+$VLLM_BIN_DIR:}$PATH" \
        SCENEEXPERT_ENV_FILE=/dev/null \
        SCENEEXPERT_TENSOR_PARALLEL_SIZE="$NUM_VLLM_GPUS" \
        SCENEEXPERT_MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        SCENEEXPERT_MODEL_ID="$MODEL_ID" \
        SCENEEXPERT_VLLM_LOG="$VLLM_LOG" \
        bash "$PROJECT_DIR/scripts/start_vllm.sh"

    if ! check_vllm_health; then
        echo "[ERROR] start_vllm.sh returned but health check still fails" >&2
        return 1
    fi
    served="$(get_served_model || true)"
    if [ "$served" != "$MODEL_ID" ]; then
        echo "[ERROR] newly started vLLM serves '$served', expected '$MODEL_ID'" >&2
        return 1
    fi
    echo "[OK] vLLM is ready"
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
        if scene_index != wanted:
            continue
        with open(destination_path, "w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            writer.writerow(header)
            writer.writerow(row)
        raise SystemExit(0)
raise SystemExit(3)
PY
}

terminate_group() {
    local pgid="$1"
    local signal="$2"
    kill "-$signal" -- "-$pgid" 2>/dev/null || true
}

if ! ensure_vllm; then
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] batch_start name=$NAME range=[$START,$END)" > "$PROGRESS_LOG"
ok=0
fail=0
skip=0
stall=0
batch_t0="$(date +%s)"

for ((idx=START; idx<END; idx++)); do
    SINGLE_CSV="$SLICE_ROOT/${SAFE_NAME}_scene_${idx}_${TS}.csv"
    SCENE_LOG="$SCENE_LOG_DIR/${SAFE_NAME}_scene_${idx}_${TS}.log"
    # Keep two path levels below the batch root so main.py's latest-run symlink
    # is created inside this batch instead of at the repository root.
    SCENE_OUTPUT="$BATCH_OUTPUT_DIR/scene_${idx}/run"

    slice_one_row "$CSV_PATH" "$SINGLE_CSV" "$idx"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        if [ "$rc" -eq 3 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] scene=$idx skip reason=missing_csv_row" >> "$PROGRESS_LOG"
            echo "[SKIP] scene $idx is not present in the CSV"
            skip=$((skip + 1))
            continue
        fi
        echo "[ERROR] failed to prepare CSV slice for scene $idx (rc=$rc)" >&2
        exit 1
    fi

    scene_t0="$(date +%s)"
    : > "$SCENE_LOG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] scene=$idx start output=$SCENE_OUTPUT" >> "$PROGRESS_LOG"
    echo ""
    echo "-- scene $idx --------------------------------------------------"

    COMMAND=(
        "$PYTHON_BIN" main.py
        "experiment=$EXPERIMENT"
        "+name=${SAFE_NAME}_scene_${idx}"
        "hydra.run.dir=$SCENE_OUTPUT"
        "experiment.csv_path=$SINGLE_CSV"
        "experiment.num_workers=1"
        "floor_plan_agent.mode=$FLOOR_PLAN_MODE"
    )
    if [ -n "$MATERIALS_DIR" ]; then
        COMMAND+=(
            "experiment.materials_retrieval_server.data_path=$MATERIALS_DIR"
            "experiment.materials_retrieval_server.embeddings_path=$MATERIALS_DIR/embeddings"
        )
    fi

    setsid env CUDA_VISIBLE_DEVICES="$SCENE_GPU_IDS" "${COMMAND[@]}" >> "$SCENE_LOG" 2>&1 &
    SCENE_PID=$!
    SCENE_PGID=$SCENE_PID

    state="running"
    state_t="$(date +%s)"
    last_growth_t="$state_t"
    last_size=0
    killed_for_stall=0

    while kill -0 "$SCENE_PID" 2>/dev/null; do
        now="$(date +%s)"
        current_size="$(stat -c %s "$SCENE_LOG" 2>/dev/null || echo 0)"
        if [ "$current_size" -gt "$last_size" ]; then
            last_size="$current_size"
            last_growth_t="$now"
        fi

        case "$state" in
            running)
                if grep -qF "$COMPLETE_MARKER" "$SCENE_LOG" 2>/dev/null; then
                    state="complete_marker"
                    state_t="$now"
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] scene=$idx completion_marker" >> "$PROGRESS_LOG"
                elif [ $((now - last_growth_t)) -ge "$STALL_THRESHOLD" ]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] scene=$idx stall signal=TERM" >> "$PROGRESS_LOG"
                    terminate_group "$SCENE_PGID" TERM
                    killed_for_stall=1
                    state="term_sent"
                    state_t="$now"
                fi
                ;;
            complete_marker)
                # Normal Task3.2 shutdown stops several local servers in a
                # finally block. Only interrupt when that cleanup also stops
                # producing log output for the full grace period.
                if [ $((now - last_growth_t)) -ge "$POST_COMPLETE_GRACE" ]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] scene=$idx post_complete_hang signal=INT" >> "$PROGRESS_LOG"
                    terminate_group "$SCENE_PGID" INT
                    state="int_sent"
                    state_t="$now"
                fi
                ;;
            int_sent)
                if [ $((now - state_t)) -ge "$SIGTERM_GRACE" ]; then
                    terminate_group "$SCENE_PGID" TERM
                    state="term_sent"
                    state_t="$now"
                fi
                ;;
            term_sent)
                if [ $((now - state_t)) -ge "$SIGTERM_GRACE" ]; then
                    terminate_group "$SCENE_PGID" KILL
                    state="kill_sent"
                    state_t="$now"
                fi
                ;;
            kill_sent)
                if [ $((now - state_t)) -ge 10 ]; then
                    break
                fi
                ;;
        esac
        sleep "$WATCHDOG_INTERVAL"
    done

    wait "$SCENE_PID" 2>/dev/null
    rc=$?
    scene_dt=$(($(date +%s) - scene_t0))

    if [ "$killed_for_stall" -eq 1 ]; then
        stall=$((stall + 1))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] scene=$idx stalled rc=$rc seconds=$scene_dt" >> "$PROGRESS_LOG"
        echo "[STALL] scene $idx stopped after ${scene_dt}s"
    elif grep -qF "$COMPLETE_MARKER" "$SCENE_LOG" 2>/dev/null; then
        ok=$((ok + 1))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] scene=$idx ok rc=$rc seconds=$scene_dt" >> "$PROGRESS_LOG"
        echo "[OK] scene $idx completed in ${scene_dt}s"
    else
        fail=$((fail + 1))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] scene=$idx failed rc=$rc seconds=$scene_dt" >> "$PROGRESS_LOG"
        echo "[FAIL] scene $idx exited rc=$rc after ${scene_dt}s"
        tail -n 12 "$SCENE_LOG" | sed 's/^/       /'
    fi
done

total_dt=$(($(date +%s) - batch_t0))
echo "[$(date '+%Y-%m-%d %H:%M:%S')] batch_done ok=$ok fail=$fail stall=$stall skip=$skip seconds=$total_dt" >> "$PROGRESS_LOG"

echo ""
echo "============================================================"
echo " Batch finished"
echo "  ok       : $ok"
echo "  failed   : $fail"
echo "  stalled  : $stall"
echo "  skipped  : $skip"
echo "  elapsed  : ${total_dt}s"
echo "  outputs  : $BATCH_OUTPUT_DIR"
echo "  logs     : $SCENE_LOG_DIR/${SAFE_NAME}_scene_*_${TS}.log"
echo "  progress : $PROGRESS_LOG"
echo "============================================================"

if [ "$ok" -eq 0 ] && [ $((fail + stall)) -gt 0 ]; then
    exit 1
fi
exit 0
