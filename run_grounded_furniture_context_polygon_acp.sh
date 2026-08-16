#!/usr/bin/env bash
# Run one full polygon-room ACP acceptance case with grounded image layout.
#
# GPU 0: Qwen3.6-27B Q8, embedding, SceneSmith/Blender
# GPU 1: Qwen-Image-Edit and GroundingDINO
#
# Usage:
#   bash run_grounded_furniture_context_polygon_acp.sh
#   CASE_FILTER= MAX_CASES=2 PIPELINE_STOP_STAGE=furniture \
#       bash run_grounded_furniture_context_polygon_acp.sh

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAUNCHER="$PROJECT_ROOT/run_parallel_furniture_context_polygon_acp.sh"
GROUNDING_LAUNCHER="$PROJECT_ROOT/scripts/start_grounding_dino_server.sh"

SCENESMITH_GPU="${SCENESMITH_GPU:-0}"
QWEN_IMAGE_EDIT_GPU="${QWEN_IMAGE_EDIT_GPU:-1}"
GROUNDING_DINO_GPU="${GROUNDING_DINO_GPU:-1}"
GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-18030}"
GROUNDING_BASE_URL="http://127.0.0.1:${GROUNDING_DINO_PORT}"
SCENE_CONCURRENCY="${SCENE_CONCURRENCY:-1}"
MAX_CASES="${MAX_CASES:-1}"
PIPELINE_STOP_STAGE="${PIPELINE_STOP_STAGE:-manipuland}"
CASE_FILTER="${CASE_FILTER-}"
RUN_ID="${RUN_ID:-grounded_polygon_acp_$(date +%Y%m%d_%H%M%S)}"
STOP_GROUNDING_DINO_ON_EXIT="${STOP_GROUNDING_DINO_ON_EXIT:-true}"
DRY_RUN="${DRY_RUN:-false}"

TASK32_ASSET_ROOT="${TASK32_ASSET_ROOT:-/mnt/afs/visitor33/Task3.2}"
EMBEDDING_SERVER="${EMBEDDING_SERVER:-$TASK32_ASSET_ROOT/bin/llama-server-cuda12-sm90}"
LLAMA_LAUNCHER="${LLAMA_LAUNCHER:-$TASK32_ASSET_ROOT/start_llama.sh}"
QWEN_IMAGE_EDIT_MODEL_DIR="${QWEN_IMAGE_EDIT_MODEL_DIR:-$TASK32_ASSET_ROOT/models/Qwen-Image-Edit}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/afs/visitor33/scenesmith-qwen/.venv/bin/python}"

GROUNDING_STARTED_BY_THIS_SCRIPT=false

normalize_bool() {
    case "${1,,}" in
        1|true|yes|y|on) printf 'true' ;;
        0|false|no|n|off|'') printf 'false' ;;
        *) return 1 ;;
    esac
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM HUP
    if [ "$GROUNDING_STARTED_BY_THIS_SCRIPT" = "true" ] \
        && [ "$STOP_GROUNDING_DINO_ON_EXIT" = "true" ]; then
        echo "[CLEANUP] stopping GroundingDINO started by this ACP job"
        GROUNDING_DINO_PORT="$GROUNDING_DINO_PORT" \
            bash "$GROUNDING_LAUNCHER" --stop || true
    fi
    exit "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

if ! STOP_GROUNDING_DINO_ON_EXIT="$(
    normalize_bool "$STOP_GROUNDING_DINO_ON_EXIT"
)"; then
    echo "[ERROR] STOP_GROUNDING_DINO_ON_EXIT must be true or false" >&2
    exit 2
fi
if ! DRY_RUN="$(normalize_bool "$DRY_RUN")"; then
    echo "[ERROR] DRY_RUN must be true or false" >&2
    exit 2
fi
for integer_setting in SCENESMITH_GPU QWEN_IMAGE_EDIT_GPU GROUNDING_DINO_GPU \
    GROUNDING_DINO_PORT SCENE_CONCURRENCY MAX_CASES; do
    integer_value="${!integer_setting}"
    if ! [[ "$integer_value" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] $integer_setting must be a non-negative integer" >&2
        exit 2
    fi
done
if [ "$SCENE_CONCURRENCY" -lt 1 ] || [ "$MAX_CASES" -lt 1 ]; then
    echo "[ERROR] SCENE_CONCURRENCY and MAX_CASES must be positive" >&2
    exit 2
fi
if [ "$SCENESMITH_GPU" = "$QWEN_IMAGE_EDIT_GPU" ]; then
    echo "[ERROR] use different GPUs for SceneSmith/Qwen and Qwen-Image-Edit" >&2
    exit 2
fi
for required_file in "$BASE_LAUNCHER" "$GROUNDING_LAUNCHER" \
    "$EMBEDDING_SERVER" "$LLAMA_LAUNCHER"; do
    if [ ! -f "$required_file" ]; then
        echo "[ERROR] required file not found: $required_file" >&2
        exit 1
    fi
done
if [ ! -f "$QWEN_IMAGE_EDIT_MODEL_DIR/model_index.json" ]; then
    echo "[ERROR] Qwen-Image-Edit model not found: $QWEN_IMAGE_EDIT_MODEL_DIR" >&2
    exit 1
fi
if [ "$GROUNDING_DINO_PORT" -lt 1 ] || [ "$GROUNDING_DINO_PORT" -gt 65535 ]; then
    echo "[ERROR] GROUNDING_DINO_PORT must be between 1 and 65535" >&2
    exit 2
fi

if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY RUN] polygon launcher validation passed"
    echo "[DRY RUN] base launcher: $BASE_LAUNCHER"
    echo "[DRY RUN] output: $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
    exit 0
fi

if curl -fsS --max-time 2 "$GROUNDING_BASE_URL/health" \
    | grep -Fq '"ready":true'; then
    echo "[GROUNDING] reusing ready service: $GROUNDING_BASE_URL"
else
    echo "[GROUNDING] starting on physical GPU $GROUNDING_DINO_GPU"
    GROUNDING_DINO_GPU_ID="$GROUNDING_DINO_GPU" \
    GROUNDING_DINO_PORT="$GROUNDING_DINO_PORT" \
        bash "$GROUNDING_LAUNCHER" --background
    GROUNDING_STARTED_BY_THIS_SCRIPT=true
fi

echo "========== GROUNDED POLYGON ACP =========="
echo "run id:                 $RUN_ID"
echo "polygon selection:      ${CASE_FILTER:-first built-in case}"
echo "pipeline stop stage:    $PIPELINE_STOP_STAGE"
echo "SceneSmith/Qwen GPU:     $SCENESMITH_GPU"
echo "Qwen-Image-Edit GPU:     $QWEN_IMAGE_EDIT_GPU"
echo "GroundingDINO GPU:       $GROUNDING_DINO_GPU"
echo "GroundingDINO endpoint:  $GROUNDING_BASE_URL"
echo "=========================================="

SCENESMITH_GPU="$SCENESMITH_GPU" \
QWEN_IMAGE_EDIT_GPU="$QWEN_IMAGE_EDIT_GPU" \
SCENE_CONCURRENCY="$SCENE_CONCURRENCY" \
MAX_CASES="$MAX_CASES" \
CASE_FILTER="$CASE_FILTER" \
RUN_ID="$RUN_ID" \
PIPELINE_STOP_STAGE="$PIPELINE_STOP_STAGE" \
FURNITURE_GROUNDED_LAYOUT_ENABLED=true \
FURNITURE_GROUNDED_LAYOUT_BASE_URL="$GROUNDING_BASE_URL" \
EMBEDDING_SERVER="$EMBEDDING_SERVER" \
LLAMA_LAUNCHER="$LLAMA_LAUNCHER" \
QWEN_IMAGE_EDIT_MODEL_DIR="$QWEN_IMAGE_EDIT_MODEL_DIR" \
PYTHON_BIN="$PYTHON_BIN" \
CRITIC_PROBE_RENDER_FINAL_VIEWS="${CRITIC_PROBE_RENDER_FINAL_VIEWS:-true}" \
    bash "$BASE_LAUNCHER"

echo "[OK] grounded polygon ACP run completed"
echo "[OK] output: $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
