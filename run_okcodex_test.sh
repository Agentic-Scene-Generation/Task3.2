#!/usr/bin/env bash
# Test okcodex image editing instead of local Qwen-Image-Edit
# Based on run_grounded_furniture_context_room_acp.sh

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAUNCHER="$PROJECT_ROOT/run_parallel_furniture_context_acp.sh"
GROUNDING_LAUNCHER="$PROJECT_ROOT/scripts/start_grounding_dino_server.sh"

SCENESMITH_GPU="${SCENESMITH_GPU:-0}"
GROUNDING_DINO_GPU="${GROUNDING_DINO_GPU:-1}"
GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-18030}"
GROUNDING_BASE_URL="http://127.0.0.1:${GROUNDING_DINO_PORT}"
SCENE_CONCURRENCY="${SCENE_CONCURRENCY:-1}"
MAX_CASES="${MAX_CASES:-1}"
PIPELINE_STOP_STAGE="${PIPELINE_STOP_STAGE:-furniture}"
CASE_FILTER="${CASE_FILTER-}"
RUN_ID="${RUN_ID:-okcodex_test_$(date +%Y%m%d_%H%M%S)}"
STOP_GROUNDING_DINO_ON_EXIT="${STOP_GROUNDING_DINO_ON_EXIT:-true}"
DRY_RUN="${DRY_RUN:-false}"

# OKCodex API configuration
OKCODEX_API_KEY="${OKCODEX_API_KEY:-sk-2859167011902c268ec87a71178a31b5a481d61722b30ef2bcb836eb23a68cad}"
OKCODEX_BASE_URL="${OKCODEX_BASE_URL:-https://api.okcodex.cn}"
OKCODEX_IMAGE_MODEL="${OKCODEX_IMAGE_MODEL:-gpt-image-1.5}"

TASK32_ASSET_ROOT="${TASK32_ASSET_ROOT:-/mnt/afs/visitor33/Task3.2}"
EMBEDDING_SERVER="${EMBEDDING_SERVER:-$TASK32_ASSET_ROOT/bin/llama-server-cuda12-sm90}"
LLAMA_LAUNCHER="${LLAMA_LAUNCHER:-$TASK32_ASSET_ROOT/start_llama.sh}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/afs/visitor33/scenesmith-sequence/.venv/bin/python}"

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
        echo "[CLEANUP] stopping GroundingDINO started by this script"
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

for required_file in "$BASE_LAUNCHER" "$GROUNDING_LAUNCHER" \
    "$EMBEDDING_SERVER" "$LLAMA_LAUNCHER"; do
    if [ ! -f "$required_file" ]; then
        echo "[ERROR] required file not found: $required_file" >&2
        exit 1
    fi
done

if [ -z "$OKCODEX_API_KEY" ]; then
    echo "[ERROR] OKCODEX_API_KEY is required" >&2
    exit 1
fi

if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY RUN] okcodex test validation passed"
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

echo "========== OKCODEX TEST =========="
echo "run id:                 $RUN_ID"
echo "room selection:         ${CASE_FILTER:-first built-in case}"
echo "pipeline stop stage:    $PIPELINE_STOP_STAGE"
echo "SceneSmith/Qwen GPU:     $SCENESMITH_GPU"
echo "GroundingDINO GPU:       $GROUNDING_DINO_GPU"
echo "GroundingDINO endpoint:  $GROUNDING_BASE_URL"
echo "OKCodex API:             $OKCODEX_BASE_URL"
echo "OKCodex Model:           $OKCODEX_IMAGE_MODEL"
echo "======================================="

# Export okcodex config
export OKCODEX_API_KEY
export OKCODEX_BASE_URL
export OKCODEX_IMAGE_MODEL
export FURNITURE_CONTEXT_IMAGE_GENERATION_ENABLED=true
export FURNITURE_CONTEXT_IMAGE_GENERATION_BACKEND=okcodex

# Use okcodex backend instead of qwen_local
SCENESMITH_GPU="$SCENESMITH_GPU" \
SCENE_CONCURRENCY="$SCENE_CONCURRENCY" \
MAX_CASES="$MAX_CASES" \
CASE_FILTER="$CASE_FILTER" \
RUN_ID="$RUN_ID" \
PIPELINE_STOP_STAGE="$PIPELINE_STOP_STAGE" \
FURNITURE_GROUNDED_LAYOUT_ENABLED=true \
FURNITURE_GROUNDED_LAYOUT_BASE_URL="$GROUNDING_BASE_URL" \
EMBEDDING_SERVER="$EMBEDDING_SERVER" \
LLAMA_LAUNCHER="$LLAMA_LAUNCHER" \
PYTHON_BIN="$PYTHON_BIN" \
CRITIC_PROBE_RENDER_FINAL_VIEWS="${CRITIC_PROBE_RENDER_FINAL_VIEWS:-true}" \
    bash "$BASE_LAUNCHER"

echo "[OK] okcodex test completed"
echo "[OK] output: $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
