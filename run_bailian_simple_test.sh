#!/bin/bash

# Simplified Bailian test - reuses okcodex test infrastructure

set -e

PROJECT_ROOT="/mnt/afs/visitor33/Task3.2"
cd "$PROJECT_ROOT"

# Read Bailian API key
BAILIAN_API_KEY=$(python3 -c "import json; print(json.load(open('/mnt/afs/visitor33/bailian.json'))['OPENAI_API_KEY'])")

# Test configuration - use okcodex test as template
export RUN_ID="bailian_test_$(date +%Y%m%d_%H%M%S)"
export MAX_CASES=1
export SCENE_CONCURRENCY=1
export SCENESMITH_GPU="${SCENESMITH_GPU:-0}"
export GROUNDING_DINO_GPU="${GROUNDING_DINO_GPU:-0}"  # Use same GPU as SceneSmith
export GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-18030}"
export PIPELINE_STOP_STAGE="${PIPELINE_STOP_STAGE:-furniture}"
export STOP_GROUNDING_DINO_ON_EXIT="${STOP_GROUNDING_DINO_ON_EXIT:-true}"

GROUNDING_LAUNCHER="$PROJECT_ROOT/scripts/start_grounding_dino_server.sh"
GROUNDING_STARTED_BY_THIS_SCRIPT=false

cleanup() {
    local exit_code=$?
    if [ "$GROUNDING_STARTED_BY_THIS_SCRIPT" = "true" ] \
        && [ "$STOP_GROUNDING_DINO_ON_EXIT" = "true" ]; then
        echo "[CLEANUP] stopping GroundingDINO started by this script"
        GROUNDING_DINO_PORT="$GROUNDING_DINO_PORT" \
            bash "$GROUNDING_LAUNCHER" --stop || true
    fi
    exit "$exit_code"
}

trap cleanup EXIT

# Bailian configuration
export BAILIAN_API_KEY="$BAILIAN_API_KEY"
export BAILIAN_IMAGE_MODEL="${BAILIAN_IMAGE_MODEL:-wan2.7-image-pro}"
export FURNITURE_CONTEXT_IMAGE_GENERATION_ENABLED=true
export FURNITURE_CONTEXT_IMAGE_GENERATION_BACKEND="bailian"

# Set dummy okcodex key to avoid config parsing errors (not used when backend=bailian)
export OKCODEX_API_KEY="${OKCODEX_API_KEY:-dummy}"

# Reuse okcodex test services configuration
export PROJECT_BIN="$PROJECT_ROOT/bin"
export EMBEDDING_SERVER="${EMBEDDING_SERVER:-$PROJECT_BIN/llama-server-cuda12-sm90}"
export LLAMA_LAUNCHER="${LLAMA_LAUNCHER:-$PROJECT_ROOT/start_llama.sh}"
export PYTHON_BIN="${PYTHON_BIN:-/mnt/afs/visitor33/scenesmith-sequence/.venv/bin/python}"
export GROUNDING_BASE_URL="${GROUNDING_BASE_URL:-http://127.0.0.1:18030}"

# Use the same launcher as okcodex
BASE_LAUNCHER="$PROJECT_ROOT/run_parallel_furniture_context_acp.sh"

echo "========================================"
echo "Bailian Test (using okcodex infrastructure)"
echo "========================================"
echo "RUN_ID: $RUN_ID"
echo "Model: $BAILIAN_IMAGE_MODEL"
echo "Cases: $MAX_CASES"
echo "GPU: $SCENESMITH_GPU"
echo "GroundingDINO GPU: $GROUNDING_DINO_GPU"
echo "========================================"

# Start GroundingDINO if not already running
if curl -fsS --max-time 2 "$GROUNDING_BASE_URL/health" 2>/dev/null | grep -Fq '"ready":true'; then
    echo "[GROUNDING] reusing ready service: $GROUNDING_BASE_URL"
else
    echo "[GROUNDING] starting on GPU $GROUNDING_DINO_GPU"
    GROUNDING_DINO_GPU_ID="$GROUNDING_DINO_GPU" \
    GROUNDING_DINO_PORT="$GROUNDING_DINO_PORT" \
        bash "$GROUNDING_LAUNCHER" --background
    GROUNDING_STARTED_BY_THIS_SCRIPT=true
fi

# Call the base launcher (it handles case file creation and service startup)
SCENESMITH_GPU="$SCENESMITH_GPU" \
SCENE_CONCURRENCY="$SCENE_CONCURRENCY" \
MAX_CASES="$MAX_CASES" \
RUN_ID="$RUN_ID" \
PIPELINE_STOP_STAGE="$PIPELINE_STOP_STAGE" \
FURNITURE_GROUNDED_LAYOUT_ENABLED=true \
FURNITURE_GROUNDED_LAYOUT_BASE_URL="$GROUNDING_BASE_URL" \
EMBEDDING_SERVER="$EMBEDDING_SERVER" \
LLAMA_LAUNCHER="$LLAMA_LAUNCHER" \
PYTHON_BIN="$PYTHON_BIN" \
CRITIC_PROBE_RENDER_FINAL_VIEWS="${CRITIC_PROBE_RENDER_FINAL_VIEWS:-true}" \
STOP_QWEN_IMAGE_EDIT_ON_EXIT=false \
    bash "$BASE_LAUNCHER"

echo ""
echo "[OK] Bailian test completed"
echo "[OK] output: $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
