#!/bin/bash

# Test scenesmith with Bailian image editor (wan2.7-image-pro)

set -e

PROJECT_ROOT="/mnt/afs/visitor33/Task3.2"
cd "$PROJECT_ROOT"

# Read Bailian API key from config file
BAILIAN_API_KEY=$(python3 -c "import json; print(json.load(open('/mnt/afs/visitor33/bailian.json'))['OPENAI_API_KEY'])")

# Test configuration
RUN_ID="${RUN_ID:-bailian_test_$(date +%Y%m%d_%H%M%S)}"
MAX_CASES="${MAX_CASES:-1}"
SCENE_CONCURRENCY="${SCENE_CONCURRENCY:-1}"
CASE_FILTER="${CASE_FILTER:-scene_000}"
SCENESMITH_GPU="${SCENESMITH_GPU:-0}"

# Bailian configuration
export BAILIAN_API_KEY="$BAILIAN_API_KEY"
export BAILIAN_IMAGE_MODEL="${BAILIAN_IMAGE_MODEL:-wan2.7-image-pro}"

# Other services
export GROUNDING_DINO_BASE_URL="${GROUNDING_DINO_BASE_URL:-http://127.0.0.1:18030}"
export EMBEDDING_SERVER="${EMBEDDING_SERVER:-http://127.0.0.1:18010}"

PARALLEL_LAUNCHER="$PROJECT_ROOT/run_parallel_rooms_no_shared_base.sh"

echo "========================================"
echo "Bailian Test Configuration"
echo "========================================"
echo "RUN_ID: $RUN_ID"
echo "Model: $BAILIAN_IMAGE_MODEL"
echo "Cases: $MAX_CASES"
echo "Concurrency: $SCENE_CONCURRENCY"
echo "GPU: $SCENESMITH_GPU"
echo "Grounding: $GROUNDING_DINO_BASE_URL"
echo "Embedding: $EMBEDDING_SERVER"
echo "========================================"

# Run with bailian backend
CUDA_VISIBLE_DEVICES="$SCENESMITH_GPU" \
LLAMA_CUDA_VISIBLE_DEVICES="$SCENESMITH_GPU" \
SCENE_CONCURRENCY="$SCENE_CONCURRENCY" \
MAX_CASES="$MAX_CASES" \
CASE_FILTER="$CASE_FILTER" \
RUN_ID="$RUN_ID" \
FURNITURE_GROUNDED_LAYOUT_ENABLED="true" \
FURNITURE_GROUNDED_LAYOUT_BASE_URL="$GROUNDING_DINO_BASE_URL" \
EMBEDDING_SERVER="$EMBEDDING_SERVER" \
CRITIC_PROBE_RENDER_FINAL_VIEWS="true" \
    bash "$PARALLEL_LAUNCHER" \
    experiment=test_critic_probe \
    furniture_agent=bailian_furniture_agent \
    experiment.pipeline_stop_stage=furniture \
    experiment.furniture_agent.designer.thinking=high \
    experiment.furniture_agent.critic.thinking=low

echo ""
echo "[OK] Bailian test completed"
echo "[OK] output: $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
