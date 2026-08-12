#!/usr/bin/env bash
# Test OKCodex integration by using the base script infrastructure
# but with okcodex backend instead of qwen-image

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# OKCodex API configuration
export OKCODEX_API_KEY="sk-2859167011902c268ec87a71178a31b5a481d61722b30ef2bcb836eb23a68cad"
export OKCODEX_BASE_URL="https://api.okcodex.cn"
export OKCODEX_IMAGE_MODEL="gpt-image-1.5"

# Disable qwen-image service startup (we're using okcodex cloud service)
export START_QWEN_IMAGE_EDIT="false"
export STOP_QWEN_IMAGE_EDIT_ON_EXIT="false"

# GroundingDINO configuration
export GROUNDING_DINO_GPU="${GROUNDING_DINO_GPU:-1}"
export GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-18030}"
export GROUNDING_DINO_BASE_URL="http://127.0.0.1:${GROUNDING_DINO_PORT}"

# Use only 1 case for quick test
export MAX_CASES="${MAX_CASES:-1}"
export SCENE_CONCURRENCY="${SCENE_CONCURRENCY:-1}"
export PIPELINE_STOP_STAGE="${PIPELINE_STOP_STAGE:-furniture}"

# Custom furniture agent config with okcodex backend
export FURNITURE_AGENT_OVERRIDE="okcodex_furniture_agent"

# Run ID
export RUN_ID="okcodex_test_$(date +%Y%m%d_%H%M%S)"

echo "===== OKCodex Integration Test ====="
echo "Run ID: $RUN_ID"
echo "Backend: okcodex (cloud API)"
echo "GroundingDINO: $GROUNDING_DINO_BASE_URL"
echo "Max cases: $MAX_CASES"
echo "Stop stage: $PIPELINE_STOP_STAGE"
echo ""

# Call the base infrastructure script
exec bash "$PROJECT_ROOT/run_parallel_furniture_context_acp.sh"
