#!/usr/bin/env bash
# Direct OKCodex test - bypasses qwen-image service startup

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARALLEL_LAUNCHER="$PROJECT_ROOT/run_parallel_rooms_no_shared_base.sh"

# GPU configuration
SCENESMITH_GPU="${SCENESMITH_GPU:-0}"
SCENE_CONCURRENCY="${SCENE_CONCURRENCY:-1}"
PIPELINE_STOP_STAGE="${PIPELINE_STOP_STAGE:-furniture}"
FURNITURE_DESIGNER_THINKING="${FURNITURE_DESIGNER_THINKING:-high}"
FURNITURE_CRITIC_THINKING="${FURNITURE_CRITIC_THINKING:-low}"
RUN_ID="okcodex_direct_test_$(date +%Y%m%d_%H%M%S)"
CASES_DIR="$PROJECT_ROOT/tmp/acp_furniture_context/$RUN_ID"
CASES_FILE="$CASES_DIR/test_case.tsv"
MAX_CASES=1

# OKCodex API configuration
export OKCODEX_API_KEY="sk-2859167011902c268ec87a71178a31b5a481d61722b30ef2bcb836eb23a68cad"
export OKCODEX_BASE_URL="https://api.okcodex.cn"
export OKCODEX_IMAGE_MODEL="gpt-image-1.5"

# Create test case file
mkdir -p "$CASES_DIR"
cat > "$CASES_FILE" <<'EOF'
test_bedroom	Simple bedroom test	A bedroom with a bed, nightstand, and wardrobe.
EOF

echo "========== OKCODEX DIRECT TEST =========="
echo "run id:                 $RUN_ID"
echo "SceneSmith/LLM GPU:     $SCENESMITH_GPU"
echo "scene concurrency:      $SCENE_CONCURRENCY"
echo "selected cases:         $MAX_CASES"
echo "pipeline stop stage:    $PIPELINE_STOP_STAGE"
echo "furniture thinking:     $FURNITURE_DESIGNER_THINKING/$FURNITURE_CRITIC_THINKING"
echo "context image backend:  okcodex"
echo "cases file:             $CASES_FILE"
echo "output:                 $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
echo "=========================================="

echo "[SCENES] starting test with okcodex backend"
CUDA_VISIBLE_DEVICES="$SCENESMITH_GPU" \
LLAMA_CUDA_VISIBLE_DEVICES="$SCENESMITH_GPU" \
SCENE_CONCURRENCY="$SCENE_CONCURRENCY" \
MAX_CASES="$MAX_CASES" \
RUN_ID="$RUN_ID" \
CRITIC_PROBE_CASES_FILE="$CASES_FILE" \
FURNITURE_CONTEXT_IMAGE_GENERATION_ENABLED="true" \
FURNITURE_CONTEXT_IMAGE_GENERATION_BACKEND="okcodex" \
FURNITURE_GROUNDED_LAYOUT_ENABLED="${FURNITURE_GROUNDED_LAYOUT_ENABLED:-true}" \
FURNITURE_GROUNDED_LAYOUT_BASE_URL="${FURNITURE_GROUNDED_LAYOUT_BASE_URL:-http://127.0.0.1:18030}" \
PIPELINE_STOP_STAGE="$PIPELINE_STOP_STAGE" \
FURNITURE_DESIGNER_THINKING="$FURNITURE_DESIGNER_THINKING" \
FURNITURE_CRITIC_THINKING="$FURNITURE_CRITIC_THINKING" \
FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS="${FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS:-false}" \
bash "$PARALLEL_LAUNCHER"

echo "[OK] okcodex test completed"
echo "[OK] output: $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
