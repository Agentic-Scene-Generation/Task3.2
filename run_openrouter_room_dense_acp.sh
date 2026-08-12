#!/usr/bin/env bash
# Batch ACP test for OpenRouter furniture context images in dense bedroom mode.
#
# GPU allocation (single H100 80GB):
#   GPU 0: Qwen llama-server, embedding service, SceneSmith rendering/collision, GroundingDINO
#
# Default usage inside a single-GPU ACP task:
#   cd /mnt/afs/visitor33/Task3.2
#   bash run_openrouter_room_dense_acp.sh
#
# Useful overrides:
#   MAX_CASES=5 bash run_openrouter_room_dense_acp.sh
#   PIPELINE_STOP_STAGE=furniture bash run_openrouter_room_dense_acp.sh
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARALLEL_LAUNCHER="$PROJECT_ROOT/run_parallel_rooms_no_shared_base.sh"
GROUNDING_LAUNCHER="$PROJECT_ROOT/scripts/start_grounding_dino_server.sh"

SCENESMITH_GPU="${SCENESMITH_GPU:-0}"
GROUNDING_DINO_GPU="${GROUNDING_DINO_GPU:-0}"
SCENE_CONCURRENCY="${SCENE_CONCURRENCY:-3}"
PIPELINE_STOP_STAGE="${PIPELINE_STOP_STAGE:-manipuland}"
FURNITURE_DESIGNER_THINKING="${FURNITURE_DESIGNER_THINKING:-high}"
FURNITURE_CRITIC_THINKING="${FURNITURE_CRITIC_THINKING:-low}"
STOP_GROUNDING_DINO_ON_EXIT="${STOP_GROUNDING_DINO_ON_EXIT:-true}"
RUN_ID="${RUN_ID:-openrouter_room_dense_acp_$(date +%Y%m%d_%H%M%S)}"
CASES_DIR="${CASES_DIR:-$PROJECT_ROOT/tmp/acp_openrouter_room_dense/$RUN_ID}"
CASES_FILE="$CASES_DIR/dense_bedroom_cases.tsv"
GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-18030}"
GROUNDING_BASE_URL="http://127.0.0.1:${GROUNDING_DINO_PORT}"
OPENROUTER_IMAGE_MODEL="${OPENROUTER_IMAGE_MODEL:-openai/gpt-image-2}"
OPENROUTER_KEY_FILE="${OPENROUTER_KEY_FILE:-/mnt/afs/visitor33/exportkey.sh}"

# Dense bedroom cases with neutral styling and varied functional layouts.
DENSE_BEDROOM_CASES=(
    $'dense_bedroom_0\tDensely furnished bedroom with storage and seating zones\tA densely furnished bedroom has a double bed centered against a wall, with a nightstand and table lamp on each side. Place a large rug beneath the bed and two armchairs with a small side table near the window. Add a tall wardrobe against one wall, a long six-drawer dresser against another wall, a vanity table with a stool, a storage bench at the foot of the bed, two floor lamps in open corners, three potted plants on suitable surfaces, and a small bookshelf filled with books.'
    $'dense_bedroom_1\tDensely furnished bedroom with reading and storage areas\tA densely furnished bedroom has a wooden double bed against a wall, with two nightstands and table lamps. A large rug covers the central floor area. Place one armchair and a reading lamp near the window and another chair in an open corner. Include a tall wardrobe, a dresser with a mirror, a vanity table with a stool, an ottoman at the foot of the bed, two tall bookcases filled with books and small objects, three floor lamps, and five potted plants distributed without blocking circulation.'
    $'dense_bedroom_2\tDensely furnished bedroom with wardrobe media and vanity zones\tA densely furnished bedroom has a king-size bed centered against a wall with matching nightstands on both sides. A large rug extends from the bed toward a seating area with two armchairs and a small side table. Include a wardrobe system along one wall, a six-drawer dresser, a vanity station with a large mirror and bench, a media console with a television, a tall bookcase, a storage ottoman, a chaise lounge in an open corner, two floor lamps, two table lamps, and four potted plants.'
    $'dense_bedroom_3\tDensely furnished bedroom with sleeping sitting and work zones\tA densely furnished bedroom has a wooden bed against a wall, flanked by two large nightstands with drawers. A rug defines the sleeping area, while two armchairs and a small table form a sitting area near the window. Add a tall wardrobe, an eight-drawer dresser with a mirror, a vanity table with a chair, a storage chest at the foot of the bed, two tall bookcases, a writing desk with an office chair, three floor lamps, two table lamps, and four potted plants.'
    $'dense_bedroom_4\tDensely furnished bedroom with display and relaxation areas\tA densely furnished bedroom has a double bed centered against a wall, with two nightstands and table lamps. Place a vanity with a large mirror and stool against the far wall, and arrange two armchairs with a small round table near the window. Include a tall wardrobe, a six-drawer chest, a chaise lounge, a bench at the foot of the bed, two standing floor lamps, a media console, a tall display cabinet, a large rug, and three potted plants.'
    $'dense_bedroom_5\tDensely furnished bedroom with multiple seating and storage areas\tA densely furnished bedroom has a double bed centered against a wall, with two wooden nightstands and table lamps. A large rug defines the bed area, while two armchairs and a small table create a seating area near the window. Add a tall wardrobe, a six-drawer dresser, a vanity table with a mirror and stool, a storage bench at the foot of the bed, a tall bookcase filled with books, two additional side tables, three floor lamps, and six potted plants placed without obstructing the door or window.'
    $'dense_bedroom_6\tDensely furnished bedroom with long storage wall and room divider\tA densely furnished bedroom has a king-size bed centered against the back wall, flanked by two nightstands with table lamps. A long twelve-drawer dresser occupies one wall, and a low cabinet serves as a media console. Add two armchairs with a side table near the window, a tall wardrobe, a vanity desk with a chair, a storage bench, a room-divider bookshelf, two floor lamps, a large area rug, and four potted plants.'
    $'dense_bedroom_7\tDensely furnished bedroom with books storage and work areas\tA densely furnished bedroom has a wooden double bed centered against a wall with a nightstand on each side. Place two tall bookcases along separate walls, a large rug in the bed area, and two armchairs with a low table in a sitting area. Add a tall wardrobe, an eight-drawer dresser, a writing desk with a chair, a vanity with a mirror, an ottoman, two additional bookshelves, three floor lamps, and five potted plants.'
    $'dense_bedroom_8\tDensely furnished bedroom with plants seating and organized storage\tA densely furnished bedroom has a low double bed against a wall with two wooden nightstands. Two armchairs face a small table near the window, with several potted plants placed around the seating area while keeping the window clear. Add a tall wardrobe, a six-drawer dresser, a vanity table with a mirror and stool, a storage bench with baskets underneath, two plant stands, three floor lamps, a large rug, a bookshelf filled with books, and additional plants on suitable surfaces.'
    $'dense_bedroom_9\tDensely furnished bedroom with reading media and storage zones\tA densely furnished bedroom has a double bed centered against a wall, with two substantial nightstands and table lamps. Use one large rug for the sleeping area and a smaller rug for a reading area with two armchairs and a side table near the window. Include a tall wardrobe, a six-drawer dresser, a vanity table with a mirror and bench, a storage trunk at the foot of the bed, a tall bookcase, a media console, two floor lamps, two table lamps, and five potted plants.'
)

MAX_CASES="${MAX_CASES:-3}"

GROUNDING_STARTED_BY_THIS_SCRIPT=false
cleanup_started=false

normalize_bool() {
    case "${1,,}" in
        1|true|yes|y|on) printf 'true' ;;
        0|false|no|n|off|'') printf 'false' ;;
        *) return 1 ;;
    esac
}

load_openrouter_api_key() {
    # An explicitly supplied environment variable takes precedence; otherwise
    # source the user-provided export file without printing the credential.
    if [ -n "${OPENROUTER_API_KEY:-}" ]; then
        return
    fi
    if [ ! -r "$OPENROUTER_KEY_FILE" ]; then
        echo "[ERROR] OpenRouter key file is not readable: $OPENROUTER_KEY_FILE" >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$OPENROUTER_KEY_FILE"
    if [ -z "${OPENROUTER_API_KEY:-}" ]; then
        echo "[ERROR] OPENROUTER_KEY_FILE must export OPENROUTER_API_KEY" >&2
        exit 1
    fi
}

cleanup() {
    local exit_code=$?
    if [ "$cleanup_started" = "true" ]; then
        return
    fi
    cleanup_started=true
    trap - EXIT INT TERM HUP
    if [ "$GROUNDING_STARTED_BY_THIS_SCRIPT" = "true" ] \
        && [ "$STOP_GROUNDING_DINO_ON_EXIT" = "true" ]; then
        echo "[CLEANUP] stopping GroundingDINO started by this ACP job"
        bash "$GROUNDING_LAUNCHER" --stop || true
    fi
    exit "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

if ! STOP_GROUNDING_DINO_ON_EXIT="$(normalize_bool "$STOP_GROUNDING_DINO_ON_EXIT")"; then
    echo "[ERROR] STOP_GROUNDING_DINO_ON_EXIT must be true or false" >&2
    exit 2
fi
for integer_setting in SCENE_CONCURRENCY MAX_CASES; do
    integer_value="${!integer_setting}"
    if ! [[ "$integer_value" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] $integer_setting must be a positive integer" >&2
        exit 2
    fi
done
if [ "$MAX_CASES" -gt "${#DENSE_BEDROOM_CASES[@]}" ]; then
    echo "[ERROR] MAX_CASES=$MAX_CASES exceeds the ${#DENSE_BEDROOM_CASES[@]} built-in cases" >&2
    exit 2
fi
case "$PIPELINE_STOP_STAGE" in
    furniture|wall_mounted|ceiling_mounted|manipuland) ;;
    *)
        echo "[ERROR] invalid PIPELINE_STOP_STAGE: $PIPELINE_STOP_STAGE" >&2
        exit 2
        ;;
esac
for required_file in "$PARALLEL_LAUNCHER" "$GROUNDING_LAUNCHER"; do
    if [ ! -f "$required_file" ]; then
        echo "[ERROR] required launcher not found: $required_file" >&2
        exit 1
    fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi is unavailable; run inside an ACP task with GPU" >&2
    exit 1
fi
VISIBLE_GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d '[:space:]')"
if [ "$VISIBLE_GPU_COUNT" -lt 1 ]; then
    echo "[ERROR] this test requires at least one visible GPU; found $VISIBLE_GPU_COUNT" >&2
    exit 1
fi

load_openrouter_api_key
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "[ERROR] OPENROUTER_API_KEY environment variable is required" >&2
    exit 1
fi
export OPENROUTER_API_KEY OPENROUTER_IMAGE_MODEL

# Optional backend sections are still composed by Hydra, so keep their
# otherwise-unused credentials resolvable without exposing real keys.
export BAILIAN_API_KEY="${BAILIAN_API_KEY:-dummy}"
export OKCODEX_API_KEY="${OKCODEX_API_KEY:-dummy}"

mkdir -p "$CASES_DIR"
: > "$CASES_FILE"
for case_row in "${DENSE_BEDROOM_CASES[@]:0:$MAX_CASES}"; do
    printf '%s\n' "$case_row" >> "$CASES_FILE"
done

echo "========== ACP OPENROUTER DENSE BEDROOM TEST =========="
echo "run id:                     $RUN_ID"
echo "visible GPUs:                $VISIBLE_GPU_COUNT"
echo "SceneSmith/LLM GPU:          $SCENESMITH_GPU"
echo "GroundingDINO GPU:           $GROUNDING_DINO_GPU"
echo "scene concurrency:           $SCENE_CONCURRENCY"
echo "selected dense bedrooms:     $MAX_CASES / ${#DENSE_BEDROOM_CASES[@]}"
echo "pipeline stop stage:         $PIPELINE_STOP_STAGE"
echo "floor plan mode:             room (rectangular)"
echo "placement-order reference:   disabled"
echo "context image backend:       openrouter"
echo "openrouter model:             ${OPENROUTER_IMAGE_MODEL:-openai/gpt-image-2}"
echo "grounded layout:             ${FURNITURE_GROUNDED_LAYOUT_ENABLED:-true}"
echo "cases file:                  $CASES_FILE"
echo "output:                      $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
echo "===================================================="
echo "selected prompts:"
while IFS=$'\t' read -r selected_case_id selected_goal _; do
    printf '  - %s: %s\n' "$selected_case_id" "$selected_goal"
done < "$CASES_FILE"

# Start GroundingDINO service if not already running
if curl -fsS --max-time 2 "$GROUNDING_BASE_URL/health" 2>/dev/null \
    | grep -Fq '"ready":true'; then
    echo "[GROUNDING] reusing ready service: $GROUNDING_BASE_URL"
else
    echo "[GROUNDING] starting on GPU $GROUNDING_DINO_GPU"
    GROUNDING_DINO_GPU_ID="$GROUNDING_DINO_GPU" \
    GROUNDING_DINO_PORT="$GROUNDING_DINO_PORT" \
        bash "$GROUNDING_LAUNCHER" --background
    GROUNDING_STARTED_BY_THIS_SCRIPT=true
fi

echo "[OPENROUTER] Using OpenRouter API backend, skipping Qwen-Image-Edit service"
echo "[SCENES] starting dense bedroom batch with OpenRouter image generation"

CUDA_VISIBLE_DEVICES="$SCENESMITH_GPU" \
LLAMA_CUDA_VISIBLE_DEVICES="$SCENESMITH_GPU" \
SCENE_CONCURRENCY="$SCENE_CONCURRENCY" \
MAX_CASES="$MAX_CASES" \
RUN_ID="$RUN_ID" \
CRITIC_PROBE_CASES_FILE="$CASES_FILE" \
FLOOR_PLAN_MODE=room \
FURNITURE_PLACEMENT_ORDER_ENABLED=false \
FURNITURE_CONTEXT_IMAGE_GENERATION_ENABLED=true \
FURNITURE_CONTEXT_IMAGE_GENERATION_BACKEND=openrouter \
FURNITURE_GROUNDED_LAYOUT_ENABLED="${FURNITURE_GROUNDED_LAYOUT_ENABLED:-true}" \
FURNITURE_GROUNDED_LAYOUT_BASE_URL="$GROUNDING_BASE_URL" \
PIPELINE_STOP_STAGE="$PIPELINE_STOP_STAGE" \
FURNITURE_DESIGNER_THINKING="$FURNITURE_DESIGNER_THINKING" \
FURNITURE_CRITIC_THINKING="$FURNITURE_CRITIC_THINKING" \
FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS="${FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS:-false}" \
bash "$PARALLEL_LAUNCHER"

echo "[OK] OpenRouter dense bedroom ACP batch completed"
echo "[OK] output: $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
