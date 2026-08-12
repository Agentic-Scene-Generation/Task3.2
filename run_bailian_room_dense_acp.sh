#!/usr/bin/env bash
# Batch ACP test for Bailian furniture context images in dense bedroom mode.
#
# GPU allocation (single H100 80GB):
#   GPU 0: Qwen llama-server, embedding service, SceneSmith rendering/collision, GroundingDINO
#
# Default usage inside a single-GPU ACP task:
#   cd /mnt/afs/visitor33/Task3.2
#   bash run_bailian_room_dense_acp.sh
#
# Useful overrides:
#   MAX_CASES=5 bash run_bailian_room_dense_acp.sh
#   PIPELINE_STOP_STAGE=furniture bash run_bailian_room_dense_acp.sh
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARALLEL_LAUNCHER="$PROJECT_ROOT/run_parallel_rooms_no_shared_base.sh"
GROUNDING_LAUNCHER="$PROJECT_ROOT/scripts/start_grounding_dino_server.sh"

SCENESMITH_GPU="${SCENESMITH_GPU:-0}"
GROUNDING_DINO_GPU="${GROUNDING_DINO_GPU:-0}"
SCENE_CONCURRENCY="${SCENE_CONCURRENCY:-2}"
PIPELINE_STOP_STAGE="${PIPELINE_STOP_STAGE:-manipuland}"
FURNITURE_DESIGNER_THINKING="${FURNITURE_DESIGNER_THINKING:-high}"
FURNITURE_CRITIC_THINKING="${FURNITURE_CRITIC_THINKING:-low}"
STOP_GROUNDING_DINO_ON_EXIT="${STOP_GROUNDING_DINO_ON_EXIT:-true}"
RUN_ID="${RUN_ID:-bailian_room_dense_acp_$(date +%Y%m%d_%H%M%S)}"
CASES_DIR="${CASES_DIR:-$PROJECT_ROOT/tmp/acp_bailian_room_dense/$RUN_ID}"
CASES_FILE="$CASES_DIR/dense_bedroom_cases.tsv"
GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-18030}"
GROUNDING_BASE_URL="http://127.0.0.1:${GROUNDING_DINO_PORT}"
BAILIAN_IMAGE_MODEL="${BAILIAN_IMAGE_MODEL:-wan2.7-image-pro}"
BAILIAN_CREDENTIALS_FILE="${BAILIAN_CREDENTIALS_FILE:-/mnt/afs/visitor33/bailian.json}"

# Dense bedroom cases - expanded from prompts_v1.csv with more furniture and complexity
DENSE_BEDROOM_CASES=(
    $'dense_scandinavian_bedroom_0\tDense Scandinavian bedroom with multiple storage and seating zones\tA densely furnished Scandinavian bedroom centers around a low double bed dressed in crisp white linens, flanked by two sleek wooden nightstands with table lamps. A plush gray rug anchors the bed area, while two matching armchairs face each other near the window for conversation. Add a tall wardrobe against one wall, a long dresser with six drawers on another wall, a vanity table with a cushioned stool, a storage bench at the foot of the bed, two floor lamps in opposite corners, three potted plants on various surfaces, and a small bookshelf filled with books.'
    $'dense_eclectic_bedroom_1\tEclectic bedroom packed with vintage furniture and colorful accents\tA furniture-rich eclectic bedroom features a vintage wooden bed against the wall, flanked by two mismatched nightstands each with unique table lamps, and a large colorful patterned rug covering most of the floor. A plush velvet armchair sits by the window with a reading lamp beside it, while a second upholstered chair occupies another corner. Include a tall vintage wardrobe, a wooden dresser with decorative mirror, a vanity table with ornate stool, an ottoman at the foot of the bed, two tall bookcases filled with books and collectibles, three floor lamps, and at least five potted plants distributed throughout the room.'
    $'dense_contemporary_bedroom_2\tContemporary bedroom with extensive storage and multiple functional zones\tThis densely furnished contemporary bedroom features a sleek king-size bed centered against the wall with matching nightstands on both sides. A large plush rug covers the floor beneath the bed extending toward a seating area with two modern armchairs and a small side table. Include a massive wardrobe system spanning one wall, a stylish six-drawer dresser, a full vanity station with large mirror and cushioned bench, a media console with TV, a tall bookcase, a storage ottoman, a chaise lounge in the corner, two floor lamps, two table lamps, and four decorative plants.'
    $'dense_traditional_bedroom_3\tTraditional bedroom with maximum furniture and classic elegance\tA heavily furnished traditional bedroom features a substantial wooden bed with ornate headboard pushed against the textured wall, flanked by two large matching nightstands with drawers. A plush area rug anchors the sleeping zone, while two vintage armchairs create a sitting area near the window with a small tea table between them. Add a tall wooden wardrobe, a matching dresser with eight drawers and mirror, a vanity table with upholstered chair, a blanket chest at the foot of the bed, two tall bookcases, a writing desk with office chair, three floor lamps, two table lamps, and at least four potted plants.'
    $'dense_art_deco_bedroom_4\tLuxurious art deco bedroom with geometric elements and abundant furniture\tA lavishly furnished art deco bedroom centers on a plush double bed with dramatic geometric headboard, flanked by two sleek mirrored nightstands with art deco lamps. A polished vanity with large tri-fold mirror and velvet stool sits against the far wall, while two elegant armchairs with a small round table occupy the window area. Include a tall wardrobe with geometric inlays, a six-drawer chest, a chaise lounge with metallic legs, an upholstered bench at the foot of the bed, two standing floor lamps with brass details, a media console, a tall display cabinet, and three decorative plants in art deco planters.'
    $'dense_coastal_bedroom_5\tCoastal bedroom packed with beach-inspired furniture and nautical accents\tA furniture-dense coastal bedroom features a white double bed with weathered wood headboard centered against a soft blue wall, flanked by two wooden nightstands with nautical lamps. A large woven rug anchors the space, with two wicker armchairs and a small table creating a seating nook by the window. Add a tall white wardrobe, a six-drawer dresser in light wood, a vanity table with rope-accented mirror and cushioned stool, a storage bench with striped cushion at the bed foot, a tall bookcase filled with books and shells, two rattan side tables, three floor lamps, and at least six potted coastal plants and palms.'
    $'dense_midcentury_bedroom_6\tMid-century modern bedroom with extensive walnut furniture collection\tThis densely arranged mid-century modern bedroom features a low-profile king bed with teak frame centered against the back wall, flanked by two sleek walnut nightstands with atomic-era lamps. A walnut dresser with twelve drawers spans one wall, while a matching credenza serves as media console. Add two molded plastic armchairs with a kidney-shaped side table near the window, a tall wardrobe, a vanity desk with modern chair, a storage bench with vinyl cushion, a room divider bookshelf, two tripod floor lamps, a large area rug, and four potted plants in ceramic planters.'
    $'dense_eclectic_bedroom_7\tEclectic bedroom overflowing with vintage books and curated furniture\tA maximalist eclectic bedroom features a vintage wooden bed centered in the room with ornate bedding, flanked by two tall bookcases against the left and right walls packed with books and curiosities. A large patterned rug covers the floor, with two mismatched armchairs and a vintage trunk serving as coffee table. Add a tall wardrobe with distressed finish, a dresser with eight drawers, a writing desk with antique chair piled with books, a vanity with decorative mirror, an ottoman, two standing bookshelves at different heights, three floor lamps in various styles, and at least five potted plants.'
    $'dense_bohemian_bedroom_8\tBohemian bedroom bursting with plants textiles and layered furniture\tA heavily decorated bohemian bedroom features a low wooden bed against the wall draped in multiple layered textiles and pillows, with two carved wooden nightstands. Two comfortable armchairs with colorful throws face each other near the window for lounging, surrounded by at least eight large potted plants including hanging varieties. Add a tall rattan wardrobe, a painted dresser with six drawers, a vanity table with macramé mirror and cushioned stool, a storage bench with woven baskets underneath, two tall plant stands with multiple plants, three floor lamps with fabric shades, multiple layered rugs, and a bookshelf filled with books and artifacts.'
    $'dense_coastal_bedroom_9\tCoastal bedroom with comprehensive white furniture and beach textures\tThis furniture-rich coastal bedroom features a white double bed with upholstered headboard centered against the wall, flanked by two substantial wooden nightstands with ceramic lamps. A large woven rug lies in the center with two additional smaller rugs layered on top, while two cushioned armchairs and a side table create a reading area by the window. Include a tall white wardrobe, a six-drawer dresser, a vanity table with ship-lap mirror and cushioned bench, a storage trunk at the bed foot, a tall bookcase with beach-themed books and decor, a media console, two floor lamps, two table lamps, and at least five coastal plants in white ceramic pots.'
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

load_bailian_api_key() {
    # An explicitly supplied environment variable takes precedence; otherwise
    # read the credential once from the local JSON file without printing it.
    if [ -n "${BAILIAN_API_KEY:-}" ]; then
        return
    fi
    if [ ! -r "$BAILIAN_CREDENTIALS_FILE" ]; then
        echo "[ERROR] Bailian credentials file is not readable: $BAILIAN_CREDENTIALS_FILE" >&2
        exit 1
    fi
    if ! command -v jq >/dev/null 2>&1; then
        echo "[ERROR] jq is required to read BAILIAN_CREDENTIALS_FILE" >&2
        exit 1
    fi
    if ! BAILIAN_API_KEY="$(jq -er '.OPENAI_API_KEY | select(type == "string" and length > 0)' "$BAILIAN_CREDENTIALS_FILE")"; then
        echo "[ERROR] BAILIAN_CREDENTIALS_FILE must contain a non-empty OPENAI_API_KEY" >&2
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

load_bailian_api_key
# Validate Bailian API key
if [ -z "${BAILIAN_API_KEY:-}" ]; then
    echo "[ERROR] BAILIAN_API_KEY environment variable is required" >&2
    exit 1
fi
export BAILIAN_API_KEY BAILIAN_IMAGE_MODEL

# Dummy OKCODEX_API_KEY for config parsing compatibility
export OKCODEX_API_KEY="${OKCODEX_API_KEY:-dummy}"

mkdir -p "$CASES_DIR"
: > "$CASES_FILE"
for case_row in "${DENSE_BEDROOM_CASES[@]:0:$MAX_CASES}"; do
    printf '%s\n' "$case_row" >> "$CASES_FILE"
done

echo "========== ACP BAILIAN DENSE BEDROOM TEST =========="
echo "run id:                     $RUN_ID"
echo "visible GPUs:                $VISIBLE_GPU_COUNT"
echo "SceneSmith/LLM GPU:          $SCENESMITH_GPU"
echo "GroundingDINO GPU:           $GROUNDING_DINO_GPU"
echo "scene concurrency:           $SCENE_CONCURRENCY"
echo "selected dense bedrooms:     $MAX_CASES / ${#DENSE_BEDROOM_CASES[@]}"
echo "pipeline stop stage:         $PIPELINE_STOP_STAGE"
echo "floor plan mode:             room (rectangular)"
echo "placement-order reference:   disabled"
echo "context image backend:       bailian"
echo "bailian model:               ${BAILIAN_IMAGE_MODEL:-wan2.7-image-pro}"
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

echo "[BAILIAN] Using Bailian API backend, skipping Qwen-Image-Edit service"
echo "[SCENES] starting dense bedroom batch with Bailian image generation"

CUDA_VISIBLE_DEVICES="$SCENESMITH_GPU" \
LLAMA_CUDA_VISIBLE_DEVICES="$SCENESMITH_GPU" \
SCENE_CONCURRENCY="$SCENE_CONCURRENCY" \
MAX_CASES="$MAX_CASES" \
RUN_ID="$RUN_ID" \
CRITIC_PROBE_CASES_FILE="$CASES_FILE" \
FLOOR_PLAN_MODE=room \
FURNITURE_PLACEMENT_ORDER_ENABLED=false \
FURNITURE_CONTEXT_IMAGE_GENERATION_ENABLED=true \
FURNITURE_CONTEXT_IMAGE_GENERATION_BACKEND=bailian \
FURNITURE_GROUNDED_LAYOUT_ENABLED="${FURNITURE_GROUNDED_LAYOUT_ENABLED:-true}" \
FURNITURE_GROUNDED_LAYOUT_BASE_URL="$GROUNDING_BASE_URL" \
PIPELINE_STOP_STAGE="$PIPELINE_STOP_STAGE" \
FURNITURE_DESIGNER_THINKING="$FURNITURE_DESIGNER_THINKING" \
FURNITURE_CRITIC_THINKING="$FURNITURE_CRITIC_THINKING" \
FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS="${FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS:-false}" \
bash "$PARALLEL_LAUNCHER"

echo "[OK] Bailian dense bedroom ACP batch completed"
echo "[OK] output: $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
