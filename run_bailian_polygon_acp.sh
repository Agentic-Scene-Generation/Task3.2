#!/usr/bin/env bash
# Batch ACP test for Bailian furniture context images in single-room polygon mode.
#
# GPU allocation (single H100 80GB):
#   GPU 0: Qwen llama-server, embedding service, SceneSmith rendering/collision, GroundingDINO
#
# Default usage inside a single-GPU ACP task:
#   cd /mnt/afs/visitor33/Task3.2
#   bash run_bailian_polygon_acp.sh
#
# Useful overrides:
#   MAX_CASES=5 bash run_bailian_polygon_acp.sh
#   PIPELINE_STOP_STAGE=furniture bash run_bailian_polygon_acp.sh

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
RUN_ID="${RUN_ID:-bailian_polygon_acp_$(date +%Y%m%d_%H%M%S)}"
CASES_DIR="${CASES_DIR:-$PROJECT_ROOT/tmp/acp_bailian_polygon/$RUN_ID}"
CASES_FILE="$CASES_DIR/polygon_furniture_cases.tsv"
GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-18030}"
GROUNDING_BASE_URL="http://127.0.0.1:${GROUNDING_DINO_PORT}"
BAILIAN_IMAGE_MODEL="${BAILIAN_IMAGE_MODEL:-wan2.7-image-pro}"
BAILIAN_CREDENTIALS_FILE="${BAILIAN_CREDENTIALS_FILE:-/mnt/afs/visitor33/bailian.json}"

# Each prompt specifies one simple polygon in counter-clockwise boundary order.
# The explicit outside regions make it easier to audit whether floor-plan,
# image-edit, and 3D placement stages incorrectly fill a concavity or use the
# polygon's axis-aligned bounding rectangle.
POLYGON_CASES=(
    $'polygon_l_living_room\tL-shaped living room: preserve the concave notch and connect two dense activity zones\tCreate exactly one irregular L-shaped living room using these ordered floor-boundary vertices in meters: [[0,0],[10,0],[10,4],[6,4],[6,9],[0,9]]. Use this exact simple polygon rather than its bounding rectangle. The rectangular notch x=6..10, y=4..9 is outside the room and must contain no floor, wall-spanning shortcut, furniture, or furniture footprint. Add one exterior door and suitable windows only on polygon boundary edges. Furnish the lower wing with two three-seat sofas, four armchairs, one large coffee table, four side tables, two ottomans, and a media console; furnish the upper-left wing with two bookcases, one reading table, four reading chairs, two floor lamps, and three plants. Keep a clear route through the inside concave corner and from the door to both wings. Every complete furniture footprint must lie inside the exact polygon.'
    $'polygon_u_library_lounge\tU-shaped library lounge: keep the deep central courtyard notch empty\tCreate exactly one U-shaped library lounge using these ordered floor-boundary vertices in meters: [[0,0],[12,0],[12,9],[8,9],[8,4],[4,4],[4,9],[0,9]]. Use the exact polygon, not the 12 m by 9 m bounding rectangle. The central notch x=4..8, y=4..9 is outside the room and must remain completely empty. Add one main door on the lower boundary and windows on valid exterior polygon edges. Place a reception desk with one task chair, two sofas, six lounge armchairs, three coffee tables, six side tables, eight bookcases, two reading tables with four chairs each, four floor lamps, and six plants. Use the left and right arms as distinct quiet reading zones and the lower connector as reception and shared lounge space. Preserve continuous circulation around both inner corners, and keep every furniture footprint strictly inside the U-shaped floor.'
    $'polygon_t_executive_center\tT-shaped executive center: dense conference, office, and lounge zones across narrow junctions\tCreate exactly one T-shaped executive office and meeting room using these ordered floor-boundary vertices in meters: [[0,0],[12,0],[12,4],[8,4],[8,10],[4,10],[4,4],[0,4]]. Do not replace it with a rectangle. The upper-left region x=0..4, y=4..10 and upper-right region x=8..12, y=4..10 are outside the room. Add doors and windows only on the true polygon boundary. In the wide lower bar place a conference table with twelve chairs, two credenzas, and two storage cabinets. In the vertical stem place an executive desk and chair, four visitor chairs, two bookcases, one sofa, two lounge chairs, one coffee table, two side tables, two floor lamps, and three plants. Keep the T junction open as the only circulation connection and ensure no object or footprint crosses either outside region.'
    $'polygon_stepped_studio\tStepped concave studio: maintain usable circulation across three offset zones\tCreate exactly one stepped polygon creative studio using these ordered floor-boundary vertices in meters: [[0,0],[10,0],[10,4],[8,4],[8,7],[5,7],[5,10],[0,10]]. Preserve every step and concave corner; do not use the polygon bounding rectangle. The regions outside the ordered boundary, including x=8..10 above y=4 and x=5..8 above y=7, must remain empty. Add one entrance and several windows on actual boundary edges. Arrange four large worktables with twelve stools in the broad lower area, a review table with six chairs and two storage cabinets in the middle step, and two desks with office chairs, four shelving units, four drawer cabinets, one sofa, two lounge chairs, one coffee table, and three plants in the upper-left area. Maintain an unobstructed route through both step transitions and keep all complete furniture footprints inside the polygon.'
    $'polygon_double_notch_dining\tDouble-notch dining gallery: high-capacity seating without bridging either exterior recess\tCreate exactly one concave dining gallery using these ordered floor-boundary vertices in meters: [[0,0],[12,0],[12,4],[9,4],[9,7],[12,7],[12,11],[0,11],[0,7],[3,7],[3,4],[0,4]]. Use this exact polygon instead of a 12 m by 11 m rectangle. The right recess x=9..12, y=4..7 and left recess x=0..3, y=4..7 are outside the room and must remain empty. Place openings only on true boundary edges. Furnish the lower zone with a long dining table and twelve chairs plus two host armchairs; use the central connector for two serving consoles and two bar carts while keeping it passable; furnish the upper zone with four round four-person tables and sixteen chairs, two sideboards, two display cabinets, two benches, and four plants. Preserve chair pull-out clearance, waiter circulation through the narrow middle, and keep every furniture footprint within the exact polygon.'
    $'polygon_cross_maker_hub\tCross-shaped maker hub: test four wings, concave corners, and a shared circulation core\tCreate exactly one cross-shaped maker hub using these ordered floor-boundary vertices in meters: [[3,0],[9,0],[9,3],[12,3],[12,8],[9,8],[9,11],[3,11],[3,8],[0,8],[0,3],[3,3]]. Preserve the exact cross footprint and all eight concave/convex transitions; do not fill the four missing corner rectangles of the 12 m by 11 m bounding box. Add a main door and windows only on actual polygon boundary edges. Put two project tables with eight stools in the center, two worktables with six stools and two tool cabinets in the south wing, a review table with six chairs and two display cabinets in the north wing, three shelving units and three storage cabinets in the west wing, and two desks with office chairs, one sofa, four lounge chairs, two coffee tables, three drawer cabinets, and four plants in the east wing. Keep the central crossing and all four wing approaches clear, with every complete furniture footprint inside the polygon.'
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
if [ "$MAX_CASES" -gt "${#POLYGON_CASES[@]}" ]; then
    echo "[ERROR] MAX_CASES=$MAX_CASES exceeds the ${#POLYGON_CASES[@]} built-in cases" >&2
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
for case_row in "${POLYGON_CASES[@]:0:$MAX_CASES}"; do
    printf '%s\n' "$case_row" >> "$CASES_FILE"
done

echo "========== ACP BAILIAN POLYGON FURNITURE CONTEXT TEST =========="
echo "run id:                     $RUN_ID"
echo "visible GPUs:                $VISIBLE_GPU_COUNT"
echo "SceneSmith/LLM GPU:          $SCENESMITH_GPU"
echo "GroundingDINO GPU:           $GROUNDING_DINO_GPU"
echo "scene concurrency:           $SCENE_CONCURRENCY"
echo "selected polygon cases:      $MAX_CASES / ${#POLYGON_CASES[@]}"
echo "pipeline stop stage:         $PIPELINE_STOP_STAGE"
echo "floor plan mode:             polygon"
echo "placement-order reference:   disabled"
echo "context image backend:       bailian"
echo "bailian model:               ${BAILIAN_IMAGE_MODEL:-wan2.7-image-pro}"
echo "grounded layout:             ${FURNITURE_GROUNDED_LAYOUT_ENABLED:-true}"
echo "cases file:                  $CASES_FILE"
echo "output:                      $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
echo "================================================================="
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
echo "[SCENES] starting polygon furniture batch with Bailian image generation"

CUDA_VISIBLE_DEVICES="$SCENESMITH_GPU" \
LLAMA_CUDA_VISIBLE_DEVICES="$SCENESMITH_GPU" \
SCENE_CONCURRENCY="$SCENE_CONCURRENCY" \
MAX_CASES="$MAX_CASES" \
RUN_ID="$RUN_ID" \
CRITIC_PROBE_CASES_FILE="$CASES_FILE" \
FLOOR_PLAN_MODE=polygon \
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

echo "[OK] Bailian polygon ACP batch completed"
echo "[OK] output: $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
