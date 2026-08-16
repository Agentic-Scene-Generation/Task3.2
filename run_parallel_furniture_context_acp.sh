#!/usr/bin/env bash
# Batch ACP test for furniture context-image generation with two GPUs.
#
# GPU allocation:
#   GPU 0: Qwen llama-server, embedding service, SceneSmith rendering/collision
#   GPU 1: one persistent Qwen-Image-Edit sidecar
#
# Default usage inside a two-GPU ACP task:
#   cd /mnt/afs/visitor33/Task3.2
#   bash run_parallel_furniture_context_acp.sh
#
# Useful overrides:
#   MAX_CASES=4 bash run_parallel_furniture_context_acp.sh
#   SCENE_CONCURRENCY=1 MAX_CASES=2 bash run_parallel_furniture_context_acp.sh
#   PIPELINE_STOP_STAGE=manipuland bash run_parallel_furniture_context_acp.sh

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARALLEL_LAUNCHER="$PROJECT_ROOT/run_parallel_rooms_no_shared_base.sh"
QWEN_IMAGE_LAUNCHER="$PROJECT_ROOT/scripts/start_qwen_image_edit_server.sh"

SCENESMITH_GPU="${SCENESMITH_GPU:-0}"
QWEN_IMAGE_EDIT_GPU="${QWEN_IMAGE_EDIT_GPU:-1}"
SCENE_CONCURRENCY="${SCENE_CONCURRENCY:-2}"
PIPELINE_STOP_STAGE="${PIPELINE_STOP_STAGE:-furniture}"
FURNITURE_DESIGNER_THINKING="${FURNITURE_DESIGNER_THINKING:-high}"
FURNITURE_CRITIC_THINKING="${FURNITURE_CRITIC_THINKING:-low}"
STOP_QWEN_IMAGE_EDIT_ON_EXIT="${STOP_QWEN_IMAGE_EDIT_ON_EXIT:-true}"
RUN_ID="${RUN_ID:-furniture_context_dense_$(date +%Y%m%d_%H%M%S)}"
CASES_DIR="${CASES_DIR:-$PROJECT_ROOT/tmp/acp_furniture_context/$RUN_ID}"
CASES_FILE="$CASES_DIR/dense_furniture_cases.tsv"

DENSE_CASES=(
    $'grand_living_room\tDense living-room zoning, conversation groups, media sightlines, and circulation\tDesign a spacious 9 m by 8 m grand contemporary living room with two three-seat sofas facing each other, four armchairs arranged as two conversational pairs, one large central coffee table, two smaller coffee tables, four side tables, two ottomans, a long media console facing the main sofa, two tall bookcases, one low display cabinet, a console table behind a sofa, two floor lamps, and four large indoor plants. Keep clear walking routes from every doorway, orient all seating toward useful conversation or media focal points, and distribute the furniture into a main conversation zone, a reading corner, and a media zone.'
    $'formal_dining_gallery\tHigh-capacity dining layout, chair access, storage access, and serving circulation\tCreate a spacious 10 m by 7 m formal dining room centered on a long table for twelve diners with twelve individual dining chairs. Add two host armchairs at the table ends, a sideboard on one long wall, two glass-front china cabinets on the opposite wall, a bar cart, a narrow serving console, two upholstered benches in a waiting corner, two large plants, and a small round auxiliary table with two chairs. Preserve generous pull-out clearance behind every chair and an unobstructed serving loop around the table.'
    $'executive_library_office\tMultiple work, meeting, reading, and storage zones with dense furniture\tBuild a large 10 m by 8 m executive library office. Include a substantial executive desk with an office chair, four visitor armchairs facing the desk, a conference table with eight conference chairs, two sofas facing a coffee table in a separate lounge zone, two side tables, six tall bookcases distributed along solid walls, three filing cabinets, one credenza behind the desk, one printer cabinet, two floor lamps, and two plants. Keep the desk, meeting, library, and lounge zones functionally distinct with clear paths between them.'
    $'luxury_bedroom_suite\tDense bedroom composition with sleeping, dressing, storage, and relaxation zones\tDesign a generous 9 m by 8 m luxury bedroom suite with a king bed centered on a solid wall, two nightstands, a wide upholstered bench at the foot of the bed, a large rug under the bed group, two dressers, two wardrobes, a vanity table with a vanity chair, a chaise lounge, two armchairs around a small round table, a low media console facing the bed, one bookcase, one storage chest, two floor lamps, and three plants. Maintain access on both sides of the bed and in front of every wardrobe and dresser.'
    $'hotel_lobby_lounge\tDense hospitality seating clusters, reception, waiting, and circulation\tCreate a spacious 12 m by 9 m boutique hotel lobby with a long reception desk and two staff chairs, three sofas, eight lounge armchairs arranged into three conversation clusters, three large coffee tables, six side tables, four ottomans, two console tables, two display cabinets, four floor lamps, six large plants, two luggage benches, and a writing desk with two chairs. Keep a clear route from the entrance to reception and from reception to all lounge zones.'
    $'active_classroom\tMaximum useful classroom furniture with explicit quantities and aisle requirements\tDesign a large 11 m by 9 m active-learning classroom containing twelve student desks with one chair at each desk, a teacher desk with a teacher chair, two demonstration tables, four mobile storage cabinets, three tall bookcases, two low supply cabinets, one reading sofa, four lounge chairs around two low tables, and two coat-storage units. Arrange the twelve desk-chair pairs in three groups of four, leave wide aisles between groups, and preserve a clear teaching zone at the front.'
    $'restaurant_dining_room\tHigh-density restaurant seating with service stations and safe circulation\tCreate a spacious 12 m by 10 m restaurant dining room with eight four-person dining tables and thirty-two dining chairs, two banquette sofas with four small tables and eight additional chairs, a host stand near the entrance, three service sideboards, two dish cabinets, two bar carts, four waiting armchairs around two small tables, and six plants used as soft zone dividers. Preserve continuous waiter circulation and chair pull-out clearance without blocking the entrance.'
    $'creative_studio\tFurniture-rich maker studio with work, storage, review, and lounge zones\tDesign a 11 m by 9 m creative maker studio with six large worktables, eighteen stools, two standing-height project tables, eight storage cabinets, four shelving units, three rolling drawer cabinets, one materials sorting table, two desks with two office chairs, a review table with six chairs, one sofa, four lounge chairs, two coffee tables, and four plants. Keep tool-storage fronts accessible and create broad routes for moving projects between all work zones.'
    $'family_media_game_room\tDense recreation room with media, board-game, reading, and social zones\tCreate a large 10 m by 9 m family media and game room with one large sectional sofa, two additional sofas, six armchairs, four ottomans, two coffee tables, four side tables, a long media console, two game tables with six chairs each, two tall bookcases, four game-storage cabinets, one snack console, two floor lamps, and four plants. Orient the main seating toward the media console while keeping the two game-table zones independently accessible.'
    $'conference_training_center\tLarge mixed conference and workshop furniture program\tDesign a 12 m by 9 m conference and training room with a central boardroom table and fourteen executive chairs, four movable training tables with three chairs each, two facilitator desks with two office chairs, four credenzas, six storage cabinets, two bookcases, two sofas, four lounge armchairs, two coffee tables, four side tables, and four plants. Maintain clear presentation sightlines, chair pull-out space, accessible storage fronts, and separate boardroom, workshop, and break-out lounge zones.'
)

if [ -z "${MAX_CASES:-}" ]; then
    MAX_CASES="${#DENSE_CASES[@]}"
fi

QWEN_STARTED_BY_THIS_SCRIPT=false
cleanup_started=false

normalize_bool() {
    case "${1,,}" in
        1|true|yes|y|on) printf 'true' ;;
        0|false|no|n|off|'') printf 'false' ;;
        *) return 1 ;;
    esac
}

cleanup() {
    local exit_code=$?
    if [ "$cleanup_started" = "true" ]; then
        return
    fi
    cleanup_started=true
    trap - EXIT INT TERM HUP

    if [ "$QWEN_STARTED_BY_THIS_SCRIPT" = "true" ] \
        && [ "$STOP_QWEN_IMAGE_EDIT_ON_EXIT" = "true" ]; then
        echo "[CLEANUP] stopping Qwen-Image-Edit started by this ACP job"
        bash "$QWEN_IMAGE_LAUNCHER" --stop || true
    fi
    exit "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

if ! STOP_QWEN_IMAGE_EDIT_ON_EXIT="$(normalize_bool "$STOP_QWEN_IMAGE_EDIT_ON_EXIT")"; then
    echo "[ERROR] STOP_QWEN_IMAGE_EDIT_ON_EXIT must be true or false" >&2
    exit 2
fi
if ! [[ "$SCENE_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] SCENE_CONCURRENCY must be a positive integer" >&2
    exit 2
fi
if ! [[ "$MAX_CASES" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] MAX_CASES must be a positive integer" >&2
    exit 2
fi
if [ "$MAX_CASES" -gt "${#DENSE_CASES[@]}" ]; then
    echo "[ERROR] MAX_CASES=$MAX_CASES exceeds the ${#DENSE_CASES[@]} built-in cases" >&2
    exit 2
fi
case "$PIPELINE_STOP_STAGE" in
    furniture|wall_mounted|ceiling_mounted|manipuland) ;;
    *)
        echo "[ERROR] invalid PIPELINE_STOP_STAGE: $PIPELINE_STOP_STAGE" >&2
        exit 2
        ;;
esac
if [ ! -f "$PARALLEL_LAUNCHER" ]; then
    echo "[ERROR] parallel launcher not found: $PARALLEL_LAUNCHER" >&2
    exit 1
fi
if [ ! -f "$QWEN_IMAGE_LAUNCHER" ]; then
    echo "[ERROR] Qwen-Image-Edit launcher not found: $QWEN_IMAGE_LAUNCHER" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi is unavailable; run inside a two-GPU ACP task" >&2
    exit 1
fi

VISIBLE_GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d '[:space:]')"
if [ "$VISIBLE_GPU_COUNT" -lt 2 ]; then
    echo "[ERROR] this test requires at least two visible GPUs; found $VISIBLE_GPU_COUNT" >&2
    exit 1
fi

mkdir -p "$CASES_DIR"
: > "$CASES_FILE"
for case_row in "${DENSE_CASES[@]}"; do
    printf '%s\n' "$case_row" >> "$CASES_FILE"
done

echo "========== ACP DENSE FURNITURE CONTEXT TEST =========="
echo "run id:                 $RUN_ID"
echo "visible GPUs:            $VISIBLE_GPU_COUNT"
echo "SceneSmith/LLM GPU:      $SCENESMITH_GPU"
echo "Qwen-Image-Edit GPU:     $QWEN_IMAGE_EDIT_GPU"
echo "scene concurrency:       $SCENE_CONCURRENCY"
echo "selected cases:          $MAX_CASES / ${#DENSE_CASES[@]}"
echo "pipeline stop stage:     $PIPELINE_STOP_STAGE"
echo "furniture thinking:      $FURNITURE_DESIGNER_THINKING/$FURNITURE_CRITIC_THINKING"
echo "context image backend:   qwen_local"
echo "cases file:              $CASES_FILE"
echo "output:                  $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
echo "======================================================"
if [ "$SCENE_CONCURRENCY" -gt 2 ]; then
    echo "[WARNING] Qwen-Image-Edit is intentionally serial; concurrency > 2 may"
    echo "          exceed the current 120-second per-request client timeout."
fi

if curl -fsS --max-time 2 "http://127.0.0.1:18020/ready" \
    | grep -Fq '"ready":true'; then
    echo "[QWEN IMAGE] reusing ready service on port 18020"
else
    echo "[QWEN IMAGE] starting one persistent worker on GPU $QWEN_IMAGE_EDIT_GPU"
    QWEN_IMAGE_EDIT_CUDA_VISIBLE_DEVICES="$QWEN_IMAGE_EDIT_GPU" \
        QWEN_IMAGE_EDIT_STARTUP_TIMEOUT_SECONDS="${QWEN_IMAGE_EDIT_STARTUP_TIMEOUT_SECONDS:-900}" \
        bash "$QWEN_IMAGE_LAUNCHER" --background
    QWEN_STARTED_BY_THIS_SCRIPT=true
fi

echo "[SCENES] starting dense furniture batch"
CUDA_VISIBLE_DEVICES="$SCENESMITH_GPU" \
LLAMA_CUDA_VISIBLE_DEVICES="$SCENESMITH_GPU" \
SCENE_CONCURRENCY="$SCENE_CONCURRENCY" \
MAX_CASES="$MAX_CASES" \
RUN_ID="$RUN_ID" \
CRITIC_PROBE_CASES_FILE="$CASES_FILE" \
FURNITURE_CONTEXT_IMAGE_GENERATION_ENABLED=true \
FURNITURE_CONTEXT_IMAGE_GENERATION_BACKEND=qwen_local \
FURNITURE_GROUNDED_LAYOUT_ENABLED="${FURNITURE_GROUNDED_LAYOUT_ENABLED:-true}" \
FURNITURE_GROUNDED_LAYOUT_BASE_URL="${FURNITURE_GROUNDED_LAYOUT_BASE_URL:-http://127.0.0.1:18030}" \
QWEN_IMAGE_EDIT_BASE_URL="${QWEN_IMAGE_EDIT_BASE_URL:-http://127.0.0.1:18020/v1}" \
PIPELINE_STOP_STAGE="$PIPELINE_STOP_STAGE" \
FURNITURE_DESIGNER_THINKING="$FURNITURE_DESIGNER_THINKING" \
FURNITURE_CRITIC_THINKING="$FURNITURE_CRITIC_THINKING" \
FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS="${FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS:-false}" \
bash "$PARALLEL_LAUNCHER"

echo "[OK] dense furniture ACP batch completed"
echo "[OK] output: $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
