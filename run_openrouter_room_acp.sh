#!/usr/bin/env bash
# Batch ACP test for OpenRouter furniture context images in rectangular room mode.
#
# GPU allocation (single H100 80GB):
#   GPU 0: Qwen llama-server, embedding service, SceneSmith rendering/collision, GroundingDINO
#
# Default usage inside a single-GPU ACP task:
#   cd /mnt/afs/visitor33/Task3.2
#   bash run_openrouter_room_acp.sh
#
# Useful overrides:
#   MAX_CASES=5 bash run_openrouter_room_acp.sh
#   PIPELINE_STOP_STAGE=furniture bash run_openrouter_room_acp.sh
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
RUN_ID="${RUN_ID:-openrouter_room_acp_$(date +%Y%m%d_%H%M%S)}"
CASES_DIR="${CASES_DIR:-$PROJECT_ROOT/tmp/acp_openrouter_room/$RUN_ID}"
CASES_FILE="$CASES_DIR/room_furniture_cases.tsv"
GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-18030}"
GROUNDING_BASE_URL="http://127.0.0.1:${GROUNDING_DINO_PORT}"
OPENROUTER_IMAGE_MODEL="${OPENROUTER_IMAGE_MODEL:-openai/gpt-image-2}"
OPENROUTER_KEY_FILE="${OPENROUTER_KEY_FILE:-/mnt/afs/visitor33/exportkey.sh}"

# Bedroom cases from prompts_v1.csv (scene_index 0-9)
BEDROOM_CASES=(
    $'scandinavian_bedroom_0\tSerene Scandinavian bedroom with low bed and minimal furniture\tA serene Scandinavian bedroom centers around a low double bed dressed in crisp white linens, flanked by two sleek wooden nightstands. A plush gray rug anchors the space beneath the bed, while a simple armchair sits by the window, inviting quiet relaxation.'
    $'eclectic_bedroom_1\tEclectic bedroom with vintage and colorful elements\tAn eclectic bedroom features a vintage wooden bed against the wall, flanked by a mismatched nightstand and a colorful rug in the center. A plush velvet armchair sits by the window, creating a cozy reading nook amidst the diverse decor.'
    $'contemporary_bedroom_2\tContemporary bedroom with wardrobe and vanity\tThis contemporary bedroom features a sleek wardrobe against the wall and a stylish vanity by the window. A plush rug in the center anchors the space, creating a cozy and modern retreat.'
    $'traditional_bedroom_3\tCozy traditional bedroom with wooden bed and armchair\tA cozy traditional bedroom features a sturdy wooden bed pushed against the textured wall, flanked by two matching nightstands. A plush area rug anchors the space, while a vintage armchair sits invitingly in the corner by the window.'
    $'art_deco_bedroom_4\tLuxurious art deco bedroom with geometric elements\tA luxurious art deco bedroom centers on a plush double bed with a geometric headboard, flanked by sleek nightstands. A polished vanity with a large mirror sits against the far wall, while an elegant armchair rests by the window, bathed in warm light.'
    $'coastal_bedroom_5\tSerene coastal bedroom with white bed and wicker furniture\tA serene coastal bedroom features a white double bed centered against the soft blue wall, flanked by wooden nightstands. A woven rug anchors the space, while a wicker armchair sits by the window, inviting relaxation with ocean views.'
    $'midcentury_bedroom_6\tMid-century modern bedroom with low-profile bed\tThis mid-century modern bedroom features a low-profile bed centered against the back wall, flanked by sleek nightstands. A walnut dresser stands in the corner, while a plush rug anchors the space near a window seat.'
    $'eclectic_bedroom_7\tEclectic bedroom with bookshelf and vintage bed\tAn eclectic bedroom features a vintage wooden bed centered in the room, flanked by a tall bookshelf against the left wall and a patterned rug beneath it, creating a cozy, curated retreat.'
    $'bohemian_bedroom_8\tBohemian bedroom with plants and eclectic textiles\tA bohemian bedroom features a cozy bed against the wall, surrounded by lush plants and eclectic textiles. An armchair sits by the window, inviting relaxation, while a wooden nightstand holds a stack of books and a small lamp.'
    $'coastal_bedroom_9\tCoastal bedroom with double bed and woven rug\tThis coastal bedroom features a white double bed centered against the wall, flanked by two wooden nightstands. A woven rug lies in the center of the room, adding texture beneath the feet while light filters through sheer curtains by the window.'
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
if [ "$MAX_CASES" -gt "${#BEDROOM_CASES[@]}" ]; then
    echo "[ERROR] MAX_CASES=$MAX_CASES exceeds the ${#BEDROOM_CASES[@]} built-in cases" >&2
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
for case_row in "${BEDROOM_CASES[@]:0:$MAX_CASES}"; do
    printf '%s\n' "$case_row" >> "$CASES_FILE"
done

echo "========== ACP OPENROUTER ROOM FURNITURE CONTEXT TEST =========="
echo "run id:                     $RUN_ID"
echo "visible GPUs:                $VISIBLE_GPU_COUNT"
echo "SceneSmith/LLM GPU:          $SCENESMITH_GPU"
echo "GroundingDINO GPU:           $GROUNDING_DINO_GPU"
echo "scene concurrency:           $SCENE_CONCURRENCY"
echo "selected room cases:         $MAX_CASES / ${#BEDROOM_CASES[@]}"
echo "pipeline stop stage:         $PIPELINE_STOP_STAGE"
echo "floor plan mode:             room (rectangular)"
echo "placement-order reference:   disabled"
echo "context image backend:       openrouter"
echo "openrouter model:             ${OPENROUTER_IMAGE_MODEL:-openai/gpt-image-2}"
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

echo "[OPENROUTER] Using OpenRouter API backend, skipping Qwen-Image-Edit service"
echo "[SCENES] starting room furniture batch with OpenRouter image generation"

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

echo "[OK] OpenRouter room ACP batch completed"
echo "[OK] output: $PROJECT_ROOT/outputs/critic_probe/$RUN_ID"
