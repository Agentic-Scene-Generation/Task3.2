#!/usr/bin/env bash
# Run SceneExpert critic-on probes in isolated processes with non-overlapping
# service ports. This script intentionally has no critic-off, embedding, or VLM
# annotation path: SceneBenchmark feedback is injected only into existing LLM
# critic prompts.
#
# Shared-base replay:
#   GENERATE_SHARED_BASE=true ... bash scripts/run_parallel_critic_on.sh
# generates OUTPUT_ROOT/shared_base and branches the critic run from it.
# To reuse a previous base, set BRANCH_FROM_SHARED_BASE=true and point
# SHARED_BASE_ROOT at that directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

EXPERIMENT="${SCENEEXPERT_EXPERIMENT:-ablation_4c_qwen3_hybrid_memory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-${SCENEEXPERT_MODEL_ID:-Qwen3.6-27B-Q8_0}}"
RUN_ID="${RUN_ID:-critic_on_$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/critic_probe/$RUN_ID}"

SCENE_BATCH_SIZE="${SCENE_BATCH_SIZE:-1}"
SCENE_WORKERS_PER_PROCESS="${SCENE_WORKERS_PER_PROCESS:-1}"
CRITIC_PROBE_PARALLEL="${CRITIC_PROBE_PARALLEL:-true}"
CRITIC_PROBE_INNER_PARALLELISM="${CRITIC_PROBE_INNER_PARALLELISM:-2}"
CRITIC_PROBE_PORT_BASE="${CRITIC_PROBE_PORT_BASE:-9000}"
CRITIC_PROBE_PORT_BLOCK_SIZE="${CRITIC_PROBE_PORT_BLOCK_SIZE:-400}"
CRITIC_PROBE_SHUTDOWN_GRACE_SECONDS="${CRITIC_PROBE_SHUTDOWN_GRACE_SECONDS:-30}"
# Continue other batches after one batch fails; the script still exits nonzero
# after all batches finish if any batch failed. Set false for fail-fast mode.
CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE="${CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE:-true}"

PIPELINE_STOP_STAGE="${PIPELINE_STOP_STAGE:-manipuland}"
# Keep strict furniture-stage validation by default. Set this to false only
# when intentionally allowing unresolved furniture hard constraints through.
# Example: FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS=false bash scripts/run_parallel_critic_on.sh
FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS="${FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS:-true}"
BRANCH_FROM_SHARED_BASE="${BRANCH_FROM_SHARED_BASE:-false}"
SHARED_BASE_STOP_STAGE="${SHARED_BASE_STOP_STAGE:-floor_plan}"
SHARED_BASE_ROOT="${SHARED_BASE_ROOT:-}"
GENERATE_SHARED_BASE="${GENERATE_SHARED_BASE:-false}"
MAX_CASES="${MAX_CASES:-0}"
CASE_FILTER="${CASE_FILTER:-}"
DRY_RUN="${DRY_RUN:-false}"
DISABLE_ARTICULATED="${SCENEEXPERT_DISABLE_ARTICULATED:-false}"
DISABLE_MATERIALS="${SCENEEXPERT_DISABLE_MATERIALS:-false}"
DISABLE_BWRAP="${SCENEEXPERT_DISABLE_BWRAP:-false}"
HSSD_RETRIEVAL_BACKEND="${HSSD_RETRIEVAL_BACKEND:-clip}"
HSSD_RENDERED_ASSET_CHOICE="${HSSD_RENDERED_ASSET_CHOICE:-false}"
# os.cpu_count() sees the host's 192 logical CPUs in the CCI container, while
# the job is limited to roughly 22 CPU cores.  Allow the caller to cap each
# isolated convex-decomposition server without changing the stable defaults.
CONVEX_MAX_OMP_THREADS="${SCENEEXPERT_CONVEX_MAX_OMP_THREADS:-}"

# Match the classmate's vLLM run. The agent code maps these values to Qwen
# directives: none/minimal -> /no_think, all other values -> /think.
# Keep them as environment overrides so an ablation can change one stage
# without editing this script.
FLOOR_PLAN_DESIGNER_THINKING="${FLOOR_PLAN_DESIGNER_THINKING:-high}"
FLOOR_PLAN_CRITIC_THINKING="${FLOOR_PLAN_CRITIC_THINKING:-high}"
FURNITURE_DESIGNER_THINKING="${FURNITURE_DESIGNER_THINKING:-low}"
FURNITURE_CRITIC_THINKING="${FURNITURE_CRITIC_THINKING:-low}"
WALL_DESIGNER_THINKING="${WALL_DESIGNER_THINKING:-none}"
WALL_CRITIC_THINKING="${WALL_CRITIC_THINKING:-none}"
CEILING_DESIGNER_THINKING="${CEILING_DESIGNER_THINKING:-none}"
CEILING_CRITIC_THINKING="${CEILING_CRITIC_THINKING:-none}"
MANIPULAND_DESIGNER_THINKING="${MANIPULAND_DESIGNER_THINKING:-none}"
MANIPULAND_CRITIC_THINKING="${MANIPULAND_CRITIC_THINKING:-none}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-123}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8002/v1}"
export OPENAI_USE_RESPONSES="false"
export SCENEEXPERT_MODEL_ID="$MODEL_NAME"

# Match the ACP hybrid-memory job environment while keeping one worker per
# shell process for forkserver-safe parallel scene runs.
export SCENEEXPERT_MEMORY_EMBEDDING_DEVICE="cpu"
export SCENEEXPERT_MEMORY_EMBEDDING_INDEX_DEVICE="cpu"
export SCENEEXPERT_MEMORY_INDEX_AUTO_BUILD_MISSING="1"
export SCENEEXPERT_MP_START_METHOD="forkserver"

normalize_bool() {
    case "${1,,}" in
        1|true|yes|y|on) printf 'true' ;;
        0|false|no|n|off|'') printf 'false' ;;
        *) return 1 ;;
    esac
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ]; then
        echo "ERROR: $name must be a positive integer, got '$value'" >&2
        exit 1
    fi
}

next_stage_after() {
    case "$1" in
        floor_plan) printf 'furniture' ;;
        furniture) printf 'wall_mounted' ;;
        wall_mounted) printf 'ceiling_mounted' ;;
        ceiling_mounted) printf 'manipuland' ;;
        *) return 1 ;;
    esac
}

pipeline_stage_index() {
    case "$1" in
        floor_plan) printf '0' ;;
        furniture) printf '1' ;;
        wall_mounted) printf '2' ;;
        ceiling_mounted) printf '3' ;;
        manipuland) printf '4' ;;
        *) return 1 ;;
    esac
}

csv_quote() {
    local value="$1"
    value=${value//\"/\"\"}
    printf '"%s"' "$value"
}

require_positive_integer SCENE_BATCH_SIZE "$SCENE_BATCH_SIZE"
require_positive_integer SCENE_WORKERS_PER_PROCESS "$SCENE_WORKERS_PER_PROCESS"
require_positive_integer CRITIC_PROBE_INNER_PARALLELISM "$CRITIC_PROBE_INNER_PARALLELISM"
require_positive_integer CRITIC_PROBE_PORT_BASE "$CRITIC_PROBE_PORT_BASE"
require_positive_integer CRITIC_PROBE_PORT_BLOCK_SIZE "$CRITIC_PROBE_PORT_BLOCK_SIZE"
require_positive_integer CRITIC_PROBE_SHUTDOWN_GRACE_SECONDS "$CRITIC_PROBE_SHUTDOWN_GRACE_SECONDS"
if [ -n "$CONVEX_MAX_OMP_THREADS" ]; then
    require_positive_integer SCENEEXPERT_CONVEX_MAX_OMP_THREADS "$CONVEX_MAX_OMP_THREADS"
fi

if [ "$CRITIC_PROBE_PORT_BLOCK_SIZE" -lt 375 ]; then
    echo "ERROR: CRITIC_PROBE_PORT_BLOCK_SIZE must be at least 375" >&2
    exit 1
fi
if ! CRITIC_PROBE_PARALLEL="$(normalize_bool "$CRITIC_PROBE_PARALLEL")"; then
    echo "ERROR: CRITIC_PROBE_PARALLEL must be true or false" >&2
    exit 1
fi
if ! BRANCH_FROM_SHARED_BASE="$(normalize_bool "$BRANCH_FROM_SHARED_BASE")"; then
    echo "ERROR: BRANCH_FROM_SHARED_BASE must be true or false" >&2
    exit 1
fi
if ! GENERATE_SHARED_BASE="$(normalize_bool "$GENERATE_SHARED_BASE")"; then
    echo "ERROR: GENERATE_SHARED_BASE must be true or false" >&2
    exit 1
fi
if ! DRY_RUN="$(normalize_bool "$DRY_RUN")"; then
    echo "ERROR: DRY_RUN must be true or false" >&2
    exit 1
fi
if ! CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE="$(normalize_bool "$CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE")"; then
    echo "ERROR: CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE must be true or false" >&2
    exit 1
fi
if ! DISABLE_ARTICULATED="$(normalize_bool "$DISABLE_ARTICULATED")"; then
    echo "ERROR: SCENEEXPERT_DISABLE_ARTICULATED must be true or false" >&2
    exit 1
fi
if ! DISABLE_MATERIALS="$(normalize_bool "$DISABLE_MATERIALS")"; then
    echo "ERROR: SCENEEXPERT_DISABLE_MATERIALS must be true or false" >&2
    exit 1
fi
if ! DISABLE_BWRAP="$(normalize_bool "$DISABLE_BWRAP")"; then
    echo "ERROR: SCENEEXPERT_DISABLE_BWRAP must be true or false" >&2
    exit 1
fi
if [[ "$HSSD_RETRIEVAL_BACKEND" != "clip" && "$HSSD_RETRIEVAL_BACKEND" != "embedding" ]]; then
    echo "ERROR: HSSD_RETRIEVAL_BACKEND must be clip or embedding" >&2
    exit 1
fi
if ! HSSD_RENDERED_ASSET_CHOICE="$(normalize_bool "$HSSD_RENDERED_ASSET_CHOICE")"; then
    echo "ERROR: HSSD_RENDERED_ASSET_CHOICE must be true or false" >&2
    exit 1
fi
if ! FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS="$(normalize_bool "$FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS")"; then
    echo "ERROR: FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS must be true or false" >&2
    exit 1
fi

# Some containers expose /usr/bin/bwrap but forbid unprivileged namespaces.
# Keep the active Python directory available while hiding only bwrap from
# BlenderServer's capability check; the server then runs without GPU namespace
# isolation and still uses its configured port ranges.
if [ "$DISABLE_BWRAP" = "true" ]; then
    PYTHON_EXEC_DIR="$(dirname "$(readlink -f "$(command -v "$PYTHON_BIN")")")"
fi

if [ "$SCENE_WORKERS_PER_PROCESS" -ne 1 ]; then
    echo "ERROR: use one worker per process to avoid fork-after-bpy-import." >&2
    exit 1
fi
if [ "$CRITIC_PROBE_PARALLEL" = "true" ] && ! command -v setsid >/dev/null 2>&1; then
    echo "ERROR: setsid is required for isolated parallel batch cleanup" >&2
    exit 1
fi

case "$PIPELINE_STOP_STAGE" in
    furniture|wall_mounted|ceiling_mounted|manipuland) ;;
    *)
        echo "ERROR: PIPELINE_STOP_STAGE must be furniture, wall_mounted, ceiling_mounted, or manipuland" >&2
        exit 1
        ;;
esac

BRANCH_START_STAGE=""
if [ "$BRANCH_FROM_SHARED_BASE" = "true" ] || [ "$GENERATE_SHARED_BASE" = "true" ]; then
    BRANCH_FROM_SHARED_BASE="true"
    case "$SHARED_BASE_STOP_STAGE" in
        floor_plan|furniture|wall_mounted|ceiling_mounted) ;;
        *)
            echo "ERROR: SHARED_BASE_STOP_STAGE must precede the target stage" >&2
            exit 1
            ;;
    esac
    BRANCH_START_STAGE="$(next_stage_after "$SHARED_BASE_STOP_STAGE")"
    if [ "$(pipeline_stage_index "$PIPELINE_STOP_STAGE")" -le "$(pipeline_stage_index "$SHARED_BASE_STOP_STAGE")" ]; then
        echo "ERROR: PIPELINE_STOP_STAGE must be after SHARED_BASE_STOP_STAGE when using a shared base" >&2
        exit 1
    fi
    if [ -z "$SHARED_BASE_ROOT" ]; then
        SHARED_BASE_ROOT="$OUTPUT_ROOT/shared_base"
    fi
    if [ "$GENERATE_SHARED_BASE" = "false" ] && [ ! -d "$SHARED_BASE_ROOT" ]; then
        echo "ERROR: SHARED_BASE_ROOT does not exist: $SHARED_BASE_ROOT" >&2
        exit 1
    fi
fi

# Parallel batches re-enter this script in a new session. Export every value
# that may have been normalized or defaulted above so the child uses exactly
# the same run configuration as the parent.
export SCENEEXPERT_EXPERIMENT="$EXPERIMENT"
export PYTHON_BIN MODEL_NAME RUN_ID OUTPUT_ROOT
export SCENE_BATCH_SIZE SCENE_WORKERS_PER_PROCESS
export CRITIC_PROBE_PARALLEL CRITIC_PROBE_INNER_PARALLELISM
export CRITIC_PROBE_PORT_BASE CRITIC_PROBE_PORT_BLOCK_SIZE
export CRITIC_PROBE_SHUTDOWN_GRACE_SECONDS
export CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE
export PIPELINE_STOP_STAGE BRANCH_FROM_SHARED_BASE SHARED_BASE_STOP_STAGE
export SHARED_BASE_ROOT GENERATE_SHARED_BASE MAX_CASES CASE_FILTER DRY_RUN
export SCENEEXPERT_DISABLE_ARTICULATED="$DISABLE_ARTICULATED"
export SCENEEXPERT_DISABLE_MATERIALS="$DISABLE_MATERIALS"
export SCENEEXPERT_DISABLE_BWRAP="$DISABLE_BWRAP"
export FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS
export HSSD_RETRIEVAL_BACKEND HSSD_RENDERED_ASSET_CHOICE
export CONVEX_MAX_OMP_THREADS
export FLOOR_PLAN_DESIGNER_THINKING FLOOR_PLAN_CRITIC_THINKING
export FURNITURE_DESIGNER_THINKING FURNITURE_CRITIC_THINKING
export WALL_DESIGNER_THINKING WALL_CRITIC_THINKING
export CEILING_DESIGNER_THINKING CEILING_CRITIC_THINKING
export MANIPULAND_DESIGNER_THINKING MANIPULAND_CRITIC_THINKING

mkdir -p "$OUTPUT_ROOT"

echo "========== PARALLEL CRITIC-ON PROBE =========="
echo "project: $PROJECT_ROOT"
echo "experiment: $EXPERIMENT"
echo "run id: $RUN_ID"
echo "output root: $OUTPUT_ROOT"
echo "model: $MODEL_NAME"
echo "OpenAI base URL: $OPENAI_BASE_URL"
echo "batch size: $SCENE_BATCH_SIZE"
echo "parallel batches: $CRITIC_PROBE_PARALLEL ($CRITIC_PROBE_INNER_PARALLELISM)"
echo "port allocation: base=$CRITIC_PROBE_PORT_BASE block=$CRITIC_PROBE_PORT_BLOCK_SIZE"
echo "continue after batch failure: $CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE"
echo "fail unresolved furniture hard constraints: $FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS"
echo "HSSD retrieval: backend=$HSSD_RETRIEVAL_BACKEND rendered_asset_choice=$HSSD_RENDERED_ASSET_CHOICE"
if [ -n "$CONVEX_MAX_OMP_THREADS" ]; then
    echo "convex decomposition max OMP threads: $CONVEX_MAX_OMP_THREADS"
fi
echo "thinking profile: floor_plan=${FLOOR_PLAN_DESIGNER_THINKING}/${FLOOR_PLAN_CRITIC_THINKING}, furniture=${FURNITURE_DESIGNER_THINKING}/${FURNITURE_CRITIC_THINKING}, wall=${WALL_DESIGNER_THINKING}/${WALL_CRITIC_THINKING}, ceiling=${CEILING_DESIGNER_THINKING}/${CEILING_CRITIC_THINKING}, manipuland=${MANIPULAND_DESIGNER_THINKING}/${MANIPULAND_CRITIC_THINKING}"
echo "shared base: $BRANCH_FROM_SHARED_BASE (generate=$GENERATE_SHARED_BASE)"
echo "==============================================="

# case_id|critic goal|prompt. Override only selection/count with CASE_FILTER
# and MAX_CASES; this keeps batch indices stable for reusable shared bases.
CASES=(
    "default_bedroom|ACP default scene 0|A bedroom with a bed, two nightstands, and a wardrobe in the corner of the room."
    "default_living_room|ACP default scene 1|A living room with a two-seater sofa against the wall, a square rug in the middle in front of the sofa, and two large plants on the floor near the sofa."
    "default_classroom|ACP default scene 2|A classroom with six student desks, each with a chair. A teacher's desk sits at the front near the chalkboard, which hangs on the wall."
    "default_rustic_bedroom|ACP default scene 3|A bedroom featuring rustic farmhouse decor with exposed wooden beams."
    "living_room_media_bottleneck|sofa-coffee-table-TV functional relation and living-room circulation bottleneck|A living room with a sofa against the back wall facing a TV stand and television on the opposite wall, a coffee table centered between the sofa and TV stand, two armchairs flanking the coffee table near each end of the sofa, and a floor lamp beside one armchair. A remote control and a few magazines lie on the coffee table, and a small rug lies between the coffee table and TV stand."
    "study_desk_access_crunch|desk-chair-monitor functional relation and study access|A study with a desk centered against the back wall, an office chair tucked under the desk, a computer monitor on the desk, two guest chairs against the side wall facing the desk, and a bookshelf on the adjacent wall. A desk lamp and a notebook sit on the desk, a pen holder next to the monitor, and a small trash can beside the desk."
    "bedroom_bedside_blockage|bed-nightstand-lamp functional relation and bed-side/wardrobe accessibility|A bedroom with a bed centered on the main wall, a nightstand with a table lamp on each side of the bed, a dresser against the opposite wall directly facing the bed, and a wardrobe placed next to the dresser. An alarm clock sits on one nightstand, a book on the other, and a small wastebasket near the dresser."
    "dining_room_service_squeeze|dining table-chair-place-setting relation and dining/sideboard accessibility|A dining room with a dining table in the center, four dining chairs arranged around it with one on each side, a sideboard against the wall behind the chairs on one side, and table settings for four including plates, cutlery, and glasses. A centerpiece vase with flowers sits in the middle of the table, and a set of coasters sits on the sideboard."
)

COMMON_ARGS=(
    "experiment.num_workers=${SCENE_WORKERS_PER_PROCESS}"
    "experiment.scene_retry_attempts=1"
    "furniture_agent.fail_stage_on_unresolved_hard_constraints=${FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS}"
    "experiment.pipeline.parallel_rooms=false"
    "experiment.pipeline.max_parallel_rooms=1"
    "experiment.scenebenchmark_critic.enabled=true"
    "experiment.scenebenchmark_critic.inject_into_llm_critic=true"
    "experiment.scenebenchmark_critic.fd_relation_proposer_mode=template"
    "experiment.scenebenchmark_critic.max_fd_relation_proposals=8"
    "floor_plan_agent.openai.reasoning_effort.designer=${FLOOR_PLAN_DESIGNER_THINKING}"
    "floor_plan_agent.openai.reasoning_effort.critic=${FLOOR_PLAN_CRITIC_THINKING}"
    "furniture_agent.openai.reasoning_effort.designer=${FURNITURE_DESIGNER_THINKING}"
    "furniture_agent.openai.reasoning_effort.critic=${FURNITURE_CRITIC_THINKING}"
    "wall_agent.openai.reasoning_effort.designer=${WALL_DESIGNER_THINKING}"
    "wall_agent.openai.reasoning_effort.critic=${WALL_CRITIC_THINKING}"
    "ceiling_agent.openai.reasoning_effort.designer=${CEILING_DESIGNER_THINKING}"
    "ceiling_agent.openai.reasoning_effort.critic=${CEILING_CRITIC_THINKING}"
    "manipuland_agent.openai.reasoning_effort.designer=${MANIPULAND_DESIGNER_THINKING}"
    "manipuland_agent.openai.reasoning_effort.critic=${MANIPULAND_CRITIC_THINKING}"
    "furniture_agent.asset_manager.hssd.retrieval_backend=${HSSD_RETRIEVAL_BACKEND}"
    "wall_agent.asset_manager.hssd.retrieval_backend=${HSSD_RETRIEVAL_BACKEND}"
    "ceiling_agent.asset_manager.hssd.retrieval_backend=${HSSD_RETRIEVAL_BACKEND}"
    "manipuland_agent.asset_manager.hssd.retrieval_backend=${HSSD_RETRIEVAL_BACKEND}"
    "furniture_agent.asset_manager.hssd.rendered_asset_choice.enabled=${HSSD_RENDERED_ASSET_CHOICE}"
    "wall_agent.asset_manager.hssd.rendered_asset_choice.enabled=${HSSD_RENDERED_ASSET_CHOICE}"
    "ceiling_agent.asset_manager.hssd.rendered_asset_choice.enabled=${HSSD_RENDERED_ASSET_CHOICE}"
    "manipuland_agent.asset_manager.hssd.rendered_asset_choice.enabled=${HSSD_RENDERED_ASSET_CHOICE}"
)

if [ -n "$CONVEX_MAX_OMP_THREADS" ]; then
    COMMON_ARGS+=(
        "furniture_agent.collision_geometry.max_omp_threads=${CONVEX_MAX_OMP_THREADS}"
        "wall_agent.collision_geometry.max_omp_threads=${CONVEX_MAX_OMP_THREADS}"
        "ceiling_agent.collision_geometry.max_omp_threads=${CONVEX_MAX_OMP_THREADS}"
        "manipuland_agent.collision_geometry.max_omp_threads=${CONVEX_MAX_OMP_THREADS}"
    )
fi

if [ "$DISABLE_ARTICULATED" = "true" ]; then
    COMMON_ARGS+=(
        "furniture_agent.asset_manager.router.strategies.articulated.enabled=false"
        "manipuland_agent.asset_manager.router.strategies.articulated.enabled=false"
        "wall_agent.asset_manager.router.strategies.articulated.enabled=false"
        "ceiling_agent.asset_manager.router.strategies.articulated.enabled=false"
        "furniture_agent.asset_manager.articulated.sources.partnet_mobility.enabled=false"
        "furniture_agent.asset_manager.articulated.sources.artvip.enabled=false"
        "manipuland_agent.asset_manager.articulated.sources.partnet_mobility.enabled=false"
        "manipuland_agent.asset_manager.articulated.sources.artvip.enabled=false"
        "wall_agent.asset_manager.articulated.sources.partnet_mobility.enabled=false"
        "wall_agent.asset_manager.articulated.sources.artvip.enabled=false"
        "ceiling_agent.asset_manager.articulated.sources.partnet_mobility.enabled=false"
        "ceiling_agent.asset_manager.articulated.sources.artvip.enabled=false"
    )
fi
if [ "$DISABLE_MATERIALS" = "true" ]; then
    COMMON_ARGS+=(
        "floor_plan_agent.materials.use_retrieval_server=false"
        "furniture_agent.asset_manager.router.strategies.thin_covering.enabled=false"
        "furniture_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false"
        "manipuland_agent.asset_manager.router.strategies.thin_covering.enabled=false"
        "manipuland_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false"
        "wall_agent.asset_manager.router.strategies.thin_covering.enabled=false"
        "wall_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false"
        "ceiling_agent.asset_manager.router.strategies.thin_covering.enabled=false"
        "ceiling_agent.asset_manager.router.strategies.thin_covering.generator.enabled=false"
    )
fi

port_args=()
build_port_args() {
    local batch_index="$1"
    local block_base=$((CRITIC_PROBE_PORT_BASE + (batch_index - 1) * CRITIC_PROBE_PORT_BLOCK_SIZE))
    if [ $((block_base + 374)) -gt 65535 ]; then
        echo "ERROR: batch $batch_index port block exceeds 65535" >&2
        exit 1
    fi
    port_args=(
        "experiment.geometry_generation_server.port=$((block_base + 5))"
        "experiment.hssd_retrieval_server.port=$((block_base + 6))"
        "experiment.articulated_retrieval_server.port=$((block_base + 7))"
        "experiment.materials_retrieval_server.port=$((block_base + 8))"
        "experiment.objaverse_retrieval_server.port=$((block_base + 9))"
        "floor_plan_agent.rendering.blender_server_port_range=[$((block_base + 100)),$((block_base + 124))]"
        "furniture_agent.rendering.blender_server_port_range=[$((block_base + 125)),$((block_base + 199))]"
        "wall_agent.rendering.blender_server_port_range=[$((block_base + 200)),$((block_base + 224))]"
        "ceiling_agent.rendering.blender_server_port_range=[$((block_base + 225)),$((block_base + 249))]"
        "manipuland_agent.rendering.blender_server_port_range=[$((block_base + 200)),$((block_base + 249))]"
        "furniture_agent.collision_geometry.server_port_range=[$((block_base + 250)),$((block_base + 324))]"
        "wall_agent.collision_geometry.server_port_range=[$((block_base + 325)),$((block_base + 349))]"
        "ceiling_agent.collision_geometry.server_port_range=[$((block_base + 350)),$((block_base + 374))]"
        "manipuland_agent.collision_geometry.server_port_range=[$((block_base + 325)),$((block_base + 374))]"
    )
}

run_batch() {
    local run_kind="$1"
    local batch_index="$2"
    shift 2
    local batch_entries=("$@")
    local batch_label
    batch_label=$(printf 'batch_%03d' "$batch_index")
    local run_root="$OUTPUT_ROOT/$run_kind/$batch_label"
    local batch_csv="$run_root/batch_cases.csv"
    local stop_stage="$PIPELINE_STOP_STAGE"
    local critic_enabled=true
    local start_stage=""
    local resume_from=""
    local shared_base_batch_root=""

    build_port_args "$batch_index"
    mkdir -p "$run_root"
    printf 'scene_index,prompt,case_id,critic_goal\n' > "$batch_csv"
    for entry in "${batch_entries[@]}"; do
        IFS='|' read -r scene_index case_id critic_goal prompt <<< "$entry"
        printf '%s,%s,%s,%s\n' "$scene_index" "$(csv_quote "$prompt")" "$(csv_quote "$case_id")" "$(csv_quote "$critic_goal")" >> "$batch_csv"
    done

    if [ "$run_kind" = "shared_base" ]; then
        stop_stage="$SHARED_BASE_STOP_STAGE"
        critic_enabled=false
    elif [ "$BRANCH_FROM_SHARED_BASE" = "true" ]; then
        start_stage="$BRANCH_START_STAGE"
        shared_base_batch_root="$SHARED_BASE_ROOT/$batch_label"
        # This script puts Hydra's scene directory below a per-batch
        # ``hydra`` directory to avoid latest-run symlink races.  The
        # single-room probe uses the batch directory directly, so accept both
        # layouts when replaying a shared base.
        if [ -d "$shared_base_batch_root/hydra" ]; then
            resume_from="$shared_base_batch_root/hydra"
        else
            resume_from="$shared_base_batch_root"
        fi
        if [ ! -d "$resume_from" ]; then
            echo "ERROR: missing reusable shared-base batch: $resume_from" >&2
            exit 1
        fi
        for entry in "${batch_entries[@]}"; do
            IFS='|' read -r scene_index _case_id _critic_goal _prompt <<< "$entry"
            if [ ! -d "$resume_from/scene_$(printf '%03d' "$scene_index")" ]; then
                echo "ERROR: shared-base scene directory not found: $resume_from/scene_$(printf '%03d' "$scene_index")" >&2
                echo "       Expected the shared base under $shared_base_batch_root/hydra or $shared_base_batch_root." >&2
                exit 1
            fi
        done
    fi

    local cmd=(
        "$PYTHON_BIN" main.py "experiment=$EXPERIMENT"
        "+name=critic_on_${batch_label}"
        "${COMMON_ARGS[@]}" "${port_args[@]}"
        "experiment.tasks=[generate_scenes]"
        "experiment.pipeline.stop_stage=${stop_stage}"
        "experiment.scenebenchmark_critic.enabled=${critic_enabled}"
        # main.py maintains a latest-run symlink two parents above the Hydra
        # output. Keep that parent unique per batch to avoid symlink races.
        "hydra.run.dir=${run_root}/hydra"
        "experiment.csv_path=${batch_csv}"
    )
    if [ -n "$start_stage" ]; then
        cmd+=("experiment.pipeline.start_stage=${start_stage}" "experiment.pipeline.resume_from_path=${resume_from}")
    fi

    echo "[$run_kind/$batch_label] ${cmd[*]}"
    if [ "$DRY_RUN" = "true" ]; then
        return 0
    fi
    if [ "$DISABLE_BWRAP" = "true" ]; then
        PATH="$PYTHON_EXEC_DIR:/usr/local/sbin:/usr/local/bin" "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
}

run_batches() {
    local run_kind="$1"
    local active_pids=()
    local active_labels=()
    local failed_group_pids=()
    local batch_index=0
    local source_batch_index=0
    local selected=0
    local batch_entries=()
    local batch_failure=0
    local cleanup_started=false

    mkdir -p "$OUTPUT_ROOT/$run_kind"

    process_group_alive() {
        ps -eo pgid=,stat= | awk -v pgid="$1" \
            '$1 == pgid && $2 !~ /^Z/ { found = 1 } END { exit !found }'
    }

    cleanup_active_batches() {
        local pid deadline any_alive
        local cleanup_pids=("${active_pids[@]}" "${failed_group_pids[@]}")
        if [ "$cleanup_started" = "true" ]; then
            return 0
        fi
        cleanup_started=true

        # Every parallel batch is started in its own session/process group.
        # Signal the whole group so Python, Blender, and retrieval-server
        # descendants cannot outlive the batch shell.
        for pid in "${cleanup_pids[@]}"; do
            kill -TERM -- "-$pid" 2>/dev/null || true
        done

        deadline=$((SECONDS + CRITIC_PROBE_SHUTDOWN_GRACE_SECONDS))
        while [ "$SECONDS" -lt "$deadline" ]; do
            any_alive=false
            for pid in "${cleanup_pids[@]}"; do
                if process_group_alive "$pid"; then
                    any_alive=true
                    break
                fi
            done
            if [ "$any_alive" = "false" ]; then
                break
            fi
            sleep 1
        done

        for pid in "${cleanup_pids[@]}"; do
            if process_group_alive "$pid"; then
                echo "WARNING: force-killing batch process group $pid" >&2
                kill -KILL -- "-$pid" 2>/dev/null || true
            fi
            wait "$pid" 2>/dev/null || true
        done
        active_pids=()
        active_labels=()
        failed_group_pids=()
    }

    on_batch_signal() {
        cleanup_active_batches
        exit "$1"
    }

    # A signal must not leave the background batch shells, Python workers, or
    # their Blender children behind. The EXIT trap is deliberately local to
    # this function so completed batches do not affect the next run kind.
    trap 'cleanup_active_batches' EXIT
    trap 'on_batch_signal 130' INT
    trap 'on_batch_signal 143' TERM
    trap 'on_batch_signal 129' HUP

    wait_one() {
        local finished_pid="" rc=0 label i pid state

        # Do not rely on wait -n -p here. If a child has already been reaped
        # by the shell, wait -n can return without a PID that matches our
        # bookkeeping array, leaving the outer wait loop stuck forever.
        # Polling also lets us recognize zombie children and reap them.
        while [ -z "$finished_pid" ]; do
            for i in "${!active_pids[@]}"; do
                pid="${active_pids[$i]}"
                state="$(ps -o stat= -p "$pid" 2>/dev/null | awk '{print $1}' || true)"
                if ! kill -0 "$pid" 2>/dev/null || [[ "$state" == Z* ]]; then
                    finished_pid="$pid"
                    if wait "$pid" 2>/dev/null; then rc=0; else rc=$?; fi
                    break
                fi
            done
            if [ -z "$finished_pid" ]; then
                sleep 1
            fi
        done

        label="pid_${finished_pid}"
        for i in "${!active_pids[@]}"; do
            if [ "${active_pids[$i]}" = "$finished_pid" ]; then
                label="${active_labels[$i]}"
                unset 'active_pids[i]' 'active_labels[i]'
                active_pids=("${active_pids[@]}")
                active_labels=("${active_labels[@]}")
                break
            fi
        done
        if [ "$rc" -ne 0 ]; then
            echo "ERROR: $run_kind/$label failed with exit code $rc" >&2
            batch_failure="$rc"
            if [ "$CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE" = "true" ]; then
                # The batch leader may have exited while native descendants
                # remain in its process group. Keep that group for cleanup
                # after all other batches have finished.
                failed_group_pids+=("$finished_pid")
                echo "WARNING: continuing remaining $run_kind batches after $run_kind/$label failure; final exit will report failure" >&2
                return 0
            fi
            # Preserve the original fail-fast behavior when explicitly disabled.
            # The batch leader may have exited while native descendants remain
            # in its process group. Keep that group in cleanup's input.
            active_pids+=("$finished_pid")
            active_labels+=("$label")
            # Fail fast. Waiting for unrelated scenes after one batch crashes
            # can keep an ACP allocation alive indefinitely if one of them is
            # also stuck in native or server shutdown code.
            cleanup_active_batches
            return "$rc"
        else
            echo "completed: $run_kind/$label"
        fi
    }

    launch() {
        local label
        label=$(printf 'batch_%03d' "$batch_index")
        if [ "$CRITIC_PROBE_PARALLEL" = "true" ]; then
            # A distinct process group makes cleanup include all descendants.
            # Re-entering this script avoids exporting shell functions/arrays.
            setsid bash "$0" --internal-run-batch "$run_kind" "$batch_index" "${batch_entries[@]}" \
                > "$OUTPUT_ROOT/$run_kind/${label}.log" 2>&1 &
            active_pids+=("$!")
            active_labels+=("$label")
            while [ "${#active_pids[@]}" -ge "$CRITIC_PROBE_INNER_PARALLELISM" ]; do wait_one; done
        else
            local rc=0
            if run_batch "$run_kind" "$batch_index" "${batch_entries[@]}"; then
                :
            else
                rc=$?
                echo "ERROR: $run_kind/batch_$(printf '%03d' "$batch_index") failed with exit code $rc" >&2
                batch_failure="$rc"
                if [ "$CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE" = "true" ]; then
                    echo "WARNING: continuing remaining $run_kind batches after $run_kind/batch_$(printf '%03d' "$batch_index") failure; final exit will report failure" >&2
                else
                    return "$rc"
                fi
            fi
        fi
    }

    for index in "${!CASES[@]}"; do
        IFS='|' read -r case_id critic_goal prompt <<< "${CASES[$index]}"
        if [ -n "$CASE_FILTER" ] && [[ "$case_id" != *"$CASE_FILTER"* ]]; then continue; fi
        if [ "$MAX_CASES" -gt 0 ] && [ "$selected" -ge "$MAX_CASES" ]; then break; fi
        source_batch_index=$((index / SCENE_BATCH_SIZE + 1))
        if [ "${#batch_entries[@]}" -gt 0 ] && [ "$batch_index" -ne "$source_batch_index" ]; then
            launch
            batch_entries=()
            batch_index=0
        fi
        if [ "$batch_index" -eq 0 ]; then batch_index="$source_batch_index"; fi
        batch_entries+=("$index|$case_id|$critic_goal|$prompt")
        selected=$((selected + 1))
        if [ "${#batch_entries[@]}" -eq "$SCENE_BATCH_SIZE" ]; then
            launch; batch_entries=(); batch_index=0
        fi
    done
    if [ "${#batch_entries[@]}" -gt 0 ]; then launch; fi
    while [ "${#active_pids[@]}" -gt 0 ]; do wait_one; done
    # A failed parallel batch can leave native descendants behind even though
    # its shell leader has exited. Reap those groups after other batches finish.
    if [ "${#failed_group_pids[@]}" -gt 0 ]; then
        cleanup_active_batches
    fi
    trap - EXIT INT TERM HUP
    if [ "$batch_failure" -ne 0 ]; then
        return "$batch_failure"
    fi
}

if [ "${1:-}" = "--internal-run-batch" ]; then
    shift
    run_batch "$@"
    exit $?
fi

if [ "$GENERATE_SHARED_BASE" = "true" ]; then
    run_batches shared_base
fi
run_batches critic_on
echo "critic-on probe complete: $OUTPUT_ROOT"
