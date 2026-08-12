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
# Output defaults to ``outputs/critic_probe/<run-id>``. Override it with
# OUTPUT_ROOT, ``--output-root <directory>``, or ``--output-dir <directory>``
# for disposable probes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Critic probes normally run the non-memory harness. Memory experiments remain
# opt-in so an omitted environment variable cannot load BGE-M3 unexpectedly.
EXPERIMENT="${SCENEEXPERT_EXPERIMENT:-ablation_3_qwen3_harness}"
if [ -n "${PYTHON_BIN:-}" ]; then
    PYTHON_BIN="$PYTHON_BIN"
else
    # A linked worktree normally shares Task3.2's environment rather than
    # carrying its own .venv. Prefer either project-local location before the
    # host interpreter so every batch has Hydra and the SceneSmith deps.
    PYTHON_BIN="python"
    for candidate in \
        "$PROJECT_ROOT/.venv/bin/python" \
        "$PROJECT_ROOT/../../Task3.2/.venv/bin/python"; do
        if [ -x "$candidate" ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
if ! "$PYTHON_BIN" -c 'import hydra' >/dev/null 2>&1; then
    echo "ERROR: PYTHON_BIN cannot import hydra: $PYTHON_BIN" >&2
    echo "       Set PYTHON_BIN to the Task3.2 virtualenv interpreter." >&2
    exit 1
fi

# A linked worktree has no local ``models`` directory. Resolve BGE-M3 before
# launching any batch so hybrid/vector-memory workers cannot fail only after
# their retrieval servers have started. The shared-model fallback is derived
# from a sibling Task3.2 checkout when one exists, not from the worktree name.
MEMORY_EMBEDDING_MODEL_DIR="${SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR:-}"
if [ -z "$MEMORY_EMBEDDING_MODEL_DIR" ]; then
    MODEL_LAYOUT_ROOT="$PROJECT_ROOT"
    if [ -d "$PROJECT_ROOT/../../Task3.2" ]; then
        MODEL_LAYOUT_ROOT="$PROJECT_ROOT/../../Task3.2"
    fi
    for candidate in \
        "${SCENEEXPERT_MODELS_DIR:-}/bge-m3" \
        "$PROJECT_ROOT/models/bge-m3" \
        "$MODEL_LAYOUT_ROOT/models/bge-m3" \
        "$(cd "$MODEL_LAYOUT_ROOT/../../.." && pwd)/share_model/Memory/bge-m3"; do
        if [ -n "$candidate" ] && [ -d "$candidate" ]; then
            MEMORY_EMBEDDING_MODEL_DIR="$candidate"
            break
        fi
    done
fi
if [ "$EXPERIMENT" = "ablation_4b_qwen3_vector_memory" ] \
    || [ "$EXPERIMENT" = "ablation_4c_qwen3_hybrid_memory" ]; then
    if [ -z "$MEMORY_EMBEDDING_MODEL_DIR" ] || [ ! -d "$MEMORY_EMBEDDING_MODEL_DIR" ]; then
        echo "ERROR: local BGE-M3 directory is required for $EXPERIMENT" >&2
        echo "       Set SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR to a valid bge-m3 directory." >&2
        exit 1
    fi
    export SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR="$MEMORY_EMBEDDING_MODEL_DIR"
fi
MODEL_NAME="${MODEL_NAME:-${SCENEEXPERT_MODEL_ID:-Qwen3.6-27B-Q8_0}}"
RUN_ID="${RUN_ID:-critic_on_$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/critic_probe/$RUN_ID}"
CASE_SET="${CASE_SET:-new3}"
SCENE_SELECTION="${SCENE_SELECTION:-all}"
SCENE_SELECTION_EXPLICIT="false"
REPLAY_FROM_PATH="${REPLAY_FROM_PATH:-}"
REPLAY_MODE="${REPLAY_MODE:-floor_plan}"
RESUME_FURNITURE_RENDER_MODE=""

usage() {
    cat <<'EOF'
Usage: bash scripts/run_parallel_critic_on.sh [options]

Options:
  --case-set <set>         case registry: new3 (default) or legacy8 (old8 alias)
  --scenes <selection>     all, or a comma-separated list chosen from:
                           the selected case registry
  --output-root <dir>      write probe output below <dir>
  --output-dir <dir>       alias for --output-root
  --resume-from <dir>      reuse prior critic batch outputs below <dir>
  --resume-mode <mode>     floor_plan (default), furniture_initial_render, or
                           furniture_latest_render
  -h, --help               show this help

Case registries:
  new3      bedroom, office, long_living_room
  legacy8   default_bedroom, default_living_room, default_classroom,
            default_rustic_bedroom, meeting_room_mixed_edge_seating,
            study_desk_access_crunch, bedroom_bedside_blockage,
            dining_room_service_squeeze

CASE_FILTER remains available for legacy substring filtering when --scenes is
not supplied. An explicit --scenes selection takes precedence. A reusable
shared base must use the same case registry and fixed scene ordering.
EOF
}

# Internal children receive positional batch payloads and must bypass the
# public option parser. The top-level parser accepts options in any order.
if [ "${1:-}" != "--internal-run-batch" ]; then
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --output-root|--output-dir)
                if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
                    echo "ERROR: $1 requires a directory" >&2
                    exit 2
                fi
                OUTPUT_ROOT="$2"
                shift 2
                ;;
            --output-root=*|--output-dir=*)
                OUTPUT_ROOT="${1#*=}"
                if [ -z "$OUTPUT_ROOT" ]; then
                    echo "ERROR: --output-root/--output-dir requires a directory" >&2
                    exit 2
                fi
                shift
                ;;
            --scenes)
                if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
                    echo "ERROR: --scenes requires all or a comma-separated scene list" >&2
                    exit 2
                fi
                SCENE_SELECTION="$2"
                SCENE_SELECTION_EXPLICIT="true"
                shift 2
                ;;
            --scenes=*)
                SCENE_SELECTION="${1#*=}"
                if [ -z "$SCENE_SELECTION" ]; then
                    echo "ERROR: --scenes requires all or a comma-separated scene list" >&2
                    exit 2
                fi
                SCENE_SELECTION_EXPLICIT="true"
                shift
                ;;
            --case-set)
                if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
                    echo "ERROR: --case-set requires new3 or legacy8" >&2
                    exit 2
                fi
                CASE_SET="$2"
                shift 2
                ;;
            --case-set=*)
                CASE_SET="${1#*=}"
                if [ -z "$CASE_SET" ]; then
                    echo "ERROR: --case-set requires new3 or legacy8" >&2
                    exit 2
                fi
                shift
                ;;
            --resume-from)
                if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
                    echo "ERROR: --resume-from requires a directory" >&2
                    exit 2
                fi
                REPLAY_FROM_PATH="$2"
                shift 2
                ;;
            --resume-from=*)
                REPLAY_FROM_PATH="${1#*=}"
                if [ -z "$REPLAY_FROM_PATH" ]; then
                    echo "ERROR: --resume-from requires a directory" >&2
                    exit 2
                fi
                shift
                ;;
            --resume-mode)
                if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
                    echo "ERROR: --resume-mode requires floor_plan, furniture_initial_render, or furniture_latest_render" >&2
                    exit 2
                fi
                REPLAY_MODE="$2"
                shift 2
                ;;
            --resume-mode=*)
                REPLAY_MODE="${1#*=}"
                if [ -z "$REPLAY_MODE" ]; then
                    echo "ERROR: --resume-mode requires floor_plan, furniture_initial_render, or furniture_latest_render" >&2
                    exit 2
                fi
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "ERROR: unknown argument: $1" >&2
                usage >&2
                exit 2
                ;;
        esac
    done
fi

case "$CASE_SET" in
    old8) CASE_SET="legacy8" ;;
    new3|legacy8) ;;
    *)
        echo "ERROR: CASE_SET must be new3 or legacy8, got '$CASE_SET'" >&2
        exit 2
        ;;
esac

SCENE_BATCH_SIZE="${SCENE_BATCH_SIZE:-1}"
SCENE_WORKERS_PER_PROCESS="${SCENE_WORKERS_PER_PROCESS:-1}"
# Native Drake/solver crashes cannot be caught in-process.  Keep two clean
# process retries by default; only failures classified as transient retry.
SCENE_RETRY_ATTEMPTS="${SCENE_RETRY_ATTEMPTS:-2}"
CRITIC_PROBE_PARALLEL="${CRITIC_PROBE_PARALLEL:-true}"
# A Qwen llama-server already reserves tens of GiB in the ACP cgroup.  Keep
# one Python scene process by default; callers can opt into more concurrency
# explicitly after checking the allocation budget.
CRITIC_PROBE_INNER_PARALLELISM="${CRITIC_PROBE_INNER_PARALLELISM:-1}"
CRITIC_PROBE_MAX_SAFE_INNER_PARALLELISM="${CRITIC_PROBE_MAX_SAFE_INNER_PARALLELISM:-1}"
CRITIC_PROBE_ALLOW_UNSAFE_PARALLELISM="${CRITIC_PROBE_ALLOW_UNSAFE_PARALLELISM:-false}"
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
INCLUDE_HOLDOUT_CASES="${INCLUDE_HOLDOUT_CASES:-false}"
DRY_RUN="${DRY_RUN:-false}"
CRITIC_PROBE_RENDER_FINAL_VIEWS="${CRITIC_PROBE_RENDER_FINAL_VIEWS:-false}"
CRITIC_PROBE_FINAL_VIEW_PARALLELISM="${CRITIC_PROBE_FINAL_VIEW_PARALLELISM:-1}"
FINAL_VIEW_PYTHON_BIN="${FINAL_VIEW_PYTHON_BIN:-$PYTHON_BIN}"
DISABLE_ARTICULATED="${SCENEEXPERT_DISABLE_ARTICULATED:-false}"
DISABLE_MATERIALS="${SCENEEXPERT_DISABLE_MATERIALS:-false}"
DISABLE_BWRAP="${SCENEEXPERT_DISABLE_BWRAP:-false}"
# The probe uses only external BlenderServer instances. Avoid importing bpy in
# the controller process, which otherwise allocates hundreds of idle threads.
SKIP_MAIN_BPY_IMPORT="${SCENEEXPERT_SKIP_MAIN_BPY_IMPORT:-true}"
HSSD_RETRIEVAL_BACKEND="${HSSD_RETRIEVAL_BACKEND:-clip}"
HSSD_RENDERED_ASSET_CHOICE="${HSSD_RENDERED_ASSET_CHOICE:-false}"
# A directory check alone is insufficient for BGE-M3: recent Transformers
# releases reject pickle checkpoints when the active Torch is too old. Load it
# once in the controller before any batch starts so an incompatible runtime
# cannot waste several retrieval-server startups and produce failed batches.
# Memory embedding is opt-in. Critic replays use the harness by default and
# must not start BGE-M3 merely because the model directory is present.
SCENEEXPERT_MEMORY_EMBEDDING_PREFLIGHT="${SCENEEXPERT_MEMORY_EMBEDDING_PREFLIGHT:-false}"
# os.cpu_count() sees the host's 192 logical CPUs in the CCI container.  A
# critic replay should never inherit the 32-thread YAML default implicitly:
# each isolated decomposition server gets a small explicit cap.
CONVEX_MAX_OMP_THREADS="${SCENEEXPERT_CONVEX_MAX_OMP_THREADS:-2}"
# Native BLAS/OpenMP libraries otherwise see all host CPUs. Six simultaneous
# BGE initializations can then create more than one thousand threads before
# scene generation even starts.
SCENEEXPERT_OMP_NUM_THREADS="${SCENEEXPERT_OMP_NUM_THREADS:-2}"

INTERNAL_RUN_BATCH="false"
if [ "${1:-}" = "--internal-run-batch" ]; then
    INTERNAL_RUN_BATCH="true"
fi

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
export OMP_NUM_THREADS="$SCENEEXPERT_OMP_NUM_THREADS"
export MKL_NUM_THREADS="$SCENEEXPERT_OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$SCENEEXPERT_OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$SCENEEXPERT_OMP_NUM_THREADS"

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

normalize_replay_source_root() {
    local requested_root="$1"
    local candidate

    if [ ! -d "$requested_root" ]; then
        echo "ERROR: --resume-from directory does not exist: $requested_root" >&2
        return 1
    fi

    # Normal runs write batches below <output>/critic_on, while a shared base
    # uses <output>/shared_base. Accept those output roots and either already
    # normalized batch root.
    for candidate in "$requested_root" "$requested_root/critic_on" "$requested_root/shared_base"; do
        if find "$candidate" -mindepth 1 -maxdepth 1 -type d \
            -name 'batch_[0-9][0-9][0-9]' -print -quit 2>/dev/null | grep -q .; then
            printf '%s' "$candidate"
            return 0
        fi
    done

    # Permit targeting one failed batch; its parent is the batch root used by
    # run_batch below.
    if [[ "$(basename "$requested_root")" =~ ^batch_[0-9]{3}$ ]]; then
        printf '%s' "$(dirname "$requested_root")"
        return 0
    fi

    echo "ERROR: --resume-from must be an output root, critic_on/shared_base root, or batch_XXX directory containing reusable batches: $requested_root" >&2
    return 1
}

require_positive_integer SCENE_BATCH_SIZE "$SCENE_BATCH_SIZE"
require_positive_integer SCENE_WORKERS_PER_PROCESS "$SCENE_WORKERS_PER_PROCESS"
if [[ ! "$SCENE_RETRY_ATTEMPTS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SCENE_RETRY_ATTEMPTS must be a non-negative integer, got '$SCENE_RETRY_ATTEMPTS'" >&2
    exit 1
fi
require_positive_integer CRITIC_PROBE_INNER_PARALLELISM "$CRITIC_PROBE_INNER_PARALLELISM"
require_positive_integer CRITIC_PROBE_MAX_SAFE_INNER_PARALLELISM "$CRITIC_PROBE_MAX_SAFE_INNER_PARALLELISM"
require_positive_integer CRITIC_PROBE_PORT_BASE "$CRITIC_PROBE_PORT_BASE"
require_positive_integer CRITIC_PROBE_PORT_BLOCK_SIZE "$CRITIC_PROBE_PORT_BLOCK_SIZE"
require_positive_integer CRITIC_PROBE_SHUTDOWN_GRACE_SECONDS "$CRITIC_PROBE_SHUTDOWN_GRACE_SECONDS"
require_positive_integer CRITIC_PROBE_FINAL_VIEW_PARALLELISM "$CRITIC_PROBE_FINAL_VIEW_PARALLELISM"
if [ -n "$CONVEX_MAX_OMP_THREADS" ]; then
    require_positive_integer SCENEEXPERT_CONVEX_MAX_OMP_THREADS "$CONVEX_MAX_OMP_THREADS"
fi
require_positive_integer SCENEEXPERT_OMP_NUM_THREADS "$SCENEEXPERT_OMP_NUM_THREADS"

if ! CRITIC_PROBE_ALLOW_UNSAFE_PARALLELISM="$(normalize_bool "$CRITIC_PROBE_ALLOW_UNSAFE_PARALLELISM")"; then
    echo "ERROR: CRITIC_PROBE_ALLOW_UNSAFE_PARALLELISM must be true or false" >&2
    exit 1
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
if [ -n "$REPLAY_FROM_PATH" ]; then
    if [ "$GENERATE_SHARED_BASE" != "false" ]; then
        echo "ERROR: --resume-from cannot be combined with GENERATE_SHARED_BASE=true" >&2
        exit 2
    fi
    case "$REPLAY_MODE" in
        floor_plan)
            RESUME_FURNITURE_RENDER_MODE=""
            ;;
        furniture_initial_render)
            RESUME_FURNITURE_RENDER_MODE="initial"
            ;;
        furniture_latest_render)
            RESUME_FURNITURE_RENDER_MODE="latest"
            ;;
        *)
            echo "ERROR: --resume-mode must be floor_plan, furniture_initial_render, or furniture_latest_render" >&2
            exit 2
            ;;
    esac
    BRANCH_FROM_SHARED_BASE="true"
    if ! SHARED_BASE_ROOT="$(normalize_replay_source_root "$REPLAY_FROM_PATH")"; then
        exit 2
    fi
    # Both furniture-render modes add a saved furniture snapshot before the
    # furniture stage; floor-plan resume starts from the persisted geometry.
    SHARED_BASE_STOP_STAGE="floor_plan"
elif [ "$REPLAY_MODE" != "floor_plan" ]; then
    echo "ERROR: --resume-mode requires --resume-from" >&2
    exit 2
fi
if ! DRY_RUN="$(normalize_bool "$DRY_RUN")"; then
    echo "ERROR: DRY_RUN must be true or false" >&2
    exit 1
fi
if ! INCLUDE_HOLDOUT_CASES="$(normalize_bool "$INCLUDE_HOLDOUT_CASES")"; then
    echo "ERROR: INCLUDE_HOLDOUT_CASES must be true or false" >&2
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
if ! SKIP_MAIN_BPY_IMPORT="$(normalize_bool "$SKIP_MAIN_BPY_IMPORT")"; then
    echo "ERROR: SCENEEXPERT_SKIP_MAIN_BPY_IMPORT must be true or false" >&2
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
if ! SCENEEXPERT_MEMORY_EMBEDDING_PREFLIGHT="$(normalize_bool "$SCENEEXPERT_MEMORY_EMBEDDING_PREFLIGHT")"; then
    echo "ERROR: SCENEEXPERT_MEMORY_EMBEDDING_PREFLIGHT must be true or false" >&2
    exit 1
fi
if ! FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS="$(normalize_bool "$FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS")"; then
    echo "ERROR: FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS must be true or false" >&2
    exit 1
fi
if ! CRITIC_PROBE_RENDER_FINAL_VIEWS="$(normalize_bool "$CRITIC_PROBE_RENDER_FINAL_VIEWS")"; then
    echo "ERROR: CRITIC_PROBE_RENDER_FINAL_VIEWS must be true or false" >&2
    exit 1
fi
if [ "$CRITIC_PROBE_RENDER_FINAL_VIEWS" = "true" ] && [ "$SCENE_BATCH_SIZE" -ne 1 ]; then
    echo "ERROR: immediate per-scene final rendering requires SCENE_BATCH_SIZE=1 (got $SCENE_BATCH_SIZE)" >&2
    echo "       Set SCENE_BATCH_SIZE=1 so each completed batch maps to exactly one scene." >&2
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
if [ "$CRITIC_PROBE_PARALLEL" = "true" ] \
    && [ "$CRITIC_PROBE_INNER_PARALLELISM" -gt "$CRITIC_PROBE_MAX_SAFE_INNER_PARALLELISM" ] \
    && [ "$CRITIC_PROBE_ALLOW_UNSAFE_PARALLELISM" != "true" ]; then
    echo "ERROR: refusing unsafe critic batch concurrency: inner=$CRITIC_PROBE_INNER_PARALLELISM (safe default max=$CRITIC_PROBE_MAX_SAFE_INNER_PARALLELISM)." >&2
    echo "       Set CRITIC_PROBE_INNER_PARALLELISM=1, or explicitly opt in with CRITIC_PROBE_ALLOW_UNSAFE_PARALLELISM=true." >&2
    exit 1
fi
if [ "$CRITIC_PROBE_PARALLEL" = "true" ] && ! command -v setsid >/dev/null 2>&1; then
    echo "ERROR: setsid is required for isolated parallel batch cleanup" >&2
    exit 1
fi
if [ "$INTERNAL_RUN_BATCH" = "false" ] \
    && [ "$DRY_RUN" = "false" ] \
    && [ "$SCENEEXPERT_MEMORY_EMBEDDING_PREFLIGHT" = "true" ] \
    && { [ "$EXPERIMENT" = "ablation_4b_qwen3_vector_memory" ] \
        || [ "$EXPERIMENT" = "ablation_4c_qwen3_hybrid_memory" ]; }; then
    echo "preflighting SceneExpert memory embedding runtime: $SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR"
    if ! "$PYTHON_BIN" -c 'from scenesmith.scene_expert.memory.embedding import SceneMemoryEmbedder; SceneMemoryEmbedder(device="cpu")'; then
        echo "ERROR: SceneExpert memory embedding preflight failed; no critic batches were started." >&2
        echo "       Use a runtime that can load the configured BGE-M3 checkpoint." >&2
        exit 1
    fi
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

validate_scene_selection() {
    local scene_id seen="," selection_count=0
    if [ "$SCENE_SELECTION" = "all" ]; then
        return 0
    fi
    # Bash's ``read -a`` drops a trailing empty field, so reject malformed CSV
    # before splitting rather than relying only on the per-item empty check.
    if [[ "$SCENE_SELECTION" == ,* \
        || "$SCENE_SELECTION" == *, \
        || "$SCENE_SELECTION" == *,,* ]]; then
        echo "ERROR: --scenes contains an empty scene ID" >&2
        exit 2
    fi
    IFS=',' read -r -a selected_scene_ids <<< "$SCENE_SELECTION"
    for scene_id in "${selected_scene_ids[@]}"; do
        if [ -z "$scene_id" ]; then
            echo "ERROR: --scenes contains an empty scene ID" >&2
            exit 2
        fi
        case "$scene_id" in
            all)
                echo "ERROR: --scenes all cannot be combined with individual scene IDs" >&2
                exit 2
                ;;
        esac
        if ! case_registry_contains "$scene_id"; then
            echo "ERROR: unknown scene ID '$scene_id' for case set '$CASE_SET'; choose an ID from: $CASE_SET_IDS" >&2
            echo "       Or use --scenes all." >&2
            exit 2
        fi
        if [[ "$seen" == *",$scene_id,"* ]]; then
            echo "ERROR: duplicate scene ID in --scenes: $scene_id" >&2
            exit 2
        fi
        seen+="$scene_id,"
        selection_count=$((selection_count + 1))
    done
    if [ "$selection_count" -eq 0 ]; then
        echo "ERROR: --scenes did not select any scenes" >&2
        exit 2
    fi
}

case_registry_contains() {
    local requested_id="$1"
    local case_entry case_id
    for case_entry in "${CASES[@]}"; do
        IFS='|' read -r case_id _ <<< "$case_entry"
        if [ "$case_id" = "$requested_id" ]; then
            return 0
        fi
    done
    return 1
}

select_case_registry() {
    case "$CASE_SET" in
        legacy8)
            # Preserved verbatim from commit 1c45466. Both ID order and
            # prompts define the persisted shared-base batch/scene mapping.
            CASES=(
                "default_bedroom|ACP default scene 0|A bedroom with a bed, two nightstands, and a wardrobe in the corner of the room."
                "default_living_room|ACP default scene 1|A living room with a two-seater sofa against the wall, a square rug in the middle in front of the sofa, and two large plants on the floor near the sofa."
                "default_classroom|ACP default scene 2|A classroom with six student desks, each with a chair. A teacher's desk sits at the front near the chalkboard, which hangs on the wall."
                "default_rustic_bedroom|ACP default scene 3|A bedroom featuring rustic farmhouse decor with exposed wooden beams."
                "meeting_room_mixed_edge_seating|conference-table mixed long-and-short-side equal chair distribution|A meeting room with one rectangular conference table centered in the room and seven office chairs. Arrange six office chairs in two equal groups of three, evenly spaced along the table's two long sides, with all chairs facing the table. Place one remaining office chair centered along one short side, facing the table. Keep the opposite short side free of chairs. Keep clear circulation around the table."
                "study_desk_access_crunch|desk-chair-monitor functional relation and study access|A study with a desk centered against the back wall, an office chair tucked under the desk, a computer monitor on the desk, two guest chairs against the side wall with their usable fronts perpendicular to the wall and facing into the room, and a bookshelf on the adjacent wall with its usable front perpendicular to the wall and facing into the room. A desk lamp and a notebook sit on the desk, a pen holder next to the monitor, and a small trash can beside the desk."
                "bedroom_bedside_blockage|bed-nightstand-lamp functional relation and bed-side/wardrobe accessibility|A bedroom with a bed centered on the main wall, a nightstand with a table lamp on each side of the bed, a dresser against the opposite wall directly facing the bed, and a wardrobe placed next to the dresser. An alarm clock sits on one nightstand, a book on the other, and a small wastebasket near the dresser."
                "dining_room_service_squeeze|dining table-chair-place-setting relation and dining/sideboard accessibility|A dining room with a dining table in the center, four dining chairs arranged around it with one on each side, a sideboard against the wall behind the chairs on one side, and table settings for four including plates, cutlery, and glasses. A centerpiece vase with flowers sits in the middle of the table, and a set of coasters sits on the sideboard."
            )
            ;;
        new3)
            CASES=(
                "bedroom|bed-nightstand flanking, dressing station alignment, storage access, and wall mirror|A functional bedroom with one bed centered against the back wall. Place two nightstands, one on each side of the bed. Place one wardrobe against a side wall. Position one low freestanding dressing table against a free wall, with one stool centered in front of and facing the dressing table. Mount one separate mirror on the wall, centered directly above the dressing table; the mirror must not be integrated into the table. Keep the entrance route clear and leave enough usable space in front of the wardrobe and dressing table."
                "office|four one-to-one desk-chair-monitor workstations, shared circulation, and corner wastebasket|A practical office with four separate desks forming four workstations. Pair each desk with exactly one office chair positioned at its usable side and facing the desk. Place exactly one computer monitor on top of each desk, for four monitors in total, with every monitor facing its paired chair. Place one freestanding water dispenser against a wall and keep its front accessible. Place one wastebasket on the floor in one corner of the room. Maintain a clear central aisle and enough clearance to use every workstation."
                "long_living_room|living-media alignment, five-seat dining edge distribution, wall storage, and distinct-corner plants|A long rectangular living room with separate living and dining areas. In the living area, place one sofa against a wall facing one TV stand on the opposite side, with one television supported on top of the TV stand. Center one coffee table between the sofa and the TV stand. In the dining area, place one rectangular dining table with five complete table settings, each including a plate, cutlery, and a drinking glass. Arrange five dining chairs around the table: two evenly spaced along each long side and one centered on one short side, all facing the table; keep the opposite short side free of chairs. Place one storage cabinet against a wall without blocking circulation. Place four large floor plants in four distinct room corners, exactly one plant per corner. Keep a clear route between the entrance, living area, dining area, and storage cabinet."
            )
            ;;
        *)
            echo "ERROR: CASE_SET must be new3 or legacy8, got '$CASE_SET'" >&2
            exit 2
            ;;
    esac

    local case_entry case_id
    CASE_SET_IDS=""
    for case_entry in "${CASES[@]}"; do
        IFS='|' read -r case_id _ <<< "$case_entry"
        if [ -n "$CASE_SET_IDS" ]; then
            CASE_SET_IDS+=", "
        fi
        CASE_SET_IDS+="$case_id"
    done
}

select_case_registry

# Reject invalid selections before creating shared-base metadata, an output
# directory, or a lock file.
validate_scene_selection

SHARED_BASE_CASE_SET_FILE=""
if [ "$BRANCH_FROM_SHARED_BASE" = "true" ]; then
    SHARED_BASE_CASE_SET_FILE="$SHARED_BASE_ROOT/.critic_on_case_set"
    if [ "$GENERATE_SHARED_BASE" = "true" ]; then
        if [ "$DRY_RUN" = "false" ]; then
            mkdir -p "$SHARED_BASE_ROOT"
            printf '%s\n' "$CASE_SET" > "$SHARED_BASE_CASE_SET_FILE"
        fi
    elif [ -f "$SHARED_BASE_CASE_SET_FILE" ]; then
        shared_base_case_set="$(tr -d '[:space:]' < "$SHARED_BASE_CASE_SET_FILE")"
        if [ "$shared_base_case_set" != "$CASE_SET" ]; then
            echo "ERROR: shared base case set is '$shared_base_case_set', but this run requested '$CASE_SET'" >&2
            echo "       Reuse only a shared base generated for the same case registry." >&2
            exit 2
        fi
    else
        echo "WARNING: reusable shared base has no .critic_on_case_set metadata; compatibility is inferred from batch files." >&2
    fi
fi

# Parallel batches re-enter this script in a new session. Export every value
# that may have been normalized or defaulted above so the child uses exactly
# the same run configuration as the parent.
export SCENEEXPERT_EXPERIMENT="$EXPERIMENT"
export PYTHON_BIN MODEL_NAME RUN_ID OUTPUT_ROOT CASE_SET CASE_SET_IDS
export SCENE_BATCH_SIZE SCENE_WORKERS_PER_PROCESS SCENE_RETRY_ATTEMPTS
export CRITIC_PROBE_PARALLEL CRITIC_PROBE_INNER_PARALLELISM
export CRITIC_PROBE_MAX_SAFE_INNER_PARALLELISM CRITIC_PROBE_ALLOW_UNSAFE_PARALLELISM
export CRITIC_PROBE_PORT_BASE CRITIC_PROBE_PORT_BLOCK_SIZE
export CRITIC_PROBE_SHUTDOWN_GRACE_SECONDS
export CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE
export CRITIC_PROBE_RENDER_FINAL_VIEWS
export CRITIC_PROBE_FINAL_VIEW_PARALLELISM
export FINAL_VIEW_PYTHON_BIN
export PIPELINE_STOP_STAGE BRANCH_FROM_SHARED_BASE SHARED_BASE_STOP_STAGE
export SHARED_BASE_ROOT GENERATE_SHARED_BASE MAX_CASES CASE_FILTER
export INCLUDE_HOLDOUT_CASES DRY_RUN SCENE_SELECTION SCENE_SELECTION_EXPLICIT
export REPLAY_FROM_PATH REPLAY_MODE RESUME_FURNITURE_RENDER_MODE
export SCENEEXPERT_DISABLE_ARTICULATED="$DISABLE_ARTICULATED"
export SCENEEXPERT_DISABLE_MATERIALS="$DISABLE_MATERIALS"
export SCENEEXPERT_DISABLE_BWRAP="$DISABLE_BWRAP"
export SCENEEXPERT_SKIP_MAIN_BPY_IMPORT="$SKIP_MAIN_BPY_IMPORT"
export FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS
export HSSD_RETRIEVAL_BACKEND HSSD_RENDERED_ASSET_CHOICE
export CONVEX_MAX_OMP_THREADS SCENEEXPERT_OMP_NUM_THREADS
export FLOOR_PLAN_DESIGNER_THINKING FLOOR_PLAN_CRITIC_THINKING
export FURNITURE_DESIGNER_THINKING FURNITURE_CRITIC_THINKING
export WALL_DESIGNER_THINKING WALL_CRITIC_THINKING
export CEILING_DESIGNER_THINKING CEILING_CRITIC_THINKING
export MANIPULAND_DESIGNER_THINKING MANIPULAND_CRITIC_THINKING

mkdir -p "$OUTPUT_ROOT"

read_cgroup_memory_value() {
    local path value
    for path in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
        if [ -r "$path" ]; then
            value=$(tr -d '[:space:]' < "$path")
            if [ -n "$value" ] && [ "$value" != "max" ]; then
                printf '%s\n' "$value"
                return 0
            fi
        fi
    done
    return 1
}

read_cgroup_memory_current() {
    local path value
    for path in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory/memory.usage_in_bytes; do
        if [ -r "$path" ]; then
            value=$(tr -d '[:space:]' < "$path")
            if [[ "$value" =~ ^[0-9]+$ ]]; then
                printf '%s\n' "$value"
                return 0
            fi
        fi
    done
    return 1
}

if [ "$INTERNAL_RUN_BATCH" = "false" ]; then
    # Serialize only re-entry into the same output root.  Independent replay
    # roots have non-overlapping service-port blocks and can safely share a
    # llama server configured for concurrent requests.
    LOCK_FILE="${CRITIC_PROBE_LOCK_FILE:-${OUTPUT_ROOT}.lock}"
    mkdir -p "$(dirname "$LOCK_FILE")"
    if ! command -v flock >/dev/null 2>&1; then
        echo "ERROR: flock is required to prevent overlapping critic probes" >&2
        exit 1
    fi
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        echo "ERROR: another critic probe already owns output lock $LOCK_FILE" >&2
        echo "       Wait for it to finish or inspect its process/log before retrying." >&2
        exit 1
    fi

    if memory_limit_bytes=$(read_cgroup_memory_value) \
        && memory_current_bytes=$(read_cgroup_memory_current); then
        # Refuse a new run when less than 15% of the cgroup remains.  This is
        # intentionally a startup guard; an explicit override is available for
        # allocations whose memory is accounted outside this cgroup.
        if [ "${CRITIC_PROBE_ALLOW_HIGH_MEMORY_START:-0}" != "1" ] \
            && [ "$memory_current_bytes" -gt $((memory_limit_bytes * 85 / 100)) ]; then
            echo "ERROR: cgroup memory is already at ${memory_current_bytes}/${memory_limit_bytes} bytes; refusing a new critic probe." >&2
            echo "       Set CRITIC_PROBE_ALLOW_HIGH_MEMORY_START=1 only after verifying stale processes are gone." >&2
            exit 1
        fi
    fi
fi

echo "========== PARALLEL CRITIC-ON PROBE =========="
echo "project: $PROJECT_ROOT"
echo "experiment: $EXPERIMENT"
echo "run id: $RUN_ID"
echo "output root: $OUTPUT_ROOT"
echo "case set: $CASE_SET"
echo "model: $MODEL_NAME"
echo "OpenAI base URL: $OPENAI_BASE_URL"
if [ -n "${SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR:-}" ]; then
    echo "SceneExpert memory embedding model: $SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR"
fi
echo "batch size: $SCENE_BATCH_SIZE"
echo "parallel batches: $CRITIC_PROBE_PARALLEL ($CRITIC_PROBE_INNER_PARALLELISM)"
echo "scene retries after transient/native failure: $SCENE_RETRY_ATTEMPTS"
echo "port allocation: base=$CRITIC_PROBE_PORT_BASE block=$CRITIC_PROBE_PORT_BLOCK_SIZE"
echo "continue after batch failure: $CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE"
echo "final-view parallelism: $CRITIC_PROBE_FINAL_VIEW_PARALLELISM"
echo "fail unresolved furniture hard constraints: $FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS"
echo "HSSD retrieval: backend=$HSSD_RETRIEVAL_BACKEND rendered_asset_choice=$HSSD_RENDERED_ASSET_CHOICE"
echo "skip controller bpy import: $SKIP_MAIN_BPY_IMPORT"
if [ -n "$CONVEX_MAX_OMP_THREADS" ]; then
    echo "convex decomposition max OMP threads: $CONVEX_MAX_OMP_THREADS"
fi
echo "native OMP/BLAS threads per scene: $SCENEEXPERT_OMP_NUM_THREADS"
if [ "$INTERNAL_RUN_BATCH" = "false" ] \
    && memory_limit_bytes=$(read_cgroup_memory_value) \
    && memory_current_bytes=$(read_cgroup_memory_current); then
    echo "cgroup memory: current=$memory_current_bytes limit=$memory_limit_bytes"
fi
echo "thinking profile: floor_plan=${FLOOR_PLAN_DESIGNER_THINKING}/${FLOOR_PLAN_CRITIC_THINKING}, furniture=${FURNITURE_DESIGNER_THINKING}/${FURNITURE_CRITIC_THINKING}, wall=${WALL_DESIGNER_THINKING}/${WALL_CRITIC_THINKING}, ceiling=${CEILING_DESIGNER_THINKING}/${CEILING_CRITIC_THINKING}, manipuland=${MANIPULAND_DESIGNER_THINKING}/${MANIPULAND_CRITIC_THINKING}"
echo "shared base: $BRANCH_FROM_SHARED_BASE (generate=$GENERATE_SHARED_BASE)"
echo "replay source: ${REPLAY_FROM_PATH:-none} (mode=$REPLAY_MODE)"
echo "holdout cases: $INCLUDE_HOLDOUT_CASES"
echo "scene selection: $SCENE_SELECTION"
echo "==============================================="

case_selected() {
    local case_id="$1"
    local filter
    if [ "$SCENE_SELECTION_EXPLICIT" = "true" ] || [ "$SCENE_SELECTION" != "all" ]; then
        if [ "$SCENE_SELECTION" = "all" ]; then
            return 0
        fi
        IFS=',' read -r -a selected_scene_ids <<< "$SCENE_SELECTION"
        for filter in "${selected_scene_ids[@]}"; do
            if [ "$case_id" = "$filter" ]; then
                return 0
            fi
        done
        return 1
    fi
    if [ -z "$CASE_FILTER" ]; then
        return 0
    fi
    IFS=',' read -r -a case_filters <<< "$CASE_FILTER"
    for filter in "${case_filters[@]}"; do
        if [ -n "$filter" ] && [[ "$case_id" == *"$filter"* ]]; then
            return 0
        fi
    done
    return 1
}

validate_shared_base_case_mapping() {
    local index entry case_id _critic_goal _prompt batch_index batch_csv selected=0

    if [ "$BRANCH_FROM_SHARED_BASE" != "true" ] \
        || [ "$GENERATE_SHARED_BASE" = "true" ]; then
        return 0
    fi

    for index in "${!CASES[@]}"; do
        entry="${CASES[$index]}"
        IFS='|' read -r case_id _critic_goal _prompt <<< "$entry"
        if ! case_selected "$case_id"; then
            continue
        fi
        if [ "$MAX_CASES" -gt 0 ] && [ "$selected" -ge "$MAX_CASES" ]; then
            break
        fi
        batch_index=$((index / SCENE_BATCH_SIZE + 1))
        batch_csv="$SHARED_BASE_ROOT/$(printf 'batch_%03d' "$batch_index")/batch_cases.csv"
        if [ ! -f "$batch_csv" ]; then
            echo "ERROR: reusable shared base has no batch manifest: $batch_csv" >&2
            echo "       Expected case '$case_id' at scene index $index for case set '$CASE_SET'." >&2
            exit 2
        fi
        if ! awk -F',' -v expected_index="$index" -v expected_id="$case_id" \
            '$1 == expected_index && index($0, "\"" expected_id "\"") { found = 1 } END { exit !found }' \
            "$batch_csv"; then
            echo "ERROR: reusable shared base batch mapping does not contain case '$case_id' at scene index $index: $batch_csv" >&2
            echo "       The base belongs to another case registry or was generated with a different ordering." >&2
            exit 2
        fi
        selected=$((selected + 1))
    done
}

validate_shared_base_case_mapping

COMMON_ARGS=(
    "experiment.num_workers=${SCENE_WORKERS_PER_PROCESS}"
    "experiment.scene_retry_attempts=${SCENE_RETRY_ATTEMPTS}"
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
    local asset_choice_audit_path="$run_root/hydra/asset_choice_audit.jsonl"

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
        if [ "$DRY_RUN" = "false" ] && [ ! -d "$resume_from" ]; then
            echo "ERROR: missing reusable shared-base batch: $resume_from" >&2
            exit 1
        fi
        if [ "$DRY_RUN" = "false" ]; then
            for entry in "${batch_entries[@]}"; do
                IFS='|' read -r scene_index _case_id _critic_goal _prompt <<< "$entry"
                if [ ! -d "$resume_from/scene_$(printf '%03d' "$scene_index")" ]; then
                    echo "ERROR: shared-base scene directory not found: $resume_from/scene_$(printf '%03d' "$scene_index")" >&2
                    echo "       Expected the shared base under $shared_base_batch_root/hydra or $shared_base_batch_root." >&2
                    exit 1
                fi
            done
        fi
    fi

    local cmd=(
        "$PYTHON_BIN" main.py "experiment=$EXPERIMENT"
        "+name=critic_on_${batch_label}"
        "${COMMON_ARGS[@]}" "${port_args[@]}"
        "experiment.tasks=[generate_scenes]"
        "experiment.pipeline.stop_stage=${stop_stage}"
        "experiment.scenebenchmark_critic.enabled=${critic_enabled}"
        "hydra.run.dir=${run_root}/hydra"
        "experiment.csv_path=${batch_csv}"
    )
    if [ -n "$start_stage" ]; then
        cmd+=("experiment.pipeline.start_stage=${start_stage}" "experiment.pipeline.resume_from_path=${resume_from}")
        if [ -n "$RESUME_FURNITURE_RENDER_MODE" ]; then
            cmd+=("experiment.pipeline.resume_furniture_from_render=${RESUME_FURNITURE_RENDER_MODE}")
        fi
    fi

    echo "[$run_kind/$batch_label] ${cmd[*]}"
    if [ "$DRY_RUN" = "true" ]; then
        return 0
    fi
    if [ "$DISABLE_BWRAP" = "true" ]; then
        HSSD_RENDERED_ASSET_CHOICE_AUDIT_PATH="$asset_choice_audit_path" \
            PATH="$PYTHON_EXEC_DIR:/usr/local/sbin:/usr/local/bin" "${cmd[@]}"
    else
        HSSD_RENDERED_ASSET_CHOICE_AUDIT_PATH="$asset_choice_audit_path" "${cmd[@]}"
    fi

    # Render only the scene represented by this completed batch. Immediate
    # rendering keeps usable views available even when a later batch fails.
    # Rendering is best-effort: a renderer/Blender failure must not change the
    # already successful scene-generation result.
    if [ "$run_kind" = "critic_on" ] \
        && [ "$CRITIC_PROBE_RENDER_FINAL_VIEWS" = "true" ] \
        && [ "$PIPELINE_STOP_STAGE" = "manipuland" ]; then
        for entry in "${batch_entries[@]}"; do
            IFS='|' read -r scene_index _case_id _critic_goal _prompt <<< "$entry"
            local scene_dir="$run_root/hydra/scene_$(printf '%03d' "$scene_index")"
            local blend_path="$scene_dir/combined_house/house.blend"
            if [ ! -f "$blend_path" ]; then
                echo "WARNING: completed $run_kind/$batch_label has no final blend; skipping render: $blend_path" >&2
                continue
            fi
            echo "[$run_kind/$batch_label] rendering final views for scene $(printf '%03d' "$scene_index")"
            if "$FINAL_VIEW_PYTHON_BIN" "$SCRIPT_DIR/render_critic_final_views.py" \
                --parallelism 1 -- "$blend_path"; then
                echo "[$run_kind/$batch_label] final views ready: $scene_dir/critic_final_views"
            else
                echo "WARNING: final-view rendering failed for $blend_path; continuing" >&2
            fi
        done
    fi
}

run_batches() {
    local run_kind="$1"
    local active_pids=()
    local active_labels=()
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

    terminate_failed_batch_group() {
        local pid="$1" deadline any_alive

        # The session leader has already been reaped by wait_one, but a
        # crashed Python process can leave its forkserver, Blender, or
        # retrieval-server descendants alive. Clean only this batch's process
        # group before starting a replacement batch.
        if ! process_group_alive "$pid"; then
            return 0
        fi
        echo "WARNING: terminating failed batch process group $pid" >&2
        kill -TERM -- "-$pid" 2>/dev/null || true
        deadline=$((SECONDS + CRITIC_PROBE_SHUTDOWN_GRACE_SECONDS))
        while [ "$SECONDS" -lt "$deadline" ]; do
            if ! process_group_alive "$pid"; then
                return 0
            fi
            sleep 1
        done
        if process_group_alive "$pid"; then
            echo "WARNING: force-killing failed batch process group $pid" >&2
            kill -KILL -- "-$pid" 2>/dev/null || true
        fi
    }

    cleanup_active_batches() {
        local pid deadline any_alive
        local cleanup_pids=("${active_pids[@]}")
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
                terminate_failed_batch_group "$finished_pid"
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
        if ! case_selected "$case_id"; then continue; fi
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
if [ "$CRITIC_PROBE_RENDER_FINAL_VIEWS" = "true" ] \
    && [ "$PIPELINE_STOP_STAGE" != "manipuland" ]; then
    echo "skipping final combined-house views: pipeline stops at $PIPELINE_STOP_STAGE"
fi
echo "critic-on probe complete: $OUTPUT_ROOT"
