#!/usr/bin/env bash
# Canonical Qwen3.8 ACP runtime and the operator entrypoint for a new 4c run.
# Prompt source is selected with CASE_SET; Full and reuse entrypoints delegate
# to this file so model, critic, memory, stage-policy, and logging settings stay
# identical across the experiment matrix.

set -euo pipefail

# =============================================================================
# USER CONFIGURATION — review every TODO(user) item before starting a run.
# =============================================================================

# --- 1. Workspace and shared infrastructure ---------------------------------
# TODO(user): Project directory containing the synchronized code to evaluate.
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2}"

# TODO(user): Our ACP account sees the task mount at /mnt/afs. Keep every
# shared path derived from this account-local mount root.
TASK3_SHARED_ROOT="${TASK3_SHARED_ROOT:-/mnt/afs/task3_2}"
FALLBACK_HELPER_DIR="${FALLBACK_HELPER_DIR:-$TASK3_SHARED_ROOT/L202500266_hrk/code}"
CRITIC_HELPER_DIR="${CRITIC_HELPER_DIR:-$PROJECT_ROOT/tmp/code}"
SHARED_HSSD_ROOT="${SHARED_HSSD_ROOT:-$TASK3_SHARED_ROOT/share_data/hsm}"
SHARED_HSSD_AUX_ROOT="${SHARED_HSSD_AUX_ROOT:-$TASK3_SHARED_ROOT/share_data/scenesmith}"
SHARED_MODEL_ROOT="${SHARED_MODEL_ROOT:-$TASK3_SHARED_ROOT/share_model}"
SHARED_SCRIPT_ROOT="${SHARED_SCRIPT_ROOT:-$TASK3_SHARED_ROOT/share_scripts}"
SCENEEXPERT_DATA_DIR="${SCENEEXPERT_DATA_DIR:-$TASK3_SHARED_ROOT/L202500276_lwz/data}"
CUDA13_LIB_DIR="${CUDA13_LIB_DIR:-$TASK3_SHARED_ROOT/L202500276_lwz/projects/Task3.2/.venv/lib/python3.11/site-packages/nvidia/cu13/lib}"

# --- 2. Qwen3.8 serving profile ---------------------------------------------
# TODO(user): Change these together when the critic team changes model builds.
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.8-27B-GGUF}"
MODEL_DIR="${MODEL_DIR:-$SHARED_MODEL_ROOT/unsloth/Qwen3.8-27B-GGUF}"
MODEL="${MODEL:-$MODEL_DIR/Qwen3.8-27B-UD-Q8_K_XL.gguf}"
MMPROJ="${MMPROJ:-$SHARED_MODEL_ROOT/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf}"
LLM_LAUNCHER="${LLM_LAUNCHER:-$SHARED_SCRIPT_ROOT/llama.cpp/run_qwen38_27b_llama_cpp.sh}"

# TODO(user): Keep false for a self-contained run whose service lifecycle is
# owned by this script. Set true only when an operator intentionally maintains
# compatible long-lived services on ports 8002 and 8014 outside this job.
REUSE_EXISTING_MODEL_SERVICES="${REUSE_EXISTING_MODEL_SERVICES:-false}"

# --- 3. Experiment cases -----------------------------------------------------
# TODO(user): CASE_SET = {new3, legacy8, sceneeval100, sceneeval500}. The default
# is the current SceneEval-100 hard protocol. SceneEval supports single_room rows.
CASE_SET="${CASE_SET:-sceneeval100}"
SCENE_SELECTION="${SCENE_SELECTION:-all}"
MAX_CASES="${MAX_CASES:-0}"
SCENEEVAL_SIZE="${SCENEEVAL_SIZE:-100}"
SCENEEVAL_ANNOTATIONS="${SCENEEVAL_ANNOTATIONS:-$PROJECT_ROOT/scripts/assets/annotations.csv}"
DIFFICULTY_SELECTION="${DIFFICULTY_SELECTION:-hard}"
# Seven workers match the validated Qwen3.8 p7 service profile. Use 1 only for
# strictly ordered cross-task memory diagnostics.
ACP_PARALLELISM="${ACP_PARALLELISM:-7}"
SCENEEXPERT_EXPERIMENT="${SCENEEXPERT_EXPERIMENT:-ablation_4c_qwen3_hybrid_memory}"

# --- 4. Independent wrapper feature switches --------------------------------
# TODO(user): These booleans control only harness/memory wrapper extensions.
# Main's critic, asset validation, designers, behavior and retry mechanisms are
# not disabled by these switches. Empty means inherit the selected experiment's
# preset; set an individual value to true/false only for that ablation axis.
SCENEEXPERT_COMPONENT_TASK_COMPILER_ENABLED="${SCENEEXPERT_COMPONENT_TASK_COMPILER_ENABLED:-}"
SCENEEXPERT_COMPONENT_HARNESS_ENABLED="${SCENEEXPERT_COMPONENT_HARNESS_ENABLED:-}"
SCENEEXPERT_COMPONENT_GLOBAL_PLANNER_ENABLED="${SCENEEXPERT_COMPONENT_GLOBAL_PLANNER_ENABLED:-}"
SCENEEXPERT_COMPONENT_PROMPT_INJECTION_ENABLED="${SCENEEXPERT_COMPONENT_PROMPT_INJECTION_ENABLED:-}"
SCENEEXPERT_COMPONENT_FAST_MEMORY_RETRIEVAL_ENABLED="${SCENEEXPERT_COMPONENT_FAST_MEMORY_RETRIEVAL_ENABLED:-}"
SCENEEXPERT_COMPONENT_MEMORY_WRITER_ENABLED="${SCENEEXPERT_COMPONENT_MEMORY_WRITER_ENABLED:-}"
SCENEEXPERT_COMPONENT_STAGE_WORKING_MEMORY_ENABLED="${SCENEEXPERT_COMPONENT_STAGE_WORKING_MEMORY_ENABLED:-}"
SCENEEXPERT_COMPONENT_VERIFIER_ENABLED="${SCENEEXPERT_COMPONENT_VERIFIER_ENABLED:-}"
SCENEEXPERT_COMPONENT_REPAIR_ENABLED="${SCENEEXPERT_COMPONENT_REPAIR_ENABLED:-false}"
SCENEEXPERT_COMPONENT_CRITIC_BRIDGE_ENABLED="${SCENEEXPERT_COMPONENT_CRITIC_BRIDGE_ENABLED:-}"
SCENEEXPERT_COMPONENT_TRACE_ENABLED="${SCENEEXPERT_COMPONENT_TRACE_ENABLED:-}"
SCENEEXPERT_COMPONENT_STRUCTURED_LLM_ENABLED="${SCENEEXPERT_COMPONENT_STRUCTURED_LLM_ENABLED:-}"
# 4c keeps observer-only Slow Memory capture off. The Full entrypoints set true.
SCENEEXPERT_COMPONENT_SLOW_MEMORY_CAPTURE_ENABLED="${SCENEEXPERT_COMPONENT_SLOW_MEMORY_CAPTURE_ENABLED:-false}"

# TODO(user): This is the real replacement for the obsolete
# SCENEEXPERT_EXECUTION_CONTROL_ENABLED variable. It limits only the wrapper's
# planner/repair budget and never changes SceneSmith agent iteration settings.
# Keep false unless budget control is the intended ablation variable.
SCENEEXPERT_COMPONENT_HARNESS_BUDGET_ENABLED="${SCENEEXPERT_COMPONENT_HARNESS_BUDGET_ENABLED:-false}"

# TODO(user): auto means "required minimum + autonomous optional design". Use
# required_only only for a strict ablation. Neither policy skips a native stage.
SCENEEXPERT_STAGE_POLICY_DEFAULT="${SCENEEXPERT_STAGE_POLICY_DEFAULT:-auto}"
SCENEEXPERT_STAGE_POLICY_FLOOR_PLAN="${SCENEEXPERT_STAGE_POLICY_FLOOR_PLAN:-}"
SCENEEXPERT_STAGE_POLICY_FURNITURE="${SCENEEXPERT_STAGE_POLICY_FURNITURE:-}"
SCENEEXPERT_STAGE_POLICY_WALL_MOUNTED="${SCENEEXPERT_STAGE_POLICY_WALL_MOUNTED:-}"
SCENEEXPERT_STAGE_POLICY_CEILING_MOUNTED="${SCENEEXPERT_STAGE_POLICY_CEILING_MOUNTED:-}"
SCENEEXPERT_STAGE_POLICY_MANIPULAND="${SCENEEXPERT_STAGE_POLICY_MANIPULAND:-}"

# --- 5. Memory experiment identity ------------------------------------------
# TODO(user): Use a new empty directory for a cold-start run; reuse exactly the
# same directory for the paired warm-memory run.
SCENEEXPERT_MEMORY_DIR="${SCENEEXPERT_MEMORY_DIR:-$PROJECT_ROOT/outputs/scene_expert_memory/sceneeval_qwen38_v1}"

# --- 6. Shared-base and pipeline policy -------------------------------------
# TODO(user): To reuse a prior floor-plan base, set GENERATE_SHARED_BASE=false
# and provide its absolute shared_base directory in REUSED_SHARED_BASE_ROOT.
GENERATE_SHARED_BASE="${GENERATE_SHARED_BASE:-true}"
BRANCH_FROM_SHARED_BASE="${BRANCH_FROM_SHARED_BASE:-true}"
REUSED_SHARED_BASE_ROOT="${REUSED_SHARED_BASE_ROOT:-}"
SHARED_BASE_STOP_STAGE="${SHARED_BASE_STOP_STAGE:-floor_plan}"
PIPELINE_STOP_STAGE="${PIPELINE_STOP_STAGE:-manipuland}"

# --- 7. Runtime resources and failure policy --------------------------------
# TODO(user): Increase parallelism only after checking GPU memory, CPU, ports
# and Blender capacity. One process is the conservative ACP default.
CRITIC_PROBE_INNER_PARALLELISM="${CRITIC_PROBE_INNER_PARALLELISM:-1}"
CRITIC_PROBE_MAX_SAFE_INNER_PARALLELISM="${CRITIC_PROBE_MAX_SAFE_INNER_PARALLELISM:-1}"
CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE="${CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE:-true}"
CRITIC_PROBE_RENDER_FINAL_VIEWS="${CRITIC_PROBE_RENDER_FINAL_VIEWS:-true}"
FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS="${FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS:-false}"
QUALITY_FAILURE_POLICY="${QUALITY_FAILURE_POLICY:-degraded}"

# TODO(user): Use main's Qwen3.8 reasoning profile unless this is a dedicated
# reasoning-effort ablation.
DESIGNER_THINKING="${DESIGNER_THINKING:-medium}"
CRITIC_THINKING="${CRITIC_THINKING:-medium}"

# TODO(user): Keep the critic team's rendered-HSSD selector enabled for the
# standard new3 run; change only for an explicit asset-selection ablation.
HSSD_RENDERED_ASSET_CHOICE="${HSSD_RENDERED_ASSET_CHOICE:-true}"
HSSD_RENDERED_ASSET_CHOICE_TOP_N="${HSSD_RENDERED_ASSET_CHOICE_TOP_N:-4}"

# =============================================================================
# END USER CONFIGURATION
# =============================================================================

RUN_CASE="${CASE_SET,,}"
case "$RUN_CASE" in
  sceneeval|scene-eval) RUN_CASE="sceneeval${SCENEEVAL_SIZE}" ;;
  sceneeval-100|sceneeval_100) RUN_CASE=sceneeval100 ;;
  sceneeval-500|sceneeval_500) RUN_CASE=sceneeval500 ;;
  old8) RUN_CASE=legacy8 ;;
esac
if [[ "$RUN_CASE" == sceneeval* ]]; then
  RUN_SCOPE="${RUN_CASE}_${DIFFICULTY_SELECTION//,/-}"
else
  RUN_SCOPE="$RUN_CASE"
fi
RUN_ID="${RUN_ID:-generate_4c_${RUN_SCOPE}_qwen38_p7_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/critic_probe/$RUN_ID}"
ACP_ENTRYPOINT="${ACP_ENTRYPOINT:-${BASH_SOURCE[0]}}"
ACP_LOG_ROOT="$PROJECT_ROOT/tmp/acp_logs"
ACP_LOG_DIR="$ACP_LOG_ROOT/$RUN_ID"
TERMINAL_LOG="$ACP_LOG_DIR/terminal.log"
EMBEDDING_LOG="$ACP_LOG_DIR/llama_embedding.log"
LLM_LOG="$ACP_LOG_DIR/llama_qwen38_27b.log"

if [[ ! -f "$CRITIC_HELPER_DIR/run_qwen3_vl_embedding_2b_llama_cpp.sh" \
    || ! -f "$CRITIC_HELPER_DIR/check_llama_cpp.sh" ]]; then
  CRITIC_HELPER_DIR="$FALLBACK_HELPER_DIR"
fi
EMBEDDING_LAUNCHER="$CRITIC_HELPER_DIR/run_qwen3_vl_embedding_2b_llama_cpp.sh"
READY_CHECKER="$CRITIC_HELPER_DIR/check_llama_cpp.sh"
MEMORY_EMBEDDING_MODEL_DIR="$SHARED_MODEL_ROOT/Memory/bge-m3"
SHARED_HSSD_ZVEC_ROOT="$SHARED_HSSD_AUX_ROOT/hssd_zvec_collection"
LOCAL_HSSD_ZVEC_ROOT="$PROJECT_ROOT/tmp/hssd_zvec_collection_qwen38"
LOCAL_HSSD_MIRROR_MARKER="$LOCAL_HSSD_ZVEC_ROOT/.mirror_complete"
LOCAL_HSSD_RUNTIME_ROOT="$PROJECT_ROOT/tmp/hssd_runtime_qwen38"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_dir() {
  [[ -d "$2" ]] || die "$1 directory does not exist: $2"
}

require_file() {
  [[ -f "$2" ]] || die "$1 file does not exist: $2"
}

require_bool() {
  case "${2,,}" in
    1|0|true|false|yes|no|on|off) ;;
    *) die "$1 must be a boolean, got '$2'" ;;
  esac
}

require_optional_bool() {
  [[ -z "$2" ]] || require_bool "$1" "$2"
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

[[ "$PROJECT_ROOT" == /* ]] || die "PROJECT_ROOT must be absolute: $PROJECT_ROOT"
[[ "$OUTPUT_ROOT" == /* ]] || die "OUTPUT_ROOT must be absolute: $OUTPUT_ROOT"
[[ "$SCENEEXPERT_MEMORY_DIR" == /* ]] || \
  die "SCENEEXPERT_MEMORY_DIR must be absolute: $SCENEEXPERT_MEMORY_DIR"
[[ "$RUN_ID" != */* && "$RUN_ID" != "." && "$RUN_ID" != ".." ]] || \
  die "RUN_ID must be one directory name: $RUN_ID"
require_dir PROJECT_ROOT "$PROJECT_ROOT"
require_dir TASK3_SHARED_ROOT "$TASK3_SHARED_ROOT"

case "$SCENEEXPERT_EXPERIMENT" in
  ablation_4c_qwen3_hybrid_memory)
    ACP_MODE=4c
    SCENEEXPERT_COMPONENT_SLOW_MEMORY_CAPTURE_ENABLED=false
    ;;
  ablation_5_qwen3_full)
    ACP_MODE=full
    SCENEEXPERT_COMPONENT_SLOW_MEMORY_CAPTURE_ENABLED=true
    ;;
  *)
    die "SCENEEXPERT_EXPERIMENT must be ablation_4c_qwen3_hybrid_memory or ablation_5_qwen3_full"
    ;;
esac

case "${CASE_SET,,}" in
  new3|legacy8) CASE_SET="${CASE_SET,,}" ;;
  old8) CASE_SET=legacy8 ;;
  sceneeval|scene-eval)
    [[ "$SCENEEVAL_SIZE" == 100 || "$SCENEEVAL_SIZE" == 500 ]] || \
      die "SCENEEVAL_SIZE must be 100 or 500, got '$SCENEEVAL_SIZE'"
    CASE_SET="sceneeval${SCENEEVAL_SIZE}"
    ;;
  sceneeval100|sceneeval-100|sceneeval_100) CASE_SET=sceneeval100; SCENEEVAL_SIZE=100 ;;
  sceneeval500|sceneeval-500|sceneeval_500) CASE_SET=sceneeval500; SCENEEVAL_SIZE=500 ;;
  *) die "CASE_SET must be new3, legacy8, sceneeval100, or sceneeval500; got '$CASE_SET'" ;;
esac
[[ -n "$SCENE_SELECTION" ]] || die "SCENE_SELECTION must not be empty"
[[ "$MAX_CASES" =~ ^[0-9]+$ ]] || die "MAX_CASES must be a non-negative integer"
[[ -z "$ACP_PARALLELISM" || "$ACP_PARALLELISM" =~ ^[1-9][0-9]*$ ]] || \
  die "ACP_PARALLELISM must be empty or a positive integer"
if [[ "$CASE_SET" == sceneeval* ]]; then
  require_file SCENEEVAL_ANNOTATIONS "$SCENEEVAL_ANNOTATIONS"
fi
[[ "$SHARED_BASE_STOP_STAGE" == "floor_plan" ]] || \
  die "normalized 4c flow requires SHARED_BASE_STOP_STAGE=floor_plan"
case "$PIPELINE_STOP_STAGE" in
  furniture|wall_mounted|ceiling_mounted|manipuland) ;;
  *) die "invalid PIPELINE_STOP_STAGE: $PIPELINE_STOP_STAGE" ;;
esac

for bool_name in \
  GENERATE_SHARED_BASE BRANCH_FROM_SHARED_BASE \
  REUSE_EXISTING_MODEL_SERVICES \
  HSSD_RENDERED_ASSET_CHOICE \
  CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE CRITIC_PROBE_RENDER_FINAL_VIEWS \
  FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS
do
  require_bool "$bool_name" "${!bool_name}"
done

for bool_name in \
  SCENEEXPERT_COMPONENT_TASK_COMPILER_ENABLED \
  SCENEEXPERT_COMPONENT_HARNESS_ENABLED \
  SCENEEXPERT_COMPONENT_HARNESS_BUDGET_ENABLED \
  SCENEEXPERT_COMPONENT_GLOBAL_PLANNER_ENABLED \
  SCENEEXPERT_COMPONENT_PROMPT_INJECTION_ENABLED \
  SCENEEXPERT_COMPONENT_FAST_MEMORY_RETRIEVAL_ENABLED \
  SCENEEXPERT_COMPONENT_MEMORY_WRITER_ENABLED \
  SCENEEXPERT_COMPONENT_STAGE_WORKING_MEMORY_ENABLED \
  SCENEEXPERT_COMPONENT_VERIFIER_ENABLED \
  SCENEEXPERT_COMPONENT_REPAIR_ENABLED \
  SCENEEXPERT_COMPONENT_CRITIC_BRIDGE_ENABLED \
  SCENEEXPERT_COMPONENT_TRACE_ENABLED \
  SCENEEXPERT_COMPONENT_STRUCTURED_LLM_ENABLED \
  SCENEEXPERT_COMPONENT_SLOW_MEMORY_CAPTURE_ENABLED
do
  require_optional_bool "$bool_name" "${!bool_name}"
done

case "$QUALITY_FAILURE_POLICY" in
  strict|degraded) ;;
  *) die "QUALITY_FAILURE_POLICY must be strict or degraded" ;;
esac

for policy_name in \
  SCENEEXPERT_STAGE_POLICY_DEFAULT \
  SCENEEXPERT_STAGE_POLICY_FLOOR_PLAN \
  SCENEEXPERT_STAGE_POLICY_FURNITURE \
  SCENEEXPERT_STAGE_POLICY_WALL_MOUNTED \
  SCENEEXPERT_STAGE_POLICY_CEILING_MOUNTED \
  SCENEEXPERT_STAGE_POLICY_MANIPULAND
do
  policy_value="${!policy_name}"
  case "$policy_value" in
    ""|auto|required_only) ;;
    *) die "$policy_name must be auto, required_only, or empty; got '$policy_value'" ;;
  esac
done

if is_true "$GENERATE_SHARED_BASE"; then
  SHARED_BASE_ROOT="$OUTPUT_ROOT/shared_base"
elif is_true "$BRANCH_FROM_SHARED_BASE"; then
  [[ -n "$REUSED_SHARED_BASE_ROOT" ]] || \
    die "REUSED_SHARED_BASE_ROOT is required when GENERATE_SHARED_BASE=false"
  [[ "$REUSED_SHARED_BASE_ROOT" == /* ]] || \
    die "REUSED_SHARED_BASE_ROOT must be absolute: $REUSED_SHARED_BASE_ROOT"
  require_dir REUSED_SHARED_BASE_ROOT "$REUSED_SHARED_BASE_ROOT"
  SHARED_BASE_ROOT="$(readlink -f "$REUSED_SHARED_BASE_ROOT")"
  require_file shared-base-case-set "$SHARED_BASE_ROOT/.critic_on_case_set"
  source_case_set="$(tr -d '[:space:]' < "$SHARED_BASE_ROOT/.critic_on_case_set")"
  [[ "$source_case_set" == "$CASE_SET" ]] || \
    die "shared base case set is '$source_case_set', requested '$CASE_SET'"
else
  SHARED_BASE_ROOT="${REUSED_SHARED_BASE_ROOT:-$OUTPUT_ROOT/shared_base}"
fi

mkdir -p "$ACP_LOG_DIR"
ln -sfn "$RUN_ID" "$ACP_LOG_ROOT/latest_${ACP_MODE}_qwen38"
exec > >(tee -a "$TERMINAL_LOG") 2>&1

echo "========== START ${ACP_MODE^^} QWEN3.8 ACP JOB =========="
echo "started_at=$(date --iso-8601=seconds)"
echo "run_id=$RUN_ID"
echo "project_root=$PROJECT_ROOT"
echo "experiment=$SCENEEXPERT_EXPERIMENT"
echo "model=$MODEL_NAME"
echo "case_set=$CASE_SET"
echo "scenes=$SCENE_SELECTION"
echo "output_root=$OUTPUT_ROOT"
echo "memory_root=$SCENEEXPERT_MEMORY_DIR"
echo "shared_base_root=$SHARED_BASE_ROOT"
echo "log_dir=$ACP_LOG_DIR"
echo "reuse_existing_model_services=$REUSE_EXISTING_MODEL_SERVICES"
echo "stage_policy_default=$SCENEEXPERT_STAGE_POLICY_DEFAULT"
echo "slow_memory_capture=$SCENEEXPERT_COMPONENT_SLOW_MEMORY_CAPTURE_ENABLED"

clashctl on 2>/dev/null || true

require_dir CRITIC_HELPER_DIR "$CRITIC_HELPER_DIR"
require_dir SHARED_HSSD_ROOT "$SHARED_HSSD_ROOT"
require_dir SHARED_HSSD_AUX_ROOT "$SHARED_HSSD_AUX_ROOT"
require_dir SHARED_MODEL_ROOT "$SHARED_MODEL_ROOT"
require_dir CUDA13_LIB_DIR "$CUDA13_LIB_DIR"
require_file EMBEDDING_LAUNCHER "$EMBEDDING_LAUNCHER"
require_file READY_CHECKER "$READY_CHECKER"
require_file LLM_LAUNCHER "$LLM_LAUNCHER"
require_file QWEN38_MODEL "$MODEL"
require_file MMPROJ "$MMPROJ"
require_file CUDA13_RUNTIME "$CUDA13_LIB_DIR/libcudart.so.13"
require_dir MEMORY_EMBEDDING_MODEL_DIR "$MEMORY_EMBEDDING_MODEL_DIR"
require_dir SCENEEXPERT_DATA_DIR "$SCENEEXPERT_DATA_DIR"

if [[ ! -f "$LOCAL_HSSD_MIRROR_MARKER" \
    || ! -f "$LOCAL_HSSD_ZVEC_ROOT/0/embedding.index.3.proxima" ]]; then
  echo "========== PREPARE LOCAL HSSD ZVEC MIRROR =========="
  require_dir SHARED_HSSD_ZVEC_ROOT "$SHARED_HSSD_ZVEC_ROOT/0"
  mkdir -p "$LOCAL_HSSD_ZVEC_ROOT"
  cp -R "$SHARED_HSSD_ZVEC_ROOT"/. "$LOCAL_HSSD_ZVEC_ROOT"/
  chmod -R u+rwX,go+rX "$LOCAL_HSSD_ZVEC_ROOT"
  require_file local-HSSD-index "$LOCAL_HSSD_ZVEC_ROOT/0/embedding.index.3.proxima"
  touch "$LOCAL_HSSD_MIRROR_MARKER"
fi

mkdir -p "$LOCAL_HSSD_RUNTIME_ROOT"
require_dir HSSD-models "$SHARED_HSSD_ROOT/hssd-models"
require_dir HSSD-preprocessed "$SHARED_HSSD_ROOT/preprocessed"
require_dir HSSD-rendered-assets "$SHARED_HSSD_AUX_ROOT/hssd_rendered_assets"
for runtime_path in hssd-models preprocessed hssd_rendered_assets hssd_zvec_collection; do
  if [[ -e "$LOCAL_HSSD_RUNTIME_ROOT/$runtime_path" \
      && ! -L "$LOCAL_HSSD_RUNTIME_ROOT/$runtime_path" ]]; then
    die "refusing to replace non-symlink runtime path: $LOCAL_HSSD_RUNTIME_ROOT/$runtime_path"
  fi
done
ln -sfnT "$SHARED_HSSD_ROOT/hssd-models" "$LOCAL_HSSD_RUNTIME_ROOT/hssd-models"
ln -sfnT "$SHARED_HSSD_ROOT/preprocessed" "$LOCAL_HSSD_RUNTIME_ROOT/preprocessed"
ln -sfnT "$SHARED_HSSD_AUX_ROOT/hssd_rendered_assets" \
  "$LOCAL_HSSD_RUNTIME_ROOT/hssd_rendered_assets"
ln -sfnT "$LOCAL_HSSD_ZVEC_ROOT" "$LOCAL_HSSD_RUNTIME_ROOT/hssd_zvec_collection"

mkdir -p "$SCENEEXPERT_MEMORY_DIR"
ln -sfn "$OUTPUT_ROOT" "$ACP_LOG_DIR/output_root"
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "run_id=$RUN_ID"
  echo "project_root=$PROJECT_ROOT"
  echo "entrypoint=$ACP_ENTRYPOINT"
  echo "mode=$ACP_MODE"
  echo "experiment=$SCENEEXPERT_EXPERIMENT"
  echo "model=$MODEL_NAME"
  echo "model_file=$MODEL"
  echo "mmproj_file=$MMPROJ"
  echo "case_set=$CASE_SET"
  echo "sceneeval_size=$SCENEEVAL_SIZE"
  echo "sceneeval_annotations=$SCENEEVAL_ANNOTATIONS"
  echo "difficulty=$DIFFICULTY_SELECTION"
  echo "scene_selection=$SCENE_SELECTION"
  echo "max_cases=$MAX_CASES"
  echo "acp_parallelism=${ACP_PARALLELISM:-environment_default}"
  echo "output_root=$OUTPUT_ROOT"
  echo "memory_root=$SCENEEXPERT_MEMORY_DIR"
  echo "shared_base_root=$SHARED_BASE_ROOT"
  echo "generate_shared_base=$GENERATE_SHARED_BASE"
  echo "slow_memory_capture=$SCENEEXPERT_COMPONENT_SLOW_MEMORY_CAPTURE_ENABLED"
  echo "harness_budget=$SCENEEXPERT_COMPONENT_HARNESS_BUDGET_ENABLED"
  echo "stage_policy_default=$SCENEEXPERT_STAGE_POLICY_DEFAULT"
  echo "stage_policy_floor_plan=${SCENEEXPERT_STAGE_POLICY_FLOOR_PLAN:-inherit}"
  echo "stage_policy_furniture=${SCENEEXPERT_STAGE_POLICY_FURNITURE:-inherit}"
  echo "stage_policy_wall_mounted=${SCENEEXPERT_STAGE_POLICY_WALL_MOUNTED:-inherit}"
  echo "stage_policy_ceiling_mounted=${SCENEEXPERT_STAGE_POLICY_CEILING_MOUNTED:-inherit}"
  echo "stage_policy_manipuland=${SCENEEXPERT_STAGE_POLICY_MANIPULAND:-inherit}"
  echo "reuse_existing_model_services=$REUSE_EXISTING_MODEL_SERVICES"
} > "$ACP_LOG_DIR/run_metadata.env"
cp "${BASH_SOURCE[0]}" "$ACP_LOG_DIR/runtime_script.sh"
if [[ "$ACP_ENTRYPOINT" != "${BASH_SOURCE[0]}" && -f "$ACP_ENTRYPOINT" ]]; then
  cp "$ACP_ENTRYPOINT" "$ACP_LOG_DIR/entrypoint_script.sh"
fi

EMBEDDING_PID=""
LLM_PID=""
RUNNER_PID=""
cleanup() {
  local exit_code="$?"
  trap - EXIT INT TERM
  for child_pid in "$RUNNER_PID" "$LLM_PID" "$EMBEDDING_PID"; do
    [[ -z "$child_pid" ]] || kill "$child_pid" 2>/dev/null || true
  done
  for child_pid in "$RUNNER_PID" "$LLM_PID" "$EMBEDDING_PID"; do
    [[ -z "$child_pid" ]] || wait "$child_pid" 2>/dev/null || true
  done
  {
    echo "exit_code=$exit_code"
    echo "finished_at=$(date --iso-8601=seconds)"
  } > "$ACP_LOG_DIR/exit_status.env"
  echo "========== FINISH ${ACP_MODE^^} QWEN3.8 ACP JOB =========="
  echo "exit_code=$exit_code"
  echo "terminal_log=$TERMINAL_LOG"
  echo "embedding_log=$EMBEDDING_LOG"
  echo "llm_log=$LLM_LOG"
  exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

EMBEDDING_PORT=8014
LLM_PORT=8002

port_accepts_connections() {
  local port="$1"
  timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$port" \
    >/dev/null 2>&1
}

show_port_owner() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :$port" >&2 || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >&2 || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser -v "${port}/tcp" >&2 || true
  fi
}

require_free_service_port() {
  local service_name="$1" port="$2"
  if ! port_accepts_connections "$port"; then
    return 0
  fi
  echo "ERROR: $service_name port $port is already accepting connections." >&2
  echo "       This ACP job did not start that listener and will not borrow it." >&2
  show_port_owner "$port"
  echo "       Stop the stale service manually, or set" >&2
  echo "       REUSE_EXISTING_MODEL_SERVICES=true only for an intentionally" >&2
  echo "       managed compatible service pair on ports 8002 and 8014." >&2
  exit 2
}

wait_for_http_health() {
  local service_name="$1" service_pid="$2" port="$3" log_path="$4"
  local timeout_seconds="${5:-7200}"
  local deadline=$((SECONDS + timeout_seconds))
  echo "Waiting for $service_name service on port $port"
  while ((SECONDS < deadline)); do
    if [[ -n "$service_pid" ]] && ! kill -0 "$service_pid" 2>/dev/null; then
      echo "ERROR: $service_name launcher PID $service_pid exited before readiness." >&2
      [[ -f "$log_path" ]] && tail -n 80 "$log_path" >&2
      exit 2
    fi
    if curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" \
      >/dev/null 2>&1; then
      echo "$service_name service is healthy on port $port"
      return 0
    fi
    sleep 5
  done
  echo "ERROR: timed out waiting for $service_name on port $port" >&2
  [[ -f "$log_path" ]] && tail -n 80 "$log_path" >&2
  exit 2
}

cd "$CRITIC_HELPER_DIR"
export LD_LIBRARY_PATH="$CUDA13_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
command -v curl >/dev/null 2>&1 || die "curl is required for service health checks"
command -v timeout >/dev/null 2>&1 || die "timeout is required for port preflight"
if is_true "$REUSE_EXISTING_MODEL_SERVICES"; then
  {
    echo "Reusing operator-managed embedding service on port $EMBEDDING_PORT"
    echo "This ACP job does not own or stop that process."
  } > "$EMBEDDING_LOG"
  {
    echo "Reusing operator-managed LLM service on port $LLM_PORT"
    echo "This ACP job does not own or stop that process."
  } > "$LLM_LOG"
else
  require_free_service_port embedding "$EMBEDDING_PORT"
  require_free_service_port LLM "$LLM_PORT"

  # MODEL/MODEL_DIR/MMPROJ are exported by some operator entrypoints for the
  # Qwen service.  The embedding helper uses the same generic variable names,
  # so inheriting them would make it load the 27B generator as an embedding
  # model.  Keep those overrides scoped to the LLM launcher below.
  nohup env -u MODEL -u MODEL_DIR -u MMPROJ -u ALIAS \
    HOST=0.0.0.0 PORT="$EMBEDDING_PORT" PARALLEL=7 THREADS_HTTP=7 \
    N_GPU_LAYERS=auto CTX_SIZE=12288 BATCH_SIZE=4096 UBATCH_SIZE=4096 \
    bash "$EMBEDDING_LAUNCHER" > "$EMBEDDING_LOG" 2>&1 &
  EMBEDDING_PID=$!

  nohup env \
    HOST=0.0.0.0 PORT="$LLM_PORT" CTX_SIZE=458752 PARALLEL=7 N_GPU_LAYERS=999 \
    CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0 VISION=true THREADS=16 THREADS_HTTP=7 \
    BATCH_SIZE=1024 UBATCH_SIZE=256 THINKING=true REASONING=auto \
    REASONING_PRESERVE=true TEMP=1.0 TOP_P=0.95 TOP_K=20 MIN_P=0.00 \
    PRESENCE_PENALTY=0.0 REPEAT_PENALTY=1.0 \
    MODEL_DIR="$MODEL_DIR" MODEL="$MODEL" MMPROJ="$MMPROJ" ALIAS="$MODEL_NAME" \
    bash "$LLM_LAUNCHER" \
    --flash-attn on --spec-type draft-mtp --spec-draft-n-max 3 \
    --no-kv-unified --cache-prompt --cache-ram 49152 --cache-idle-slots \
    --ctx-checkpoints 64 --slot-prompt-similarity 0.5 \
    > "$LLM_LOG" 2>&1 &
  LLM_PID=$!
fi

{
  echo "EMBEDDING_PID=$EMBEDDING_PID"
  echo "LLM_PID=$LLM_PID"
} > "$ACP_LOG_DIR/service_pids.env"

wait_for_http_health embedding "$EMBEDDING_PID" "$EMBEDDING_PORT" \
  "$EMBEDDING_LOG" 7200
wait_for_http_health LLM "$LLM_PID" "$LLM_PORT" "$LLM_LOG" 7200
PORT="$LLM_PORT" WAIT_TIMEOUT=60 EXPECTED_MODEL="$MODEL_NAME" bash "$READY_CHECKER"
if [[ -n "$LLM_PID" ]] && ! kill -0 "$LLM_PID" 2>/dev/null; then
  die "LLM launcher exited after readiness; see $LLM_LOG"
fi
if [[ -n "$EMBEDDING_PID" ]] && ! kill -0 "$EMBEDDING_PID" 2>/dev/null; then
  die "embedding launcher exited after readiness; see $EMBEDDING_LOG"
fi

cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$TASK3_SHARED_ROOT/L202500276_lwz/projects/Task3.2-main/.venv/bin/python"
fi
[[ -x "$PYTHON_BIN" ]] || die "no usable SceneSmith Python: $PYTHON_BIN"

RUNNER_ARGS=(
  --case-set "$CASE_SET"
  --scenes "$SCENE_SELECTION"
  --output-root "$OUTPUT_ROOT"
)
if [[ "$CASE_SET" == sceneeval* ]]; then
  RUNNER_ARGS+=(--difficulty "$DIFFICULTY_SELECTION")
fi
if [[ -n "$ACP_PARALLELISM" ]]; then
  RUNNER_ARGS+=(--parallelism "$ACP_PARALLELISM")
fi

env \
  PYTHON_BIN="$PYTHON_BIN" \
  SCENEEXPERT_EXPERIMENT="$SCENEEXPERT_EXPERIMENT" \
  MODEL_NAME="$MODEL_NAME" \
  FLOOR_PLAN_DESIGNER_THINKING="$DESIGNER_THINKING" FLOOR_PLAN_CRITIC_THINKING="$CRITIC_THINKING" \
  FURNITURE_DESIGNER_THINKING="$DESIGNER_THINKING" FURNITURE_CRITIC_THINKING="$CRITIC_THINKING" \
  WALL_DESIGNER_THINKING="$DESIGNER_THINKING" WALL_CRITIC_THINKING="$CRITIC_THINKING" \
  CEILING_DESIGNER_THINKING="$DESIGNER_THINKING" CEILING_CRITIC_THINKING="$CRITIC_THINKING" \
  MANIPULAND_DESIGNER_THINKING="$DESIGNER_THINKING" MANIPULAND_CRITIC_THINKING="$CRITIC_THINKING" \
  OPENAI_API_KEY=sk-123 OPENAI_BASE_URL="http://127.0.0.1:$LLM_PORT/v1" \
  OPENAI_USE_RESPONSES=false HF_HOME="$PROJECT_ROOT/.cache/huggingface" \
  SCENEEXPERT_DATA_DIR="$SCENEEXPERT_DATA_DIR" \
  SCENEEXPERT_HSSD_DATA_DIR="$LOCAL_HSSD_RUNTIME_ROOT" \
  HSSD_RETRIEVAL_BACKEND=embedding \
  HSSD_RENDERED_ASSET_CHOICE="$HSSD_RENDERED_ASSET_CHOICE" \
  HSSD_RENDERED_ASSET_CHOICE_TOP_N="$HSSD_RENDERED_ASSET_CHOICE_TOP_N" \
  HSSD_RENDERED_ASSETS_DIR="$LOCAL_HSSD_RUNTIME_ROOT/hssd_rendered_assets" \
  HSSD_ZVEC_COLLECTION_PATH="$LOCAL_HSSD_RUNTIME_ROOT/hssd_zvec_collection" \
  HSSD_EMBEDDING_BASE_URL="http://127.0.0.1:$EMBEDDING_PORT" \
  SCENEEXPERT_MEMORY_DIR="$SCENEEXPERT_MEMORY_DIR" \
  SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR="$MEMORY_EMBEDDING_MODEL_DIR" \
  SCENEEXPERT_MEMORY_EMBEDDING_DEVICE=cpu \
  SCENEEXPERT_MEMORY_EMBEDDING_INDEX_DEVICE=cpu \
  SCENEEXPERT_MEMORY_INDEX_AUTO_BUILD_MISSING=1 \
  SCENEEXPERT_MEMORY_INDEX_REQUIRE_READY=true \
  SCENEEXPERT_MEMORY_EMBEDDING_PREFLIGHT=true \
  SCENEEXPERT_COMPONENT_TASK_COMPILER_ENABLED="$SCENEEXPERT_COMPONENT_TASK_COMPILER_ENABLED" \
  SCENEEXPERT_COMPONENT_HARNESS_ENABLED="$SCENEEXPERT_COMPONENT_HARNESS_ENABLED" \
  SCENEEXPERT_COMPONENT_HARNESS_BUDGET_ENABLED="$SCENEEXPERT_COMPONENT_HARNESS_BUDGET_ENABLED" \
  SCENEEXPERT_COMPONENT_GLOBAL_PLANNER_ENABLED="$SCENEEXPERT_COMPONENT_GLOBAL_PLANNER_ENABLED" \
  SCENEEXPERT_COMPONENT_PROMPT_INJECTION_ENABLED="$SCENEEXPERT_COMPONENT_PROMPT_INJECTION_ENABLED" \
  SCENEEXPERT_COMPONENT_FAST_MEMORY_RETRIEVAL_ENABLED="$SCENEEXPERT_COMPONENT_FAST_MEMORY_RETRIEVAL_ENABLED" \
  SCENEEXPERT_COMPONENT_MEMORY_WRITER_ENABLED="$SCENEEXPERT_COMPONENT_MEMORY_WRITER_ENABLED" \
  SCENEEXPERT_COMPONENT_STAGE_WORKING_MEMORY_ENABLED="$SCENEEXPERT_COMPONENT_STAGE_WORKING_MEMORY_ENABLED" \
  SCENEEXPERT_COMPONENT_VERIFIER_ENABLED="$SCENEEXPERT_COMPONENT_VERIFIER_ENABLED" \
  SCENEEXPERT_COMPONENT_REPAIR_ENABLED="$SCENEEXPERT_COMPONENT_REPAIR_ENABLED" \
  SCENEEXPERT_COMPONENT_CRITIC_BRIDGE_ENABLED="$SCENEEXPERT_COMPONENT_CRITIC_BRIDGE_ENABLED" \
  SCENEEXPERT_COMPONENT_TRACE_ENABLED="$SCENEEXPERT_COMPONENT_TRACE_ENABLED" \
  SCENEEXPERT_COMPONENT_STRUCTURED_LLM_ENABLED="$SCENEEXPERT_COMPONENT_STRUCTURED_LLM_ENABLED" \
  SCENEEXPERT_COMPONENT_SLOW_MEMORY_CAPTURE_ENABLED="$SCENEEXPERT_COMPONENT_SLOW_MEMORY_CAPTURE_ENABLED" \
  SCENEEXPERT_STAGE_POLICY_DEFAULT="$SCENEEXPERT_STAGE_POLICY_DEFAULT" \
  SCENEEXPERT_STAGE_POLICY_FLOOR_PLAN="$SCENEEXPERT_STAGE_POLICY_FLOOR_PLAN" \
  SCENEEXPERT_STAGE_POLICY_FURNITURE="$SCENEEXPERT_STAGE_POLICY_FURNITURE" \
  SCENEEXPERT_STAGE_POLICY_WALL_MOUNTED="$SCENEEXPERT_STAGE_POLICY_WALL_MOUNTED" \
  SCENEEXPERT_STAGE_POLICY_CEILING_MOUNTED="$SCENEEXPERT_STAGE_POLICY_CEILING_MOUNTED" \
  SCENEEXPERT_STAGE_POLICY_MANIPULAND="$SCENEEXPERT_STAGE_POLICY_MANIPULAND" \
  SCENEEVAL_SIZE="$SCENEEVAL_SIZE" \
  SCENEEVAL_ANNOTATIONS="$SCENEEVAL_ANNOTATIONS" \
  DIFFICULTY_SELECTION="$DIFFICULTY_SELECTION" \
  SCENEEXPERT_DISABLE_ARTICULATED=1 SCENEEXPERT_DISABLE_MATERIALS=1 \
  SCENEEXPERT_DISABLE_BWRAP=1 SCENEEXPERT_CONVEX_MAX_OMP_THREADS=2 \
  CRITIC_PROBE_PARALLEL=true \
  CRITIC_PROBE_INNER_PARALLELISM="$CRITIC_PROBE_INNER_PARALLELISM" \
  CRITIC_PROBE_MAX_SAFE_INNER_PARALLELISM="$CRITIC_PROBE_MAX_SAFE_INNER_PARALLELISM" \
  CRITIC_PROBE_ALLOW_UNSAFE_PARALLELISM=false \
  CRITIC_PROBE_PORT_BASE=13000 CRITIC_PROBE_PORT_BLOCK_SIZE=400 \
  SCENE_BATCH_SIZE=1 SCENE_WORKERS_PER_PROCESS=1 \
  GENERATE_SHARED_BASE="$GENERATE_SHARED_BASE" \
  BRANCH_FROM_SHARED_BASE="$BRANCH_FROM_SHARED_BASE" \
  SHARED_BASE_STOP_STAGE="$SHARED_BASE_STOP_STAGE" \
  SHARED_BASE_ROOT="$SHARED_BASE_ROOT" \
  MAX_CASES="$MAX_CASES" \
  FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS="$FAIL_STAGE_ON_UNRESOLVED_HARD_CONSTRAINTS" \
  QUALITY_FAILURE_POLICY="$QUALITY_FAILURE_POLICY" \
  PIPELINE_STOP_STAGE="$PIPELINE_STOP_STAGE" \
  CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE="$CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE" \
  CRITIC_PROBE_RENDER_FINAL_VIEWS="$CRITIC_PROBE_RENDER_FINAL_VIEWS" \
  RUN_ID="$RUN_ID" OUTPUT_ROOT="$OUTPUT_ROOT" \
  bash scripts/run_parallel_critic_on.sh "${RUNNER_ARGS[@]}" &
RUNNER_PID=$!
echo "RUNNER_PID=$RUNNER_PID" >> "$ACP_LOG_DIR/service_pids.env"

# Stop the batch tree promptly if a service owned by this ACP job dies. Without
# this guard every in-flight OpenAI request performs its own retries and turns
# one service failure into many misleading per-scene failures.
while kill -0 "$RUNNER_PID" 2>/dev/null; do
  if [[ -n "$LLM_PID" ]] && ! kill -0 "$LLM_PID" 2>/dev/null; then
    echo "ERROR: owned LLM service exited while the scene runner was active." >&2
    tail -n 80 "$LLM_LOG" >&2 || true
    kill -TERM "$RUNNER_PID" 2>/dev/null || true
    wait "$RUNNER_PID" 2>/dev/null || true
    RUNNER_PID=""
    exit 2
  fi
  if [[ -n "$EMBEDDING_PID" ]] && ! kill -0 "$EMBEDDING_PID" 2>/dev/null; then
    echo "ERROR: owned embedding service exited while the scene runner was active." >&2
    tail -n 80 "$EMBEDDING_LOG" >&2 || true
    kill -TERM "$RUNNER_PID" 2>/dev/null || true
    wait "$RUNNER_PID" 2>/dev/null || true
    RUNNER_PID=""
    exit 2
  fi
  sleep 10
done

if wait "$RUNNER_PID"; then
  RUNNER_PID=""
  exit 0
else
  runner_exit_code=$?
  RUNNER_PID=""
  exit "$runner_exit_code"
fi
