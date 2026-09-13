#!/usr/bin/env bash
# Full-mode entrypoint for a new run. Scene generation is identical to 4c;
# observer-only Slow Memory capture is the sole additional runtime component.
# Dataset export and DPO training remain separate offline commands.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# USER CONFIGURATION — review every TODO(user) item before starting the run.
# =============================================================================

# TODO(user): Project directory containing the synchronized code to evaluate.
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2}"

# TODO(user): CASE_SET = {new3, legacy8, sceneeval100, sceneeval500}.
CASE_SET="${CASE_SET:-sceneeval100}"
SCENE_SELECTION="${SCENE_SELECTION:-all}"
MAX_CASES="${MAX_CASES:-0}"
SCENEEVAL_SIZE="${SCENEEVAL_SIZE:-100}"
DIFFICULTY_SELECTION="${DIFFICULTY_SELECTION:-hard}"
ACP_PARALLELISM="${ACP_PARALLELISM:-7}"

# TODO(user): Keep this root stable across paired Full runs that share one Fast
# Memory bank. Use a new empty root for a cold-memory sequence.
SCENEEXPERT_MEMORY_DIR="${SCENEEXPERT_MEMORY_DIR:-$PROJECT_ROOT/outputs/scene_expert_memory/sceneeval_full_qwen38_v1}"

# auto = prompt-explicit required minimum + autonomous optional asset design.
SCENEEXPERT_STAGE_POLICY_DEFAULT="${SCENEEXPERT_STAGE_POLICY_DEFAULT:-auto}"

BASE_ACP_SCRIPT="${BASE_ACP_SCRIPT:-$SCRIPT_DIR/acp_qwen38_4c_generate.sh}"

# =============================================================================
# END USER CONFIGURATION
# =============================================================================

case "${CASE_SET,,}" in
  sceneeval|scene-eval) RUN_CASE="sceneeval${SCENEEVAL_SIZE}" ;;
  sceneeval100|sceneeval-100|sceneeval_100) RUN_CASE=sceneeval100 ;;
  sceneeval500|sceneeval-500|sceneeval_500) RUN_CASE=sceneeval500 ;;
  new3|legacy8|old8) RUN_CASE="${CASE_SET,,}" ;;
  *) echo "ERROR: invalid CASE_SET: $CASE_SET" >&2; exit 2 ;;
esac

DIFFICULTY_LABEL="${DIFFICULTY_SELECTION//,/-}"
RUN_ID="${RUN_ID:-generate_full_${RUN_CASE}_${DIFFICULTY_LABEL}_qwen38_p7_$(date +%Y%m%d_%H%M%S)}"

[[ -f "$BASE_ACP_SCRIPT" ]] || {
  echo "ERROR: canonical ACP runtime not found: $BASE_ACP_SCRIPT" >&2
  exit 2
}

exec env \
  PROJECT_ROOT="$PROJECT_ROOT" \
  RUN_ID="$RUN_ID" \
  CASE_SET="$CASE_SET" \
  SCENE_SELECTION="$SCENE_SELECTION" \
  MAX_CASES="$MAX_CASES" \
  SCENEEVAL_SIZE="$SCENEEVAL_SIZE" \
  DIFFICULTY_SELECTION="$DIFFICULTY_SELECTION" \
  ACP_PARALLELISM="$ACP_PARALLELISM" \
  SCENEEXPERT_MEMORY_DIR="$SCENEEXPERT_MEMORY_DIR" \
  SCENEEXPERT_EXPERIMENT=ablation_5_qwen3_full \
  SCENEEXPERT_STAGE_POLICY_DEFAULT="$SCENEEXPERT_STAGE_POLICY_DEFAULT" \
  SCENEEXPERT_COMPONENT_HARNESS_BUDGET_ENABLED=false \
  SCENEEXPERT_COMPONENT_REPAIR_ENABLED=false \
  SCENEEXPERT_COMPONENT_SLOW_MEMORY_CAPTURE_ENABLED=true \
  SCENEEXPERT_SLOW_MEMORY_MAX_RESPONSE_CHARS=1048576 \
  GENERATE_SHARED_BASE=true \
  BRANCH_FROM_SHARED_BASE=true \
  ACP_ENTRYPOINT="${ACP_ENTRYPOINT:-${BASH_SOURCE[0]}}" \
  bash "$BASE_ACP_SCRIPT"
