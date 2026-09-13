#!/usr/bin/env bash
# Qwen3.8-only Slow Memory recollection entrypoint.
#
# This starts a fresh, ordered Full-mode trajectory collection and immediately
# exports an auditable DPO probe. It intentionally does not claim to be the
# future decision-level paired-candidate collector: existing Full capture has
# one native Designer execution per decision. The export is therefore a quality
# and pairing-yield diagnostic, not a training launch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ACP-local project mount. Override only when the operator's mount differs.
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2}"
FULL_LAUNCHER="${FULL_LAUNCHER:-$SCRIPT_DIR/acp_qwen38_full_generate.sh}"
TASK3_SHARED_ROOT="${TASK3_SHARED_ROOT:-/mnt/afs/task3_2}"

# This batch is deliberately Qwen3.8-only. Generic repository fallbacks remain
# untouched; a mismatched alias is a collection error rather than silent drift.
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.8-27B-GGUF}"
EXPECTED_MODEL_NAME="unsloth/Qwen3.8-27B-GGUF"
if [[ -z "${QWEN38_MODEL_ROOT:-}" ]]; then
  if [[ -f "$TASK3_SHARED_ROOT/share_model/Qwen/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf" ]]; then
    QWEN38_MODEL_ROOT="$TASK3_SHARED_ROOT/share_model/Qwen/Qwen3.8-27B-GGUF"
  else
    QWEN38_MODEL_ROOT="$TASK3_SHARED_ROOT/share_model/unsloth/Qwen3.8-27B-GGUF"
  fi
fi
MODEL_DIR="${MODEL_DIR:-$QWEN38_MODEL_ROOT}"
MODEL="${MODEL:-$MODEL_DIR/Qwen3.8-27B-UD-Q8_K_XL.gguf}"
MMPROJ="${MMPROJ:-$MODEL_DIR/mmproj-F16.gguf}"

# Begin with a bounded ordered batch. Set MAX_CASES=0 only after the first
# collection audit has been reviewed. Ordered scenes keep Writer updates causal:
# an update from a completed scene may affect a later scene, never a sibling run.
CASE_SET="${CASE_SET:-sceneeval100}"
SCENE_SELECTION="${SCENE_SELECTION:-all}"
MAX_CASES="${MAX_CASES:-12}"
SCENEEVAL_SIZE="${SCENEEVAL_SIZE:-100}"
DIFFICULTY_SELECTION="${DIFFICULTY_SELECTION:-hard}"
ACP_PARALLELISM="${ACP_PARALLELISM:-1}"
RUN_ID="${RUN_ID:-recollect_qwen38_${CASE_SET}_$(date +%Y%m%d_%H%M%S)}"

# Never reuse an old data batch or an old live bank by default.
COLLECTION_ROOT="${COLLECTION_ROOT:-$PROJECT_ROOT/outputs/slow_memory/$RUN_ID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$COLLECTION_ROOT/runs}"
SCENEEXPERT_MEMORY_DIR="${SCENEEXPERT_MEMORY_DIR:-$COLLECTION_ROOT/memory}"
# Keep review artifacts inside OUTPUT_ROOT so the canonical ACP log bundle's
# output_root link includes them when the operator archives the run.
COLLECTION_REVIEW_DIR="$OUTPUT_ROOT/collection"
MIN_DPO_PAIRS="${MIN_DPO_PAIRS:-0}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ "$MODEL_NAME" == "$EXPECTED_MODEL_NAME" ]] || die \
  "MODEL_NAME must be $EXPECTED_MODEL_NAME for this collection, got $MODEL_NAME"
[[ -f "$FULL_LAUNCHER" ]] || die "Full ACP launcher not found: $FULL_LAUNCHER"
[[ -d "$PROJECT_ROOT" ]] || die "PROJECT_ROOT does not exist: $PROJECT_ROOT"
[[ "$ACP_PARALLELISM" == "1" ]] || die \
  "ACP_PARALLELISM must be 1: MemoryWriter is intentionally enabled between scenes"
[[ ! -e "$COLLECTION_ROOT" ]] || die \
  "COLLECTION_ROOT already exists; choose a new RUN_ID or COLLECTION_ROOT"
[[ ! -e "$OUTPUT_ROOT" ]] || die \
  "OUTPUT_ROOT already exists; choose a fresh output directory"
[[ "$MIN_DPO_PAIRS" =~ ^[0-9]+$ ]] || die "MIN_DPO_PAIRS must be nonnegative"
[[ "$OUTPUT_ROOT" == /* ]] || die "OUTPUT_ROOT must be absolute: $OUTPUT_ROOT"

# Fail before paying for generation if the offline collection tools cannot run.
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$TASK3_SHARED_ROOT/L202500276_lwz/projects/Task3.2-main/.venv/bin/python"
fi
[[ -x "$PYTHON_BIN" ]] || die "No collection Python at $PYTHON_BIN"
(cd "$PROJECT_ROOT" && "$PYTHON_BIN" -c \
  'from scenesmith.scene_expert.slow_memory.collection import audit_collection')

mkdir -p "$COLLECTION_ROOT" "$COLLECTION_REVIEW_DIR"
collection_exit() {
  local code=$?
  printf '%s\n' "exit_code=$code" "min_dpo_pairs=$MIN_DPO_PAIRS" \
    "finished_at=$(date --iso-8601=seconds)" \
    > "$COLLECTION_REVIEW_DIR/collection_exit_status.env"
}
trap collection_exit EXIT
printf '%s\n' \
  "collection_kind=qwen38_full_observer_recollection" \
  "pair_collection_status=not_implemented" \
  "model_name=$MODEL_NAME" \
  "model_file=$MODEL" \
  "mmproj_file=$MMPROJ" \
  "case_set=$CASE_SET" \
  "scene_selection=$SCENE_SELECTION" \
  "max_cases=$MAX_CASES" \
  "acp_parallelism=$ACP_PARALLELISM" \
  "memory_writer=enabled_between_completed_scenes" \
  "slow_memory_capture=enabled" \
  > "$COLLECTION_ROOT/collection_manifest.env"
cp "$COLLECTION_ROOT/collection_manifest.env" "$COLLECTION_REVIEW_DIR/collection_manifest.env"
cp "${BASH_SOURCE[0]}" "$COLLECTION_REVIEW_DIR/collection_entrypoint.sh"

echo "Starting Qwen3.8 Slow Memory recollection: $RUN_ID"
echo "Collection root: $COLLECTION_ROOT"

generation_exit=0
env \
  PROJECT_ROOT="$PROJECT_ROOT" \
  TASK3_SHARED_ROOT="$TASK3_SHARED_ROOT" \
  RUN_ID="$RUN_ID" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  MODEL_NAME="$MODEL_NAME" \
  MODEL_DIR="$MODEL_DIR" \
  MODEL="$MODEL" \
  MMPROJ="$MMPROJ" \
  CASE_SET="$CASE_SET" \
  SCENE_SELECTION="$SCENE_SELECTION" \
  MAX_CASES="$MAX_CASES" \
  SCENEEVAL_SIZE="$SCENEEVAL_SIZE" \
  DIFFICULTY_SELECTION="$DIFFICULTY_SELECTION" \
  ACP_PARALLELISM="$ACP_PARALLELISM" \
  PYTHON_BIN="$PYTHON_BIN" \
  ACP_ENTRYPOINT="${BASH_SOURCE[0]}" \
  SCENEEXPERT_MEMORY_DIR="$SCENEEXPERT_MEMORY_DIR" \
  SCENEEXPERT_COMPONENT_MEMORY_WRITER_ENABLED=true \
  SCENEEXPERT_COMPONENT_SLOW_MEMORY_CAPTURE_ENABLED=true \
  bash "$FULL_LAUNCHER" || generation_exit=$?

audit_exit=0
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/audit_sceneexpert_collection.py" \
  --run-root "$OUTPUT_ROOT" \
  --output-dir "$COLLECTION_REVIEW_DIR" \
  --expected-model "$EXPECTED_MODEL_NAME" \
  --min-pairs "$MIN_DPO_PAIRS" || audit_exit=$?

echo "Collection artifacts: $COLLECTION_REVIEW_DIR"
echo "Generation exit: $generation_exit; collection audit exit: $audit_exit"
echo "MIN_DPO_PAIRS=0 validates observer collection only; inspect dpo_export_ready separately."
if [[ "$generation_exit" != "0" ]]; then
  exit "$generation_exit"
fi
exit "$audit_exit"
