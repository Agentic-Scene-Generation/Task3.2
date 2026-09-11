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

# This batch is deliberately Qwen3.8-only. Generic repository fallbacks remain
# untouched; a mismatched alias is a collection error rather than silent drift.
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.8-27B-GGUF}"
EXPECTED_MODEL_NAME="unsloth/Qwen3.8-27B-GGUF"
MODEL="${MODEL:-Qwen3.8-27B-UD-Q8_K_XL.gguf}"
MMPROJ="${MMPROJ:-mmproj-F16.gguf}"

# Begin with a bounded ordered batch. Set MAX_CASES=0 only after the first
# export has been reviewed. Ordered scenes keep Writer updates causal: a memory
# update from a completed scene may affect a later scene, never a sibling run.
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
DPO_PROBE_DIR="${DPO_PROBE_DIR:-$COLLECTION_ROOT/dpo_probe}"

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

mkdir -p "$COLLECTION_ROOT"
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

echo "Starting Qwen3.8 Slow Memory recollection: $RUN_ID"
echo "Collection root: $COLLECTION_ROOT"

env \
  PROJECT_ROOT="$PROJECT_ROOT" \
  RUN_ID="$RUN_ID" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  MODEL_NAME="$MODEL_NAME" \
  MODEL="$MODEL" \
  MMPROJ="$MMPROJ" \
  CASE_SET="$CASE_SET" \
  SCENE_SELECTION="$SCENE_SELECTION" \
  MAX_CASES="$MAX_CASES" \
  SCENEEVAL_SIZE="$SCENEEVAL_SIZE" \
  DIFFICULTY_SELECTION="$DIFFICULTY_SELECTION" \
  ACP_PARALLELISM="$ACP_PARALLELISM" \
  SCENEEXPERT_MEMORY_DIR="$SCENEEXPERT_MEMORY_DIR" \
  SCENEEXPERT_COMPONENT_MEMORY_WRITER_ENABLED=true \
  SCENEEXPERT_COMPONENT_SLOW_MEMORY_CAPTURE_ENABLED=true \
  bash "$FULL_LAUNCHER"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || die "No collection Python at $PYTHON_BIN"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/export_sceneexpert_dpo.py" \
  --trajectory-source "$OUTPUT_ROOT" \
  --output-dir "$DPO_PROBE_DIR" \
  --allow-empty

echo "Recollection completed. Inspect: $DPO_PROBE_DIR/manifest.json"
echo "This probe is not a training dataset unless it contains real exact-context pairs."
