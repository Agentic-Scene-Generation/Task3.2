#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
TRAIN_PYTHON="${TRAIN_PYTHON:-$PROJECT_ROOT/.venv_dpo/bin/python}"
CAMPAIGN_ID="${CAMPAIGN_ID:-qwen38_sceneeval_dpo_019}"
RUN_ID="${RUN_ID:-qwen38_dpo_initial_020}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ && "$CAMPAIGN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
BASE_MODEL="${BASE_MODEL:-/mnt/afs/task3_2/share_model/Qwen/Qwen3.8-27B}"
[[ -f "$BASE_MODEL/config.json" ]] || { echo "Missing local base model: $BASE_MODEL" >&2; exit 2; }
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT/outputs/slow_memory/$CAMPAIGN_ID/datasets/verified_relative_v1}"
OUTPUT_DIR="$PROJECT_ROOT/outputs/slow_memory/$RUN_ID"
PROFILE="${PROFILE:-furniture_initial}"
[[ -x "$TRAIN_PYTHON" ]] || { echo 'Create the isolated environment using requirements_train.txt first.' >&2; exit 2; }
resume_args=()
mkdir -p "$(dirname "$OUTPUT_DIR")"
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  [[ -d "$RESUME_CHECKPOINT" ]] || { echo 'Resume checkpoint does not exist.' >&2; exit 2; }
  resume_args=(--resume-from-checkpoint "$RESUME_CHECKPOINT")
  mkdir -p "$OUTPUT_DIR"
else
  mkdir "$OUTPUT_DIR" || { echo 'Use a fresh RUN_ID or an explicit RESUME_CHECKPOINT.' >&2; exit 2; }
fi
finish() { local code=$?; printf 'exit_code=%s\n' "$code" > "$OUTPUT_DIR/exit_status.env"; }
trap finish EXIT
"$TRAIN_PYTHON" scripts/train_sceneexpert_dpo.py \
  --model "$BASE_MODEL" --dataset-dir "$DATASET_DIR" --output-dir "$OUTPUT_DIR" \
  --profile "$PROFILE" --dry-run 2>&1 | tee "$OUTPUT_DIR/preflight.log"
"$TRAIN_PYTHON" scripts/train_sceneexpert_dpo.py \
  --model "$BASE_MODEL" --dataset-dir "$DATASET_DIR" --output-dir "$OUTPUT_DIR" \
  --profile "$PROFILE" "${resume_args[@]}" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
