#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_ID="${RUN_ID:-qwen38_sceneeval_dpo_019}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
COLLECTION_ROOT="$PROJECT_ROOT/outputs/slow_memory/$RUN_ID"
export SCENEEXPERT_CODE_PROVENANCE_GIT_ENABLED=false
"$PYTHON_BIN" scripts/prepare_sceneexpert_campaign.py \
  --output "$COLLECTION_ROOT/campaign.json" \
  --train "${TRAIN_TASKS:-128}" --validation "${VALIDATION_TASKS:-24}" --test "${TEST_TASKS:-32}"
"$PYTHON_BIN" scripts/run_sceneexpert_campaign.py --root "$COLLECTION_ROOT" \
  --action "${CAMPAIGN_ACTION:-collect}" --parallelism "${ACP_PARALLELISM:-2}" \
  --max-tasks "${MAX_TASKS:-0}" --chunk-size "${CHUNK_SIZE:-12}"
