#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_ID="${RUN_ID:-qwen38_sceneeval_dpo_019}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
COLLECTION_ROOT="$PROJECT_ROOT/outputs/slow_memory/$RUN_ID"
export SCENEEXPERT_CODE_PROVENANCE_GIT_ENABLED=false
INVOCATION="$(date -u +%Y%m%dT%H%M%SZ)_$$"
CONTROL_DIR="$PROJECT_ROOT/tmp/acp_logs/$RUN_ID/$INVOCATION"
mkdir -p "$CONTROL_DIR"
collection_exit=0
if "$PYTHON_BIN" scripts/prepare_sceneexpert_campaign.py \
  --output "$COLLECTION_ROOT/campaign.json" \
  --train "${TRAIN_TASKS:-128}" --validation "${VALIDATION_TASKS:-24}" --test "${TEST_TASKS:-32}" \
  > "$CONTROL_DIR/preparation.log" 2>&1; then
  if "$PYTHON_BIN" scripts/run_sceneexpert_campaign.py --root "$COLLECTION_ROOT" \
    --action "${CAMPAIGN_ACTION:-collect}" --parallelism "${ACP_PARALLELISM:-2}" \
    --max-tasks "${MAX_TASKS:-0}" --chunk-size "${CHUNK_SIZE:-12}" \
    2>&1 | tee "$CONTROL_DIR/collection.log"; then
    collection_exit=0
  else
    collection_exit=$?
  fi
else
  collection_exit=$?
  cat "$CONTROL_DIR/preparation.log" >&2
fi
printf 'collection_exit_code=%s\n' "$collection_exit" > "$CONTROL_DIR/collection_exit_status.env"

# Package after the collector closes its logs, including failed-run diagnostics.
# Packaging's own output stays outside input roots to avoid archiving a live log.
mkdir -p "$PROJECT_ROOT/tmp/results/slow_memory"
PACKAGE_PATH="${PACKAGE_PATH:-$PROJECT_ROOT/tmp/results/slow_memory/${RUN_ID}_review_${INVOCATION}.tar.gz}"
mkdir -p "$(dirname "$PACKAGE_PATH")"
package_exit=0
if "$PYTHON_BIN" scripts/package_sceneexpert_results.py \
  --project-root "$PROJECT_ROOT" --run-id "$RUN_ID" --output "$PACKAGE_PATH" \
  --max-file-mib "${PACKAGE_MAX_FILE_MIB:-32}" \
  --max-total-mib "${PACKAGE_MAX_TOTAL_MIB:-512}" \
  > "$PACKAGE_PATH.packaging.log" 2>&1; then
  if "$PYTHON_BIN" scripts/package_sceneexpert_results.py --verify "$PACKAGE_PATH" \
    >> "$PACKAGE_PATH.packaging.log" 2>&1; then
    echo "Verified review package: $PACKAGE_PATH"
  else
    package_exit=$?
  fi
else
  package_exit=$?
fi
printf 'collection_exit_code=%s\npackage_exit_code=%s\npackage_path=%s\n' \
  "$collection_exit" "$package_exit" "$PACKAGE_PATH" > "$CONTROL_DIR/workflow_exit_status.env"
if [[ "$package_exit" != 0 ]]; then
  cat "$PACKAGE_PATH.packaging.log" >&2
fi
[[ "$collection_exit" == 0 ]] || exit "$collection_exit"
exit "$package_exit"
