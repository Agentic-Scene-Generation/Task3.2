#!/usr/bin/env bash
# Read-only policy diagnostics; never starts training or scene tools.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_ID="${RUN_ID:-qwen38_dpo_policy_audit_024}"
TRAIN_RUN_ID="${TRAIN_RUN_ID:-qwen38_dpo_pilot_021}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ && "$TRAIN_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv_dpo/bin/python}"
OUTPUT="$PROJECT_ROOT/outputs/slow_memory/$RUN_ID"
[[ ! -e "$OUTPUT" ]] || { echo "RUN_ID=$RUN_ID already exists at $OUTPUT; choose a new RUN_ID to preserve recorded results" >&2; exit 2; }
export SCENEEXPERT_CODE_PROVENANCE_GIT_ENABLED=false PYTHONUNBUFFERED=1
LOG_ROOT="$PROJECT_ROOT/tmp/acp_logs/$RUN_ID"
mkdir -p "$LOG_ROOT"
diagnostic_exit=0
"$PYTHON_BIN" -u scripts/diagnose_sceneexpert_dpo.py \
  --training-run "$PROJECT_ROOT/outputs/slow_memory/$TRAIN_RUN_ID" \
  --output-dir "$OUTPUT" --max-pairs "${MAX_PAIRS:-0}" \
  2>&1 | tee "$LOG_ROOT/diagnostic.log" || diagnostic_exit=$?
mkdir -p "$OUTPUT"
printf 'diagnostic_exit_code=%s\n' "$diagnostic_exit" > "$OUTPUT/exit_status.env"
PACKAGE_PATH="$PROJECT_ROOT/tmp/results/slow_memory/${RUN_ID}_review_$(date -u +%Y%m%dT%H%M%SZ)_$$.tar.gz"
package_exit=0
"$PYTHON_BIN" scripts/package_sceneexpert_results.py --project-root "$PROJECT_ROOT" \
  --run-id "$RUN_ID" --output "$PACKAGE_PATH" \
  --max-file-mib "${PACKAGE_MAX_FILE_MIB:-8}" --max-total-mib "${PACKAGE_MAX_TOTAL_MIB:-32}" \
  > "$PACKAGE_PATH.packaging.log" 2>&1 || package_exit=$?
if [[ "$package_exit" == 0 ]]; then
  "$PYTHON_BIN" scripts/package_sceneexpert_results.py --verify "$PACKAGE_PATH" \
    >> "$PACKAGE_PATH.packaging.log" 2>&1 || package_exit=$?
fi
printf 'diagnostic_exit_code=%s\npackage_exit_code=%s\npackage_path=%s\n' \
  "$diagnostic_exit" "$package_exit" "$PACKAGE_PATH" > "$LOG_ROOT/workflow_exit_status.env"
cat "$LOG_ROOT/workflow_exit_status.env"
if [[ "$diagnostic_exit" != 0 ]]; then
  echo "Diagnostic returned $diagnostic_exit; inspect $OUTPUT/policy_diagnostics.json for evaluation completion and reproduction failure, or $LOG_ROOT/diagnostic.log for runtime errors" >&2
fi
[[ "$diagnostic_exit" == 0 ]] || exit "$diagnostic_exit"
exit "$package_exit"
