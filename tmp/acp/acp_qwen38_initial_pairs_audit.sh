#!/usr/bin/env bash
# Audit retained server evidence into a new run; no generation or rescoring.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_ID="${RUN_ID:?Set a new audit RUN_ID}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:?Set the retained collection SOURCE_RUN_ID}"
EXPECTED_GROUPS="${EXPECTED_GROUPS:-4}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ && "$SOURCE_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run ID' >&2; exit 2; }
[[ "$RUN_ID" != "$SOURCE_RUN_ID" ]] || { echo 'Use a separate audit RUN_ID' >&2; exit 2; }
[[ "$EXPECTED_GROUPS" =~ ^[1-4]$ ]] || { echo 'EXPECTED_GROUPS must be 1..4' >&2; exit 2; }
SOURCE_ROOT="$PROJECT_ROOT/outputs/slow_memory/$SOURCE_RUN_ID"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/slow_memory/$RUN_ID"
LOG_DIR="$PROJECT_ROOT/tmp/acp_logs/$RUN_ID"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "No audit Python: $PYTHON_BIN" >&2; exit 2; }
[[ -d "$SOURCE_ROOT/runs/paired_initial" ]] || { echo 'Source pair directory is missing' >&2; exit 2; }
[[ ! -e "$OUTPUT_ROOT" && ! -e "$LOG_DIR" ]] || { echo 'Audit run already exists; choose a new RUN_ID' >&2; exit 2; }
# Use the existing collection lock without truncating or creating source files.
[[ -f "$SOURCE_ROOT/runs.lock" ]] || { echo 'Source collection lock is missing; verify the source path' >&2; exit 2; }
command -v flock >/dev/null || { echo 'flock is required for an inactive-source audit' >&2; exit 2; }
exec 9<"$SOURCE_ROOT/runs.lock"
flock -n 9 || { echo 'Source collection is still running; wait for it to finish' >&2; exit 2; }
mkdir -p "$LOG_DIR"
trap 'code=$?; printf "exit_code=%s\nfinished_at=%s\n" "$code" "$(date --iso-8601=seconds)" > "$LOG_DIR/exit_status.env"' EXIT
export SCENEEXPERT_CODE_PROVENANCE_GIT_ENABLED=false
cp "${BASH_SOURCE[0]}" "$LOG_DIR/entrypoint_script.sh"
printf '%s\n' "run_id=$RUN_ID" "source_run_id=$SOURCE_RUN_ID" \
  "source_pair_root=$SOURCE_ROOT/runs/paired_initial" "expected_groups=$EXPECTED_GROUPS" \
  'mode=retained_evidence_audit' 'model_calls=0' 'native_rescoring=false' \
  > "$LOG_DIR/run_metadata.env"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_sceneexpert_initial_pairs.py" \
  --audit-root "$SOURCE_ROOT/runs/paired_initial" \
  --audit-output "$OUTPUT_ROOT/runs/paired_initial" \
  --expected-groups "$EXPECTED_GROUPS" --min-pairs 1 \
  2>&1 | tee "$LOG_DIR/pair_audit.log"
