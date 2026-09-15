#!/usr/bin/env bash
# Refresh deterministic evidence from retained raw snapshots; no LLM services.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_ID="${RUN_ID:-qwen38_initial_pairs_008_rescore_010}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-qwen38_initial_pairs_008}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ && "$SOURCE_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run ID' >&2; exit 2; }
[[ "$RUN_ID" != "$SOURCE_RUN_ID" ]] || { echo 'Use a new output RUN_ID' >&2; exit 2; }
COLLECTION_ROOT="$PROJECT_ROOT/outputs/slow_memory/$RUN_ID"
[[ ! -e "$COLLECTION_ROOT" ]] || { echo 'Output already exists; use a new RUN_ID' >&2; exit 2; }
SOURCE_PAIR_ROOT="${SOURCE_PAIR_ROOT:-$PROJECT_ROOT/outputs/slow_memory/$SOURCE_RUN_ID/runs/paired_initial}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=/mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-main/.venv/bin/python
fi
[[ -x "$PYTHON_BIN" ]] || { echo 'Use the scene runtime Python with Drake installed' >&2; exit 2; }
LOG_DIR="${ACP_LOG_ROOT:-$PROJECT_ROOT/tmp/acp_logs}/$RUN_ID"
mkdir -p "$LOG_DIR"
finish() {
  local code=$?
  printf '%s\n' "exit_code=$code" "finished_at=$(date -Is)" > "$LOG_DIR/exit_status.env"
}
trap finish EXIT
export ACP_ENTRYPOINT="${BASH_SOURCE[0]}"
export SCENEEXPERT_CODE_PROVENANCE_GIT_ENABLED=false
cp "${BASH_SOURCE[0]}" "$LOG_DIR/entrypoint_script.sh"
printf '%s\n' "run_id=$RUN_ID" "source_pair_root=$SOURCE_PAIR_ROOT" \
  'mode=raw_candidate_rescore' 'model_calls=0' > "$LOG_DIR/run_metadata.env"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_sceneexpert_initial_pairs.py" \
  --rescore-source "$SOURCE_PAIR_ROOT" \
  --rescore-output "$COLLECTION_ROOT/runs/paired_initial" \
  2>&1 | tee "$LOG_DIR/rescore.log"
