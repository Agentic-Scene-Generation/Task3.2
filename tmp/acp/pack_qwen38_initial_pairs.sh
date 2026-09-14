#!/usr/bin/env bash
# Read-only review packaging: no GPU/model startup, source mutation, or Git.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_ID="${RUN_ID:?Set RUN_ID to the experiment to package}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
[[ -x "$PYTHON_BIN" ]] || { echo 'Python 3.11+ is required for packaging' >&2; exit 2; }
PACKAGE_PATH="${PACKAGE_PATH:-$PROJECT_ROOT/tmp/downloads/${RUN_ID}_review_$(date +%Y%m%d_%H%M%S).tar.gz}"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/package_sceneexpert_results.py" \
  --project-root "$PROJECT_ROOT" --run-id "$RUN_ID" --output "$PACKAGE_PATH" \
  --collection-root "${COLLECTION_ROOT:-$PROJECT_ROOT/outputs/slow_memory/$RUN_ID}" \
  --log-root "${ACP_LOG_DIR:-$PROJECT_ROOT/tmp/acp_logs/$RUN_ID}" \
  --max-file-mib "${PACKAGE_MAX_FILE_MIB:-32}" \
  --max-total-mib "${PACKAGE_MAX_TOTAL_MIB:-512}"
