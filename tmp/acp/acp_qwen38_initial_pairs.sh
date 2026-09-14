#!/usr/bin/env bash
# Two fresh, simple scene tasks; canonical A plus one isolated furniture shadow B.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2}"
RUN_ID="${RUN_ID:-qwen38_initial_pairs_008_$(date +%Y%m%d_%H%M%S)}"
PAIR_GROUPS="${PAIR_GROUPS:-2}"
[[ "$PAIR_GROUPS" =~ ^[1-4]$ ]] || { echo 'PAIR_GROUPS must be 1..4' >&2; exit 2; }
[[ "${ACP_PARALLELISM:-1}" == 1 ]] || { echo 'Initial pair pilot requires ACP_PARALLELISM=1' >&2; exit 2; }
COLLECTION_ROOT="${COLLECTION_ROOT:-$PROJECT_ROOT/outputs/slow_memory/$RUN_ID}"
OUTPUT_ROOT="$COLLECTION_ROOT/runs"
PAIR_ROOT="$OUTPUT_ROOT/paired_initial"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=/mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-main/.venv/bin/python
fi
[[ -x "$PYTHON_BIN" ]] || { echo "No collection Python: $PYTHON_BIN" >&2; exit 2; }
[[ ! -e "$COLLECTION_ROOT" ]] || { echo "Collection already exists: $COLLECTION_ROOT" >&2; exit 2; }
# ACP receives a manually synchronized source tree. Content identity is required;
# Git metadata, repository ownership and network access are not runtime inputs.
export SCENEEXPERT_CODE_PROVENANCE_GIT_ENABLED=false
SCENEEXPERT_PAIR_SOURCE_HASH="$(cd "$PROJECT_ROOT" && "$PYTHON_BIN" -c 'from scenesmith.scene_expert.slow_memory.paired_provenance import collect_pair_code_provenance; print(collect_pair_code_provenance()["source_bundle_hash"])')"
[[ "$SCENEEXPERT_PAIR_SOURCE_HASH" =~ ^[0-9a-f]{64}$ ]] || { echo 'Invalid pair source fingerprint' >&2; exit 2; }
export SCENEEXPERT_PAIR_SOURCE_HASH
echo "Pair source SHA256: $SCENEEXPERT_PAIR_SOURCE_HASH"
PAIR_LOG_DIR="${ACP_LOG_ROOT:-$PROJECT_ROOT/tmp/acp_logs}/$RUN_ID"
mkdir -p "$PAIR_LOG_DIR"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_sceneexpert_initial_pairs.py" \
  --preflight --preflight-report "$PAIR_LOG_DIR/pair_preflight.json" \
  2>&1 | tee "$PAIR_LOG_DIR/pair_preflight.log"
(cd "$PROJECT_ROOT" && "$PYTHON_BIN" -c 'from scenesmith.scene_expert.slow_memory.paired_runtime import open_initial_pair; from scenesmith.furniture_agents.stateful_furniture_agent import StatefulFurnitureAgent; from openai import DefaultAsyncHttpxClient')

# Observer audit and pair audit are separate gates. A zero-pair observer probe
# cannot make this wrapper succeed: the final pair gate always requires >=1 pair.
generation_exit=0
env PROJECT_ROOT="$PROJECT_ROOT" RUN_ID="$RUN_ID" COLLECTION_ROOT="$COLLECTION_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
  CASE_SET="${CASE_SET:-legacy8}" SCENE_SELECTION="${SCENE_SELECTION:-default_bedroom,default_living_room}" \
  DIFFICULTY_SELECTION="${DIFFICULTY_SELECTION:-all}" MAX_CASES="$PAIR_GROUPS" \
  ACP_PARALLELISM=1 MIN_DPO_PAIRS=0 \
  SCENEEXPERT_INITIAL_PAIRS_DIR="$PAIR_ROOT" SCENEEXPERT_PAIR_MAX_GROUPS="$PAIR_GROUPS" \
  SCENEEXPERT_PAIR_TIMEOUT="${SCENEEXPERT_PAIR_TIMEOUT:-3600}" \
  bash "$SCRIPT_DIR/acp_qwen38_slow_memory_recollect.sh" || generation_exit=$?

pair_exit=0
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_sceneexpert_initial_pairs.py" \
  --audit-root "$PAIR_ROOT" --expected-groups "$PAIR_GROUPS" --min-pairs 1 || pair_exit=$?
mkdir -p "$PAIR_ROOT"
cp "${BASH_SOURCE[0]}" "$PAIR_ROOT/entrypoint.sh"
printf '%s\n' 'collection_kind=furniture_initial_independent_pairs' \
  "expected_groups=$PAIR_GROUPS" 'candidates_per_group=2' 'canonical_candidate=A' \
  'scoring=raw_candidate_main_deterministic_checks' 'min_pairs=1' \
  "source_bundle_hash=$SCENEEXPERT_PAIR_SOURCE_HASH" 'git_metadata=disabled' \
  > "$PAIR_ROOT/pair_manifest.env"
printf '%s\n' "generation_exit=$generation_exit" "pair_audit_exit=$pair_exit" \
  > "$PAIR_ROOT/exit_status.env"
echo "Pair artifacts: $PAIR_ROOT"
if [[ "$generation_exit" != 0 ]]; then exit "$generation_exit"; fi
exit "$pair_exit"
