#!/usr/bin/env bash
# Shared observations followed by independently executed natural design actions.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
export RUN_ID="${RUN_ID:-qwen38_design_pairs_026}"
export DECISION_BOUNDARY=first_mutation
export TRAIN_TASKS="${TRAIN_TASKS:-128}" VALIDATION_TASKS="${VALIDATION_TASKS:-24}" TEST_TASKS="${TEST_TASKS:-32}"
export MAX_TASKS="${MAX_TASKS:-2}" CHUNK_SIZE="${CHUNK_SIZE:-2}" ACP_PARALLELISM="${ACP_PARALLELISM:-1}"
export MODEL_NAME=unsloth/Qwen3.8-27B-GGUF
export MODEL_DIR="${MODEL_DIR:-/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF}"
export MODEL="${MODEL:-$MODEL_DIR/Qwen3.8-27B-UD-Q8_K_XL.gguf}"
export MMPROJ="${MMPROJ:-$MODEL_DIR/mmproj-F16.gguf}"
export PACKAGE_MAX_FILE_MIB="${PACKAGE_MAX_FILE_MIB:-16}" PACKAGE_MAX_TOTAL_MIB="${PACKAGE_MAX_TOTAL_MIB:-128}"
export SCENEEXPERT_PAIR_TIMEOUT="${SCENEEXPERT_PAIR_TIMEOUT:-7200}"
bash "$PROJECT_ROOT/tmp/acp/acp_qwen38_dpo_campaign.sh"
