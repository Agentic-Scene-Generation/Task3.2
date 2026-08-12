#!/usr/bin/env bash
# Dense room ACP run with AIJWS GPT-5.5 (Responses API) and OpenRouter GPT Image 2.
# Local Qwen3.6-27B is disabled; local Qwen3-VL embedding remains enabled.
#
# Usage:
#   cd /mnt/afs/visitor33/Task3.2
#   MAX_CASES=3 PIPELINE_STOP_STAGE=furniture bash ./run_aijws_gpt55_room_dense_acp.sh
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DENSE_RUNNER="$PROJECT_ROOT/run_openrouter_room_dense_acp.sh"
PROXY_STARTER="$PROJECT_ROOT/start_openrouter_proxy.sh"
AIJWS_KEY_FILE="${AIJWS_KEY_FILE:-/mnt/afs/visitor33/apikeys/javis.json}"
AIJWS_BASE_URL="${AIJWS_BASE_URL:-https://api.aijws.com}"
AIJWS_MODEL="${AIJWS_MODEL:-gpt-5.5}"

if [ ! -f "$DENSE_RUNNER" ]; then
    echo "[ERROR] dense room runner is missing: $DENSE_RUNNER" >&2
    exit 1
fi
if [ ! -f "$PROXY_STARTER" ]; then
    echo "[ERROR] OpenRouter proxy starter is missing: $PROXY_STARTER" >&2
    exit 1
fi
if [ ! -r "$AIJWS_KEY_FILE" ]; then
    echo "[ERROR] AIJWS key file is unreadable: $AIJWS_KEY_FILE" >&2
    exit 1
fi

# Image editing still calls OpenRouter, so configure its local proxy first.
# shellcheck source=/dev/null
source "$PROXY_STARTER"

AIJWS_API_KEY="$(python3 - "$AIJWS_KEY_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle).get("OPENAI_API_KEY", "")
if not isinstance(value, str) or not value.strip():
    raise SystemExit("javis key file requires a non-empty OPENAI_API_KEY")
print(value)
PY
)"

# Designer, critic, VLM, TaskCompiler, and GlobalPlanner use AIJWS GPT-5.5.
export REMOTE_LLM=true
export REMOTE_LLM_MODEL_CHECK=false
export EXPECTED_MODEL="$AIJWS_MODEL"
export OPENAI_API_KEY="$AIJWS_API_KEY"
export OPENAI_BASE_URL="$AIJWS_BASE_URL"
export OPENAI_USE_RESPONSES=true
export SCENEEXPERT_FORCE_REASONING_EFFORT="${SCENEEXPERT_FORCE_REASONING_EFFORT:-xhigh}"
export SCENEEXPERT_OPENAI_DEFAULT_HEADERS_JSON='{"x-openai-actor-authorization":"local-image-extension"}'

# Furniture context-image editing remains independent on OpenRouter.
export OPENROUTER_IMAGE_MODEL="${OPENROUTER_IMAGE_MODEL:-openai/gpt-image-2}"
export RUN_ID="${RUN_ID:-aijws_gpt55_room_dense_acp_$(date +%Y%m%d_%H%M%S)}"

cat <<SUMMARY
============= ACP AIJWS GPT-5.5 DENSE ROOM =============
run id:                   $RUN_ID
reasoning/VLM model:       $EXPECTED_MODEL
reasoning/VLM endpoint:    $OPENAI_BASE_URL (Responses API)
reasoning effort:          $SCENEEXPERT_FORCE_REASONING_EFFORT
context image model:       $OPENROUTER_IMAGE_MODEL (OpenRouter)
local generative LLM:      disabled
local embedding model:     Qwen3-VL-Embedding-2B-Q8_0
==========================================================
SUMMARY

exec bash "$DENSE_RUNNER"
