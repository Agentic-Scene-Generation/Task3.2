#!/usr/bin/env bash
# Dense room ACP run using OpenRouter for both GPT-5.2 reasoning and GPT Image 2.
# Local Qwen3.6-27B is not started; local Qwen3-VL embedding remains enabled.
#
# Usage inside one ACP task:
#   cd /mnt/afs/visitor33/Task3.2
#   source ./start_openrouter_proxy.sh
#   PIPELINE_STOP_STAGE=furniture bash ./run_openrouter_gpt52_room_dense_acp.sh
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DENSE_RUNNER="$PROJECT_ROOT/run_openrouter_room_dense_acp.sh"
OPENROUTER_KEY_FILE="${OPENROUTER_KEY_FILE:-/mnt/afs/visitor33/exportkey.sh}"
OPENROUTER_LLM_MODEL="${OPENROUTER_LLM_MODEL:-openai/gpt-5.2}"
OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"

if [ ! -f "$DENSE_RUNNER" ]; then
    echo "[ERROR] dense OpenRouter runner not found: $DENSE_RUNNER" >&2
    exit 1
fi

# The dedicated proxy script must be sourced so requests inherits its exports.
if [ -z "${HTTPS_PROXY:-${https_proxy:-}}" ]; then
    echo "[ERROR] HTTPS proxy is not configured in this shell." >&2
    echo "        Run: source $PROJECT_ROOT/start_openrouter_proxy.sh" >&2
    exit 1
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    if [ ! -r "$OPENROUTER_KEY_FILE" ]; then
        echo "[ERROR] OpenRouter key file is not readable: $OPENROUTER_KEY_FILE" >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$OPENROUTER_KEY_FILE"
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "[ERROR] OPENROUTER_API_KEY is required" >&2
    exit 1
fi

# Scene designer/critic/VLM requests use GPT-5.2 through OpenRouter.
export REMOTE_LLM=true
export EXPECTED_MODEL="$OPENROUTER_LLM_MODEL"
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="$OPENROUTER_BASE_URL"
export OPENAI_USE_RESPONSES=false
export SCENEEXPERT_LLM_PROVIDER=openrouter

# Context image generation remains on GPT Image 2 unless explicitly overridden.
export OPENROUTER_IMAGE_MODEL="${OPENROUTER_IMAGE_MODEL:-openai/gpt-image-2}"
export RUN_ID="${RUN_ID:-openrouter_gpt52_room_dense_acp_$(date +%Y%m%d_%H%M%S)}"

cat <<SUMMARY
========== ACP OPENROUTER GPT-5.2 DENSE ROOM ==========
run id:                   $RUN_ID
reasoning/VLM model:       $EXPECTED_MODEL
reasoning/VLM endpoint:    $OPENAI_BASE_URL
context image model:       $OPENROUTER_IMAGE_MODEL
local generative LLM:      disabled
local embedding model:     Qwen3-VL-Embedding-2B-Q8_0
========================================================
SUMMARY

exec bash "$DENSE_RUNNER"
