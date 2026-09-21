#!/usr/bin/env bash
# Install into a separate environment; preserve the scene-generation runtime.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
TRAIN_ENV="${TRAIN_ENV:-$PROJECT_ROOT/.venv_dpo}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$PROJECT_ROOT/configurations/slow_memory/requirements_train.txt}"
# Ignore host-wide extra indexes (the ACP image may contain an unreachable NGC
# index). A site mirror remains selectable through TRAIN_PIP_INDEX_URL.
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL="${TRAIN_PIP_INDEX_URL:-https://pypi.org/simple}"
export PIP_EXTRA_INDEX_URL=""
[[ -d "$TRAIN_ENV" ]] || "$BOOTSTRAP_PYTHON" -m venv "$TRAIN_ENV"
"$TRAIN_ENV/bin/python" -m pip install --upgrade pip
"$TRAIN_ENV/bin/python" -m pip install -r "$REQUIREMENTS_FILE"
"$TRAIN_ENV/bin/python" -m pip freeze > "$TRAIN_ENV/sceneexpert_requirements.lock.txt"
"$TRAIN_ENV/bin/python" -c 'from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor; from trl import DPOTrainer, DPOConfig; from peft import LoraConfig; import torch; print("DPO runtime imports passed; CUDA:", torch.cuda.is_available())'
