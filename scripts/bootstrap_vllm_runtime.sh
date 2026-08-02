#!/bin/bash
# Build a reproducible vLLM server environment without changing SceneSmith's
# application environment. The two runtimes require different Torch/CUDA ABIs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VLLM_VERSION="${SCENEEXPERT_VLLM_VERSION:-0.22.1}"
TORCH_BACKEND="${SCENEEXPERT_VLLM_TORCH_BACKEND:-cu129}"
VLLM_VENV_PATH="${SCENEEXPERT_VLLM_VENV_PATH:-$PROJECT_DIR/.venv-vllm-${VLLM_VERSION}-${TORCH_BACKEND}}"
if [[ "$VLLM_VENV_PATH" != /* ]]; then
    VLLM_VENV_PATH="$PROJECT_DIR/$VLLM_VENV_PATH"
fi
PYTHON_VERSION="${SCENEEXPERT_VLLM_PYTHON_VERSION:-3.11}"
PIP_INDEX_URL="${SCENEEXPERT_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_INDEX_URL="${SCENEEXPERT_VLLM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_BACKEND}}"
VLLM_WHEELHOUSE="${SCENEEXPERT_VLLM_WHEELHOUSE:-}"
FORCE_REBUILD="${SCENEEXPERT_VLLM_FORCE_REBUILD:-0}"
VLLM_PYTHON="$VLLM_VENV_PATH/bin/python"

check_runtime() {
    [ -x "$VLLM_PYTHON" ] || return 1
    PYTHONDONTWRITEBYTECODE=1 "$VLLM_PYTHON" \
        "$PROJECT_DIR/scripts/check_runtime_compatibility.py" \
        --scope server \
        --expected-vllm-version "$VLLM_VERSION" \
        --expected-torch-backend "$TORCH_BACKEND"
}

if [ "$FORCE_REBUILD" != "1" ] && check_runtime; then
    echo "vLLM runtime is ready: $VLLM_VENV_PATH"
    exit 0
fi

mkdir -p "$(dirname "$VLLM_VENV_PATH")"
LOCK_FILE="${VLLM_VENV_PATH}.bootstrap.lock"
if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK_FILE"
    flock 9
    if [ "$FORCE_REBUILD" != "1" ] && check_runtime; then
        echo "vLLM runtime was prepared by another process: $VLLM_VENV_PATH"
        exit 0
    fi
fi

echo "Preparing isolated vLLM runtime:"
echo "  path: $VLLM_VENV_PATH"
echo "  vLLM: $VLLM_VERSION"
echo "  Torch backend: $TORCH_BACKEND"

if command -v uv >/dev/null 2>&1; then
    UV_REINSTALL_ARGS=()
    UV_TORCH_ARGS=()
    if uv pip install --help 2>&1 | grep -q -- "--torch-backend"; then
        UV_TORCH_ARGS+=(--torch-backend "$TORCH_BACKEND")
    fi
    if [ ! -x "$VLLM_PYTHON" ]; then
        uv venv --python "$PYTHON_VERSION" "$VLLM_VENV_PATH"
    else
        # The environment exists but failed its native ABI check (or an
        # explicit rebuild was requested), so replace every resolved wheel.
        UV_REINSTALL_ARGS+=(--reinstall)
    fi
    if [ -n "$VLLM_WHEELHOUSE" ]; then
        uv pip install \
            --python "$VLLM_PYTHON" \
            --upgrade \
            "${UV_REINSTALL_ARGS[@]}" \
            --no-index \
            --find-links "$VLLM_WHEELHOUSE" \
            "vllm==$VLLM_VERSION"
    else
        uv pip install \
            --python "$VLLM_PYTHON" \
            --upgrade \
            "${UV_REINSTALL_ARGS[@]}" \
            --index-url "$PIP_INDEX_URL" \
            --extra-index-url "$TORCH_INDEX_URL" \
            "${UV_TORCH_ARGS[@]}" \
            "vllm==$VLLM_VERSION"
    fi
else
    BOOTSTRAP_PYTHON="${SCENEEXPERT_VLLM_BOOTSTRAP_PYTHON:-python3}"
    PIP_REINSTALL_ARGS=()
    if [ ! -x "$VLLM_PYTHON" ]; then
        "$BOOTSTRAP_PYTHON" -m venv "$VLLM_VENV_PATH"
    else
        PIP_REINSTALL_ARGS+=(--force-reinstall)
    fi
    if [ -n "$VLLM_WHEELHOUSE" ]; then
        "$VLLM_PYTHON" -m pip install \
            --upgrade \
            "${PIP_REINSTALL_ARGS[@]}" \
            --no-index \
            --find-links "$VLLM_WHEELHOUSE" \
            "vllm==$VLLM_VERSION"
    else
        "$VLLM_PYTHON" -m pip install \
            --upgrade \
            "${PIP_REINSTALL_ARGS[@]}" \
            --index-url "$PIP_INDEX_URL" \
            --extra-index-url "$TORCH_INDEX_URL" \
            "vllm==$VLLM_VERSION"
    fi
fi

check_runtime
echo "Isolated vLLM runtime prepared successfully: $VLLM_VENV_PATH"
