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
VLLM_WHEEL_URL="${SCENEEXPERT_VLLM_WHEEL_URL:-}"
VLLM_WHEEL_SHA256="${SCENEEXPERT_VLLM_WHEEL_SHA256:-}"
FORCE_REBUILD="${SCENEEXPERT_VLLM_FORCE_REBUILD:-0}"
VLLM_PYTHON="$VLLM_VENV_PATH/bin/python"

resolve_vllm_install_target() {
    if [ -n "$VLLM_WHEELHOUSE" ]; then
        # CUDA variants use a PEP 440 local version. Requiring it explicitly
        # prevents an offline wheelhouse from silently selecting the default
        # CUDA 13 wheel for a CUDA 12.9 Torch runtime.
        printf 'vllm==%s+%s\n' "$VLLM_VERSION" "$TORCH_BACKEND"
        return
    fi

    if [ -n "$VLLM_WHEEL_URL" ]; then
        if [ -n "$VLLM_WHEEL_SHA256" ]; then
            printf '%s#sha256=%s\n' "$VLLM_WHEEL_URL" "$VLLM_WHEEL_SHA256"
        else
            printf '%s\n' "$VLLM_WHEEL_URL"
        fi
        return
    fi

    local machine_arch
    machine_arch="$(uname -m)"
    case "$VLLM_VERSION:$TORCH_BACKEND:$machine_arch" in
        0.22.1:cu129:x86_64)
            printf '%s\n' \
                "https://github.com/vllm-project/vllm/releases/download/v0.22.1/vllm-0.22.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=365ee929afd73bb5d146235b65053fa948788ec2ee00a2c3e957d3f43bf2b0cd"
            ;;
        0.22.1:cu129:aarch64|0.22.1:cu129:arm64)
            printf '%s\n' \
                "https://github.com/vllm-project/vllm/releases/download/v0.22.1/vllm-0.22.1%2Bcu129-cp38-abi3-manylinux_2_28_aarch64.whl#sha256=b4cef4bf6264372d61382ebeb36e2be7183e3e736769f1d53fa4c897a0be8ce7"
            ;;
        *)
            echo "ERROR: no verified vLLM binary is registered for version=$VLLM_VERSION, backend=$TORCH_BACKEND, arch=$machine_arch." >&2
            echo "Set SCENEEXPERT_VLLM_WHEEL_URL to an ABI-matched release wheel and optionally set SCENEEXPERT_VLLM_WHEEL_SHA256." >&2
            return 1
            ;;
    esac
}

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
VLLM_INSTALL_TARGET="$(resolve_vllm_install_target)"
echo "  vLLM wheel: $VLLM_INSTALL_TARGET"

if command -v uv >/dev/null 2>&1; then
    UV_INDEX_ARGS=(--index-url "$PIP_INDEX_URL")
    if uv pip install --help 2>&1 | grep -q -- "--torch-backend"; then
        # uv routes only packages in the PyTorch ecosystem to this backend and
        # keeps general dependencies (for example packaging) on the main
        # index. Do not also add the PyTorch index as an extra index: it would
        # take priority for every package and can make resolution unsatisfiable.
        UV_INDEX_ARGS+=(--torch-backend "$TORCH_BACKEND")
    else
        # Older uv releases lack package-scoped Torch routing. Match pip's
        # cross-index resolution so a stale packaging wheel on the PyTorch
        # index cannot hide a compatible version from the main mirror.
        UV_INDEX_ARGS+=(
            --extra-index-url "$TORCH_INDEX_URL"
            --index-strategy unsafe-best-match
        )
    fi
    # Reaching this branch means the environment is absent, failed its native
    # ABI preflight, or was explicitly rebuilt. Clear it transactionally so
    # CUDA 13 packages from a previous wheel can never leak into CUDA 12.9.
    UV_VENV_ARGS=(--python "$PYTHON_VERSION")
    if [ -d "$VLLM_VENV_PATH" ]; then
        UV_VENV_ARGS+=(--clear)
    fi
    uv venv "${UV_VENV_ARGS[@]}" "$VLLM_VENV_PATH"
    if [ -n "$VLLM_WHEELHOUSE" ]; then
        uv pip install \
            --python "$VLLM_PYTHON" \
            --upgrade \
            --no-index \
            --find-links "$VLLM_WHEELHOUSE" \
            "$VLLM_INSTALL_TARGET"
    else
        uv pip install \
            --python "$VLLM_PYTHON" \
            --upgrade \
            "${UV_INDEX_ARGS[@]}" \
            "$VLLM_INSTALL_TARGET"
    fi
else
    BOOTSTRAP_PYTHON="${SCENEEXPERT_VLLM_BOOTSTRAP_PYTHON:-python3}"
    VENV_CLEAR_ARGS=()
    if [ -d "$VLLM_VENV_PATH" ]; then
        VENV_CLEAR_ARGS+=(--clear)
    fi
    "$BOOTSTRAP_PYTHON" -m venv "${VENV_CLEAR_ARGS[@]}" "$VLLM_VENV_PATH"
    if [ -n "$VLLM_WHEELHOUSE" ]; then
        "$VLLM_PYTHON" -m pip install \
            --upgrade \
            --no-index \
            --find-links "$VLLM_WHEELHOUSE" \
            "$VLLM_INSTALL_TARGET"
    else
        "$VLLM_PYTHON" -m pip install \
            --upgrade \
            --index-url "$PIP_INDEX_URL" \
            --extra-index-url "$TORCH_INDEX_URL" \
            "$VLLM_INSTALL_TARGET"
    fi
fi

check_runtime
echo "Isolated vLLM runtime prepared successfully: $VLLM_VENV_PATH"
