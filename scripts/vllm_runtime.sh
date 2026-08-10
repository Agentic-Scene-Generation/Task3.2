#!/bin/bash
# Shared vLLM runtime resolution for SceneExpert launch scripts.
# Source this file, then call sceneexpert_resolve_vllm_runtime PROJECT_DIR.

sceneexpert_resolve_vllm_runtime() {
    local project_dir="$1"
    local expected_version="${SCENEEXPERT_VLLM_VERSION:-0.22.1}"
    local venv_path="${SCENEEXPERT_VLLM_VENV_PATH:-}"
    local auto_bootstrap="${SCENEEXPERT_VLLM_AUTO_BOOTSTRAP:-0}"
    local requested_executable="${SCENEEXPERT_VLLM_EXECUTABLE:-}"
    local requested_python="${SCENEEXPERT_VLLM_PYTHON:-}"

    if [ -n "$venv_path" ] && [[ "$venv_path" != /* ]]; then
        venv_path="$project_dir/$venv_path"
    fi
    if [ -n "$requested_executable" ] && [[ "$requested_executable" != /* ]]; then
        requested_executable="$project_dir/$requested_executable"
    fi
    if [ -n "$requested_python" ] && [[ "$requested_python" != /* ]]; then
        requested_python="$project_dir/$requested_python"
    fi

    if [ "$auto_bootstrap" = "1" ]; then
        bash "$project_dir/scripts/bootstrap_vllm_runtime.sh"
    fi

    if [ -n "$requested_executable" ]; then
        SCENEEXPERT_RESOLVED_VLLM_EXECUTABLE="$requested_executable"
        if [ -n "$requested_python" ]; then
            SCENEEXPERT_RESOLVED_VLLM_PYTHON="$requested_python"
        elif [ -x "$(dirname "$requested_executable")/python" ]; then
            SCENEEXPERT_RESOLVED_VLLM_PYTHON="$(dirname "$requested_executable")/python"
        else
            SCENEEXPERT_RESOLVED_VLLM_PYTHON="$(command -v python)"
        fi
        SCENEEXPERT_RESOLVED_VLLM_LAUNCH_MODE="cli"
    elif [ -n "$venv_path" ]; then
        SCENEEXPERT_RESOLVED_VLLM_EXECUTABLE="$venv_path/bin/vllm"
        SCENEEXPERT_RESOLVED_VLLM_PYTHON="$venv_path/bin/python"
        SCENEEXPERT_RESOLVED_VLLM_LAUNCH_MODE="cli"
    elif command -v vllm >/dev/null 2>&1; then
        SCENEEXPERT_RESOLVED_VLLM_EXECUTABLE="$(command -v vllm)"
        SCENEEXPERT_RESOLVED_VLLM_PYTHON="${requested_python:-$(command -v python)}"
        SCENEEXPERT_RESOLVED_VLLM_LAUNCH_MODE="cli"
    elif python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('vllm.entrypoints.openai.api_server') else 1)" >/dev/null 2>&1; then
        SCENEEXPERT_RESOLVED_VLLM_EXECUTABLE=""
        SCENEEXPERT_RESOLVED_VLLM_PYTHON="$(command -v python)"
        SCENEEXPERT_RESOLVED_VLLM_LAUNCH_MODE="python-module"
    else
        echo "ERROR: no vLLM server runtime is available."
        echo "Prepare the isolated CUDA runtime with:"
        echo "  bash scripts/bootstrap_vllm_runtime.sh"
        return 1
    fi

    if [ ! -x "$SCENEEXPERT_RESOLVED_VLLM_PYTHON" ]; then
        echo "ERROR: vLLM Python is not executable: $SCENEEXPERT_RESOLVED_VLLM_PYTHON"
        return 1
    fi
    if [ "$SCENEEXPERT_RESOLVED_VLLM_LAUNCH_MODE" = "cli" ] \
        && [ ! -x "$SCENEEXPERT_RESOLVED_VLLM_EXECUTABLE" ]; then
        echo "ERROR: vLLM executable is not available: $SCENEEXPERT_RESOLVED_VLLM_EXECUTABLE"
        echo "Prepare it with: bash scripts/bootstrap_vllm_runtime.sh"
        return 1
    fi

    if [ "${SCENEEXPERT_SKIP_PYTHON_PREFLIGHT:-0}" != "1" ]; then
        echo "  Running vLLM native CUDA ABI preflight..."
        PYTHONDONTWRITEBYTECODE=1 "$SCENEEXPERT_RESOLVED_VLLM_PYTHON" \
            "$project_dir/scripts/check_runtime_compatibility.py" \
            --scope server \
            --expected-vllm-version "$expected_version" \
            --expected-torch-backend "${SCENEEXPERT_VLLM_TORCH_BACKEND:-cu129}"
    fi

    export SCENEEXPERT_RESOLVED_VLLM_EXECUTABLE
    export SCENEEXPERT_RESOLVED_VLLM_PYTHON
    export SCENEEXPERT_RESOLVED_VLLM_LAUNCH_MODE
}
