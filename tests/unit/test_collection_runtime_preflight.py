"""CPU startup checks catch stale paths before any model/scene process starts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import preflight_sceneexpert_cuda_runtime as runtime

ROOT = Path(__file__).resolve().parents[2]


def libraries(path: Path, names=runtime.REQUIRED_LIBRARIES) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for name in names:
        (path / name).touch()
    return path


def test_current_project_runtime_precedes_training_environment(tmp_path):
    preferred = libraries(tmp_path / ".venv/lib/python3.11/site-packages/nvidia/cu13/lib")
    libraries(tmp_path / ".venv_dpo/lib/python3.11/site-packages/nvidia/cu13/lib")
    assert runtime.resolve_runtime(tmp_path)[0] == preferred


def test_training_environment_supplies_cuda13_when_collection_only_has_cuda12(tmp_path):
    libraries(
        tmp_path / ".venv/lib/python3.11/site-packages/nvidia/cu13/lib",
        ("libcudart.so.12",),
    )
    expected = libraries(tmp_path / ".venv_dpo/lib/python3.12/site-packages/nvidia/cu13/lib")
    selected, inspected = runtime.resolve_runtime(tmp_path)
    assert selected == expected
    assert inspected[0]["missing_libraries"] == list(runtime.REQUIRED_LIBRARIES)


def test_explicit_invalid_path_never_silently_falls_back(tmp_path):
    libraries(tmp_path / ".venv_dpo/lib/python3.11/site-packages/nvidia/cu13/lib")
    with pytest.raises(ValueError, match="No complete CUDA 13 runtime"):
        runtime.resolve_runtime(tmp_path, str(tmp_path / "obsolete Task3.2"))


def test_explicit_path_requires_cublas_and_lt_not_just_cudart(tmp_path):
    incomplete = libraries(tmp_path / "partial", ("libcudart.so.13",))
    with pytest.raises(ValueError, match="libcublasLt.so.13"):
        runtime.resolve_runtime(tmp_path, str(incomplete))


@pytest.mark.parametrize("rc,text", [(0, "libcublas.so.13 => not found"), (1, "not an ELF")])
def test_binary_link_failure_is_fatal_even_when_ldd_returns_zero(tmp_path, monkeypatch, rc, text):
    binary = tmp_path / "llama-server"
    binary.write_text("binary")
    binary.chmod(0o755)
    monkeypatch.setattr(runtime.os, "access", lambda *_: True)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=rc, stdout=text, stderr=""
    ))
    with pytest.raises(ValueError, match="Unresolved native dependencies"):
        runtime.check_binary(binary, tmp_path)


def test_cli_retains_failure_report_and_nonzero_exit(tmp_path):
    report = tmp_path / "audit/runtime_preflight.json"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/preflight_sceneexpert_cuda_runtime.py"),
        "--project-root", str(tmp_path), "--cuda13-lib-dir", str(tmp_path / "gone"),
        "--report", str(report),
    ], capture_output=True, text=True)
    assert result.returncode == 2 and result.stdout == ""
    audit = json.loads(report.read_text())
    assert audit["status"] == "failed" and audit["explicit_override"] is True
    assert "gone" in audit["error"]


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("bash"), reason="native Bash runtime")
def test_real_launcher_cpu_preflight_stops_before_model_services_and_asset_copy(tmp_path):
    shared = tmp_path / "shared"
    project = tmp_path / "current project"
    helpers = project / "tmp/code"
    helpers.mkdir(parents=True)
    for name in ["run_qwen3_vl_embedding_2b_llama_cpp.sh", "check_llama_cpp.sh"]:
        (helpers / name).touch()
    for directory in ["share_data/hsm/hssd-models", "share_data/hsm/preprocessed",
                      "share_data/scenesmith/hssd_rendered_assets", "share_model/Memory/bge-m3"]:
        (shared / directory).mkdir(parents=True)
    (project / "data").mkdir()
    model = shared / "share_model/unsloth/Qwen3.8-27B-GGUF"
    model.mkdir(parents=True)
    for name in ["Qwen3.8-27B-UD-Q8_K_XL.gguf", "mmproj-F16.gguf"]:
        (model / name).touch()
    llama = shared / "share_scripts/llama.cpp"
    llama.mkdir(parents=True)
    (llama / "run_qwen38_27b_llama_cpp.sh").touch()
    binary = llama / "llama-server"
    binary.write_text("#!/bin/sh\nexit 99\n")
    binary.chmod(0o755)
    ldd = tmp_path / "bin/ldd"
    ldd.parent.mkdir()
    ldd.write_text("#!/bin/sh\necho 'all dependencies resolved'\n")
    ldd.chmod(0o755)
    expected = libraries(project / ".venv_dpo/lib/python3.11/site-packages/nvidia/cu13/lib")
    target = project / "scripts/preflight_sceneexpert_cuda_runtime.py"
    target.parent.mkdir()
    shutil.copyfile(ROOT / "scripts/preflight_sceneexpert_cuda_runtime.py", target)
    script = tmp_path / "runtime.sh"
    script.write_text((ROOT / "tmp/acp/acp_qwen38_4c_generate.sh").read_text())
    env = {**os.environ, "PROJECT_ROOT": str(project), "TASK3_SHARED_ROOT": str(shared),
           "SCENEEXPERT_DATA_DIR": str(project / "data"), "PYTHON_BIN": sys.executable,
           "EMBEDDING_LLAMA_DIR": str(llama), "CASE_SET": "legacy8", "RUN_ID": "cpu_only",
           "SCENEEXPERT_RUNTIME_PREFLIGHT_ONLY": "true", "CUDA13_LIB_DIR": "",
           "PATH": str(ldd.parent) + os.pathsep + os.environ["PATH"]}
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    audit = json.loads((project / "tmp/acp_logs/cpu_only/runtime_preflight.json").read_text())
    assert audit["status"] == "passed" and audit["cuda13_lib_dir"] == str(expected)
    assert len(audit["binary_checks"]) == 2
    assert not (project / "tmp/hssd_zvec_collection_qwen38").exists()
    assert not (project / "tmp/acp_logs/cpu_only/service_pids.env").exists()
