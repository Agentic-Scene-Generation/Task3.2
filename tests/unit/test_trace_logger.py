"""Regression coverage for trace reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scenesmith.scene_expert.schemas import FullVerifyReport
from scenesmith.scene_expert.trace_logger import TraceLogger, collect_code_provenance


def test_collect_code_provenance_hashes_requested_source_files(tmp_path) -> None:
    source = tmp_path / "tracked.py"
    source.write_text("answer = 42\n", encoding="utf-8")

    provenance = collect_code_provenance(
        repo_root=tmp_path,
        source_paths=["tracked.py", "missing.py", "../outside.py"],
    )

    assert provenance["repo_root"] == str(tmp_path)
    assert provenance["source_hashes"] == {
        "tracked.py": hashlib.sha256(b"answer = 42\n").hexdigest()
    }
    assert provenance["git_revision"] == ""
    assert provenance["dirty"] is None


def test_collect_code_provenance_uses_absolute_git_in_minimal_worker_path(
    monkeypatch,
) -> None:
    git_executable = shutil.which("git")
    if git_executable is None:
        pytest.skip("Git is unavailable in this test environment.")
    repo_root = Path(__file__).resolve().parents[2]
    expected_revision = subprocess.run(
        [git_executable, "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    monkeypatch.delenv("GIT", raising=False)
    monkeypatch.setenv("PATH", "")
    provenance = collect_code_provenance(repo_root=repo_root)

    assert provenance["git_revision"] == expected_revision
    assert provenance["dirty"] is not None
    assert {
        "scenesmith/manipuland_agents/cross_stage_inventory.py",
        "scenesmith/manipuland_agents/tools/manipuland_tools.py",
        "scenesmith/scenebenchmark_critic/asset_library_annotations.py",
        "scenesmith/scenebenchmark_critic/metrics/functional_dependency/builder.py",
        "scenesmith/scenebenchmark_critic/metrics/functional_dependency/relations.py",
    } <= set(provenance["source_hashes"])


def test_trace_logger_persists_code_provenance_in_partial_and_final_traces(
    tmp_path,
) -> None:
    provenance = {
        "git_revision": "deadbeef",
        "dirty": True,
        "source_hashes": {"scenesmith/example.py": "abc123"},
    }
    logger = TraceLogger(
        output_dir=str(tmp_path),
        scene_index=7,
        prompt="A reproducible test scene.",
        code_provenance=provenance,
        component_flags={"memory_writer": True, "harness_budget": False},
    )

    partial_path = logger.save_partial(status="running")
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    assert partial["code_provenance"] == provenance
    assert partial["component_flags"] == {
        "memory_writer": True,
        "harness_budget": False,
    }

    final_trace = logger.finalize(FullVerifyReport(), exports={})
    assert final_trace["code_provenance"] == provenance
    assert final_trace["component_flags"]["memory_writer"] is True
    final_path = logger.save(final_trace)
    assert (
        json.loads(final_path.read_text(encoding="utf-8"))["code_provenance"]
        == provenance
    )
