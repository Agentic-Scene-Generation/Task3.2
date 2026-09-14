"""Pair identity must survive manual deployment while rejecting changed code."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scenesmith.scene_expert import trace_logger
from scenesmith.scene_expert.slow_memory import paired_runtime
from scenesmith.scene_expert.slow_memory.paired import write_json
from scenesmith.scene_expert.slow_memory.paired_provenance import (
    _REQUIRED_SOURCES,
    collect_pair_code_provenance,
    verify_pair_code_provenance,
)


@pytest.fixture
def deployed_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "manual-copy"
    for name in (
        *_REQUIRED_SOURCES,
        "scenesmith/prompts/data/scene_expert/task_compiler.yaml",
        "scenesmith/scene_expert/helper.py",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture: original\n")
    monkeypatch.delenv("SCENEEXPERT_PAIR_SOURCE_HASH", raising=False)
    monkeypatch.delenv("ACP_ENTRYPOINT", raising=False)
    return root


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("Git unavailable"),
        subprocess.CalledProcessError(
            128, ["git", "rev-parse", "HEAD"], stderr="detected dubious ownership"
        ),
    ],
)
def test_identity_works_without_git_or_trusted_ownership(
    deployed_tree: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(args)
        raise failure

    monkeypatch.setattr(subprocess, "run", unavailable)
    monkeypatch.setattr(subprocess, "check_output", unavailable)
    monkeypatch.setattr(trace_logger, "_git_executable", unavailable)
    # No .git at first; unusable Git metadata must not change runtime identity.
    before = collect_pair_code_provenance(deployed_tree)
    (deployed_tree / ".git").write_text("gitdir: /unavailable/foreign-owned-checkout")
    after = verify_pair_code_provenance(before, repo_root=deployed_tree)
    assert before == after
    assert len(after["source_bundle_hash"]) == 64
    assert after["identity_kind"] == "source_bundle_sha256"
    assert "git_revision" not in after
    assert calls == []


def test_canonical_trace_honors_git_disabled_for_pair_workers(
    deployed_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCENEEXPERT_CODE_PROVENANCE_GIT_ENABLED", "false")

    def forbidden():
        pytest.fail("Git discovery must not run in the paired ACP process tree")

    monkeypatch.setattr(trace_logger, "_git_executable", forbidden)
    provenance = trace_logger.collect_code_provenance(deployed_tree)
    assert provenance["source_hashes"]
    assert provenance["git_metadata_enabled"] is False
    assert provenance["git_revision"] == provenance["git_status_hash"] == ""
    assert provenance["dirty"] is None


def test_identity_is_portable_and_ignores_outputs_and_git_metadata(
    deployed_tree: Path,
) -> None:
    expected = collect_pair_code_provenance(deployed_tree)
    clone = deployed_tree.parent / "another-mount"
    shutil.copytree(deployed_tree, clone)
    for name in expected["source_hashes"]:
        path = clone / name
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    for name in (
        "outputs/trace.json",
        "tmp/results/run.log",
        "scenesmith/__pycache__/helper.pyc",
        ".git/HEAD",
        "notes.md",
    ):
        path = clone / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("runtime data or metadata")
    assert verify_pair_code_provenance(expected, repo_root=clone) == expected


@pytest.mark.parametrize(
    "name",
    [
        "scenesmith/scene_expert/slow_memory/paired_runtime.py",
        "scenesmith/scene_expert/helper.py",
        "scenesmith/prompts/data/scene_expert/task_compiler.yaml",
        "configurations/scene_expert/base_scene_expert.yaml",
        "tmp/acp/acp_qwen38_initial_pairs.sh",
        "pyproject.toml",
    ],
)
def test_changed_runtime_source_is_rejected_without_git(
    deployed_tree: Path, name: str
) -> None:
    expected = collect_pair_code_provenance(deployed_tree)
    (deployed_tree / name).write_text("changed after candidate A started")
    with pytest.raises(ValueError, match="source changed"):
        verify_pair_code_provenance(expected, repo_root=deployed_tree)


@pytest.mark.parametrize("change", ["add", "delete", "missing-required"])
def test_source_set_changes_are_detected(deployed_tree: Path, change: str) -> None:
    expected = collect_pair_code_provenance(deployed_tree)
    if change == "add":
        (deployed_tree / "scenesmith/new_runtime_helper.py").write_text("new module")
    elif change == "delete":
        (deployed_tree / "scenesmith/scene_expert/helper.py").unlink()
    else:
        (deployed_tree / "scripts/collect_sceneexpert_initial_pairs.py").unlink()
    with pytest.raises(ValueError, match="source changed|source files missing"):
        verify_pair_code_provenance(expected, repo_root=deployed_tree)


def test_startup_fingerprint_is_pinned_through_canonical_capture(
    deployed_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    startup = collect_pair_code_provenance(deployed_tree)
    monkeypatch.setenv("SCENEEXPERT_PAIR_SOURCE_HASH", startup["source_bundle_hash"])
    verify_pair_code_provenance(repo_root=deployed_tree)
    (deployed_tree / "scenesmith/scene_expert/helper.py").write_text(
        "resynchronized during generation"
    )
    with pytest.raises(ValueError, match="since ACP preflight"):
        verify_pair_code_provenance(repo_root=deployed_tree)


def test_shadow_rejects_code_drift_before_importing_native_runtime(
    deployed_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    group = deployed_tree.parent / "group"
    write_json(
        group / "snapshot.json",
        {"code_provenance": collect_pair_code_provenance(deployed_tree)},
    )
    (deployed_tree / "scenesmith/scene_expert/helper.py").write_text("changed")
    monkeypatch.setattr(paired_runtime, "REPO", deployed_tree)
    # This host has no Drake. Importing native dependencies before the source
    # check would fail here instead of giving the intended source-drift error.
    with pytest.raises(ValueError, match="source changed"):
        asyncio.run(paired_runtime.run_shadow(group))


def test_native_safety_cannot_mark_changed_code_complete(
    deployed_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    group = deployed_tree.parent / "group"
    pair = paired_runtime.InitialPair(
        group, {"code_provenance": collect_pair_code_provenance(deployed_tree)}
    )
    (deployed_tree / "scenesmith/scene_expert/helper.py").write_text(
        "changed during native safety"
    )
    monkeypatch.setattr(paired_runtime, "REPO", deployed_tree)
    with pytest.raises(ValueError, match="source changed"):
        pair.capture_returned(SimpleNamespace(), "", "A")
    assert not (group / "A/result.json").exists()


def test_legacy_git_only_snapshot_is_not_silently_replayed(deployed_tree: Path) -> None:
    with pytest.raises(ValueError, match="lacks content-based code identity"):
        verify_pair_code_provenance(
            {"git_revision": "old-commit"}, repo_root=deployed_tree
        )
