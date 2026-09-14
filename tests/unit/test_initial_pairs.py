"""Independent state, evidence, request, and failure gates for the initial pilot."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from scenesmith.scene_expert.slow_memory.paired import (
    EXPECTED_MODEL,
    audit_pairs,
    copy_scene_tree,
    deterministic_verdict,
    digest,
    read_json,
    reserve_group,
    tree_hashes,
    write_json,
    validate_tool_execution,
)
from scenesmith.scene_expert.slow_memory.paired_runtime import shadow_environment
from scenesmith.scene_expert.slow_memory.paired_wire import client_options
from scenesmith.scene_expert.slow_memory.schemas import (
    PreferenceEvidence,
    TrajectoryRecord,
    TrajectoryOutcome,
)


def _group(root: Path) -> Path:
    group = reserve_group(root, "bedroom", 2)
    assert group is not None
    input_scene = group / "input_scene"
    input_scene.mkdir()
    (input_scene / "geometry.sdf").write_text("immutable geometry")
    snapshot = {"files": tree_hashes(input_scene), "model": EXPECTED_MODEL}
    write_json(group / "snapshot.json", snapshot)
    write_json(group / "status.json", {"status": "completed"})
    write_json(
        group / "continuation_proof.json",
        {
            "canonical_before": "a",
            "canonical_after": "a",
            "assets_before": "files",
            "assets_after": "files",
            "memory_before": "m",
            "memory_after": "m",
        },
    )
    request = {
        "model": EXPECTED_MODEL,
        "messages": [{"role": "user", "content": "Place a bed"}],
        "tools": [],
    }
    for candidate, score, failures in (("A", 1.0, 0), ("B", 0.5, 1)):
        directory = group / candidate
        raw_files = copy_scene_tree(input_scene, directory / "raw_scene")
        state = {"candidate_geometry": candidate}
        report = {
            "summary": {
                "scene_summary": {
                    "total_checks": 2,
                    "fail": failures,
                    "unknown": 0,
                    "score": score,
                }
            }
        }
        verdict = "accepted" if failures == 0 else "rejected"
        write_json(directory / "raw_state.json", state)
        write_json(directory / "returned_state.json", state)
        write_json(directory / "safety.json", {"message": ""})
        write_json(directory / "report.json", report)
        write_json(directory / "first_request.json", request)
        write_json(
            directory / "result.json",
            {
                "status": "completed",
                "model": EXPECTED_MODEL,
                "snapshot_hash": digest(snapshot),
                "raw_state_hash": digest(state),
                "evaluation_state_hash": digest(state),
                "returned_state_hash": digest(state),
                "safety_hash": digest({"message": ""}),
                "raw_files": raw_files,
                "verdict": verdict,
                "score": score,
            },
        )
        record = TrajectoryRecord(
            trajectory_id=candidate,
            created_at="2026-09-14T00:00:00Z",
            run_id=candidate,
            scene_id="bedroom",
            task_id="bedroom",
            context_hash=digest(snapshot),
            model_id=EXPECTED_MODEL,
            stage="furniture",
            agent_role="designer",
            event="request_initial_design",
            task_type="designer_initial",
            prompt="Place a bed",
            response="Placed bed " + candidate,
            response_hash=candidate,
            spatial_context={"initial_snapshot_sha256": digest(snapshot)},
            source_refs=["report.json"],
            evidence=PreferenceEvidence(
                evidence_id=candidate,
                kind="deterministic",
                authoritative=True,
                verdict=verdict,
                quality_score=score,
                report_ref="report.json",
                source="main_raw_candidate_deterministic_checks",
                details={"raw_state_sha256": digest(state)},
            ),
            outcome=TrajectoryOutcome(
                execution_complete=True,
                tool_call_valid=True,
                hard_passed=failures == 0,
                hard_violation_count=failures,
                deterministic_score=score,
                causal_link_verified=True,
            ),
        )
        target = directory / "slow_memory/trajectories.jsonl"
        target.parent.mkdir()
        target.write_text(record.model_dump_json() + "\n", encoding="utf-8")
        result = read_json(directory / "result.json")
        result["trajectory_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        write_json(directory / "result.json", result)
    return group


def test_independent_candidate_pair_exports_without_changing_canonical(
    tmp_path: Path,
) -> None:
    group = _group(tmp_path)
    original = (group / "A/raw_state.json").read_bytes()
    result = audit_pairs(tmp_path, expected_groups=1)
    assert result["gate_passed"] is True
    assert result["candidate_count"] == 2
    assert result["eligible_pair_count"] == 1
    assert result["canonical_candidate"] == "A"
    assert (group / "A/raw_state.json").read_bytes() == original


@pytest.mark.parametrize(
    "change",
    ["request", "snapshot", "raw_state", "raw_asset", "report", "memory", "failed"],
)
def test_any_broken_proof_quarantines_the_whole_pair(
    tmp_path: Path, change: str
) -> None:
    group = _group(tmp_path)
    if change == "request":
        write_json(
            group / "B/first_request.json",
            {
                "model": EXPECTED_MODEL,
                "messages": [{"role": "user", "content": "different history"}],
            },
        )
    elif change == "snapshot":
        (group / "input_scene/geometry.sdf").write_text("changed")
    elif change == "raw_state":
        write_json(group / "B/raw_state.json", {"restored_previous_success": True})
    elif change == "raw_asset":
        (group / "B/raw_scene/geometry.sdf").write_text("changed")
    elif change == "report":
        report = read_json(group / "B/report.json")
        report["summary"]["scene_summary"]["score"] = 1.0
        write_json(group / "B/report.json", report)
    elif change == "memory":
        proof = read_json(group / "continuation_proof.json")
        proof["memory_after"] = "shadow wrote to bank"
        write_json(group / "continuation_proof.json", proof)
    else:
        write_json(group / "status.json", {"status": "shadow_failed"})
    result = audit_pairs(tmp_path, expected_groups=1)
    assert not result["gate_passed"]
    assert result["eligible_pair_count"] == 0


def test_missing_candidate_is_not_a_synthetic_negative(tmp_path: Path) -> None:
    group = _group(tmp_path)
    (group / "B/result.json").unlink()
    result = audit_pairs(tmp_path, expected_groups=1)
    assert result["eligible_pair_count"] == 0
    assert not result["gate_passed"]


def test_private_files_are_independent_and_slots_are_bounded(tmp_path: Path) -> None:
    group = _group(tmp_path)
    (group / "B/raw_scene/geometry.sdf").write_text("B-only mutation")
    assert (group / "A/raw_scene/geometry.sdf").read_text() == "immutable geometry"
    assert (group / "input_scene/geometry.sdf").read_text() == "immutable geometry"
    assert reserve_group(tmp_path, "bedroom", 2) is None
    assert reserve_group(tmp_path, "living_room", 2) is not None
    assert reserve_group(tmp_path, "extra", 2) is None


def test_unknown_evidence_cannot_become_an_authoritative_negative() -> None:
    with pytest.raises(ValueError):
        deterministic_verdict(
            {
                "summary": {
                    "scene_summary": {"total_checks": 2, "unknown": 1, "score": 0}
                }
            }
        )


@pytest.mark.parametrize(
    "output",
    [
        "ReadTimeout: request timed out",
        "An error occurred while running the tool. Please try again. Error: unavailable",
        "CUDA out of memory",
    ],
)
def test_tool_infrastructure_errors_are_not_preference_negatives(output: str) -> None:
    with pytest.raises(ValueError):
        validate_tool_execution({"tool_results": [{"output": output}]})
    validate_tool_execution(
        {
            "tool_results": [
                {
                    "output": '{"success": false, "message": "object would exceed room bounds"}'
                }
            ]
        }
    )


def test_shadow_environment_cannot_write_the_shared_bank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR", "/public/bank")
    monkeypatch.setenv("SCENEEXPERT_LLM_DEBUG_PATH", "/canonical/debug")
    monkeypatch.setenv("SCENEBENCHMARK_CRITIC_TIMING_PATH", "/canonical/timing")
    env = shadow_environment(tmp_path)
    assert env["SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR"] != "/public/bank"
    assert env["SCENEEXPERT_ACTIVE_MEMORY_BANK_READ_ONLY"] == "true"
    assert env["SCENEEXPERT_COMPONENT_MEMORY_WRITER_ENABLED"] == "false"
    assert "SCENEEXPERT_LLM_DEBUG_PATH" not in env
    assert "SCENEBENCHMARK_CRITIC_TIMING_PATH" not in env
    assert os.environ["SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR"] == "/public/bank"
    assert client_options() == {}
