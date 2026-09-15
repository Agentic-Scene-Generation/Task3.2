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
    validate_pair_inputs,
    write_json,
    validate_tool_execution,
)
from scenesmith.scene_expert.slow_memory.paired_runtime import shadow_environment
from scenesmith.scene_expert.slow_memory.paired_scoring import (
    SCORING_PROTOCOL,
    save_scoring_proof,
)
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
    snapshot = {"files": tree_hashes(input_scene), "model": EXPECTED_MODEL, "cfg": {}}
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
        physics = {
            "available": True,
            "source_phase": "sceneexpert_raw_candidate",
            "scene_hash": "geometry",
            "raw_state_sha256": digest(state),
            "collisions": [],
        }
        report["case_pack"] = {"physics_evidence": physics}
        save_scoring_proof(
            directory,
            report,
            {
                "schema_version": SCORING_PROTOCOL,
                "raw_state_sha256": digest(state),
                "evaluation_state_sha256": digest(state),
                "scene_content_hash": "geometry",
                "physics_sha256": digest(physics),
                "report_sha256": digest(report),
            },
            raw_files,
        )
        write_json(directory / "raw_state.json", state)
        write_json(directory / "returned_state.json", state)
        write_json(directory / "safety.json", {"message": ""})
        write_json(directory / "report.json", report)
        write_json(directory / "first_request.json", request)
        write_json(
            directory / "result.json",
            {
                "status": "completed",
                "scoring_protocol": SCORING_PROTOCOL,
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
    assert result["status"] == "completed_with_pairs"
    assert result["candidate_verdict_counts"] == {"accepted": 1, "rejected": 1}
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
    assert result["status"] == "execution_failed"


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


@pytest.mark.parametrize(
    "verdict,count,reason",
    [
        ("rejected", 2, "no_accepted_candidate"),
        ("accepted", 2, "no_eligible_preference_contrast"),
        ("accepted", 1, "missing_exact_context_counterpart"),
    ],
)
def test_pair_diagnostics_distinguish_contrast_from_missing_execution(
    tmp_path: Path, verdict: str, count: int, reason: str
) -> None:
    from scenesmith.scene_expert.slow_memory.dpo import (
        build_preference_pairs,
        load_trajectories,
    )

    group = _group(tmp_path)
    records, errors = load_trajectories(
        [group / name / "slow_memory/trajectories.jsonl" for name in ("A", "B")]
    )
    assert not errors
    for record in records:
        record.evidence.verdict = verdict
    pairs, diagnostics = build_preference_pairs(records[:count])
    assert not pairs
    assert [row["reason"] for row in diagnostics] == [reason]


@pytest.mark.parametrize("artifact", ["evaluation_proof.json", "physics.json"])
def test_fresh_physics_proof_is_mandatory_for_export(
    tmp_path: Path, artifact: str
) -> None:
    group = _group(tmp_path)
    (group / "B" / artifact).unlink()
    assert not validate_pair_inputs(tmp_path, require_fresh_physics=False)[2]
    audit = audit_pairs(tmp_path)
    assert audit["execution_integrity_passed"] is False
    assert audit["candidate_count"] == audit["eligible_pair_count"] == 0


def _mock_rescore_physics(
    monkeypatch: pytest.MonkeyPatch, *, corrupt_state: bool = False
) -> None:
    from scenesmith.scene_expert.slow_memory import paired_rescore

    def restore(directory: Path, snapshot: dict, *, source_candidate: Path) -> dict:
        from scenesmith.scene_expert.slow_memory.paired_restoration import (
            save_restoration_proof,
        )

        state = read_json(directory / "raw_state.json")
        save_restoration_proof(directory, state, state, state, [])
        return state

    monkeypatch.setattr(paired_rescore, "restore_raw_scene", restore)

    def fresh(state: dict, cfg: object) -> tuple[dict, dict]:
        state_hash = "wrong-state" if corrupt_state else digest(state)
        physics = {
            "available": True,
            "source_phase": "sceneexpert_raw_candidate",
            "scene_hash": "fresh-geometry",
            "raw_state_sha256": state_hash,
            "collisions": [],
        }
        report = {
            "case_pack": {"physics_evidence": physics},
            "summary": {
                "scene_summary": {
                    "total_checks": 2,
                    "unknown": 0,
                    "fail": 0,
                    "score": 1.0,
                }
            },
        }
        return report, {
            "schema_version": SCORING_PROTOCOL,
            "raw_state_sha256": state_hash,
            "evaluation_state_sha256": state_hash,
            "scene_content_hash": "fresh-geometry",
            "physics_sha256": digest(physics),
            "report_sha256": digest(report),
        }

    monkeypatch.setattr(paired_rescore, "score_raw_candidate", fresh)


def test_offline_rescore_preserves_original_executions_and_keeps_no_contrast_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scenesmith.scene_expert.slow_memory.paired_rescore import rescore_pairs

    source, output = tmp_path / "008", tmp_path / "009"
    group = _group(source)
    before = tree_hashes(source)
    _mock_rescore_physics(monkeypatch)
    audit = rescore_pairs(source, output)
    assert tree_hashes(source) == before
    assert audit["execution_integrity_passed"] is True
    assert audit["candidate_count"] == 2
    assert audit["eligible_pair_count"] == 0
    assert not audit["gate_passed"] and not audit["preference_gate_passed"]
    assert audit["status"] == "completed_no_pairs"
    rescore_status = read_json(output / "rescore_status.json")
    assert rescore_status["operation_succeeded"] is True
    assert rescore_status["preference_gate_passed"] is False
    assert rescore_status["outcome"] == "completed_no_pairs"
    manifest = read_json(output / "rescore_manifest.json")
    assert manifest["model_calls"] == 0
    assert manifest["candidates"][1]["old_verdict"] == "rejected"
    assert manifest["candidates"][1]["new_verdict"] == "accepted"
    old = read_json(group / "B/slow_memory/trajectories.jsonl")
    new = read_json(output / group.name / "B/slow_memory/trajectories.jsonl")
    assert old["trajectory_id"] != new["trajectory_id"]
    for key in (
        "prompt",
        "response",
        "response_hash",
        "context_hash",
        "spatial_context",
        "tool_calls",
        "tool_results",
    ):
        assert old.get(key) == new.get(key)
    assert new["evidence"]["verdict"] == "accepted"
    assert (
        read_json(output / group.name / "B/rescore_origin.json")["result"]["verdict"]
        == "rejected"
    )


@pytest.mark.parametrize(
    "failure", ["tampered_asset", "restored_state", "existing_output"]
)
def test_offline_rescore_rejects_unverifiable_or_overwritten_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from scenesmith.scene_expert.slow_memory.paired_rescore import rescore_pairs

    source, output = tmp_path / "008", tmp_path / "009"
    group = _group(source)
    _mock_rescore_physics(monkeypatch, corrupt_state=failure == "restored_state")
    if failure == "tampered_asset":
        (group / "B/raw_scene/geometry.sdf").write_text("changed")
    if failure == "existing_output":
        output.mkdir()
        (output / "keep.txt").write_text("existing run")
    before = tree_hashes(source)
    with pytest.raises(ValueError):
        rescore_pairs(source, output)
    assert tree_hashes(source) == before
    if failure == "existing_output":
        assert (output / "keep.txt").read_text() == "existing run"
    if failure == "restored_state":
        assert read_json(output / "rescore_status.json")["status"] == "failed"
        assert not (output / "dpo/all.jsonl").exists()
