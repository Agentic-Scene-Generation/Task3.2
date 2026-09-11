"""Synthetic CPU evidence tests; not evidence of a real-model Memory gain."""

from __future__ import annotations

import hashlib
import json

import pytest

from scenesmith.scene_expert.memory.diagnostics import diagnose_scene
from scenesmith.scene_expert.memory.injection import build_memory_injection_bundle
from scenesmith.scene_expert.memory.state import build_memory_scene_state
from scenesmith.scene_expert.memory.usage import collect_memory_usage
from scenesmith.scene_expert.schemas import MemoryPack, StageBrief
from tests.unit.test_memory_context import candidate, choice, scene, task
from tests.unit.test_memory_evaluation import fixture, write


def prepared(tmp_path, fault=""):
    source = candidate(tmp_path)
    pack = MemoryPack(
        selections=[source], current_scene_state=build_memory_scene_state(scene())
    )
    choices = [choice(source)]
    if fault == "planner_not_accepted":
        choices = []
    elif fault == "planner_rejected":
        choices = [
            choice(source, decision="rejected", reason="Not useful for this task")
        ]
    elif fault == "source_hash_mismatch":
        choices = [choice(source, source_content_hash="wrong-hash")]
    brief = StageBrief(
        stage="furniture", stage_objective="Office", memory_adaptations=choices
    )
    bundle = build_memory_injection_bundle(
        stage="furniture", stage_brief=brief, memory_pack=pack, task_spec=task()
    )
    return {
        "stages": {
            "furniture": {
                "retrieval": pack.model_dump(mode="json"),
                "injection": bundle.model_dump(mode="json"),
            }
        }
    }, bundle


@pytest.mark.parametrize(
    "fault", ["planner_not_accepted", "planner_rejected", "source_hash_mismatch"]
)
def test_saved_rejection_reason_survives_replay_and_usage(tmp_path, fault, caplog):
    activity, bundle = prepared(tmp_path, fault)
    assert not bundle.accepted_items
    write(tmp_path / "scene_expert/memory_activity.json", activity)
    report = diagnose_scene(tmp_path)
    row = report["usage"]["items"][0]
    assert row["adaptation_rejection_reasons"] == [fault]
    assert row["adaptation_decision_observed"]
    assert report["stages"][0]["replay_matches_recorded"] is True
    assert report["summary"]["accepted"] == report["summary"]["delivered"] == 0
    assert fault in caplog.text


def test_missing_old_decisions_are_unknown_not_rejected(tmp_path):
    activity, _ = prepared(tmp_path)
    activity["stages"]["furniture"]["injection"].pop("adaptation_decisions")
    result = collect_memory_usage(tmp_path, activity)
    assert result["items"][0]["adaptation_rejection_reasons"] == []
    assert not result["items"][0]["adaptation_decision_observed"]
    assert result["items"][0]["accepted"]


def test_positive_acceptance_delivery_and_action_without_any_model_or_bank_write(
    tmp_path,
):
    # The Planner response and tool outcomes are deliberate synthetic fixtures.
    activity, bundle = prepared(tmp_path)
    _, payload, reference = fixture(tmp_path)
    payload["prompt"] = "Native designer request\n" + bundle.accepted_items[0].text
    write(reference, payload)
    write(tmp_path / "scene_expert/memory_activity.json", activity)
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    result = diagnose_scene(tmp_path)
    assert result["stages"][0]["replay_matches_recorded"]
    assert result["summary"] == {
        "retrieved": 1,
        "accepted": 1,
        "delivered": 1,
        "action_observed": 1,
    }
    assert result["usage"]["items"][0]["causal_benefit"] is None
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert before == after


def test_absent_runtime_artifact_cannot_pass_diagnostic(tmp_path):
    result = diagnose_scene(tmp_path)
    assert result["warnings"]
    assert result["summary"]["accepted"] == 0


@pytest.mark.parametrize(
    "activity",
    [
        {"stages": [], "stage_attempt_history": [{"stage": "furniture"}]},
        {"stages": {"furniture": None}},
        {"stages": {"furniture": {"injection": {"accepted_items": [None]}}}},
    ],
)
def test_malformed_optional_evidence_produces_warning_not_success(tmp_path, activity):
    write(tmp_path / "scene_expert/memory_activity.json", activity)
    result = diagnose_scene(tmp_path)
    assert result["warnings"]
    assert result["summary"]["delivered"] == 0


def test_scoped_role_audit_exposes_missing_anchor(tmp_path):
    from tests.unit.test_memory_relation_scope import fixture as scoped_fixture

    source, selection, pack, task_spec = scoped_fixture(tmp_path)
    selection.bindings = selection.bindings[:1]
    brief = StageBrief(
        stage="furniture", stage_objective="Bedroom", memory_adaptations=[selection]
    )
    bundle = build_memory_injection_bundle(
        stage="furniture", stage_brief=brief, memory_pack=pack, task_spec=task_spec
    )
    write(
        tmp_path / "scene_expert/memory_activity.json",
        {
            "stages": {
                "furniture": {
                    "retrieval": pack.model_dump(),
                    "injection": bundle.model_dump(),
                }
            }
        },
    )
    audit = diagnose_scene(tmp_path)["stages"][0]["binding_audit"][0]
    assert audit["source_relation_indices"] == [0]
    assert audit["unbound_source_roles"] == ["wall"]
    assert "wardrobe" not in audit["required_source_roles"]


def test_replay_disagreement_is_not_silently_accepted(tmp_path):
    activity, _ = prepared(tmp_path, "planner_not_accepted")
    activity["stages"]["furniture"]["injection"]["adaptation_decisions"] = []
    write(tmp_path / "scene_expert/memory_activity.json", activity)
    report = diagnose_scene(tmp_path)
    assert "replay_differs_from_recorded_acceptance" in report["warnings"]


def test_metrics_export_retains_final_adaptation_reasons(tmp_path):
    from scenesmith.scene_expert.run_metrics import (
        collect_run_metrics,
        write_run_metrics,
    )

    root = tmp_path / "run"
    scene_dir = root / "critic_on/batch_001/hydra/scene_000"
    (root / "assigned_cases.csv").parent.mkdir(parents=True)
    (root / "assigned_cases.csv").write_text(
        "batch_id,scene_index,prompt,case_id\nbatch_001,0,Office,c1\n"
    )
    activity, _ = prepared(tmp_path, "source_hash_mismatch")
    write(scene_dir / "scene_expert/memory_activity.json", activity)
    write(scene_dir / "scene_status.json", {"status": "failed", "attempt": 1})
    report = collect_run_metrics(root)
    assert report["summary"]["memory_adaptation_rejection_reasons"] == {
        "source_hash_mismatch": 1
    }
    assert report["summary"]["memory_adaptation_rejection_count"] == 1
    assert report["summary"]["memory_adaptation_decision_unknown_count"] == 0
    write_run_metrics(report)
    export = json.loads((root / "metrics/memory_usage.json").read_text())
    assert export["cases"][0]["items"][0]["adaptation_rejection_reasons"] == [
        "source_hash_mismatch"
    ]
