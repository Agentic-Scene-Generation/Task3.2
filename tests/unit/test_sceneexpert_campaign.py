"""Bulk collection partitions, relative quality gates and training scope contracts."""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scenesmith.scene_expert.slow_memory.campaign import (
    freeze_memory,
    prepare_campaign,
    save_campaign,
    task_id,
)
from scenesmith.scene_expert.slow_memory.dpo import (
    build_preference_pairs,
    export_dpo_dataset,
)
from scenesmith.scene_expert.slow_memory.paired import reserve_group
from scenesmith.scene_expert.slow_memory.relative import report_profile
from scenesmith.scene_expert.slow_memory.schemas import (
    DPOPreferencePair,
    TrajectoryRecord,
)
from scenesmith.scene_expert.slow_memory.training import (
    apply_training_profile,
    evaluate_training_promotion,
)


def record(name, *, failures=2, score=0.8, depth=0.01):
    report = {
        "case_pack": {
            "intent_contract": {"constraints": [{"constraint_id": "bed_count"}]},
            "physics_evidence": {
                "available": True,
                "collisions": [{"penetration_depth_m": depth}],
            },
        },
        "summary": {"scene_summary": {"score": score, "fail": failures, "unknown": 0}},
    }
    return TrajectoryRecord(
        trajectory_id=name,
        created_at="now",
        run_id=name,
        scene_id="scene",
        task_id="task",
        context_hash="same",
        stage="furniture",
        agent_role="designer",
        event="initial",
        task_type="designer_initial",
        prompt="place a bed",
        response=name,
        response_hash=name,
        source_refs=["raw_state.json"],
        evidence={
            "evidence_id": name,
            "verdict": "rejected" if failures else "accepted",
            "authoritative": True,
            "kind": "deterministic",
            "report_ref": "report.json",
            "quality_score": score,
            "details": {"relative_profile": report_profile(report)},
        },
        outcome={
            "execution_complete": True,
            "tool_call_valid": True,
            "hard_passed": failures == 0,
            "hard_violation_count": failures,
            "deterministic_score": score,
        },
    )


def test_relative_imperfect_winner_retains_observed_rejected_verdict():
    a, b = record("a"), record("b", failures=4, score=0.6, depth=0.02)
    assert build_preference_pairs([a, b])[0] == []
    pairs, errors = build_preference_pairs(
        [a, b], preference_policy="verified_relative_v1"
    )
    assert not errors and len(pairs) == 1
    assert pairs[0].chosen_evidence.verdict == "rejected"
    assert pairs[0].chosen_outcome.hard_passed is False
    DPOPreferencePair.model_validate_json(pairs[0].model_dump_json())


@pytest.mark.parametrize(
    "defect",
    [
        "worse_physics",
        "context",
        "execution",
        "missing_profile",
        "tie",
        "unknown",
        "tampered_count",
    ],
)
def test_relative_policy_never_relaxes_evidence_integrity(defect):
    a, b = record("a"), record("b", failures=4, score=0.6, depth=0.02)
    if defect == "worse_physics":
        a.evidence.details["relative_profile"]["max_penetration_m"] = 0.05
    elif defect == "context":
        a.context_hash = "different"
    elif defect == "execution":
        a.outcome.execution_complete = False
    elif defect == "missing_profile":
        a.evidence.details.clear()
    elif defect == "tie":
        b = record("b")
    elif defect == "unknown":
        a.evidence.details["relative_profile"]["unknown_checks"] = 1
    else:
        a.outcome.hard_violation_count = 1
    assert not build_preference_pairs([a, b], preference_policy="verified_relative_v1")[
        0
    ]


def test_relative_accepts_measured_degraded_difference_between_passing_candidates():
    a, b = record("a", failures=0, score=0.99), record("b", failures=0, score=0.92)
    assert (
        len(build_preference_pairs([a, b], preference_policy="verified_relative_v1")[0])
        == 1
    )


def test_campaign_is_frozen_disjoint_and_matches_collector_task_ids(tmp_path):
    plan = prepare_campaign(Path("scripts/assets/annotations.csv"))
    assert plan["counts"] == {"train": 128, "validation": 24, "test": 32}
    assert len(plan["split_assignments"]) == 184
    assert all(row["sceneeval_id"] >= 100 for row in plan["tasks"])
    assert plan == prepare_campaign(Path("scripts/assets/annotations.csv"))
    assert (
        task_id(" a   bed ")
        == "task_" + hashlib.sha256(json.dumps("a bed").encode()).hexdigest()[:16]
    )
    path = tmp_path / "campaign.json"
    save_campaign(path, plan)
    save_campaign(path, plan)
    with pytest.raises(ValueError):
        save_campaign(path, {**plan, "seed": 43})


def test_campaign_identity_survives_windows_server_newlines(tmp_path):
    source = Path("scripts/assets/annotations.csv").read_bytes().replace(b"\r\n", b"\n")
    linux, windows = tmp_path / "lf.csv", tmp_path / "crlf.csv"
    linux.write_bytes(source)
    windows.write_bytes(source.replace(b"\n", b"\r\n"))
    assert prepare_campaign(linux) == prepare_campaign(windows)


def test_bulk_group_slots_are_unique_under_parallel_reservation(tmp_path):
    with ThreadPoolExecutor(8) as pool:
        groups = list(
            pool.map(lambda i: reserve_group(tmp_path, str(i), 128), range(32))
        )
    assert len(set(groups)) == 32 and all(group.is_dir() for group in groups)
    with pytest.raises(ValueError):
        reserve_group(tmp_path, "bad", 513)


def test_first_turn_export_excludes_environment_responses_and_obeys_fixed_split(
    tmp_path,
):
    a, b = record("a", failures=0), record("b", failures=3, score=0.5)
    for row in (a, b):
        row.completion_messages = [
            {"role": "assistant", "content": row.trajectory_id},
            {"role": "tool", "content": "environment result"},
            {"role": "assistant", "content": "later action"},
        ]
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(row.model_dump_json() for row in (a, b)), encoding="utf-8"
    )
    output = tmp_path / "out"
    manifest = export_dpo_dataset(
        trajectory_sources=[source],
        output_dir=output,
        split_assignments={"task": "validation"},
        completion_view="first_turn",
    )
    assert manifest["stats"]["split_counts"] == {"train": 0, "validation": 1, "test": 0}
    pair = json.loads((output / "validation.jsonl").read_text())
    assert pair["chosen"] == [{"role": "assistant", "content": "a"}]
    assert "environment result" not in json.dumps(pair["rejected"])


def test_initial_training_profile_removes_unsupported_coverage_without_affecting_full():
    original = {
        "data": {"minimum_unique_train_stages": 3},
        "training": {},
        "quality_gate": {},
    }
    before = deepcopy(original)
    initial = apply_training_profile(original, "furniture_initial")
    assert original == before
    assert initial["data"]["minimum_unique_train_stages"] == 1
    assert initial["data"]["minimum_train_pairs"] == 16
    assert initial["quality_gate"]["require_validation"] is True
    assert initial["training"]["loss_type"] == "sigmoid"
    smoke = apply_training_profile(original, "pipeline_smoke")
    assert smoke["training"]["max_steps"] == 2
    assert smoke["data"]["minimum_train_pairs"] == 1
    assert smoke["publish"]["push_to_hub"] is False
    assert not evaluate_training_promotion(
        smoke, evaluation_metrics={"eval_rewards/accuracies": 1.0}
    )["promotable"]


def test_memory_is_initialized_once_and_verified_on_resume(tmp_path):
    plan = prepare_campaign(
        Path("scripts/assets/annotations.csv"), train=2, validation=1, test=1
    )
    memory = freeze_memory(tmp_path, plan)
    assert freeze_memory(tmp_path, plan) == memory
    marker = tmp_path / "memory_snapshot.json"
    data = json.loads(marker.read_text())
    data["snapshot"]["bank_id"] = "tampered"
    marker.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="frozen memory changed"):
        freeze_memory(tmp_path, plan)


def test_relative_pair_cannot_relabel_unknown_or_unverified_policy():
    a, b = record("a"), record("b", failures=3, score=0.5, depth=0.02)
    pair = build_preference_pairs([a, b], preference_policy="verified_relative_v1")[0][
        0
    ]
    payload = pair.model_dump()
    payload["chosen_evidence"]["verdict"] = "unknown"
    with pytest.raises(ValueError):
        DPOPreferencePair.model_validate(payload)
    payload = pair.model_dump()
    payload["provenance"]["preference_policy"] = "strict"
    with pytest.raises(ValueError):
        DPOPreferencePair.model_validate(payload)


def test_first_turn_omits_pairs_whose_only_difference_is_downstream(tmp_path):
    a, b = record("a", failures=0), record("b", failures=3, score=0.5)
    for row in (a, b):
        row.completion_messages = [
            {"role": "assistant", "content": "observe"},
            {"role": "tool", "content": "scene"},
            {"role": "assistant", "content": row.trajectory_id},
        ]
    path = tmp_path / "source.jsonl"
    path.write_text("\n".join(r.model_dump_json() for r in (a, b)))
    result = export_dpo_dataset(
        trajectory_sources=[path],
        output_dir=tmp_path / "out",
        completion_view="first_turn",
    )
    assert sum(result["stats"]["split_counts"].values()) == 0
    assert result["stats"]["raw_preference_pair_count"] == 1


def test_tool_call_ids_do_not_create_a_preference(tmp_path):
    a, b = record("a", failures=0), record("b", failures=3, score=0.5)
    for row in (a, b):
        row.completion_messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": row.trajectory_id,
                        "type": "function",
                        "function": {"name": "observe_scene", "arguments": {}},
                    }
                ],
            }
        ]
    path = tmp_path / "source.jsonl"
    path.write_text("\n".join(r.model_dump_json() for r in (a, b)))
    result = export_dpo_dataset(
        trajectory_sources=[path],
        output_dir=tmp_path / "out",
        completion_view="first_turn",
    )
    assert result["stats"]["eligible_pair_count"] == 0


@pytest.mark.parametrize("infrastructure_failure", [False, True, "masked_zero_exit"])
def test_campaign_chunks_resume_only_missing_tasks(
    tmp_path, monkeypatch, infrastructure_failure
):
    from scenesmith.scene_expert.slow_memory import paired_runtime
    from scripts import run_sceneexpert_campaign as runner

    plan = prepare_campaign(
        Path("scripts/assets/annotations.csv"), train=4, validation=1, test=1
    )
    save_campaign(tmp_path / "campaign.json", plan)
    completed = set()
    launched = []
    mapping = {str(row["sceneeval_id"]): row["task_id"] for row in plan["tasks"]}
    monkeypatch.setitem(
        sys.modules,
        "fcntl",
        SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *_: None),
    )
    monkeypatch.setattr(
        runner, "collect_pair_code_provenance", lambda: {"source_bundle_hash": "fixed"}
    )
    monkeypatch.setattr(runner, "validated_sources", lambda _: ([], set(completed), []))
    monkeypatch.setattr(
        paired_runtime, "snapshot_codec_preflight", lambda: {"valid": True}
    )

    def execute(*_, env, **__):
        assert env["PIPELINE_STOP_STAGE"] == "furniture"
        assert env["SCENEEXPERT_MEMORY_READ_ONLY"] == "true"
        assert env["SCENEEXPERT_COMPONENT_MEMORY_WRITER_ENABLED"] == "false"
        ids = env["SCENE_SELECTION"].split(",")
        launched.extend(ids)
        if not infrastructure_failure:
            completed.update(mapping[value] for value in ids)
        return SimpleNamespace(returncode=1 if infrastructure_failure is True else 0)

    monkeypatch.setattr(
        runner, "run_collection", lambda _, env: execute(env=env).returncode
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["campaign", "--root", str(tmp_path), "--max-tasks", "4", "--chunk-size", "2"],
    )
    result = runner.main()
    if infrastructure_failure:
        assert result == 2 and len(launched) == 2
        return
    assert result == 0 and len(launched) == 4
    report = json.loads((tmp_path / "campaign_audit.json").read_text())
    assert (
        report["selected_missing_task_ids"] == []
        and len(report["missing_task_ids"]) == 1
    )
    monkeypatch.setattr(
        sys, "argv", ["campaign", "--root", str(tmp_path), "--chunk-size", "2"]
    )
    assert runner.main() == 0
    assert len(launched) == len(set(launched)) == 5
    assert not any(
        mapping[value] == row["task_id"]
        for value in launched
        for row in plan["tasks"]
        if row["split"] == "test"
    )
