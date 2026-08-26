from __future__ import annotations

import json

from pathlib import Path
from tempfile import TemporaryDirectory

from scenesmith.scene_expert.experiment_identity import (
    stable_config_hash,
    stable_experiment_signature,
)
from scenesmith.scene_expert.memory.activity import MemoryActivityLogger
from scenesmith.scene_expert.memory.injection import build_memory_injection_bundle
from scenesmith.scene_expert.memory.schemas import MEMORY_SCHEMA_VERSION, SuccessCase
from scenesmith.scene_expert.schemas import (
    MemoryPack,
    RetrievedMemorySelection,
    SceneTaskSpec,
    StageBrief,
    StageExecutionEvidence,
    StageVerifyReport,
)


def _task_spec() -> SceneTaskSpec:
    return SceneTaskSpec(
        room_type="classroom",
        style="modern",
        required_large_objects=["student desk", "chair"],
    )


def test_v2_record_loads_with_conservative_v3_defaults() -> None:
    record = SuccessCase.model_validate(
        {
            "schema_version": "sceneexpert.memory.v2",
            "case_id": "legacy_case",
            "room_type": "classroom",
            "stage": "furniture",
        }
    )

    assert MEMORY_SCHEMA_VERSION == "sceneexpert.memory.v3"
    assert record.schema_version == "sceneexpert.memory.v2"
    assert record.spatial_relations == []
    assert record.provenance.trace_id == ""


def test_experiment_signature_ignores_runtime_paths_and_ports() -> None:
    first = {
        "experiment": {"name": "ablation_4c", "quality_failure_policy": "degraded"},
        "output_dir": "/run/one",
        "blender_server_port": 12000,
        "scene_expert": {"memory_dir": "/memory/one", "mode": "harness_memory"},
    }
    second = {
        "experiment": {"name": "ablation_4c", "quality_failure_policy": "degraded"},
        "output_dir": "/run/two",
        "blender_server_port": 14000,
        "scene_expert": {"memory_dir": "/memory/two", "mode": "harness_memory"},
    }

    assert stable_config_hash(first) != stable_config_hash(second)
    assert stable_experiment_signature(first) == stable_experiment_signature(second)
    second["experiment"]["quality_failure_policy"] = "strict"
    assert stable_experiment_signature(first) != stable_experiment_signature(second)


def test_experiment_signature_ignores_batch_labels_and_port_ranges() -> None:
    first = {
        "name": "critic_on_batch_001",
        "experiment": {"pipeline": {"stop_stage": "manipuland"}},
        "scene_expert": {"ports": {"server_port_range": [13000, 13399]}},
    }
    second = {
        "name": "critic_on_batch_002",
        "experiment": {"pipeline": {"stop_stage": "manipuland"}},
        "scene_expert": {"ports": {"server_port_range": [13400, 13799]}},
    }

    assert stable_experiment_signature(first) == stable_experiment_signature(second)
    second["experiment"]["name"] = "semantically_distinct_experiment"
    assert stable_experiment_signature(first) != stable_experiment_signature(second)


def test_canonical_bundle_does_not_repeat_memory_directives() -> None:
    success = "[Positive/furniture] Keep desks aligned to the teaching wall."
    failure = "[Avoid/furniture] Do not face student chairs away from desks."
    skill = "[Skill: align_student_seating]\nSteps:\n  1. Bind chairs to desks."
    pack = MemoryPack(
        success_hints=[success],
        failure_hints=[failure],
        skill_texts=[skill],
        placement_reference="Reference relation: chair faces desk.",
        success_case_ids=["success_1"],
        failure_case_ids=["failure_1"],
        skill_names=["align_student_seating"],
    )
    bundle = build_memory_injection_bundle(
        stage="furniture",
        stage_brief=StageBrief(
            stage="furniture",
            stage_objective="Create a usable classroom.",
        ),
        memory_pack=pack,
    )

    assert bundle.final_text.count(success) == 1
    assert bundle.final_text.count(failure) == 1
    assert bundle.final_text.count(skill) == 1
    assert bundle.final_text.count(pack.placement_reference) == 1
    assert bundle.selected_memory_ids == [
        "success_1",
        "failure_1",
        "align_student_seating",
    ]


def test_memory_activity_links_selection_injection_and_outcome() -> None:
    with TemporaryDirectory() as temp_dir:
        logger = MemoryActivityLogger(
            temp_dir,
            scene_id="scene_003",
            task_spec=_task_spec(),
            experiment_signature="stable-signature",
            task_id="task_current",
            run_id="run_current",
        )
        pack = MemoryPack(
            success_case_ids=["success_1"],
            memory_bank_id="bank_1",
            memory_bank_revision=7,
            selections=[
                RetrievedMemorySelection(
                    memory_id="success_1",
                    memory_type="success",
                    rank=1,
                    score=0.83,
                    source_path="/memory/success_cases.jsonl",
                    source_task_ids=["task_previous"],
                )
            ],
        )
        bundle = build_memory_injection_bundle(
            stage="furniture",
            stage_brief=StageBrief(
                stage="furniture",
                stage_objective="Create a usable classroom.",
            ),
            memory_pack=pack,
        )
        evidence = StageExecutionEvidence(
            retrieved_memory_ids=["success_1"],
            designer_prompt_contains_memory=True,
            final_injection_hash="abc123",
        )
        logger.record_pre_stage(
            stage="furniture",
            memory_pack=pack,
            relation_context=None,
            planner_trace={"status": "ok"},
            injection_bundle=bundle,
            execution_evidence=evidence,
        )
        logger.record_post_stage(
            stage="furniture",
            verify_report=StageVerifyReport(stage="furniture", pass_stage=True),
            repair_actions=[],
            scene_state_path="/run/scene_003/final_furniture",
        )

        payload = json.loads(
            (Path(temp_dir) / "memory_activity.json").read_text(encoding="utf-8")
        )

    stage = payload["stages"]["furniture"]
    assert stage["retrieval"]["selections"][0]["source_task_ids"] == ["task_previous"]
    assert stage["injection"]["selected_memory_ids"] == ["success_1"]
    assert stage["utility_observations"][0]["outcome"] == "unknown"
    assert stage["utility_observations"][0]["stage_passed"] is True
    assert stage["utility_observations"][0]["injected"] is True
    assert stage["utility_observations"][0]["task_id"] == "task_current"
    assert stage["utility_observations"][0]["run_id"] == "run_current"
