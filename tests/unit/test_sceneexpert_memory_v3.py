from __future__ import annotations

import json

from pathlib import Path
from tempfile import TemporaryDirectory

from scenesmith.scene_expert.experiment_identity import (
    shared_base_scene_identity,
    stable_config_hash,
    stable_control_signature,
    stable_experiment_signature,
    stable_source_bundle_hash,
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
    assert record.promotion_scope == "scene"
    assert record.source_scene_passed is True
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


def test_experiment_signature_ignores_per_scene_prompt_and_identity() -> None:
    first = {
        "experiment": {
            "name": "ablation_4c",
            "prompt": "A classroom.",
            "scene_id": 1,
            "pipeline": {"stop_stage": "manipuland"},
        }
    }
    second = {
        "experiment": {
            "name": "ablation_4c",
            "prompt": "A bedroom.",
            "scene_id": 99,
            "pipeline": {"stop_stage": "manipuland"},
        }
    }

    assert stable_config_hash(first) != stable_config_hash(second)
    assert stable_experiment_signature(first) == stable_experiment_signature(second)


def test_experiment_signature_keeps_agent_prompt_template_selection() -> None:
    first = {
        "experiment": {"prompts": ["A classroom."]},
        "furniture_agent": {"agents": {"designer": {"prompt": "DESIGNER_AGENT"}}},
    }
    second = {
        "experiment": {"prompts": ["A bedroom."]},
        "furniture_agent": {
            "agents": {"designer": {"prompt": "EXPERIMENTAL_DESIGNER_AGENT"}}
        },
    }

    assert stable_experiment_signature(first) != stable_experiment_signature(second)


def test_source_bundle_hash_is_path_separator_and_order_stable() -> None:
    first = {"scenesmith/a.py": "aaa", "scenesmith/b.py": "bbb"}
    second = {"scenesmith\\b.py": "bbb", "scenesmith\\a.py": "aaa"}

    assert stable_source_bundle_hash(first) == stable_source_bundle_hash(second)


def test_control_signature_ignores_only_declared_memory_treatment() -> None:
    baseline = {
        "scene_expert": {
            "mode": "full",
            "evaluation": {"arm": "memory_off", "pair_id": "pair-a"},
            "components": {
                "fast_memory_retrieval": {"enabled": False},
                "memory_writer": {"enabled": False},
            },
        },
        "furniture_agent": {"openai": {"model": "qwen-test"}},
    }
    treatment = json.loads(json.dumps(baseline))
    treatment["scene_expert"]["evaluation"]["arm"] = "memory_on"
    treatment["scene_expert"]["components"]["fast_memory_retrieval"] = {"enabled": True}

    assert stable_experiment_signature(baseline) != stable_experiment_signature(
        treatment
    )
    assert stable_control_signature(
        baseline, controlled_dimension="fast_memory_retrieval"
    ) == stable_control_signature(
        treatment, controlled_dimension="fast_memory_retrieval"
    )

    treatment["scene_expert"]["components"]["memory_writer"] = {"enabled": True}
    assert stable_control_signature(
        baseline, controlled_dimension="fast_memory_retrieval"
    ) != stable_control_signature(
        treatment, controlled_dimension="fast_memory_retrieval"
    )


def test_shared_base_identity_is_path_independent_and_content_sensitive(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first" / "scene_003"
    second = tmp_path / "second" / "scene_003"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "house_layout.json").write_text('{"rooms": 1}', encoding="utf-8")
    (second / "house_layout.json").write_text('{"rooms": 1}', encoding="utf-8")

    first_identity = shared_base_scene_identity(tmp_path / "first", scene_index=3)
    second_identity = shared_base_scene_identity(tmp_path / "second", scene_index=3)
    assert first_identity["fingerprint"] == second_identity["fingerprint"]

    (second / "house_layout.json").write_text('{"rooms": 2}', encoding="utf-8")
    assert (
        first_identity["fingerprint"]
        != shared_base_scene_identity(tmp_path / "second", scene_index=3)["fingerprint"]
    )


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
