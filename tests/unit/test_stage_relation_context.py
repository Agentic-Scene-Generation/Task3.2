"""Coverage for hard-intent projection and advisory HSSD relation policy."""

from __future__ import annotations

import gzip
import json
from copy import deepcopy
from types import SimpleNamespace

from scenesmith.scene_expert.global_planner import (
    GlobalPlanner,
    _add_floor_plan_reservation_guidance,
)
from scenesmith.scene_expert.hooks import (
    _attach_stage_relation_context,
    _format_stage_relation_context,
)
from scenesmith.scene_expert.relation_context import StageRelationProjector
from scenesmith.scene_expert.schemas import (
    HarnessContext,
    MemoryPack,
    SceneTaskSpec,
    StageBudget,
    StageRelationContext,
    StageBrief,
)
from scenesmith.scene_expert.task_compiler import TaskCompiler
from scenesmith.scenebenchmark_critic import asset_library_annotations
from scenesmith.scenebenchmark_critic.intent_contract import (
    attach_intent_contract_to_case_pack,
    build_intent_contract,
)
from scenesmith.scenebenchmark_critic.intent_schema import validate_intent_contract
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.builder import (
    _build_explicit_target_relation_checks,
)


def _response(content: str = "", *, reasoning_content: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                )
            )
        ],
        usage=None,
    )


def _client_with_responses(responses: list[SimpleNamespace]):
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    return (
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        calls,
    )


def _task_spec(**updates) -> SceneTaskSpec:
    data = {
        "room_type": "bedroom",
        "style": "standard",
        "required_large_objects": ["bed", "nightstand"],
    }
    data.update(updates)
    return SceneTaskSpec(**data)


def _relation(
    relation_type: str,
    target: str,
    *,
    confidence: float = 0.86,
    provenance: str = "seed_rule:test",
) -> dict:
    return {
        "relation_type": relation_type,
        "target_kind": "asset_category",
        "target_category": target,
        "distance_range_m": [0.0, 0.35],
        "relative_facing": "source_front_points_to_target",
        "relative_position": "beside",
        "height_relation": "same_floor",
        "confidence": confidence,
        "provenance": provenance,
    }


def _write_lookup(tmp_path) -> str:
    records = {}
    for index in range(5):
        priors = [
            _relation("beside", "nightstand"),
            _relation("beside", "mattress"),
            _relation(
                "funeval_used_with",
                "nightstand",
                confidence=0.95,
                provenance="asset_record:funeval_functional_dependency:commonsense_llm",
            ),
        ]
        if index < 3:
            priors.append(_relation("against", "wall", confidence=0.90))
        records[f"bed-{index}"] = {
            "category": "bed",
            "category_key": "bed",
            "relation_priors": priors,
        }
        records[f"chair-{index}"] = {
            "category": "chair",
            "category_key": "chair",
            "relation_priors": [
                _relation("faces", "desk"),
                _relation("faces", "table"),
            ],
        }
    path = tmp_path / "hssd_lookup.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(records, stream)
    return str(path)


def test_projection_preserves_full_contract_and_exact_stage_ids(tmp_path) -> None:
    contract = {
        "constraints": [
            {
                "constraint_id": "furniture-1",
                "stage": "furniture",
                "relation": "faces",
                "subjects": {"category": "bed"},
                "targets": {"category": "room"},
            },
            {
                "constraint_id": "wall-1",
                "stage": "wall_mounted",
                "relation": "on_wall",
                "subjects": {"category": "mirror"},
                "targets": {"category": "wall"},
            },
        ]
    }
    original = deepcopy(contract)
    context = StageRelationProjector(lookup_path=_write_lookup(tmp_path)).project(
        stage="furniture",
        task_spec=_task_spec(),
        intent_contract=contract,
    )

    assert context.hard_constraint_ids == ["furniture-1"]
    assert context.hard_constraints[0] == contract["constraints"][0]
    assert context.projection_coverage == 1.0
    assert contract == original

    injected = _format_stage_relation_context(context)
    assert "furniture-1" in injected
    assert "wall-1" not in injected
    assert injected.index("Advisory HSSD") < injected.index("Hard Intent Contract")


def test_floor_plan_projects_only_explicit_future_wall_anchors(tmp_path) -> None:
    contract = {
        "constraints": [
            {
                "constraint_id": "bed-back-wall",
                "stage": "furniture",
                "relation": "centered_on_wall",
                "subjects": {"category": "bed", "count": 1},
                "targets": {"category": "wall", "role": "back"},
            },
            {
                "constraint_id": "mirror-wall",
                "stage": "wall_mounted",
                "relation": "on_wall",
                "subjects": {"category": "mirror", "count": 1},
                "targets": {"category": "wall"},
            },
            {
                "constraint_id": "stool-front",
                "stage": "furniture",
                "relation": "in_front_of",
                "subjects": {"category": "stool", "count": 1},
                "targets": {"category": "dressing_table", "count": 1},
            },
        ]
    }

    context = StageRelationProjector(lookup_path=_write_lookup(tmp_path)).project(
        stage="floor_plan",
        task_spec=_task_spec(required_large_objects=["bed", "stool"]),
        intent_contract=contract,
    )

    assert context.hard_constraints == []
    assert [item["constraint_id"] for item in context.floor_plan_reservations] == [
        "bed-back-wall",
        "mirror-wall",
    ]
    assert context.floor_plan_manifest is not None
    assert context.floor_plan_manifest.enabled is False


def test_floor_plan_manifest_projects_media_zones_and_explicit_windows(
    tmp_path,
) -> None:
    contract = {
        "constraints": [
            {
                "constraint_id": "living-media",
                "stage": "furniture",
                "relation": "across_from",
                "subjects": {"category": "sofa", "count": 1},
                "targets": {"category": "tv_stand", "count": 1},
            },
            {
                "constraint_id": "windows-required",
                "stage": "floor_plan",
                "relation": "required_count",
                "subjects": {"category": "window", "count": 2},
            },
        ]
    }
    context = StageRelationProjector(
        lookup_path=_write_lookup(tmp_path),
        floor_plan_reservation_gate_enabled=True,
    ).project(
        stage="floor_plan",
        task_spec=_task_spec(
            required_large_objects=["sofa", "tv_stand"],
            functional_zones=["living_zone", "dining_zone"],
        ),
        intent_contract=contract,
    )

    manifest = context.floor_plan_manifest
    assert manifest is not None and manifest.enabled
    assert manifest.explicit_window_required
    assert manifest.explicit_window_count == 2
    assert [item.kind for item in manifest.reservations].count(
        "opposed_anchor_pair"
    ) == 1
    assert (
        sum(
            item.min_zone_area_m2
            for item in manifest.reservations
            if item.kind == "functional_zone"
        )
        == 12.0
    )


def test_floor_plan_brief_reserves_opening_free_wall_segments() -> None:
    context = HarnessContext(
        stage="floor_plan",
        task_spec=_task_spec(required_large_objects=["bed", "wardrobe"]),
        memory_pack=MemoryPack(),
        relation_context=StageRelationContext(
            stage="floor_plan",
            floor_plan_reservations=[
                {
                    "constraint_id": "bed-back-wall",
                    "relation": "centered_on_wall",
                    "subjects": {"category": "bed"},
                    "targets": {"category": "wall", "role": "back"},
                },
                {
                    "constraint_id": "wardrobe-side-wall",
                    "relation": "against_wall",
                    "subjects": {"category": "wardrobe"},
                    "targets": {"category": "wall", "role": "side"},
                },
            ],
        ),
    )

    brief = _add_floor_plan_reservation_guidance(
        StageBrief(
            stage="floor_plan",
            stage_objective="Create a bedroom layout.",
            constraints_for_designer=["Keep the entrance accessible."],
        ),
        context,
    )

    text = "\n".join(brief.constraints_for_designer).lower()
    assert "opening-free usable wall segments" in text
    assert "bed centered on the back wall" in text
    assert "wardrobe on a side wall" in text
    assert "doors or windows" in text
    assert (
        "unobstructed by doors or windows" in "\n".join(brief.checks_for_critic).lower()
    )


def test_designer_gets_stage_only_json_and_critic_keeps_full_contract() -> None:
    contract = build_intent_contract(
        "A bedroom with a bed centered on the main wall and a mirror."
    )
    contract["constraints"].append(
        {
            "relation": "on_wall",
            "subjects": {"category": "mirror", "count": 1},
            "targets": {"category": "wall", "count": 1},
            "source": "explicit_prompt",
            "evidence_span": "a mirror mounted on the side wall",
        }
    )
    contract = validate_intent_contract(contract)
    furniture_rows = [
        row for row in contract["constraints"] if row["stage"] == "furniture"
    ]
    other_rows = [row for row in contract["constraints"] if row["stage"] != "furniture"]
    assert furniture_rows and other_rows
    context = StageRelationContext(
        stage="furniture",
        hard_constraints=furniture_rows,
        contract_constraint_count=len(contract["constraints"]),
        projected_constraint_count=len(furniture_rows),
    )
    scene = SimpleNamespace(
        text_description=contract["prompt"],
        scene_expert_original_description=contract["prompt"],
        metadata={},
    )

    _attach_stage_relation_context(
        scene,
        relation_context=context,
        intent_contract=contract,
        task_spec=_task_spec(),
    )

    assert all(row["constraint_id"] in scene.text_description for row in furniture_rows)
    assert all(row["constraint_id"] not in scene.text_description for row in other_rows)
    assert scene.metadata["scenebenchmark_intent_contract"] == contract
    case_pack: dict = {}
    attach_intent_contract_to_case_pack(scene, case_pack)
    assert case_pack["intent_contract"] == contract


def test_hssd_policy_activates_seed_and_audits_suppression(tmp_path) -> None:
    context = StageRelationProjector(lookup_path=_write_lookup(tmp_path)).project(
        stage="furniture",
        task_spec=_task_spec(),
        intent_contract={"constraints": []},
    )

    active = {
        (item.subject_selector.category, item.relation, item.target_selector.category)
        for item in context.advisory_hssd_priors
        if item.target_selector is not None
    }
    assert ("bed", "beside", "nightstand") in active
    reasons = [
        (item.prior.relation, item.prior.target_selector, item.reason)
        for item in context.suppressed_priors
    ]
    assert any(
        relation == "beside" and target is None and reason == "missing_target"
        for relation, target, reason in reasons
    )
    assert any(
        relation == "funeval_used_with" and reason == "low_confidence"
        for relation, _target, reason in reasons
    )
    assert any(
        relation == "against" and reason == "low_confidence"
        for relation, _target, reason in reasons
    )


def test_hssd_target_ambiguity_and_hard_orientation_conflict(tmp_path) -> None:
    contract = {
        "constraints": [
            {
                "constraint_id": "guest-faces-room",
                "stage": "furniture",
                "relation": "faces",
                "subjects": {"category": "guest_chair"},
                "targets": {"category": "room"},
            }
        ]
    }
    context = StageRelationProjector(lookup_path=_write_lookup(tmp_path)).project(
        stage="furniture",
        task_spec=_task_spec(
            room_type="study",
            required_large_objects=[
                "guest_chair",
                "desk",
                "coffee_table",
                "dining_table",
            ],
        ),
        intent_contract=contract,
    )

    assert any(
        item.reason == "hard_conflict"
        and item.prior.target_selector is not None
        and item.prior.target_selector.category == "desk"
        and item.conflicting_constraint_ids == ["guest-faces-room"]
        for item in context.suppressed_priors
    )
    assert any(
        item.reason == "ambiguous_target" and item.prior.relation == "faces"
        for item in context.suppressed_priors
    )
    assert not context.advisory_hssd_priors


def test_inventory_and_compatible_position_hard_constraints_do_not_suppress_prior(
    tmp_path,
) -> None:
    contract = {
        "constraints": [
            {
                "constraint_id": "bed-count",
                "stage": "furniture",
                "relation": "required_count",
                "subjects": {"category": "bed", "count": 1},
            },
            {
                "constraint_id": "nightstand-count",
                "stage": "furniture",
                "relation": "required_count",
                "subjects": {"category": "nightstand", "count": 2},
            },
            {
                "constraint_id": "bed-wall",
                "stage": "furniture",
                "relation": "against_wall",
                "subjects": {"category": "bed"},
                "targets": {"category": "wall"},
            },
        ]
    }
    context = StageRelationProjector(lookup_path=_write_lookup(tmp_path)).project(
        stage="furniture",
        task_spec=_task_spec(),
        intent_contract=contract,
    )

    assert any(
        item.subject_selector.category == "bed"
        and item.relation == "beside"
        and item.target_selector is not None
        and item.target_selector.category == "nightstand"
        for item in context.advisory_hssd_priors
    )
    assert not any(
        item.reason == "hard_conflict"
        and item.prior.subject_selector.category == "bed"
        and item.prior.relation == "beside"
        for item in context.suppressed_priors
    )


def test_hard_wall_pose_suppresses_hssd_orientation_prior(tmp_path) -> None:
    context = StageRelationProjector(lookup_path=_write_lookup(tmp_path)).project(
        stage="furniture",
        task_spec=_task_spec(
            room_type="study",
            required_large_objects=["guest_chair", "desk"],
        ),
        intent_contract={
            "constraints": [
                {
                    "constraint_id": "guest-wall-front",
                    "stage": "furniture",
                    "relation": "against_wall",
                    "subjects": {"category": "guest_chair"},
                    "targets": {"category": "wall"},
                }
            ]
        },
    )

    assert any(
        item.reason == "hard_conflict"
        and item.prior.subject_selector.category == "guest_chair"
        and item.prior.relation == "faces"
        and item.prior.target_selector is not None
        and item.prior.target_selector.category == "desk"
        and item.conflicting_constraint_ids == ["guest-wall-front"]
        for item in context.suppressed_priors
    )


def test_global_planner_sees_relations_and_retries_strict_schema() -> None:
    invalid = json.dumps(
        {
            "stage": "furniture",
            "stage_objective": "Place furniture",
            "unexpected": True,
        }
    )
    valid = json.dumps(
        {
            "stage": "furniture",
            "stage_objective": "Place furniture",
            "recommended_skills": [],
            "constraints_for_designer": [],
            "checks_for_critic": [],
            "failure_patterns_to_avoid": [],
        }
    )
    client, calls = _client_with_responses(
        [_response(invalid), _response(reasoning_content=valid)]
    )
    planner = object.__new__(GlobalPlanner)
    planner._model = "test"
    planner._max_tokens = 512
    planner._temperature = 0.0
    planner._client = client
    planner.last_trace = {}
    relation_context = StageRelationContext(
        stage="furniture",
        hard_constraints=[
            {
                "constraint_id": "hard-1",
                "stage": "furniture",
                "relation": "faces",
            }
        ],
        contract_constraint_count=1,
        projected_constraint_count=1,
    )
    context = HarnessContext(
        stage="furniture",
        task_spec=_task_spec(),
        memory_pack=MemoryPack(),
        relation_context=relation_context,
        stage_budget=StageBudget(),
    )

    brief = planner.generate_stage_brief(context, original_task="A bedroom")

    assert brief.stage == "furniture"
    assert [item["status"] for item in planner.last_trace["attempts"]] == [
        "error",
        "ok",
    ]
    assert "hard-1" in calls[0]["messages"][1]["content"]
    assert "Previous candidate" in calls[1]["messages"][-1]["content"]
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["strict"] is True


def test_empty_stage_only_opens_for_hard_constraint() -> None:
    planner = object.__new__(GlobalPlanner)
    planner._model = "test"
    planner._max_tokens = 512
    planner._temperature = 0.0
    planner.last_trace = {}
    client, calls = _client_with_responses([])
    planner._client = client
    empty_spec = _task_spec(required_wall_objects=[])
    context = HarnessContext(
        stage="wall_mounted",
        task_spec=empty_spec,
        memory_pack=MemoryPack(),
        relation_context=StageRelationContext(stage="wall_mounted"),
    )

    fallback = planner.generate_stage_brief(context)
    assert "empty wall_mounted stage" in fallback.stage_objective
    assert not calls

    valid = json.dumps(
        {
            "stage": "wall_mounted",
            "stage_objective": "Honor hard wall intent",
            "recommended_skills": [],
            "constraints_for_designer": [],
            "checks_for_critic": [],
            "failure_patterns_to_avoid": [],
        }
    )
    planner._client, calls = _client_with_responses([_response(valid)])
    context.relation_context = StageRelationContext(
        stage="wall_mounted",
        hard_constraints=[
            {
                "constraint_id": "wall-hard",
                "stage": "wall_mounted",
                "relation": "on_wall",
            }
        ],
    )
    assert planner.generate_stage_brief(context).stage == "wall_mounted"
    assert len(calls) == 1


def test_task_compiler_strict_retry_rejects_extra_fields() -> None:
    invalid = json.dumps(
        {"room_type": "bedroom", "style": "standard", "invented": True}
    )
    valid = json.dumps(
        {
            "room_type": "bedroom",
            "style": "standard",
            "required_large_objects": ["bed"],
        }
    )
    client, calls = _client_with_responses(
        [_response(invalid), _response(reasoning_content=valid)]
    )
    compiler = object.__new__(TaskCompiler)
    compiler._model = "test"
    compiler._max_tokens = 512
    compiler._temperature = 0.0
    compiler._client = client
    compiler.last_trace = {}

    spec = compiler.compile("A bedroom with a bed.")

    assert spec.compiler_status == "ok"
    assert [item["status"] for item in compiler.last_trace["attempts"]] == [
        "error",
        "ok",
    ]
    assert calls[0]["response_format"]["json_schema"]["strict"] is True
    assert "Previous candidate" in calls[1]["messages"][-1]["content"]


def test_structured_output_double_failure_uses_minimal_fallbacks() -> None:
    compiler_client, compiler_calls = _client_with_responses(
        [_response("{}"), _response("{}")]
    )
    compiler = object.__new__(TaskCompiler)
    compiler._model = "test"
    compiler._max_tokens = 512
    compiler._temperature = 0.0
    compiler._client = compiler_client
    compiler.last_trace = {}

    spec = compiler.compile("A bedroom with a bed and two nightstands.")

    assert spec.compiler_status == "degraded"
    assert len(compiler_calls) == 2
    assert [item["status"] for item in compiler.last_trace["attempts"]] == [
        "error",
        "error",
        "deterministic_fallback",
    ]

    planner_client, planner_calls = _client_with_responses(
        [_response("{}"), _response("{}")]
    )
    planner = object.__new__(GlobalPlanner)
    planner._model = "test"
    planner._max_tokens = 512
    planner._temperature = 0.0
    planner._client = planner_client
    planner.last_trace = {}
    relation_context = StageRelationContext(
        stage="furniture",
        hard_constraints=[
            {
                "constraint_id": "bed-wall",
                "stage": "furniture",
                "relation": "against_wall",
            }
        ],
    )
    context = HarnessContext(
        stage="furniture",
        task_spec=spec,
        memory_pack=MemoryPack(),
        relation_context=relation_context,
    )

    brief = planner.generate_stage_brief(context, original_task="A bedroom")

    assert brief.stage == "furniture"
    assert len(planner_calls) == 2
    assert [item["status"] for item in planner.last_trace["attempts"]] == [
        "error",
        "error",
        "minimal_fallback",
    ]
    assert planner.last_trace["hard_constraint_ids"] == ["bed-wall"]


def _object(object_id: str, category: str, x: float) -> dict:
    return {
        "id": object_id,
        "name": category,
        "category": category,
        "category_norm": category,
        "bbox_world": {
            "center": [x, 0.0, 0.5],
            "size": [0.8, 0.8, 1.0],
            "min": [x - 0.4, -0.4, 0.0],
            "max": [x + 0.4, 0.4, 1.0],
        },
    }


def test_hssd_checks_are_auxiliary_and_manual_checks_remain_core(monkeypatch) -> None:
    monkeypatch.setattr(
        asset_library_annotations,
        "_eligible_hssd_relation_prior",
        lambda _record, relation: (
            str(relation.get("provenance") or "").startswith("seed_rule:"),
            1.0,
        ),
    )
    record = {
        "category": "bed",
        "relation_priors": [
            _relation("beside", "nightstand"),
            _relation(
                "funeval_used_with",
                "nightstand",
                confidence=0.95,
                provenance="asset_record:funeval_functional_dependency:commonsense_llm",
            ),
        ],
    }
    annotation = asset_library_annotations.build_scenebenchmark_annotation(record)
    dependencies = annotation["functional_hints"]["functional_dependencies"]
    assert len(dependencies) == 1
    assert dependencies[0]["scoring_tier"] == "auxiliary"
    assert dependencies[0]["provenance"].startswith("seed_rule:")

    bed = _object("bed-1", "bed", 0.0)
    bed["functional_hints"] = {
        "explicit_target_relation": ["nightstand"],
        "hssd_relation_prior_scoring_tier": "auxiliary",
    }
    nightstand = _object("nightstand-1", "nightstand", 0.9)
    checks = _build_explicit_target_relation_checks(
        {}, {"bed-1": bed, "nightstand-1": nightstand}, set()
    )
    assert checks and checks[0]["scoring_tier"] == "auxiliary"

    bed["functional_hints"].pop("hssd_relation_prior_scoring_tier")
    checks = _build_explicit_target_relation_checks(
        {}, {"bed-1": bed, "nightstand-1": nightstand}, set()
    )
    assert checks and checks[0]["scoring_tier"] == "core"
