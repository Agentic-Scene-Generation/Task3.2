"""Focused regression coverage for the always-hard v2 intent contract."""

from __future__ import annotations

import json

from types import SimpleNamespace

import pytest

from pydantic import ValidationError

from scenesmith.scenebenchmark_critic import adapter
from scenesmith.scene_expert.schemas import SceneTaskSpec
from scenesmith.scene_expert.task_compiler import TaskCompiler, _SYSTEM_PROMPT
from scenesmith.experiments import indoor_scene_generation
from scenesmith.scenebenchmark_critic.config import CriticConfig
from scenesmith.scenebenchmark_critic.evaluator import run_case_pack_checks
from scenesmith.scenebenchmark_critic.intent_contract import (
    _DIRECT_FD_EVALUATORS,
    SCHEMA_VERSION,
    apply_contract_execution_states,
    attach_intent_contract_to_case_pack,
    augment_contract_checks,
    bound_ids,
    build_intent_contract,
    is_hard_constraint,
    selected_ids,
    selector_match_count,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
    _EXTENSION_EVALUATORS,
)
from scenesmith.scenebenchmark_critic.relation_registry import (
    PUBLIC_RELATIONS,
    RELATION_REGISTRY,
    repair_relation_types,
    validate_relation_registry,
)
from scenesmith.scenebenchmark_critic.furniture_relation_repair import (
    _FURNITURE_REPAIR_STRATEGIES,
    _REPAIR_TARGET_HANDLERS,
)
from scenesmith.scenebenchmark_critic.reports import (
    format_markdown_report,
    format_prompt_context,
)
from scenesmith.scenebenchmark_critic.prompt_context import format_agent_prompt_context


def _record(
    object_id: str,
    category: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float] = (0.6, 0.6, 0.8),
    *,
    yaw_deg: float = 0.0,
) -> dict:
    lower = [center[index] - size[index] / 2.0 for index in range(3)]
    upper = [center[index] + size[index] / 2.0 for index in range(3)]
    return {
        "id": object_id,
        "category": category,
        "category_norm": category,
        "name": category,
        "yaw_deg": yaw_deg,
        "bbox_world": {
            "center": list(center),
            "size": list(size),
            "min": lower,
            "max": upper,
        },
    }


def test_relation_registry_is_complete_and_is_task_compiler_vocabulary() -> None:
    validate_relation_registry()
    assert PUBLIC_RELATIONS == frozenset(RELATION_REGISTRY)
    assert all(spec.target_arity in {0, 1, 2} for spec in RELATION_REGISTRY.values())
    assert all(spec.earliest_stage for spec in RELATION_REGISTRY.values())
    assert all(spec.evaluator for spec in RELATION_REGISTRY.values())
    assert all(
        spec.dependency_binding in {"same_endpoints", "subject", "any_endpoint"}
        for spec in RELATION_REGISTRY.values()
    )
    assert all(name in _SYSTEM_PROMPT for name in PUBLIC_RELATIONS)
    assert all(
        spec.repair_strategy
        for spec in RELATION_REGISTRY.values()
        if spec.repair_strategy is not None
    )
    evaluator_names = {spec.evaluator for spec in RELATION_REGISTRY.values()}
    implemented_evaluators = (
        set(_EXTENSION_EVALUATORS)
        | set(_DIRECT_FD_EVALUATORS)
        | {"faces", "aligned_with", "paired_with"}
    )
    assert evaluator_names == implemented_evaluators


def test_registered_furniture_repairs_have_dispatch_handlers() -> None:
    assert set(_REPAIR_TARGET_HANDLERS) == set(
        repair_relation_types(strategies=_FURNITURE_REPAIR_STRATEGIES)
    )


def test_typed_task_spec_rejects_unknown_relation_and_model_constraint_id() -> None:
    base = {
        "room_type": "office",
        "style": "standard",
        "intent_constraints": [
            {
                "relation": "teleport_beside",
                "subjects": {"category": "chair"},
                "targets": {"category": "desk"},
                "source": "model_inferred",
                "inference_reason": "workstation use",
            }
        ],
    }
    with pytest.raises(ValidationError, match="Unknown intent relation"):
        SceneTaskSpec.model_validate(base)

    base["intent_constraints"][0].update(
        {
            "relation": "near",
            "constraint_id": "model_owned_id",
        }
    )
    with pytest.raises(ValidationError, match="constraint_id"):
        SceneTaskSpec.model_validate(base)


def test_stable_ids_and_model_inferred_constraints_are_hard() -> None:
    task_spec = {
        "compiler_status": "ok",
        "compiler_spec_version": "scenesmith.task_compiler.v2",
        "intent_constraints": [
            {
                "relation": "near",
                "subjects": {"category": "chair", "count": 1},
                "targets": {"category": "desk", "count": 1},
                "source": "model_inferred",
                "confidence": 0.4,
                "inference_reason": "a work chair must remain near its desk",
            }
        ],
    }
    first = build_intent_contract(
        "An office with a chair and desk.", task_spec=task_spec
    )
    second = build_intent_contract(
        "An office with a chair and desk.", task_spec=task_spec
    )
    assert first["constraints"] == second["constraints"]
    inferred = next(row for row in first["constraints"] if row["relation"] == "near")
    assert is_hard_constraint(inferred)
    assert inferred["strength"] == "hard"
    assert inferred["confidence"] == 0.4
    assert inferred["inference_reason"] == "a work chair must remain near its desk"


def test_invalid_model_contract_uses_deterministic_degraded_fallback() -> None:
    raw = json.dumps(
        {
            "room_type": "bedroom",
            "style": "standard",
            "required_large_objects": ["bed"],
            "intent_constraints": [
                {
                    "relation": "unknown_relation",
                    "subjects": {"category": "bed"},
                    "targets": {"category": "wall"},
                    "source": "model_inferred",
                    "inference_reason": "invalid test output",
                }
            ],
        }
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=raw))], usage=None
    )
    compiler = object.__new__(TaskCompiler)
    compiler._model = "test-model"
    compiler._max_tokens = 256
    compiler._temperature = 0.0
    compiler._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )

    spec = compiler.compile("A bedroom with a bed against the wall.")

    assert spec.compiler_status == "degraded"
    assert "Unknown intent relation" in spec.compiler_failure_reason
    assert {row.relation for row in spec.intent_constraints} >= {
        "against_wall",
        "required_count",
    }


def test_v1_cache_rebuilds_for_current_prompt_and_compiler_spec() -> None:
    scene = SimpleNamespace(
        text_description="An office with a chair near a desk.",
        room_type="office",
        scene_expert_task_spec={
            "compiler_status": "ok",
            "intent_constraints": [
                {
                    "relation": "near",
                    "subjects": {"category": "chair", "count": 1},
                    "targets": {"category": "desk", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "chair near a desk",
                }
            ],
        },
        scenebenchmark_intent_contract={
            "schema_version": "scenesmith.intent_contract.v1",
            "prompt_sha256": "stale",
            "constraints": [],
        },
    )
    case_pack = {"room_type": "office"}

    contract = attach_intent_contract_to_case_pack(scene, case_pack)

    assert contract["schema_version"] == SCHEMA_VERSION
    assert contract["constraints"]
    assert contract["task_compiler_spec_sha256"]
    assert scene.scenebenchmark_intent_contract == contract


def test_wall_relation_is_pending_at_furniture_and_hard_fails_at_final() -> None:
    contract = build_intent_contract(
        "A room with a painting on the wall.",
        task_spec={
            "compiler_status": "ok",
            "intent_constraints": [
                {
                    "relation": "on_wall",
                    "subjects": {"category": "painting", "count": 1},
                    "targets": {"category": "wall"},
                    "source": "explicit_prompt",
                    "evidence_span": "painting on the wall",
                }
            ],
        },
    )
    base = {
        "intent_contract": contract,
        "scene_geometry": {
            "rooms": [{"id": "room", "bbox": {"min": [-2, -2, 0], "max": [2, 2, 2.7]}}],
            "objects": [_record("wall_0", "wall", (0.0, 2.0, 1.35), (4.0, 0.1, 2.7))],
        },
        "checks": [],
    }
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    furniture = run_case_pack_checks({**base, "stage": "scene_after_furniture"}, config)
    assert (
        next(row for row in furniture if row["relation_type"] == "on_wall")[
            "contract_state"
        ]
        == "pending"
    )

    final_pack = {**base, "stage": "final_scene", "intent_contract": contract.copy()}
    final = run_case_pack_checks(final_pack, config)
    final_row = next(row for row in final if row["relation_type"] == "on_wall")
    assert final_row["label"] == "fail"
    assert final_row["contract_state"] == "failed"
    assert not {row["state"] for row in final_pack["intent_contract"]["execution"]} & {
        "pending",
        "blocked",
    }


def test_furniture_relation_to_wall_object_waits_for_wall_stage() -> None:
    contract = build_intent_contract(
        "A teacher desk sits near a chalkboard.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": ["teacher_desk"],
            "required_wall_objects": ["chalkboard"],
            "intent_constraints": [
                {
                    "relation": "near",
                    "subjects": {"category": "teacher_desk", "count": 1},
                    "targets": {"category": "instructional_surface", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "teacher desk sits near a chalkboard",
                }
            ],
        },
    )
    constraint = next(
        row for row in contract["constraints"] if row["relation"] == "near"
    )
    base = {
        "intent_contract": contract,
        "scene_geometry": {
            "objects": [
                _record("teacher_desk_0", "teacher_desk", (0.0, 1.5, 0.4)),
                _record("north_wall", "wall", (0.0, 2.0, 1.35), (4.0, 0.1, 2.7)),
            ]
        },
        "checks": [],
    }
    config = CriticConfig(enabled=True, metrics=("functional_dependency",))

    furniture = run_case_pack_checks({**base, "stage": "furniture"}, config)
    furniture_row = next(
        row
        for row in furniture
        if (row.get("evidence") or {}).get("intent_constraint", {}).get("constraint_id")
        == constraint["constraint_id"]
    )

    wall_pack = {**base, "stage": "wall_mounted", "intent_contract": contract.copy()}
    wall = run_case_pack_checks(wall_pack, config)
    wall_row = next(
        row
        for row in wall
        if (row.get("evidence") or {}).get("intent_constraint", {}).get("constraint_id")
        == constraint["constraint_id"]
    )

    assert constraint["stage"] == "wall_mounted"
    assert furniture_row["contract_state"] == "pending"
    assert furniture_row["scoring_tier"] == "auxiliary"
    assert wall_row["contract_state"] == "failed"
    assert wall_row["label"] == "fail"


def test_task_inventory_sets_generic_manipuland_constraint_stage() -> None:
    contract = build_intent_contract(
        "A magazine lies on the coffee table.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": ["coffee_table"],
            "required_small_objects": ["magazine"],
            "intent_constraints": [
                {
                    "relation": "on_top_of",
                    "subjects": {"category": "magazine", "count": 1},
                    "targets": {"category": "coffee_table", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "magazine lies on the coffee table",
                }
            ],
        },
    )
    constraint = next(
        row
        for row in contract["constraints"]
        if row["relation"] == "on_top_of" and row["subjects"]["category"] == "magazine"
    )
    case_pack = {
        "stage": "furniture",
        "intent_contract": contract,
        "scene_geometry": {
            "rooms": [{"id": "room", "bbox": {"min": [-2, -2, 0], "max": [2, 2, 2.7]}}],
            "objects": [_record("coffee_table_0", "coffee_table", (0.0, 0.0, 0.4))],
        },
        "checks": [],
    }

    results = run_case_pack_checks(
        case_pack, CriticConfig(enabled=True, metrics=("functional_dependency",))
    )
    binding = next(
        row
        for row in results
        if (row.get("evidence") or {}).get("intent_constraint", {}).get("constraint_id")
        == constraint["constraint_id"]
    )

    assert constraint["stage"] == "manipuland"
    assert binding["contract_state"] == "pending"
    assert binding["scoring_tier"] == "auxiliary"


def test_future_stage_required_count_is_pending_until_its_stage() -> None:
    contract = build_intent_contract(
        "Two monitors are required.",
        task_spec={
            "compiler_status": "ok",
            "required_small_objects": ["monitor", "monitor"],
            "intent_constraints": [
                {
                    "relation": "required_count",
                    "subjects": {
                        "category": "monitor",
                        "count": 2,
                        "quantifier": "at_least",
                    },
                    "source": "explicit_prompt",
                    "evidence_span": "Two monitors",
                }
            ],
        },
    )
    case_pack = {
        "stage": "scene_after_furniture",
        "intent_contract": contract,
        "scene_geometry": {
            "rooms": [{"id": "office", "bbox": {"min": [-2, -2, 0], "max": [2, 2, 3]}}],
            "objects": [_record("desk_0", "desk", (0.0, 0.0, 0.4))],
        },
        "checks": [],
    }

    results = run_case_pack_checks(
        case_pack, CriticConfig(enabled=True, metrics=("functional_dependency",))
    )
    row = next(item for item in results if item["relation_type"] == "required_count")

    assert row["contract_state"] == "pending"
    assert row["label"] == "unknown"
    assert row["scoring_tier"] == "auxiliary"


def test_plural_evidence_uses_minimum_subject_cardinality() -> None:
    contract = build_intent_contract(
        "A few magazines lie on the coffee table.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": ["coffee_table"],
            "required_small_objects": ["magazine"],
            "intent_constraints": [
                {
                    "relation": "on_top_of",
                    "subjects": {"category": "magazine", "count": 1},
                    "targets": {"category": "coffee_table", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "a few magazines lie on the coffee table",
                }
            ],
        },
    )
    constraint = next(
        row
        for row in contract["constraints"]
        if row["relation"] == "on_top_of" and row["subjects"]["category"] == "magazine"
    )

    assert constraint["subjects"]["quantifier"] == "minimum"


def test_selector_does_not_use_known_object_description_as_category() -> None:
    remote = _record("remote_control_0", "remote_control", (0.0, 0.0, 0.1))
    remote["description"] = "Remote control for a television"
    television = _record("television_0", "television", (0.0, 1.0, 1.0))

    assert selected_ids({"category": "television"}, [remote, television]) == [
        "television_0"
    ]


def test_known_category_prevents_description_only_selector_match() -> None:
    plant = _record("plant_0", "plant", (0.0, 0.0, 0.8))
    artwork = _record("botanical_print_0", "wall_art", (0.0, 1.0, 1.2))
    artwork["description"] = "Botanical print showing a tropical plant"
    desk = _record("study_desk_0", "study_desk", (1.0, 0.0, 0.4))
    lamp = _record("desk_lamp_0", "desk_lamp", (1.0, 0.0, 0.9))

    assert selected_ids({"category": "plant"}, [plant, artwork]) == ["plant_0"]
    assert selected_ids({"category": "desk"}, [desk, lamp]) == ["study_desk_0"]
    assert selected_ids({"category": "table_lamp"}, [desk, lamp]) == ["desk_lamp_0"]


def test_one_of_repeated_targets_uses_existential_binding() -> None:
    contract = build_intent_contract(
        "A floor lamp is beside one armchair.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": ["floor_lamp", "armchair", "armchair"],
            "intent_constraints": [
                {
                    "relation": "next_to",
                    "subjects": {"category": "floor_lamp", "count": 1},
                    "targets": {"category": "armchair", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "beside one armchair",
                }
            ],
        },
    )
    constraint = next(
        row for row in contract["constraints"] if row["relation"] == "next_to"
    )
    case_pack = {
        "stage": "furniture",
        "intent_contract": contract,
        "scene_geometry": {
            "rooms": [{"id": "room", "bbox": {"min": [-4, -4, 0], "max": [4, 4, 2.7]}}],
            "objects": [
                _record("floor_lamp_0", "floor_lamp", (0.0, 0.0, 0.8), (0.2, 0.2, 1.6)),
                _record("armchair_0", "armchair", (3.0, 0.0, 0.45)),
                _record("armchair_1", "armchair", (0.5, 0.0, 0.45)),
            ],
        },
        "checks": [],
    }

    results = run_case_pack_checks(
        case_pack, CriticConfig(enabled=True, metrics=("functional_dependency",))
    )
    rows = [
        row
        for row in results
        if (row.get("evidence") or {}).get("intent_constraint", {}).get("constraint_id")
        == constraint["constraint_id"]
    ]

    assert constraint["targets"]["quantifier"] == "minimum"
    assert rows
    assert all(row["label"] == "pass" for row in rows)
    assert all(row.get("contract_state") == "passed" for row in rows)


def test_selector_binds_specific_sofa_asset_to_sofa_family() -> None:
    sofa = _record("two_seater_sofa_0", "two_seater_sofa", (0.0, 0.0, 0.4))
    sofa["metadata"] = {"semantic_name": "two_seater_sofa"}

    assert selected_ids({"category": "sofa"}, [sofa]) == ["two_seater_sofa_0"]


def test_stack_members_count_toward_contract_cardinality() -> None:
    standalone_magazine = _record("magazine_0", "magazine", (0.2, 0.0, 0.5))
    stacked_magazines = _record("stack_0", "magazine", (-0.2, 0.0, 0.5))
    stacked_magazines["metadata"] = {
        "composite_type": "stack",
        "member_assets": [
            {"asset_id": "magazine_lifestyle_0", "name": "magazine"},
            {"asset_id": "magazine_architecture_0", "name": "magazine_architecture"},
        ],
    }
    selector = {"category": "magazine", "count": 3, "quantifier": "all"}

    assert selector_match_count(selector, [standalone_magazine, stacked_magazines]) == 3
    assert bound_ids(selector, [standalone_magazine, stacked_magazines]) == [
        "magazine_0",
        "stack_0",
    ]


def test_adapter_canonicalizes_styled_semantic_asset_category() -> None:
    asset = SimpleNamespace(
        metadata={"semantic_name": "farmhouse_bed"},
        object_id="farmhouse_bed_0",
        name="farmhouse_bed",
        description="rustic farmhouse double bed",
    )

    assert adapter._category_for_object(asset) == "bed"


def test_failed_upstream_relation_blocks_downstream_until_final() -> None:
    contract = build_intent_contract(
        "A chair near and facing a desk.",
        task_spec={
            "compiler_status": "ok",
            "intent_constraints": [
                {
                    "relation": relation,
                    "subjects": {"category": "chair", "count": 1},
                    "targets": {"category": "desk", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "chair near and facing a desk",
                }
                for relation in ("near", "faces")
            ],
        },
    )
    case_pack = {
        "stage": "furniture",
        "intent_contract": contract,
        "scene_geometry": {
            "rooms": [
                {"id": "office", "bbox": {"min": [-4, -4, 0], "max": [4, 4, 2.7]}}
            ],
            "objects": [
                _record("chair_0", "chair", (0.0, 0.0, 0.45)),
                _record("desk_0", "desk", (0.0, 3.0, 0.4)),
            ],
        },
        "checks": [],
    }

    results = run_case_pack_checks(
        case_pack,
        CriticConfig(enabled=True, metrics=("functional_dependency",)),
    )

    near = next(
        row for row in results if row["relation_type"] == "generic_near_relation"
    )
    faces = next(
        row for row in results if row["relation_type"] == "seating_to_work_surface"
    )
    assert near["contract_state"] == "failed"
    assert faces["contract_state"] == "blocked"
    assert faces["scoring_tier"] == "auxiliary"


def test_unrelated_near_failure_does_not_block_faces_dependency() -> None:
    contract = build_intent_contract(
        "A chair near a plant and another chair facing a desk.",
        task_spec={
            "compiler_status": "ok",
            "intent_constraints": [
                {
                    "relation": "near",
                    "subjects": {"category": "chair", "role": "lounge"},
                    "targets": {"category": "plant", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "lounge chair near plant",
                },
                {
                    "relation": "faces",
                    "subjects": {"category": "chair", "role": "work"},
                    "targets": {"category": "desk", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "work chair facing desk",
                },
            ],
        },
    )
    near = next(row for row in contract["constraints"] if row["relation"] == "near")
    faces = next(row for row in contract["constraints"] if row["relation"] == "faces")
    results = [
        {"label": "fail", "evidence": {"intent_constraint": near}, "diagnostics": {}},
        {"label": "pass", "evidence": {"intent_constraint": faces}, "diagnostics": {}},
    ]

    apply_contract_execution_states(
        {"stage": "furniture", "intent_contract": contract}, results
    )

    assert results[0]["contract_state"] == "failed"
    assert results[1]["contract_state"] == "passed"
    assert results[1]["diagnostics"]["dependency_constraint_ids"] == []


def test_contract_near_uses_registry_threshold() -> None:
    contract = build_intent_contract(
        "A teacher desk is near a chalkboard.",
        task_spec={
            "compiler_status": "ok",
            "intent_constraints": [
                {
                    "relation": "near",
                    "subjects": {"category": "teacher_desk", "count": 1},
                    "targets": {
                        "category": "instructional_surface",
                        "count": 1,
                    },
                    "source": "explicit_prompt",
                    "evidence_span": "teacher desk is near a chalkboard",
                }
            ],
        },
    )
    case_pack = {
        "stage": "final_scene",
        "intent_contract": contract,
        "scene_geometry": {
            "rooms": [
                {"id": "classroom", "bbox": {"min": [-4, -4, 0], "max": [4, 4, 3]}}
            ],
            "objects": [
                _record("teacher_desk_0", "teacher_desk", (0.0, 0.0, 0.4)),
                _record(
                    "chalkboard_0",
                    "instructional_surface",
                    (0.0, 1.34, 1.4),
                    (2.0, 0.2, 1.0),
                ),
            ],
        },
        "checks": [],
    }

    results = run_case_pack_checks(
        case_pack, CriticConfig(enabled=True, metrics=("functional_dependency",))
    )
    row = next(
        item for item in results if item["relation_type"] == "generic_near_relation"
    )

    assert row["label"] == "pass"
    assert row["contract_state"] == "passed"


def test_floor_support_contract_uses_floor_attachment_evaluator() -> None:
    contract = build_intent_contract(
        "A plant is on the floor.",
        task_spec={
            "compiler_status": "ok",
            "intent_constraints": [
                {
                    "relation": "on_top_of",
                    "subjects": {"category": "plant", "count": 1},
                    "targets": {"category": "floor", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "plant is on the floor",
                }
            ],
        },
    )
    case_pack = {
        "stage": "final_scene",
        "intent_contract": contract,
        "scene_geometry": {
            "rooms": [{"id": "room", "bbox": {"min": [-2, -2, 0], "max": [2, 2, 3]}}],
            "objects": [
                _record("plant_0", "plant", (0.0, 0.0, 0.4)),
                _record("floor_0", "floor", (0.0, 0.0, -0.05), (4.0, 4.0, 0.1)),
            ],
        },
        "checks": [],
    }

    results = run_case_pack_checks(
        case_pack, CriticConfig(enabled=True, metrics=("functional_dependency",))
    )
    row = next(item for item in results if item["relation_type"] == "object_on_floor")

    assert row["label"] == "pass"
    assert row["contract_state"] == "passed"


def test_floor_item_near_furniture_uses_planar_near_relation() -> None:
    contract = build_intent_contract(
        "A small wastebasket sits on the floor near the dresser.",
        task_spec={
            "compiler_status": "ok",
            "intent_constraints": [
                {
                    "relation": "near",
                    "subjects": {"category": "wastebasket", "count": 1},
                    "targets": {"category": "dresser", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "wastebasket near the dresser",
                }
            ],
        },
    )
    case_pack = {
        "stage": "final_scene",
        "intent_contract": contract,
        "scene_geometry": {
            "rooms": [{"id": "room", "bbox": {"min": [-2, -2, 0], "max": [2, 2, 3]}}],
            "objects": [
                _record(
                    "dresser_0",
                    "dresser",
                    (0.0, -1.8, 0.45),
                    (1.2, 0.5, 0.9),
                ),
                _record(
                    "wastebasket_0",
                    "wastebasket",
                    (-1.1, -1.6, 0.13),
                    (0.25, 0.25, 0.25),
                ),
            ],
        },
        "checks": [],
    }

    results = run_case_pack_checks(
        case_pack, CriticConfig(enabled=True, metrics=("functional_dependency",))
    )
    row = next(
        item for item in results if item["relation_type"] == "generic_near_relation"
    )

    assert row["label"] == "pass"
    assert row["contract_state"] == "passed"


def test_other_repeated_target_uses_existential_binding() -> None:
    contract = build_intent_contract(
        "An alarm clock sits on one nightstand and a book on the other.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": ["nightstand", "nightstand"],
            "required_small_objects": ["book"],
            "intent_constraints": [
                {
                    "relation": "on_top_of",
                    "subjects": {"category": "book", "count": 1},
                    "targets": {"category": "nightstand"},
                    "source": "explicit_prompt",
                    "evidence_span": "book on the other",
                }
            ],
        },
    )
    constraint = next(
        item
        for item in contract["constraints"]
        if item["subjects"]["category"] == "book"
    )

    assert constraint["targets"] == {
        "category": "nightstand",
        "count": 1,
        "quantifier": "minimum",
    }


def test_equal_all_groups_bind_without_explicit_counts() -> None:
    contract = build_intent_contract(
        "Student chairs are paired with student desks.",
        task_spec={
            "compiler_status": "ok",
            "intent_constraints": [
                {
                    "relation": "paired_with",
                    "subjects": {"category": "student_chair", "quantifier": "all"},
                    "targets": {"category": "student_desk", "quantifier": "all"},
                    "source": "explicit_prompt",
                    "evidence_span": "chairs are paired with desks",
                }
            ],
        },
    )
    case_pack = {
        "stage": "final_scene",
        "intent_contract": contract,
        "scene_geometry": {
            "rooms": [
                {"id": "classroom", "bbox": {"min": [-4, -4, 0], "max": [4, 4, 3]}}
            ],
            "objects": [
                _record("student_chair_0", "chair", (0.0, 0.0, 0.4)),
                _record("student_desk_0", "desk", (0.0, 0.7, 0.4)),
                _record("student_chair_1", "chair", (2.0, 0.0, 0.4)),
                _record("student_desk_1", "desk", (2.0, 0.7, 0.4)),
            ],
        },
        "checks": [],
    }

    results = run_case_pack_checks(
        case_pack, CriticConfig(enabled=True, metrics=("functional_dependency",))
    )
    constraint_id = contract["constraints"][0]["constraint_id"]
    rows = [
        row
        for row in results
        if (row.get("evidence") or {}).get("intent_constraint", {}).get("constraint_id")
        == constraint_id
    ]

    assert rows
    assert not any(row["check_id"].endswith("__binding") for row in rows)
    assert {
        row["primary_object"]
        for row in rows
        if row["relation_type"] == "seating_to_work_surface"
    } == {"student_chair_0", "student_chair_1"}


def test_report_exposes_contract_execution_audit() -> None:
    payload = {
        "scope": "scene",
        "stage": "furniture",
        "gate": {"label": "report_only"},
        "summary": {"scene_summary": {}},
        "case_pack": {
            "intent_contract": {
                "execution": [
                    {
                        "constraint_id": "intent_near_1234",
                        "relation": "near",
                        "source": "model_inferred",
                        "inference_reason": "Chairs need to remain near their desks.",
                        "state": "failed",
                        "subject_ids": ["chair_0"],
                        "target_ids": ["desk_0"],
                        "dependency_constraint_ids": ["intent_count_5678"],
                        "repair_strategy": None,
                    }
                ]
            }
        },
        "results": [],
    }

    report = format_markdown_report(payload)

    assert "`intent_near_1234` [model_inferred] `near`: **failed**" in report
    assert "subjects=chair_0; targets=desk_0" in report
    assert "dependencies=intent_count_5678:unknown; repair=none" in report
    assert "rationale=Chairs need to remain near their desks." in report


def test_prompt_only_includes_actionable_contract_failures() -> None:
    def result(state: str, label: str, constraint_id: str) -> dict:
        return {
            "check_id": f"intent_contract__{constraint_id}",
            "metric": "functional_dependency",
            "label": label,
            "primary_object": "chair_0",
            "related_objects": ["desk_0"],
            "reason": f"{state} relation",
            "contract_state": state,
            "scoring_tier": "core" if state == "failed" else "auxiliary",
            "evidence": {
                "intent_constraint": {
                    "constraint_id": constraint_id,
                    "source": "explicit_prompt",
                    "relation": "near",
                }
            },
        }

    context = format_prompt_context(
        {
            "case_pack": {
                "intent_contract": {
                    "execution": [
                        {
                            "constraint_id": "actionable",
                            "state": "failed",
                            "subject_ids": ["chair_0"],
                            "dependency_constraint_ids": ["required_chair"],
                        }
                    ]
                }
            },
            "results": [
                result("failed", "fail", "actionable"),
                result("pending", "unknown", "later_stage"),
                result("blocked", "unknown", "upstream_blocked"),
            ],
        }
    )

    assert (
        "Contract: actionable source=explicit_prompt relation=near state=failed "
        "bound_subjects=chair_0 dependencies=required_chair:unknown"
    ) in context
    assert "later_stage" not in context
    assert "upstream_blocked" not in context


def test_prompt_includes_model_inference_rationale() -> None:
    payload = {
        "case_pack": {
            "intent_contract": {
                "execution": [
                    {
                        "constraint_id": "inferred_near",
                        "state": "failed",
                        "subject_ids": ["chair_0"],
                        "dependency_constraint_ids": [],
                    }
                ]
            }
        },
        "results": [
            {
                "check_id": "intent_contract__inferred_near",
                "metric": "functional_dependency",
                "label": "fail",
                "primary_object": "chair_0",
                "related_objects": ["desk_0"],
                "reason": "chair is too far from desk",
                "contract_state": "failed",
                "scoring_tier": "core",
                "evidence": {
                    "intent_constraint": {
                        "constraint_id": "inferred_near",
                        "source": "model_inferred",
                        "relation": "near",
                        "inference_reason": "A work chair must remain near its desk.",
                    }
                },
            }
        ],
    }

    assert "Contract rationale: A work chair must remain near its desk." in (
        format_prompt_context(payload)
    )


def test_prompt_excludes_non_contract_degraded_results() -> None:
    payload = {
        "case_pack": {
            "scene_geometry": {"objects": []},
            "intent_contract": {"execution": []},
        },
        "results": [
            {
                "check_id": "legacy_degraded",
                "metric": "functional_dependency",
                "label": "degraded",
                "primary_object": "chair_0",
                "related_objects": ["desk_0"],
                "reason": "legacy result",
                "contract_state": "failed",
                "scoring_tier": "core",
            }
        ],
    }

    assert "no degraded or failed checks" in format_prompt_context(payload)
    assert "no degraded or failed checks" in format_agent_prompt_context(
        payload, agent_type="furniture"
    )


def test_agent_prompt_does_not_inject_passing_contracts() -> None:
    payload = {
        "case_pack": {
            "scene_geometry": {"objects": []},
            "checks": [
                {
                    "check_id": "orientation_pass",
                    "check_source": "scenesmith_orientation_contract",
                    "subject_id": "chair_0",
                    "target_ids": ["desk_0"],
                    "relation_type": "seating_to_work_surface",
                }
            ],
            "intent_contract": {"execution": []},
        },
        "results": [
            {
                "check_id": "orientation_pass",
                "metric": "functional_dependency",
                "label": "pass",
                "primary_object": "chair_0",
                "related_objects": ["desk_0"],
                "relation_type": "seating_to_work_surface",
                "scoring_tier": "core",
            }
        ],
    }

    context = format_agent_prompt_context(payload, agent_type="furniture")

    assert "no degraded or failed checks" in context
    assert "result=pass" not in context
    assert "chair_0" not in context


def test_final_scene_critic_report_runs_only_when_configured(
    tmp_path, monkeypatch
) -> None:
    scene = SimpleNamespace(room_id="living_room")
    expected = {"stage": "final_scene"}
    calls: list[dict] = []

    monkeypatch.setattr(
        indoor_scene_generation,
        "critic_config_from_any",
        lambda _cfg: CriticConfig(enabled=True),
    )
    monkeypatch.setattr(
        indoor_scene_generation,
        "write_room_stage_report",
        lambda received_scene, output_dir, **kwargs: calls.append(
            {
                "scene": received_scene,
                "output_dir": output_dir,
                **kwargs,
            }
        )
        or expected,
    )

    result = indoor_scene_generation._write_final_critic_report(
        scene, tmp_path, {"scenebenchmark_critic": {"enabled": True}}
    )

    assert result == expected
    assert calls == [
        {
            "scene": scene,
            "output_dir": tmp_path / "scenebenchmark_critic" / "final_scene",
            "config": CriticConfig(enabled=True),
            "stage": "final_scene",
            "raw_config": {"scenebenchmark_critic": {"enabled": True}},
        }
    ]


def test_office_each_relations_normalize_to_equal_inventory_groups() -> None:
    prompt = (
        "A collaborative office with two desks facing each other across the center, "
        "an office chair and computer monitor at each desk, a filing cabinet "
        "against one wall, and a printer on top of the filing cabinet."
    )
    task_spec = {
        "compiler_status": "ok",
        "required_large_objects": [
            "desk",
            "desk",
            "office_chair",
            "office_chair",
            "filing_cabinet",
        ],
        "required_small_objects": ["computer_monitor", "computer_monitor", "printer"],
        "intent_constraints": [
            {
                "relation": "across_from",
                "subjects": {"category": "desk", "count": 1},
                "targets": {"category": "desk"},
                "source": "explicit_prompt",
                "evidence_span": "two desks facing each other across the center",
            },
            {
                "relation": "centered_in_room",
                "subjects": {"category": "desk", "count": 2},
                "targets": {"category": "room"},
                "source": "explicit_prompt",
                "evidence_span": "across the center",
            },
            {
                "relation": "on_top_of",
                "subjects": {"category": "computer_monitor", "count": 1},
                "targets": {"category": "desk"},
                "source": "explicit_prompt",
                "evidence_span": "computer monitor at each desk",
            },
        ],
    }

    constraints = build_intent_contract(prompt, task_spec=task_spec)["constraints"]
    across = next(row for row in constraints if row["relation"] == "across_from")
    support = next(
        row
        for row in constraints
        if row["relation"] == "on_top_of" and row["subjects"]["category"] == "monitor"
    )

    assert across["subjects"]["count"] == across["targets"]["count"] == 2
    assert support["subjects"]["count"] == support["targets"]["count"] == 2
    assert support["stage"] == "manipuland"
    assert not any(
        row["relation"] == "centered_in_room" and row["subjects"]["category"] == "desk"
        for row in constraints
    )


def test_centered_in_room_keeps_explicit_per_object_group_instruction() -> None:
    constraints = build_intent_contract(
        "Each desk is centered in its room.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": ["desk", "desk"],
            "intent_constraints": [
                {
                    "relation": "centered_in_room",
                    "subjects": {"category": "desk", "count": 2},
                    "targets": {"category": "room"},
                    "source": "explicit_prompt",
                    "evidence_span": "each desk is centered in its room",
                }
            ],
        },
    )["constraints"]

    assert any(row["relation"] == "centered_in_room" for row in constraints)


def test_equal_support_groups_offer_all_targets_to_each_subject() -> None:
    contract = build_intent_contract(
        "Two monitors, one on each desk.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": ["desk", "desk"],
            "required_small_objects": ["monitor", "monitor"],
            "intent_constraints": [
                {
                    "relation": "on_top_of",
                    "subjects": {"category": "monitor", "count": 1},
                    "targets": {"category": "desk"},
                    "source": "explicit_prompt",
                    "evidence_span": "one on each desk",
                }
            ],
        },
    )
    case_pack = {
        "intent_contract": contract,
        "scene_geometry": {
            "objects": [
                _record("desk_0", "desk", (-1.0, 0.0, 0.4)),
                _record("desk_1", "desk", (1.0, 0.0, 0.4)),
                _record("monitor_0", "monitor", (-1.0, 0.0, 1.0)),
                _record("monitor_1", "monitor", (1.0, 0.0, 1.0)),
            ]
        },
        "checks": [],
    }

    assert augment_contract_checks(case_pack)
    checks = [
        row
        for row in case_pack["checks"]
        if row.get("check_source") == "intent_contract"
        and row.get("relation_type") == "object_on_support"
    ]
    assert len(checks) == 2
    assert all(set(row["target_ids"]) == {"desk_0", "desk_1"} for row in checks)


def test_same_category_across_from_checks_mutual_facing() -> None:
    contract = build_intent_contract(
        "Two desks face each other.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": ["desk", "desk"],
            "intent_constraints": [
                {
                    "relation": "across_from",
                    "subjects": {"category": "desk", "count": 1},
                    "targets": {"category": "desk"},
                    "source": "explicit_prompt",
                    "evidence_span": "Two desks face each other",
                }
            ],
        },
    )
    case_pack = {
        "stage": "furniture",
        "intent_contract": contract,
        "scene_geometry": {
            "rooms": [{"id": "room", "bbox": {"min": [-3, -3, 0], "max": [3, 3, 2.7]}}],
            "objects": [
                _record("desk_0", "desk", (0.0, -1.0, 0.4), yaw_deg=0.0),
                _record("desk_1", "desk", (0.0, 1.0, 0.4), yaw_deg=180.0),
            ],
        },
        "checks": [],
    }

    results = run_case_pack_checks(
        case_pack, CriticConfig(enabled=True, metrics=("functional_dependency",))
    )
    across = next(row for row in results if row["relation_type"] == "across_from")

    assert across["label"] == "pass"
    assert across["contract_state"] == "passed"
    assert max(across["diagnostics"]["mutual_facing_error_deg"]) <= 1.0


def test_work_surfaces_across_from_allow_outward_work_sides() -> None:
    contract = build_intent_contract(
        "Two desks are across from each other.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": ["desk", "desk"],
            "intent_constraints": [
                {
                    "relation": "across_from",
                    "subjects": {"category": "desk", "count": 2},
                    "targets": {"category": "desk", "count": 2},
                    "source": "explicit_prompt",
                    "evidence_span": "Two desks are across from each other",
                }
            ],
        },
    )
    case_pack = {
        "stage": "furniture",
        "intent_contract": contract,
        "scene_geometry": {
            "rooms": [{"id": "room", "bbox": {"min": [-3, -3, 0], "max": [3, 3, 2.7]}}],
            "objects": [
                _record("desk_0", "desk", (0.0, -1.0, 0.4), yaw_deg=180.0),
                _record("desk_1", "desk", (0.0, 1.0, 0.4), yaw_deg=0.0),
            ],
        },
        "checks": [],
    }

    results = run_case_pack_checks(
        case_pack, CriticConfig(enabled=True, metrics=("functional_dependency",))
    )
    across = next(row for row in results if row["relation_type"] == "across_from")

    assert across["label"] == "pass"
    assert across["diagnostics"]["work_surface_pair"] is True
    assert max(across["diagnostics"]["axis_alignment_error_deg"]) <= 1.0
    assert across["diagnostics"]["front_axis_opposition_error_deg"] <= 1.0


def test_unrelated_faces_failure_does_not_block_across_from() -> None:
    contract = build_intent_contract(
        "Two desks are across from each other, with a monitor facing an office chair.",
        task_spec={
            "compiler_status": "ok",
            "intent_constraints": [
                {
                    "relation": "across_from",
                    "subjects": {"category": "desk", "count": 2},
                    "targets": {"category": "desk", "count": 2},
                    "source": "explicit_prompt",
                    "evidence_span": "Two desks are across from each other",
                },
                {
                    "relation": "faces",
                    "subjects": {"category": "computer_monitor", "count": 1},
                    "targets": {"category": "office_chair", "count": 1},
                    "source": "model_inferred",
                    "inference_reason": "A monitor should face its user's chair.",
                },
            ],
        },
    )
    across = next(
        row for row in contract["constraints"] if row["relation"] == "across_from"
    )
    faces = next(row for row in contract["constraints"] if row["relation"] == "faces")
    results = [
        {
            "label": "pass",
            "evidence": {"intent_constraint": across},
            "diagnostics": {},
        },
        {
            "label": "fail",
            "evidence": {"intent_constraint": faces},
            "diagnostics": {},
        },
    ]

    apply_contract_execution_states(
        {"stage": "final", "intent_contract": contract}, results
    )

    assert results[0]["contract_state"] == "passed"
    assert results[0]["diagnostics"]["dependency_constraint_ids"] == []
    assert results[1]["contract_state"] == "failed"


def _clear_workstation_aisle_result(*, include_blocker: bool) -> dict:
    contract = build_intent_contract(
        "Keep a clear walking path between the two workstations.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": ["desk", "desk"],
            "intent_constraints": [
                {
                    "relation": "clear_access",
                    "subjects": {"category": "desk", "count": 2},
                    "targets": {"category": "desk", "count": 2},
                    "source": "explicit_prompt",
                    "evidence_span": "clear walking path between the workstations",
                }
            ],
        },
    )
    objects = [
        _record("desk_0", "desk", (0.0, -1.0, 0.4), (1.2, 0.6, 0.8)),
        _record("desk_1", "desk", (0.0, 1.0, 0.4), (1.2, 0.6, 0.8)),
        _record("chair_0", "office_chair", (0.0, -1.6, 0.4), (0.5, 0.5, 0.8)),
        _record("chair_1", "office_chair", (0.0, 1.6, 0.4), (0.5, 0.5, 0.8)),
        _record("monitor_0", "computer_monitor", (0.0, -1.0, 1.0), (0.5, 0.2, 0.6)),
        _record("monitor_1", "computer_monitor", (0.0, 1.0, 1.0), (0.5, 0.2, 0.6)),
    ]
    if include_blocker:
        objects.append(_record("aisle_cart", "cart", (0.0, 0.0, 0.3), (0.4, 0.4, 0.6)))
    case_pack = {
        "stage": "furniture",
        "intent_contract": contract,
        "scene_geometry": {
            "rooms": [
                {"id": "office", "bbox": {"min": [-3, -3, 0], "max": [3, 3, 2.7]}}
            ],
            "objects": objects,
        },
        "checks": [],
    }
    results = run_case_pack_checks(
        case_pack, CriticConfig(enabled=True, metrics=("functional_dependency",))
    )
    return next(
        row
        for row in results
        if row["relation_type"] == "clear_access"
        and row["diagnostics"].get("evaluation_mode") == "between_workstations"
    )


def test_workstation_clear_access_uses_central_aisle_not_desk_front_zones() -> None:
    result = _clear_workstation_aisle_result(include_blocker=False)

    assert result["label"] == "pass"
    assert result["diagnostics"]["blocking_ids"] == []
    assert result["diagnostics"]["free_depth_m"] >= 0.8


def test_workstation_clear_access_rejects_aisle_blocker() -> None:
    result = _clear_workstation_aisle_result(include_blocker=True)

    assert result["label"] == "fail"
    assert result["diagnostics"]["blocking_ids"] == ["aisle_cart"]


def test_behind_uses_target_rear_axis_and_is_repairable() -> None:
    contract = build_intent_contract(
        "An office chair sits behind the reception desk.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": ["office_chair", "reception_desk"],
            "intent_constraints": [
                {
                    "relation": "behind",
                    "subjects": {"category": "office_chair", "count": 1},
                    "targets": {"category": "reception_desk", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "office chair sits behind the reception desk",
                }
            ],
        },
    )
    case_pack = {
        "stage": "furniture",
        "intent_contract": contract,
        "scene_geometry": {
            "rooms": [
                {"id": "reception", "bbox": {"min": [-3, -3, 0], "max": [3, 3, 2.7]}}
            ],
            "objects": [
                _record("desk_0", "reception_desk", (0.0, 0.0, 0.4), yaw_deg=0.0),
                _record("chair_0", "office_chair", (0.0, -1.0, 0.4), yaw_deg=0.0),
            ],
        },
        "checks": [],
    }

    results = run_case_pack_checks(
        case_pack, CriticConfig(enabled=True, metrics=("functional_dependency",))
    )
    result = next(
        row for row in results if row["relation_type"] == "rear_axis_alignment"
    )

    assert result["label"] == "pass"
    assert result["diagnostics"]["target_relation_axis_xy"] == [-0.0, -1.0]
    assert "rear_axis_alignment" in repair_relation_types(
        strategies=_FURNITURE_REPAIR_STRATEGIES
    )


def test_middle_of_support_rejects_malformed_wall_constraint() -> None:
    contract = build_intent_contract(
        "A centerpiece vase with flowers sits in the middle of the dining table.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": ["dining_table"],
            "required_small_objects": ["vase", "flowers"],
            "intent_constraints": [
                {
                    "relation": "centered_on_wall",
                    "subjects": {"category": "vase", "count": 1},
                    "targets": {"category": "dining_table", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "vase sits in the middle of the dining table",
                }
            ],
        },
    )

    assert not any(
        row["relation"] in {"centered_on_wall", "centered_in_room"}
        for row in contract["constraints"]
    )
    support = next(
        row
        for row in contract["constraints"]
        if row["relation"] == "on_top_of" and row["subjects"]["category"] == "vase"
    )
    assert support["targets"]["category"] == "dining_table"


def test_place_setting_component_count_is_minimum_and_checks_every_component() -> None:
    contract = build_intent_contract(
        "Set table settings for four including cutlery.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": ["dining_table"],
            "required_small_objects": ["cutlery"] * 4,
            "intent_constraints": [
                {
                    "relation": "on_top_of",
                    "subjects": {"category": "cutlery", "count": 4},
                    "targets": {"category": "dining_table", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "table settings for four including cutlery",
                }
            ],
        },
    )
    support = next(
        row
        for row in contract["constraints"]
        if row["relation"] == "on_top_of" and row["subjects"]["category"] == "cutlery"
    )
    objects = [_record("dining_table_0", "dining_table", (0.0, 0.0, 0.4))] + [
        _record(f"cutlery_{index}", "cutlery", (0.0, 0.0, 0.9)) for index in range(8)
    ]
    case_pack = {
        "intent_contract": contract,
        "scene_geometry": {"objects": objects},
        "checks": [],
    }

    assert support["subjects"]["quantifier"] == "minimum"
    assert bound_ids(support["subjects"], objects) == [
        f"cutlery_{index}" for index in range(8)
    ]
    assert augment_contract_checks(case_pack)
    assert (
        len(
            [
                row
                for row in case_pack["checks"]
                if row["relation_type"] == "object_on_support"
            ]
        )
        == 8
    )


def test_model_inferred_compound_component_direction_is_dropped() -> None:
    contract = build_intent_contract(
        "A vase with flowers.",
        task_spec={
            "compiler_status": "ok",
            "required_small_objects": ["vase", "flowers"],
            "intent_constraints": [
                {
                    "relation": "in_front_of",
                    "subjects": {"category": "flowers", "count": 1},
                    "targets": {"category": "vase", "count": 1},
                    "source": "model_inferred",
                    "inference_reason": "Flowers emerge from a vase.",
                }
            ],
        },
    )
    assert not any(row["relation"] == "in_front_of" for row in contract["constraints"])


def test_behind_one_side_selects_a_satisfying_target_from_group() -> None:
    prompt = "A sideboard sits behind the chairs on one side."
    contract = build_intent_contract(
        prompt,
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": [
                "sideboard",
                "dining_chair",
                "dining_chair",
                "dining_chair",
                "dining_chair",
            ],
            "intent_constraints": [
                {
                    "relation": "behind",
                    "subjects": {"category": "sideboard", "count": 1},
                    # Compilers may retain a typed endpoint even when the
                    # evidence uses the broad noun "chairs".
                    "targets": {
                        "category": "dining_chair",
                        "count": 4,
                        "quantifier": "all",
                    },
                    "source": "explicit_prompt",
                    "evidence_span": "behind the chairs on one side",
                }
            ],
        },
    )
    behind = next(row for row in contract["constraints"] if row["relation"] == "behind")
    assert behind["targets"]["count"] == 1
    assert behind["targets"]["quantifier"] == "minimum"

    case_pack = {
        "stage": "furniture",
        "intent_contract": contract,
        "scene_geometry": {
            "rooms": [
                {"id": "dining", "bbox": {"min": [-3, -3, 0], "max": [3, 3, 2.7]}}
            ],
            "objects": [
                _record("sideboard_0", "sideboard", (0.0, 1.8, 0.4)),
                _record("chair_0", "dining_chair", (0.0, 0.7, 0.45), yaw_deg=180.0),
                _record("chair_1", "dining_chair", (0.0, -0.7, 0.45), yaw_deg=0.0),
                _record("chair_2", "dining_chair", (0.9, 0.0, 0.45), yaw_deg=90.0),
                _record("chair_3", "dining_chair", (-0.9, 0.0, 0.45), yaw_deg=-90.0),
            ],
        },
        "checks": [],
    }

    results = run_case_pack_checks(
        case_pack, CriticConfig(enabled=True, metrics=("functional_dependency",))
    )
    result = next(
        row for row in results if row["relation_type"] == "rear_axis_alignment"
    )
    assert result["label"] == "pass"
    assert result["related_objects"] == ["chair_0"]
    assert result["diagnostics"]["existential_target_selection"] is True


def test_unrelated_one_side_clause_does_not_weaken_plural_target() -> None:
    contract = build_intent_contract(
        "A sideboard sits behind the dining chairs; place a lamp on one side.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": [
                "sideboard",
                "dining_chair",
                "dining_chair",
                "lamp",
            ],
            "intent_constraints": [
                {
                    "relation": "behind",
                    "subjects": {"category": "sideboard", "count": 1},
                    "targets": {"category": "dining_chair"},
                    "source": "explicit_prompt",
                    "evidence_span": (
                        "behind the dining chairs; place a lamp on one side"
                    ),
                }
            ],
        },
    )

    behind = next(row for row in contract["constraints"] if row["relation"] == "behind")
    assert behind["targets"].get("quantifier") != "minimum"


def test_unique_typed_inventory_deduplicates_broad_relation_endpoint() -> None:
    prompt = "A sideboard sits behind the dining chairs on one side."
    contract = build_intent_contract(
        prompt,
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": [
                "sideboard",
                "dining_chair",
                "dining_chair",
                "dining_chair",
                "dining_chair",
            ],
            "intent_constraints": [
                {
                    "relation": "behind",
                    "subjects": {"category": "sideboard", "count": 1},
                    "targets": {"category": "chair"},
                    "source": "explicit_prompt",
                    "evidence_span": "behind the dining chairs on one side",
                }
            ],
        },
    )

    behind = [row for row in contract["constraints"] if row["relation"] == "behind"]

    assert len(behind) == 1
    assert behind[0]["targets"] == {
        "category": "dining_chair",
        "count": 1,
        "quantifier": "minimum",
    }


def test_mixed_typed_inventory_keeps_broad_relation_endpoint() -> None:
    contract = build_intent_contract(
        "A sideboard sits behind a chair.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": [
                "sideboard",
                "dining_chair",
                "office_chair",
            ],
            "intent_constraints": [
                {
                    "relation": "behind",
                    "subjects": {"category": "sideboard", "count": 1},
                    "targets": {"category": "chair", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "sideboard sits behind a chair",
                }
            ],
        },
    )

    behind = [row for row in contract["constraints"] if row["relation"] == "behind"]

    assert len(behind) == 1
    assert behind[0]["targets"]["category"] == "chair"


def test_wall_decor_identity_does_not_bind_as_furniture() -> None:
    bed = _record("bed_0", "bed", (0.0, 0.0, 0.4))
    decor = _record("bedroom_art_bed_0", "unknown", (0.0, 2.0, 1.5))
    decor["name"] = "bedroom art above bed"
    decor["object_type"] = "wall_mounted"
    decor["functional_hints"] = {"scene_object_type": "wall_mounted"}

    assert selected_ids({"category": "bed"}, [bed, decor]) == ["bed_0"]


def test_between_binds_two_instances_of_the_same_anchor_category() -> None:
    contract = build_intent_contract(
        "A side table sits between the two guest chairs.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": ["side_table", "guest_chair", "guest_chair"],
            "intent_constraints": [
                {
                    "relation": "between",
                    "subjects": {"category": "side_table", "count": 1},
                    "targets": {
                        "category": "guest_chair",
                        "count": 2,
                        "secondary_category": "guest_chair",
                        "secondary_count": 2,
                    },
                    "source": "explicit_prompt",
                    "evidence_span": "side table sits between the two guest chairs",
                }
            ],
        },
    )
    case_pack = {
        "stage": "furniture",
        "intent_contract": contract,
        "scene_geometry": {
            "rooms": [
                {"id": "reception", "bbox": {"min": [-3, -3, 0], "max": [3, 3, 2.7]}}
            ],
            "objects": [
                _record("guest_0", "guest_chair", (-1.0, 0.0, 0.4)),
                _record("guest_1", "guest_chair", (1.0, 0.0, 0.4)),
                _record("table_0", "side_table", (0.0, 0.0, 0.4)),
            ],
        },
        "checks": [],
    }

    results = run_case_pack_checks(
        case_pack, CriticConfig(enabled=True, metrics=("functional_dependency",))
    )
    result = next(row for row in results if row["relation_type"] == "between_alignment")

    assert result["label"] == "pass"
    assert result["related_objects"] == ["guest_0", "guest_1"]


@pytest.mark.parametrize("blocked", [False, True])
def test_clear_access_from_entrance_uses_connected_walkable_space(
    blocked: bool,
) -> None:
    contract = build_intent_contract(
        "Keep an open route from the entrance to the desk.",
        task_spec={
            "compiler_status": "ok",
            "required_large_objects": ["reception_desk"],
            "intent_constraints": [
                {
                    "relation": "clear_access",
                    "subjects": {"category": "entrance", "count": 1},
                    "targets": {"category": "reception_desk", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "open route from the entrance to the desk",
                }
            ],
        },
    )
    objects = [_record("desk_0", "reception_desk", (0.0, 2.0, 0.4))]
    if blocked:
        objects.append(
            _record("barrier_0", "cabinet", (0.0, 0.0, 0.5), (5.2, 0.3, 1.0))
        )
    case_pack = {
        "stage": "furniture",
        "intent_contract": contract,
        "scene_geometry": {
            "rooms": [
                {"id": "reception", "bbox": {"min": [-3, -3, 0], "max": [3, 3, 2.7]}}
            ],
            "objects": objects,
            "scene_shell": {
                "doors": [
                    {"id": "door_0", "opening_type": "door", "center": [0.0, -3.0, 1.0]}
                ]
            },
        },
        "checks": [],
    }

    results = run_case_pack_checks(
        case_pack, CriticConfig(enabled=True, metrics=("functional_dependency",))
    )
    result = next(
        row
        for row in results
        if row["relation_type"] == "clear_access"
        and row["diagnostics"].get("evaluation_mode") == "entrance_route"
    )

    assert result["label"] == ("fail" if blocked else "pass")
    if blocked:
        assert "barrier_0" in result["diagnostics"]["blocking_ids"]
