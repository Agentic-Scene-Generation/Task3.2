"""Focused coverage for the independent v4 intent contract and edge geometry."""

from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace

import pytest

from pydantic import ValidationError

from scenesmith.scenebenchmark_critic.intent_schema import (
    INTENT_COMPILER_SPEC_VERSION,
    INTENT_CONTRACT_SCHEMA_VERSION,
    canonical_selector_category,
    intent_contract_json_schema,
    validate_intent_contract,
)
from scenesmith.scenebenchmark_critic.intent_compiler import (
    IntentCompilationError,
    IntentCompiler,
)
from scenesmith.scenebenchmark_critic.intent_contract import (
    apply_contract_execution_states,
    bound_ids,
    build_intent_contract,
    intent_contract_required_counts,
    selected_ids,
    selector_match_count,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
    _binding_state_result,
    _evaluate_required_count,
    evaluate_intent_contract_extensions,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.edge_distribution import (
    evaluate_edge_distribution,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.relations import (
    _relation_target_is_valid,
)
from scenesmith.scene_expert import hooks
from scenesmith.scene_expert.schemas import SceneTaskSpec


def _edge_relation(
    *,
    subject_count: int = 7,
    groups: list[dict] | None = None,
    target: dict | None = None,
    orientation: str = "toward_target",
) -> dict:
    return {
        "relation": "edge_distribution",
        "subjects": {"category": "office_chair", "count": subject_count},
        "targets": target or {"category": "conference_table", "count": 1},
        "edge_frame": "target_local_rectangle",
        "groups": groups
        or [
            {
                "edge_class": "long",
                "counts_per_edge": [3, 3],
                "spacing": "equal_segments",
            },
            {
                "edge_class": "short",
                "counts_per_edge": [1, 0],
                "spacing": "equal_segments",
            },
        ],
        "orientation": orientation,
        "source": "explicit_prompt",
        "evidence_span": "seven office chairs around the conference table",
    }


def _contract(relation: dict) -> dict:
    return {
        "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
        "prompt": "seven office chairs around the conference table",
        "constraints": [relation],
    }


def _record(
    object_id: str,
    category: str,
    center: tuple[float, float],
    size: tuple[float, float, float],
    *,
    yaw_deg: float = 0.0,
    footprint_shape: str | None = None,
) -> dict:
    x, y = center
    return {
        "id": object_id,
        "category": category,
        "category_norm": category,
        "name": category,
        "yaw_deg": yaw_deg,
        "bbox_world": {
            "center": [x, y, size[2] / 2.0],
            "size": list(size),
            "min": [x - size[0] / 2.0, y - size[1] / 2.0, 0.0],
            "max": [x + size[0] / 2.0, y + size[1] / 2.0, size[2]],
        },
        **(
            {"functional_hints": {"footprint_shape": footprint_shape}}
            if footprint_shape
            else {}
        ),
    }


def _rotated_meeting_case(
    *,
    yaw_deg: float = 30.0,
    orientation: str = "toward_target",
    spacing: str = "equal_segments",
) -> dict:
    center = (1.0, -2.0)
    yaw = math.radians(yaw_deg)
    tangent_x = (math.cos(yaw), math.sin(yaw))
    tangent_y = (-math.sin(yaw), math.cos(yaw))
    target = _record(
        "conference_table_0",
        "conference_table",
        center,
        (4.0, 1.2, 0.75),
        yaw_deg=yaw_deg,
    )
    subjects: list[dict] = []
    slots = [
        ("front", -4.0 / 3.0, yaw_deg),
        ("front", 0.0, yaw_deg),
        ("front", 4.0 / 3.0, yaw_deg),
        ("back", -4.0 / 3.0, yaw_deg + 180.0),
        ("back", 0.0, yaw_deg + 180.0),
        ("back", 4.0 / 3.0, yaw_deg + 180.0),
        ("left", 0.0, yaw_deg - 90.0),
    ]
    for index, (edge, tangent, subject_yaw) in enumerate(slots):
        normal = 0.95 if edge == "back" else -0.95 if edge == "front" else 0.0
        if edge == "left":
            local_x, local_y = -2.35, tangent
        elif edge == "right":
            local_x, local_y = 2.35, tangent
        else:
            local_x, local_y = tangent, normal
        subjects.append(
            _record(
                f"office_chair_{index}",
                "office_chair",
                (
                    center[0] + local_x * tangent_x[0] + local_y * tangent_y[0],
                    center[1] + local_x * tangent_x[1] + local_y * tangent_y[1],
                ),
                (0.6, 0.6, 0.9),
                yaw_deg=subject_yaw,
            )
        )
    relation = _edge_relation(
        orientation=orientation,
        groups=[
            {
                "edge_class": "long",
                "counts_per_edge": [3, 3],
                "spacing": spacing,
            },
            {
                "edge_class": "short",
                "counts_per_edge": [1, 0],
                "spacing": spacing,
            },
        ],
    )
    return {
        "intent_contract": _contract(relation),
        "scene_geometry": {"objects": [target, *subjects]},
    }


def test_schema_normalizes_edge_counts_and_supports_dining_shape() -> None:
    relation = _edge_relation(
        subject_count=4,
        groups=[
            {"edge_class": "short", "counts_per_edge": [1, 1]},
            {"edge_class": "long", "counts_per_edge": [1, 1]},
        ],
    )

    result = validate_intent_contract(_contract(relation))

    groups = result["constraints"][0]["groups"]
    assert groups[0]["counts_per_edge"] == [1, 1]
    assert groups[1]["counts_per_edge"] == [1, 1]


def test_schema_canonicalizes_dining_aliases_and_derives_manipuland_stage() -> None:
    result = validate_intent_contract(
        {
            "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
            "prompt": "dining room",
            "constraints": [
                {
                    "relation": "surround",
                    "subjects": {"category": "dining_chairs", "count": 4},
                    "targets": {"category": "dining_table", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "four dining chairs around the table",
                },
                {
                    "relation": "on_top_of",
                    "subjects": {"category": "centerpiece_vase", "count": 1},
                    "targets": {"category": "dining_table", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "centerpiece vase on the table",
                },
                {
                    "relation": "on_top_of",
                    "subjects": {"category": "coasters", "count": 1},
                    "targets": {"category": "sideboard", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "coasters on the sideboard",
                },
                {
                    "relation": "required_count",
                    "subjects": {"category": "table_settings", "count": 4},
                    "source": "explicit_prompt",
                    "evidence_span": "table settings for four",
                },
            ],
        }
    )

    constraints = result["constraints"]
    assert constraints[0]["subjects"]["category"] == "dining_chair"
    assert constraints[1]["subjects"]["category"] == "vase"
    assert constraints[1]["stage"] == "manipuland"
    assert constraints[2]["subjects"]["category"] == "coaster"
    assert constraints[2]["stage"] == "manipuland"
    assert constraints[3]["subjects"]["category"] == "table_setting"
    assert constraints[3]["stage"] == "manipuland"


@pytest.mark.parametrize(
    ("raw_category", "canonical_category"),
    [
        ("large_plants", "large_plant"),
        ("two_seater_sofas", "two_seater_sofa"),
        ("office_chairs", "office_chair"),
        ("batteries", "battery"),
    ],
)
def test_schema_normalizes_common_plural_selector_categories(
    raw_category: str, canonical_category: str
) -> None:
    assert canonical_selector_category(raw_category) == canonical_category


def test_plural_large_plant_selector_binds_all_matching_objects() -> None:
    objects = [
        _record("large_plant_0", "large_plant", (1.0, 1.0), (0.5, 0.5, 1.2)),
        _record("large_plant_1", "large_plant", (1.0, -1.0), (0.5, 0.5, 1.2)),
    ]

    selector = {"category": "large_plants", "count": 2}

    assert selected_ids(selector, objects) == ["large_plant_0", "large_plant_1"]
    assert selector_match_count(selector, objects) == 2


def test_modified_selector_can_bind_a_generic_retrieved_asset() -> None:
    objects = [_record("rug_0", "rug", (0.0, 0.0), (1.8, 1.8, 0.03))]

    selector = {"category": "square_rug", "count": 1}

    assert selected_ids(selector, objects) == ["rug_0"]
    assert bound_ids(selector, objects) == ["rug_0"]


def test_selector_matches_generator_added_numeric_category_suffix() -> None:
    objects = [
        _record("large_plant_0", "large_plant", (1.0, 1.0), (0.5, 0.5, 1.2)),
        _record(
            "large_plant_2_0",
            "large_plant_2",
            (1.0, -1.0),
            (0.5, 0.5, 1.2),
        ),
    ]

    selector = {"category": "large_plant", "count": 2, "quantifier": "exactly"}

    assert selected_ids(selector, objects) == ["large_plant_0", "large_plant_2_0"]
    assert bound_ids(selector, objects) == ["large_plant_0", "large_plant_2_0"]


def test_schema_derives_manipuland_stage_for_common_small_objects() -> None:
    result = validate_intent_contract(
        {
            "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
            "prompt": "bedside objects",
            "constraints": [
                {
                    "relation": "on_top_of",
                    "subjects": {"category": "table_lamp", "count": 2},
                    "targets": {"category": "nightstand", "count": 2},
                    "source": "explicit_prompt",
                    "evidence_span": "table lamps on the nightstands",
                },
                {
                    "relation": "on_top_of",
                    "subjects": {"category": "alarm_clock", "count": 1},
                    "targets": {"category": "nightstand", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "an alarm clock on one nightstand",
                },
                {
                    "relation": "near",
                    "subjects": {"category": "wastebasket", "count": 1},
                    "targets": {"category": "dresser", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "a small wastebasket near the dresser",
                },
                {
                    "relation": "on_top_of",
                    "subjects": {"category": "computer_monitor", "count": 1},
                    "targets": {"category": "desk", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "a computer monitor on the desk",
                },
            ],
        }
    )

    assert [constraint["stage"] for constraint in result["constraints"]] == [
        "manipuland",
        "manipuland",
        "manipuland",
        "manipuland",
    ]


def test_furniture_stage_keeps_manipuland_contract_pending() -> None:
    contract = validate_intent_contract(
        {
            "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
            "prompt": "a vase on a dining table",
            "constraints": [
                {
                    "relation": "on_top_of",
                    "subjects": {"category": "centerpiece_vase", "count": 1},
                    "targets": {"category": "dining_table", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "a vase on a dining table",
                }
            ],
        }
    )
    case_pack = {"stage": "furniture", "intent_contract": contract}

    results = apply_contract_execution_states(case_pack, [])

    assert results[0]["contract_state"] == "pending"
    assert results[0]["label"] == "unknown"
    assert results[0]["scoring_tier"] == "auxiliary"


def test_table_setting_count_uses_minimum_component_count() -> None:
    constraint = {
        "constraint_id": "intent_table_setting",
        "relation": "required_count",
        "subjects": {"category": "table_setting", "count": 4},
    }
    objects = [
        _record(f"plate_{index}", "plate", (0.0, 0.0), (0.2, 0.2, 0.03))
        for index in range(4)
    ]
    objects.extend(
        _record(f"cutlery_{index}", "fork", (0.0, 0.0), (0.2, 0.05, 0.03))
        for index in range(8)
    )
    objects.extend(
        _record(f"glass_{index}", "wine_glass", (0.0, 0.0), (0.1, 0.1, 0.2))
        for index in range(3)
    )

    result = _evaluate_required_count(constraint, objects, "core")

    assert result["label"] == "fail"
    assert result["diagnostics"]["component_counts"] == {
        "plate": 4,
        "cutlery": 8,
        "glass": 3,
    }
    assert result["diagnostics"]["observed_count"] == 3


@pytest.mark.parametrize(
    "relation",
    [
        _edge_relation(subject_count=6),
        _edge_relation(
            groups=[
                {"edge_class": "long", "counts_per_edge": [-1, 2]},
                {"edge_class": "short", "counts_per_edge": [1, 0]},
            ]
        ),
        _edge_relation(
            groups=[
                {"edge_class": "long", "counts_per_edge": [3, 3]},
                {"edge_class": "long", "counts_per_edge": [1, 0]},
            ]
        ),
        _edge_relation(target={"category": "conference_table", "count": 2}),
        _edge_relation(
            target={
                "category": "conference_table",
                "count": 1,
                "secondary_category": "sideboard",
            }
        ),
        {
            **_edge_relation(),
            "relation": "one_per_side",
        },
    ],
)
def test_schema_rejects_invalid_edge_contracts(relation: dict) -> None:
    with pytest.raises((ValidationError, ValueError)):
        validate_intent_contract(_contract(relation))


def test_rotated_rectangle_edge_distribution_passes_and_checks_inward_facing() -> None:
    result = evaluate_edge_distribution(_rotated_meeting_case())[0]

    assert result["label"] == "pass"
    assert len(result["diagnostics"]["topology_repair_slots"]) == 7

    wrong_orientation = _rotated_meeting_case()
    wrong_orientation["scene_geometry"]["objects"][1]["yaw_deg"] += 180.0
    failed = evaluate_edge_distribution(wrong_orientation)[0]
    assert failed["label"] == "fail"
    assert "face target inward" in failed["reason"]


def test_unconstrained_spacing_does_not_require_fixed_tangent_positions() -> None:
    case_pack = _rotated_meeting_case(
        orientation="unconstrained", spacing="unconstrained"
    )
    chair = case_pack["scene_geometry"]["objects"][1]
    chair["bbox_world"]["center"][0] += 0.45
    chair["bbox_world"]["min"][0] += 0.45
    chair["bbox_world"]["max"][0] += 0.45

    assert evaluate_edge_distribution(case_pack)[0]["label"] == "pass"


@pytest.mark.parametrize(
    "target_update",
    [
        {"extra_subject": True},
        {"extra_target": True},
        {"target_size": (2.0, 2.0, 0.75)},
        {"target_name": "round conference table"},
        {"target_shape": "l_shape"},
    ],
)
def test_ambiguous_edge_bindings_are_unresolved(target_update: dict) -> None:
    case_pack = _rotated_meeting_case()
    objects = case_pack["scene_geometry"]["objects"]
    if target_update.get("extra_subject"):
        objects.append({**objects[1], "id": "office_chair_extra"})
    if target_update.get("extra_target"):
        objects.append({**objects[0], "id": "conference_table_extra"})
    if target_update.get("target_size"):
        objects[0]["bbox_world"]["size"] = list(target_update["target_size"])
    if target_update.get("target_name"):
        objects[0]["name"] = target_update["target_name"]
    if target_update.get("target_shape"):
        objects[0]["functional_hints"] = {
            "footprint_shape": target_update["target_shape"]
        }

    result = evaluate_edge_distribution(case_pack)[0]

    assert result["label"] == "unresolved"
    assert result["diagnostics"]["unresolved"] is True


def test_unique_generic_table_is_a_fallback_for_specialized_selector() -> None:
    generic_table = _record("table_0", "table", (0.0, 0.0), (3.2, 1.0, 0.75))
    generic_table["metadata"] = {"semantic_name": "table"}
    selector = {"category": "conference_table", "count": 1}

    assert selected_ids(selector, [generic_table]) == ["table_0"]
    assert selector_match_count(selector, [generic_table]) == 1


def test_exact_specialized_candidate_precedes_generic_fallback() -> None:
    generic_table = _record("table_0", "table", (0.0, 0.0), (3.2, 1.0, 0.75))
    specialized_table = _record(
        "conference_table_0",
        "conference_table",
        (2.0, 0.0),
        (3.2, 1.0, 0.75),
    )
    selector = {"category": "conference_table", "count": 1}

    assert selected_ids(selector, [generic_table, specialized_table]) == [
        "conference_table_0"
    ]
    assert selector_match_count(selector, [generic_table, specialized_table]) == 1


def test_furniture_selector_ignores_later_stage_decor_with_parent_category() -> None:
    dresser = _record("dresser_0", "dresser", (0.0, 0.0), (1.2, 0.5, 0.85))
    tabletop_mirror = _record(
        "dresser_mirror_tabletop_0", "dresser", (0.0, 0.0), (0.35, 0.15, 0.4)
    )
    tabletop_mirror.update(
        {
            "object_type": "manipuland",
            "metadata": {"semantic_name": "dresser_mirror_tabletop"},
            "functional_hints": {"scene_object_type": "manipuland"},
        }
    )

    selector = {"category": "dresser", "count": 1, "quantifier": "exactly"}

    assert selected_ids(selector, [dresser, tabletop_mirror]) == ["dresser_0"]
    assert selector_match_count(selector, [dresser, tabletop_mirror]) == 1


def test_bedside_lamp_semantic_name_is_valid_for_nightstand_support() -> None:
    lamp = _record("bedside_lamp_0", "lamp", (0.0, 0.0), (0.15, 0.15, 0.3))
    lamp["metadata"] = {"semantic_name": "bedside_lamp"}
    nightstand = _record("nightstand_0", "nightstand", (0.0, 0.0), (0.5, 0.4, 0.7))
    nightstand["functional_hints"] = {
        "scene_object_type": "furniture",
        "category_group": "storage_surface",
    }

    assert _relation_target_is_valid(lamp, nightstand, "lamp_to_surface")


def test_existential_target_member_does_not_fail_contract_binding() -> None:
    contract = validate_intent_contract(
        {
            "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
            "prompt": "an alarm clock on one of two nightstands",
            "constraints": [
                {
                    "relation": "on_top_of",
                    "subjects": {"category": "alarm_clock", "count": 1},
                    "targets": {
                        "category": "nightstand",
                        "count": 1,
                        "quantifier": "at_least",
                    },
                    "source": "explicit_prompt",
                    "evidence_span": "an alarm clock on one of two nightstands",
                }
            ],
        }
    )
    objects = [
        _record("alarm_clock_0", "alarm_clock", (0.0, 0.0), (0.1, 0.1, 0.1)),
        _record("nightstand_0", "nightstand", (-1.0, 0.0), (0.5, 0.4, 0.7)),
        _record("nightstand_1", "nightstand", (1.0, 0.0), (0.5, 0.4, 0.7)),
    ]

    assert (
        _binding_state_result(
            {"stage": "final", "intent_contract": contract},
            contract["constraints"][0],
            objects,
        )
        is None
    )


def test_singular_target_wording_is_normalized_to_existential_binding() -> None:
    contract = validate_intent_contract(
        {
            "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
            "prompt": "a floor lamp beside one armchair",
            "constraints": [
                {
                    "relation": "near",
                    "subjects": {"category": "floor_lamp", "count": 1},
                    "targets": {"category": "armchair", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "a floor lamp beside one armchair",
                }
            ],
        }
    )

    assert contract["constraints"][0]["targets"]["quantifier"] == "minimum"
    objects = [
        _record("floor_lamp_0", "floor_lamp", (0.0, 0.0), (0.4, 0.4, 1.6)),
        _record("armchair_0", "armchair", (0.5, 0.0), (0.7, 0.7, 0.9)),
        _record("armchair_1", "armchair", (2.5, 0.0), (0.7, 0.7, 0.9)),
    ]

    assert (
        _binding_state_result(
            {"stage": "final", "intent_contract": contract},
            contract["constraints"][0],
            objects,
        )
        is None
    )


def test_collective_subject_does_not_fail_contract_binding() -> None:
    contract = validate_intent_contract(
        {
            "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
            "prompt": "a set of coasters on a sideboard",
            "constraints": [
                {
                    "relation": "on_top_of",
                    "subjects": {
                        "category": "coaster",
                        "count": 1,
                        "quantifier": "at_least",
                    },
                    "targets": {"category": "sideboard", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "a set of coasters on the sideboard",
                }
            ],
        }
    )
    objects = [
        _record(f"coaster_{index}", "coaster", (0.0, 0.0), (0.1, 0.1, 0.02))
        for index in range(4)
    ]
    objects.append(_record("sideboard_0", "sideboard", (0.0, 0.0), (1.2, 0.4, 0.8)))

    assert (
        _binding_state_result(
            {"stage": "final", "intent_contract": contract},
            contract["constraints"][0],
            objects,
        )
        is None
    )


@pytest.mark.parametrize("target_category", ["back_wall", "adjacent_wall"])
def test_on_wall_uses_physical_wall_for_room_relative_target(
    target_category: str,
) -> None:
    constraint = {
        "constraint_id": "on_wall__chalkboard",
        "relation": "on_wall",
        "subjects": {"category": "chalkboard", "count": 1},
        "targets": {"category": target_category, "count": 1},
        "source": "explicit_prompt",
        "confidence": 1.0,
        "evidence_span": "chalkboard hangs on the back wall",
    }
    objects = [
        _record("chalkboard_0", "chalkboard", (0.0, 1.9), (1.4, 0.1, 0.9)),
        _record("wall_0", "wall", (0.0, 2.0), (4.0, 0.1, 2.8)),
    ]

    assert _binding_state_result({"stage": "final"}, constraint, objects) is None


def test_furniture_relation_endpoints_supply_contract_inventory_counts() -> None:
    scene = SimpleNamespace(
        scenebenchmark_intent_contract={
            "constraints": [
                {
                    "relation": "paired_with",
                    "stage": "furniture",
                    "strength": "hard",
                    "subjects": {"category": "office_chair", "count": 1},
                    "targets": {"category": "desk", "count": 1},
                },
                {
                    "relation": "on_top_of",
                    "stage": "manipuland",
                    "strength": "hard",
                    "subjects": {"category": "desk_lamp", "count": 1},
                    "targets": {"category": "desk", "count": 1},
                },
            ]
        }
    )

    assert intent_contract_required_counts(scene) == {
        "office_chair": 1,
        "desk": 1,
    }


def test_faces_room_is_evaluated_as_an_interior_direction() -> None:
    constraint = {
        "constraint_id": "guest_chair_faces_room",
        "relation": "faces",
        "subjects": {"category": "guest_chair", "count": 1},
        "targets": {"category": "room", "count": 1},
        "source": "explicit_prompt",
        "confidence": 1.0,
        "evidence_span": "guest chair facing into the room",
    }
    case_pack = {
        "stage": "furniture",
        "intent_contract": {"constraints": [constraint]},
        "scene_geometry": {
            "rooms": [{"bbox": {"min": [-2.0, -2.0], "max": [2.0, 2.0]}}],
            "objects": [
                _record(
                    "guest_chair_0",
                    "guest_chair",
                    (0.0, -1.5),
                    (0.5, 0.5, 0.9),
                    yaw_deg=0.0,
                )
            ],
        },
    }

    result = evaluate_intent_contract_extensions(case_pack)[0]

    assert result["label"] == "pass"
    assert result["relation_type"] == "faces"
    assert result["diagnostics"]["facing_target_xy_m"] == [0.0, 0.0]


def test_edge_distribution_accepts_unique_generic_target_asset() -> None:
    case_pack = _rotated_meeting_case()
    target = case_pack["scene_geometry"]["objects"][0]
    target.update(
        {
            "id": "table_0",
            "name": "table",
            "category": "table",
            "category_norm": "table",
            "metadata": {"semantic_name": "table"},
        }
    )

    result = evaluate_edge_distribution(case_pack)[0]

    assert result["label"] == "pass"
    assert result["primary_object"] == "table_0"


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _compiler_with_responses(responses: list[SimpleNamespace]) -> IntentCompiler:
    compiler = object.__new__(IntentCompiler)
    compiler._model = "test-model"
    compiler._max_tokens = 256
    compiler._temperature = 0.0
    compiler.last_trace = {}
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    compiler._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    compiler._test_calls = calls
    return compiler


def test_task_spec_rejects_removed_intent_constraints_field() -> None:
    with pytest.raises(ValidationError, match="intent_constraints"):
        SceneTaskSpec.model_validate(
            {
                "room_type": "office",
                "style": "standard",
                "intent_constraints": [],
            }
        )


def test_intent_compiler_retries_once_without_task_spec_input() -> None:
    compiler = _compiler_with_responses(
        [
            _response('{"constraints": [{"relation": "one_per_side"}]}'),
            _response('{"constraints": []}'),
        ]
    )

    result = compiler.compile("A meeting room with a conference table.")

    assert result["schema_version"] == INTENT_CONTRACT_SCHEMA_VERSION
    assert result["retry_count"] == 1
    assert compiler.last_trace["retry_count"] == 1
    assert [item["status"] for item in compiler.last_trace["attempts"]] == [
        "error",
        "ok",
    ]
    first_user_message = compiler._test_calls[0]["messages"][1]["content"]
    assert "Original scene prompt:" in first_user_message
    assert "TaskSpec" not in first_user_message

    for call in compiler._test_calls:
        assert call["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "intent_contract",
                "strict": True,
                "schema": intent_contract_json_schema(),
            },
        }


def test_intent_compiler_canonicalizes_two_object_side_wording_to_flanking() -> None:
    compiler = _compiler_with_responses(
        [
            _response(
                '{"constraints": [{"relation": "edge_distribution", '
                '"subjects": {"category": "nightstand", "count": 2}, '
                '"targets": {"category": "bed", "count": 1}, '
                '"edge_frame": "target_local_rectangle", '
                '"groups": [{"edge_class": "short", '
                '"counts_per_edge": [1, 1]}], '
                '"orientation": "unconstrained", '
                '"source": "explicit_prompt", '
                '"evidence_span": "a nightstand on each side of the bed"}]}'
            )
        ]
    )

    result = compiler.compile("A bedroom with a nightstand on each side of the bed.")

    relation = result["constraints"][0]
    assert relation["relation"] == "flanking"
    assert relation["groups"] == []
    assert relation.get("edge_frame") is None
    assert relation.get("orientation") is None


def test_intent_compiler_retry_spells_out_unary_target_cardinality() -> None:
    compiler = _compiler_with_responses(
        [
            _response(
                '{"constraints": [{"relation": "near", '
                '"subjects": {"category": "chair"}, '
                '"targets": {"category": "desk", '
                '"secondary_category": "table"}, '
                '"source": "explicit_prompt", '
                '"evidence_span": "chair near desk"}]}'
            ),
            _response('{"constraints": []}'),
        ]
    )

    compiler.compile("A chair is near a desk.")

    retry_message = compiler._test_calls[1]["messages"][1]["content"]
    assert "remove its secondary_category" in retry_message
    assert "exactly one primary target selector" in retry_message


def test_intent_compiler_falls_back_after_semantic_json_failures() -> None:
    invalid = (
        '{"constraints": [{"relation": "centered_in_room", '
        '"subjects": {"category": "conference_table"}, '
        '"source": "explicit_prompt", '
        '"evidence_span": "table centered in the room"}]}'
    )
    compiler = _compiler_with_responses([_response(invalid), _response(invalid)])

    result = compiler.compile("A bedroom with a bed centered on the main wall.")

    assert result["retry_count"] == 1
    assert result["warnings"]
    assert compiler.last_trace["status"] == "fallback"
    centered = [
        row for row in result["constraints"] if row["relation"] == "centered_on_wall"
    ]
    assert centered
    assert centered[0]["targets"]["category"] == "wall"


def test_intent_compiler_fallback_keeps_media_contract_schema_valid() -> None:
    invalid = (
        '{"constraints": [{"relation": "faces", '
        '"subjects": {"category": "sofa"}, '
        '"source": "explicit_prompt", '
        '"evidence_span": "sofa faces the television"}]}'
    )
    compiler = _compiler_with_responses([_response(invalid), _response(invalid)])

    result = compiler.compile(
        "A living room with a sofa facing a TV stand and television on the opposite wall."
    )

    assert compiler.last_trace["status"] == "fallback"
    television_relation = next(
        row
        for row in result["constraints"]
        if row["relation"] == "on_top_of"
        and row["subjects"]["category"] == "television"
    )
    assert television_relation["inference_reason"]


def test_deterministic_contract_recognizes_floor_near_manipulands() -> None:
    prompt = (
        "A bedroom with a bed, an alarm clock on one nightstand, a book on the "
        "other, and a small wastebasket near the dresser."
    )

    contract = build_intent_contract(prompt)

    assert any(
        row["relation"] == "on_top_of"
        and row["subjects"]["category"] == "alarm_clock"
        and row["targets"]["category"] == "nightstand"
        for row in contract["constraints"]
    )
    assert any(
        row["relation"] == "near"
        and row["subjects"]["category"] == "wastebasket"
        and row["targets"]["category"] == "dresser"
        for row in contract["constraints"]
    )
    validated = validate_intent_contract(contract)
    assert all(
        row["stage"] == "manipuland"
        for row in validated["constraints"]
        if row["subjects"]["category"] in {"alarm_clock", "wastebasket"}
    )


def test_deterministic_contract_recognizes_room_center_contains_wording() -> None:
    contract = build_intent_contract("The center of the room contains a bed.")

    assert [row["relation"] for row in contract["constraints"]] == ["centered_in_room"]
    assert contract["constraints"][0]["subjects"]["category"] == "bed"


def test_schema_rejects_direction_that_only_qualifies_a_wall() -> None:
    prompt = (
        "A dining room with a dining table in the center, four dining chairs "
        "arranged around it, and a sideboard against the wall behind the chairs "
        "on one side."
    )
    relation = {
        "relation": "behind",
        "subjects": {"category": "sideboard", "count": 1},
        "targets": {
            "category": "dining_chair",
            "count": 1,
            "quantifier": "at_least",
        },
        "source": "explicit_prompt",
        "evidence_span": "behind the chairs on one side",
    }

    with pytest.raises(ValidationError, match="wall-relative directional relation"):
        validate_intent_contract(
            {
                "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
                "prompt": prompt,
                "constraints": [relation],
            }
        )


def test_intent_compiler_retries_wall_qualified_directional_relation() -> None:
    prompt = (
        "A dining room with four dining chairs and a sideboard against the wall "
        "behind the chairs on one side."
    )
    compiler = _compiler_with_responses(
        [
            _response(
                '{"constraints": [{"relation": "behind", '
                '"subjects": {"category": "sideboard", "count": 1}, '
                '"targets": {"category": "dining_chair", "count": 1, '
                '"quantifier": "at_least"}, '
                '"source": "explicit_prompt", '
                '"evidence_span": "behind the chairs on one side"}]}'
            ),
            _response(
                '{"constraints": [{"relation": "against_wall", '
                '"subjects": {"category": "sideboard", "count": 1}, '
                '"targets": {"category": "wall", "count": 1}, '
                '"source": "explicit_prompt", '
                '"evidence_span": "sideboard against the wall"}]}'
            ),
        ]
    )

    result = compiler.compile(prompt)

    assert result["retry_count"] == 1
    assert [row["relation"] for row in result["constraints"]] == ["against_wall"]
    retry_message = compiler._test_calls[1]["messages"][1]["content"]
    assert "wall-relative directional relation" in retry_message
    assert "Do not convert 'X against the wall" in retry_message


def test_legacy_contract_parser_keeps_wall_qualified_behind_as_wall_relation() -> None:
    contract = build_intent_contract(
        "A dining room with a sideboard against the wall behind the chairs on one side."
    )

    sideboard_relations = [
        row
        for row in contract["constraints"]
        if row.get("subjects", {}).get("category") == "sideboard"
    ]
    assert [row["relation"] for row in sideboard_relations] == ["against_wall"]


def test_intent_compiler_fails_after_second_invalid_response() -> None:
    compiler = _compiler_with_responses([_response("no json"), _response("still bad")])

    with pytest.raises(IntentCompilationError):
        compiler.compile("A room with a desk.")

    assert compiler.last_trace["status"] == "error"
    assert compiler.last_trace["retry_count"] == 1
    assert len(compiler.last_trace["attempts"]) == 2


def test_intent_compiler_is_disabled_without_critic_request(
    monkeypatch, tmp_path
) -> None:
    class UnexpectedCompiler:
        def __init__(self, **_kwargs):
            raise AssertionError("critic-off must not construct IntentCompiler")

    monkeypatch.setattr(hooks, "IntentCompiler", UnexpectedCompiler)

    result = hooks._compile_intent_contract_if_enabled(
        prompt="A bedroom with a bed.",
        scene_id=0,
        output_dir=tmp_path,
        cfg_dict={"scenebenchmark_critic": {"enabled": False}},
    )

    assert result == ({}, {})


def test_intent_compiler_cache_uses_prompt_and_spec_only(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    class FakeCompiler:
        SPEC_VERSION = INTENT_COMPILER_SPEC_VERSION

        def __init__(self, **_kwargs):
            self.last_trace = {}

        def compile(self, prompt: str) -> dict:
            calls.append(prompt)
            return {
                "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "intent_compiler_spec_version": INTENT_COMPILER_SPEC_VERSION,
                "constraints": [],
                "retry_count": 0,
                "warnings": [],
            }

    monkeypatch.setattr(hooks, "IntentCompiler", FakeCompiler)
    cfg = {"scenebenchmark_critic": {"enabled": True}}
    hooks._compile_intent_contract_if_enabled(
        prompt="A room with a desk.",
        scene_id=0,
        output_dir=tmp_path,
        cfg_dict=cfg,
    )
    hooks._compile_intent_contract_if_enabled(
        prompt="A room with a bed.",
        scene_id=0,
        output_dir=tmp_path,
        cfg_dict=cfg,
    )

    assert calls == ["A room with a desk.", "A room with a bed."]
