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
from scenesmith.scenebenchmark_critic.intent_compiler import IntentCompiler
from scenesmith.scenebenchmark_critic.intent_contract import (
    augment_contract_checks,
    apply_contract_execution_states,
    bound_ids,
    build_intent_contract,
    intent_contract_required_counts,
    _nearest_wall_ids,
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
    evaluate_functional_dependency,
    _relation_target_is_valid,
)
from scenesmith.scenebenchmark_critic.metrics.spatial_accessibility.companions import (
    attach_expected_access_companions,
)
from scenesmith.scenebenchmark_critic.core.geometry import load_geometry
from scenesmith.scenebenchmark_critic.adapter import _category_for_object
from scenesmith.scene_expert import hooks
from scenesmith.scene_expert.schemas import SceneTaskSpec


def test_repair_placeholder_uses_stable_compound_name_as_category() -> None:
    placeholder = SimpleNamespace(
        object_id="water_dispenser_repair_placeholder_0",
        object_type=SimpleNamespace(value="furniture"),
        name="water_dispenser",
        description="deterministic placeholder water_dispenser",
        metadata={"repair_placeholder": True},
    )

    assert _category_for_object(placeholder) == "water_dispenser"


def test_wall_anchor_binds_to_nearest_boundary_not_wall_center() -> None:
    sofa = _record("sofa_0", "sofa", (-1.73, 3.01), (0.88, 2.12, 0.85))
    west_wall = _record("west_wall", "wall", (-2.225, 0.0), (0.05, 10.0, 2.8))
    north_wall = _record("north_wall", "wall", (0.0, 4.975), (4.5, 0.05, 2.8))

    assert _nearest_wall_ids(["sofa_0"], [sofa, west_wall, north_wall]) == ["west_wall"]


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


def test_schema_canonicalizes_instructional_surface_aliases_to_wall_stage() -> None:
    result = validate_intent_contract(
        {
            "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
            "prompt": "A classroom with a chalkboard on the wall.",
            "constraints": [
                {
                    "relation": "required_count",
                    "subjects": {"category": "chalkboard", "count": 1},
                    "source": "explicit_prompt",
                    "evidence_span": "a chalkboard",
                }
            ],
        }
    )

    constraint = result["constraints"][0]
    assert constraint["subjects"]["category"] == "instructional_surface"
    assert constraint["stage"] == "wall_mounted"


@pytest.mark.parametrize(
    ("raw_category", "canonical_category"),
    [
        ("large_plants", "large_plant"),
        ("two_seater_sofas", "two_seater_sofa"),
        ("teacher's desk", "teacher_desk"),
        ("office_chairs", "office_chair"),
        ("whiteboard", "instructional_surface"),
        ("entrance_route", "entrance"),
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


def test_relation_local_secondary_role_falls_back_to_unique_anchor_category() -> None:
    constraint = {
        "constraint_id": "coffee_table_between_media",
        "relation": "centered_between",
        "stage": "furniture",
        "strength": "hard",
        "subjects": {"category": "coffee_table", "count": 1},
        "targets": {
            "category": "sofa",
            "count": 1,
            "secondary_category": "tv_stand",
            "secondary_count": 1,
            # This labels the endpoint's relation purpose rather than an
            # attribute emitted by the retrieved TV-stand asset.
            "secondary_role": "anchor",
        },
        "source": "explicit_prompt",
        "evidence_span": "Center one coffee table between the sofa and TV stand",
    }
    objects = [
        _record("sofa_0", "sofa", (-2.0, 0.0), (2.0, 0.8, 0.8)),
        _record("tv_stand_0", "tv_stand", (2.0, 0.0), (1.4, 0.5, 0.6)),
        _record("coffee_table_0", "coffee_table", (0.0, 0.0), (1.0, 0.6, 0.5)),
    ]
    case_pack = {
        "stage": "furniture",
        "intent_contract": {"constraints": [constraint]},
        "scene_geometry": {"objects": objects},
    }

    assert _binding_state_result(case_pack, constraint, objects) is None
    result = evaluate_intent_contract_extensions(case_pack)
    assert [(item["relation_type"], item["label"]) for item in result] == [
        ("centered_between_alignment", "pass")
    ]


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


def test_wall_mounted_object_can_be_centered_above_furniture_without_wall_centering() -> (
    None
):
    constraints = [
        {
            "constraint_id": "mirror_on_wall",
            "relation": "on_wall",
            "subjects": {"category": "mirror", "count": 1},
            "targets": {"category": "wall", "count": 1},
            "source": "explicit_prompt",
            "evidence_span": "mount one mirror on the wall",
        },
        {
            "constraint_id": "mirror_above_dressing_table",
            "relation": "centered_above",
            "subjects": {"category": "mirror", "count": 1},
            "targets": {"category": "dressing_table", "count": 1},
            "source": "explicit_prompt",
            "evidence_span": "mirror centered directly above the dressing table",
        },
    ]
    mirror = _record("mirror_0", "mirror", (-2.0, 0.6), (0.05, 0.6, 0.9), yaw_deg=-90)
    mirror["bbox_world"]["center"][2] = 1.35
    mirror["bbox_world"]["min"][2] = 0.9
    mirror["bbox_world"]["max"][2] = 1.8
    case_pack = {
        "stage": "wall_mounted",
        "intent_contract": {"constraints": constraints},
        "scene_geometry": {
            "rooms": [{"bbox": {"min": [-2.0, -2.0], "max": [2.0, 2.0]}}],
            "objects": [
                _record("west_wall", "wall", (-2.0, 0.0), (0.1, 4.0, 2.8)),
                _record(
                    "dressing_table_0",
                    "dressing_table",
                    (-1.7, 0.6),
                    (1.2, 0.5, 0.75),
                    yaw_deg=-90,
                ),
                mirror,
            ],
        },
    }

    results = evaluate_intent_contract_extensions(case_pack)
    by_relation = {result["relation_type"]: result for result in results}

    assert by_relation["mounted_to_wall"]["label"] == "pass"
    assert by_relation["centered_above"]["label"] == "pass"
    assert (
        by_relation["centered_above"]["diagnostics"]["alignment_axis"] == "wall_tangent"
    )

    mirror["bbox_world"]["center"][1] = 1.2
    mirror["bbox_world"]["min"][1] = 0.9
    mirror["bbox_world"]["max"][1] = 1.5
    shifted = {
        result["relation_type"]: result
        for result in evaluate_intent_contract_extensions(case_pack)
    }
    assert shifted["centered_above"]["label"] == "fail"


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


def _entrance_to_room_case(*, route_blocker: dict | None = None) -> dict:
    constraint = {
        "constraint_id": "entrance_route",
        "relation": "clear_access",
        "subjects": {"category": "entrance", "count": 1},
        "targets": {"category": "room", "count": 1},
        "source": "explicit_prompt",
        "strength": "hard",
        "evidence_span": "keep the entrance route clear",
    }
    return {
        "stage": "furniture",
        "intent_contract": {"constraints": [constraint]},
        "scene_geometry": {
            "rooms": [{"bbox": {"min": [0.0, 0.0], "max": [8.0, 5.0]}}],
            "scene_shell": {
                "doors": [
                    {
                        "id": "door_0",
                        "opening_type": "door",
                        "center": [0.0, 2.5, 1.05],
                        "width": 0.9,
                    }
                ]
            },
            "objects": [route_blocker] if route_blocker is not None else [],
        },
    }


def test_clear_access_routes_entrance_to_virtual_room_interior() -> None:
    result = evaluate_intent_contract_extensions(_entrance_to_room_case())[0]

    assert result["label"] == "pass"
    assert result["primary_object"] == "door_0"
    assert result["related_objects"] == ["room"]
    assert result["diagnostics"]["destination_id"] == "room"
    assert result["diagnostics"]["destination_target_xy_m"] == [4.0, 2.5]
    assert result["diagnostics"]["destination_walkable_xy_m"] == [4.0, 2.5]


def test_clear_access_reports_when_entrance_cannot_reach_room_interior() -> None:
    blocker = _record(
        "cabinet_0",
        "cabinet",
        (2.0, 2.5),
        (0.3, 5.0, 1.2),
    )

    result = evaluate_intent_contract_extensions(
        _entrance_to_room_case(route_blocker=blocker)
    )[0]

    assert result["label"] == "fail"
    assert result["diagnostics"]["destination_id"] == "room"
    assert result["diagnostics"]["blocking_ids"] == ["cabinet_0"]


def test_clear_access_normalizes_compiler_virtual_route_to_entrance_route() -> None:
    case_pack = _entrance_to_room_case()
    constraint = case_pack["intent_contract"]["constraints"][0]
    constraint["subjects"] = {
        "category": "route",
        "count": 1,
        "quantifier": "exactly",
        "role": "circulation_path",
    }
    constraint["targets"] = {
        "category": "entrance",
        "count": 1,
        "quantifier": "exactly",
    }

    result = evaluate_intent_contract_extensions(case_pack)[0]

    assert result["label"] == "pass"
    assert result["primary_object"] == "door_0"
    assert result["related_objects"] == ["room"]
    assert result["diagnostics"]["evaluation_mode"] == "entrance_route"


def test_clear_access_virtual_route_still_reports_a_real_blocker() -> None:
    blocker = _record(
        "cabinet_0",
        "cabinet",
        (2.0, 2.5),
        (0.3, 5.0, 1.2),
    )
    case_pack = _entrance_to_room_case(route_blocker=blocker)
    constraint = case_pack["intent_contract"]["constraints"][0]
    constraint["subjects"] = {
        "category": "route",
        "count": 1,
        "quantifier": "exactly",
        "role": "circulation_path",
    }
    constraint["targets"] = {
        "category": "entrance",
        "count": 1,
        "quantifier": "exactly",
    }

    result = evaluate_intent_contract_extensions(case_pack)[0]

    assert result["label"] == "fail"
    assert result["diagnostics"]["blocking_ids"] == ["cabinet_0"]


def test_clear_access_allows_a_hard_front_facing_functional_occupant() -> None:
    dressing_table = _record(
        "dressing_table_0",
        "dressing_table",
        (0.0, 0.0),
        (1.2, 0.5, 0.75),
        yaw_deg=0.0,
    )
    stool = _record(
        "stool_0",
        "stool",
        (0.0, 0.5),
        (0.4, 0.4, 0.45),
        yaw_deg=180.0,
    )
    access = {
        "constraint_id": "access_dressing_table",
        "relation": "clear_access",
        "subjects": {"category": "dressing_table", "count": 1},
        "targets": {"category": "room", "count": 1},
        "source": "explicit_prompt",
        "strength": "hard",
        "evidence_span": "usable space in front of the dressing table",
    }
    front = {
        "constraint_id": "stool_in_front_of_dressing_table",
        "relation": "in_front_of",
        "subjects": {"category": "stool", "count": 1},
        "targets": {"category": "dressing_table", "count": 1},
        "source": "explicit_prompt",
        "strength": "hard",
        "evidence_span": "stool centered in front of the dressing table",
    }
    faces = {
        "constraint_id": "stool_faces_dressing_table",
        "relation": "faces",
        "subjects": {"category": "stool", "count": 1},
        "targets": {"category": "dressing_table", "count": 1},
        "source": "explicit_prompt",
        "strength": "hard",
        "evidence_span": "stool facing the dressing table",
    }
    case_pack = {
        "stage": "furniture",
        "intent_contract": {"constraints": [access, front, faces]},
        "scene_geometry": {"objects": [dressing_table, stool]},
    }

    result = next(
        row
        for row in evaluate_intent_contract_extensions(case_pack)
        if row["relation_type"] == "clear_access"
    )

    assert result["label"] == "pass"
    assert result["diagnostics"]["authorized_occupant_ids"] == ["stool_0"]
    assert result["diagnostics"]["blocking_ids"] == []


def test_clear_access_allows_explicitly_paired_workstation_chairs() -> None:
    desks = [
        _record("desk_0", "desk", (-1.0, 0.0), (1.2, 0.7, 0.75), yaw_deg=0.0),
        _record("desk_1", "desk", (1.0, 0.0), (1.2, 0.7, 0.75), yaw_deg=0.0),
    ]
    chairs = [
        _record(
            "office_chair_0",
            "office_chair",
            (-1.0, 0.5),
            (0.5, 0.5, 1.0),
            yaw_deg=180.0,
        ),
        _record(
            "office_chair_1",
            "office_chair",
            (1.0, 0.5),
            (0.5, 0.5, 1.0),
            yaw_deg=180.0,
        ),
    ]
    access = {
        "constraint_id": "access_workstations",
        "relation": "clear_access",
        "subjects": {"category": "desk", "count": 2},
        "targets": {"category": "room", "count": 1},
        "source": "explicit_prompt",
        "strength": "hard",
        "evidence_span": "enough clearance to use every workstation",
    }
    paired = {
        "constraint_id": "pair_workstations",
        "relation": "paired_with",
        "subjects": {"category": "office_chair", "count": 2},
        "targets": {"category": "desk", "count": 2},
        "source": "explicit_prompt",
        "strength": "hard",
        "evidence_span": "each desk with exactly one office chair",
    }
    faces = {
        "constraint_id": "face_workstations",
        "relation": "faces",
        "subjects": {"category": "office_chair", "count": 2},
        "targets": {"category": "desk", "count": 2},
        "source": "explicit_prompt",
        "strength": "hard",
        "evidence_span": "every chair facing its paired desk",
    }
    case_pack = {
        "stage": "furniture",
        "checks": [
            {
                "metric": "functional_dependency",
                "relation_type": "seating_to_work_surface",
                "subject_id": "office_chair_0",
                "target_ids": ["desk_0"],
            },
            {
                "metric": "functional_dependency",
                "relation_type": "seating_to_work_surface",
                "subject_id": "office_chair_1",
                "target_ids": ["desk_1"],
            },
        ],
        "intent_contract": {"constraints": [access, paired, faces]},
        "scene_geometry": {"objects": [*desks, *chairs]},
    }

    results = [
        row
        for row in evaluate_intent_contract_extensions(case_pack)
        if row["relation_type"] == "clear_access"
    ]

    assert [row["label"] for row in results] == ["pass", "pass"]
    assert [row["diagnostics"]["authorized_occupant_ids"] for row in results] == [
        ["office_chair_0"],
        ["office_chair_1"],
    ]


def test_in_front_of_requires_clearance_for_both_object_footprints() -> None:
    dressing_table = _record(
        "dressing_table_0",
        "dressing_table",
        (0.0, 0.0),
        (1.2, 0.5, 0.75),
        yaw_deg=0.0,
    )
    relation = {
        "constraint_id": "stool_in_front_of_dressing_table",
        "relation": "in_front_of",
        "subjects": {"category": "stool", "count": 1},
        "targets": {"category": "dressing_table", "count": 1},
        "source": "explicit_prompt",
        "strength": "hard",
        "evidence_span": "stool centered in front of the dressing table",
    }

    def evaluate_stool_at(y: float) -> dict:
        stool = _record(
            "stool_0",
            "stool",
            (0.0, y),
            (0.4, 0.4, 0.45),
            yaw_deg=180.0,
        )
        return evaluate_intent_contract_extensions(
            {
                "stage": "furniture",
                "intent_contract": {"constraints": [relation]},
                "scene_geometry": {"objects": [dressing_table, stool]},
            }
        )[0]

    overlapping = evaluate_stool_at(0.20)
    clear = evaluate_stool_at(0.49)

    assert overlapping["label"] == "degraded"
    assert overlapping["diagnostics"]["minimum_forward_distance_m"] == pytest.approx(
        0.48
    )
    assert clear["label"] == "pass"


def _paired_workstation_axial_case(*, misplaced_chair: bool = False) -> dict:
    desks = []
    chairs = []
    for index, x in enumerate((-3.0, -1.0, 1.0, 3.0)):
        desks.append(
            _record(
                f"desk_{index}",
                "desk",
                (x, 0.0),
                (1.2, 0.7, 0.75),
                yaw_deg=0.0,
            )
        )
        chair_x = x + (0.8 if misplaced_chair and index == 2 else 0.0)
        chairs.append(
            _record(
                f"office_chair_{index}",
                "office_chair",
                (chair_x, 0.65),
                (0.5, 0.5, 1.0),
                yaw_deg=180.0,
            )
        )
    selector_chairs = {
        "category": "office_chair",
        "count": 4,
        "quantifier": "all",
    }
    selector_desks = {"category": "desk", "count": 4, "quantifier": "all"}
    paired = {
        "constraint_id": "paired_workstations",
        "relation": "paired_with",
        "subjects": selector_chairs,
        "targets": selector_desks,
        "source": "explicit_prompt",
        "strength": "hard",
        "evidence_span": "pair each desk with exactly one office chair",
    }
    in_front = {
        "constraint_id": "chairs_in_front_of_desks",
        "relation": "in_front_of",
        "subjects": selector_chairs,
        "targets": selector_desks,
        "source": "explicit_prompt",
        "strength": "hard",
        "evidence_span": "each chair is in front of its paired desk",
    }
    return {
        "stage": "furniture",
        "room_type": "office",
        "task_instruction": (
            "An office with four desks. Pair each desk with exactly one office chair."
        ),
        "intent_contract": {"constraints": [paired, in_front]},
        "scene_geometry": {
            "rooms": [{"bbox": {"min": [-5.0, -2.0], "max": [5.0, 2.0]}}],
            "objects": [*desks, *chairs],
        },
    }


def test_in_front_of_evaluates_equal_workstation_groups_one_to_one() -> None:
    results = evaluate_intent_contract_extensions(_paired_workstation_axial_case())
    axial = [row for row in results if row["relation_type"] == "front_axis_alignment"]

    assert len(axial) == 4
    assert [row["label"] for row in axial] == ["pass"] * 4
    assert {row["primary_object"] for row in axial} == {
        f"office_chair_{index}" for index in range(4)
    }
    assert {row["diagnostics"]["group_pairing"] for row in axial} == {
        "global_minimum_cost_one_to_one"
    }


def test_in_front_of_equal_workstation_group_keeps_strict_axis_failure() -> None:
    results = evaluate_intent_contract_extensions(
        _paired_workstation_axial_case(misplaced_chair=True)
    )
    axial = [row for row in results if row["relation_type"] == "front_axis_alignment"]
    failed = [row for row in axial if row["label"] == "fail"]

    assert len(axial) == 4
    assert [row["primary_object"] for row in failed] == ["office_chair_2"]
    assert failed[0]["diagnostics"]["lateral_offset_m"] == pytest.approx(0.8)


def test_clear_access_keeps_unrelated_object_as_a_blocker() -> None:
    dressing_table = _record(
        "dressing_table_0",
        "dressing_table",
        (0.0, 0.0),
        (1.2, 0.5, 0.75),
        yaw_deg=0.0,
    )
    stool = _record(
        "stool_0",
        "stool",
        (0.0, 0.5),
        (0.4, 0.4, 0.45),
        yaw_deg=180.0,
    )
    nightstand = _record(
        "nightstand_0",
        "nightstand",
        (0.55, 0.55),
        (0.4, 0.4, 0.6),
    )
    case_pack = {
        "stage": "furniture",
        "intent_contract": {
            "constraints": [
                {
                    "constraint_id": "access_dressing_table",
                    "relation": "clear_access",
                    "subjects": {"category": "dressing_table", "count": 1},
                    "targets": {"category": "room", "count": 1},
                    "source": "explicit_prompt",
                    "strength": "hard",
                    "evidence_span": "usable space in front of the dressing table",
                },
                {
                    "constraint_id": "stool_in_front_of_dressing_table",
                    "relation": "in_front_of",
                    "subjects": {"category": "stool", "count": 1},
                    "targets": {"category": "dressing_table", "count": 1},
                    "source": "explicit_prompt",
                    "strength": "hard",
                    "evidence_span": "stool centered in front of the dressing table",
                },
                {
                    "constraint_id": "stool_faces_dressing_table",
                    "relation": "faces",
                    "subjects": {"category": "stool", "count": 1},
                    "targets": {"category": "dressing_table", "count": 1},
                    "source": "explicit_prompt",
                    "strength": "hard",
                    "evidence_span": "stool facing the dressing table",
                },
            ]
        },
        "scene_geometry": {"objects": [dressing_table, stool, nightstand]},
    }

    result = next(
        row
        for row in evaluate_intent_contract_extensions(case_pack)
        if row["relation_type"] == "clear_access"
    )

    assert result["label"] == "fail"
    assert result["diagnostics"]["authorized_occupant_ids"] == ["stool_0"]
    assert result["diagnostics"]["blocking_ids"] == ["nightstand_0"]


def test_spatial_accessibility_allows_hard_bound_seating_companion() -> None:
    dressing_table = _record(
        "dressing_table_0",
        "dressing_table",
        (0.0, 0.0),
        (1.2, 0.5, 0.75),
        yaw_deg=0.0,
    )
    stool = _record(
        "stool_0",
        "stool",
        (0.0, 0.5),
        (0.4, 0.4, 0.45),
        yaw_deg=180.0,
    )
    case_pack = {
        "checks": [
            {
                "check_id": "spatial_accessibility__dressing_table_0",
                "metric": "spatial_accessibility",
                "subject_id": "dressing_table_0",
            }
        ],
        "intent_contract": {
            "constraints": [
                {
                    "relation": "in_front_of",
                    "subjects": {"category": "stool", "count": 1},
                    "targets": {"category": "dressing_table", "count": 1},
                    "source": "explicit_prompt",
                    "strength": "hard",
                },
                {
                    "relation": "faces",
                    "subjects": {"category": "stool", "count": 1},
                    "targets": {"category": "dressing_table", "count": 1},
                    "source": "explicit_prompt",
                    "strength": "hard",
                },
            ]
        },
    }

    attach_expected_access_companions(
        case_pack,
        {"dressing_table_0": dressing_table, "stool_0": stool},
    )

    assert case_pack["checks"][0]["expected_companion_ids"] == ["stool_0"]


def test_spatial_accessibility_requires_both_hard_seating_relations() -> None:
    dressing_table = _record(
        "dressing_table_0",
        "dressing_table",
        (0.0, 0.0),
        (1.2, 0.5, 0.75),
        yaw_deg=0.0,
    )
    stool = _record(
        "stool_0",
        "stool",
        (0.0, 0.5),
        (0.4, 0.4, 0.45),
        yaw_deg=180.0,
    )
    case_pack = {
        "checks": [
            {
                "check_id": "spatial_accessibility__dressing_table_0",
                "metric": "spatial_accessibility",
                "subject_id": "dressing_table_0",
            }
        ],
        "intent_contract": {
            "constraints": [
                {
                    "relation": "faces",
                    "subjects": {"category": "stool", "count": 1},
                    "targets": {"category": "dressing_table", "count": 1},
                    "source": "explicit_prompt",
                    "strength": "hard",
                }
            ]
        },
    }

    attach_expected_access_companions(
        case_pack,
        {"dressing_table_0": dressing_table, "stool_0": stool},
    )

    assert "expected_companion_ids" not in case_pack["checks"][0]


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


def _compiler_with_responses(
    responses: list[SimpleNamespace | BaseException],
) -> IntentCompiler:
    compiler = object.__new__(IntentCompiler)
    compiler._model = "test-model"
    compiler._max_tokens = 256
    compiler._temperature = 0.0
    compiler.last_trace = {}
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

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


def test_intent_schema_requires_target_for_endpoint_relations() -> None:
    relation_schema = intent_contract_json_schema()["$defs"]["IntentRelation"]
    conditions = relation_schema["allOf"]

    unary_condition = next(
        condition
        for condition in conditions
        if "flanking" in condition["if"]["properties"]["relation"]["enum"]
    )
    binary_condition = next(
        condition
        for condition in conditions
        if "between" in condition["if"]["properties"]["relation"]["enum"]
    )

    assert "targets" in unary_condition["then"]["required"]
    assert unary_condition["then"]["properties"]["targets"] == {
        "$ref": "#/$defs/IntentSelector"
    }
    assert "targets" in binary_condition["then"]["required"]
    assert binary_condition["then"]["properties"]["targets"]["allOf"][1] == {
        "required": ["secondary_category"]
    }
    zero_arity_condition = next(
        condition
        for condition in conditions
        if "required_count" in condition["if"]["properties"]["relation"]["enum"]
    )
    assert "targets" in relation_schema["required"]
    assert zero_arity_condition["then"]["properties"]["targets"] == {"type": "null"}


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


def test_intent_compiler_does_not_upgrade_inventory_to_flanking() -> None:
    invented = (
        '{"constraints": [{"relation": "flanking", '
        '"subjects": {"category": "nightstand", "count": 2}, '
        '"targets": {"category": "bed", "count": 1}, '
        '"source": "explicit_prompt", '
        '"evidence_span": "two nightstands"}]}'
    )
    compiler = _compiler_with_responses([_response(invented), _response(invented)])

    result = compiler.compile(
        "A bedroom with a bed, two nightstands, and a wardrobe in the corner."
    )

    assert compiler.last_trace["status"] == "fallback"
    assert [item["status"] for item in compiler.last_trace["attempts"]] == [
        "error",
        "error",
        "deterministic_fallback",
    ]
    assert all(row["relation"] != "flanking" for row in result["constraints"])
    assert any(
        row["relation"] == "required_count"
        and row["subjects"]["category"] == "nightstand"
        and row["subjects"]["count"] == 2
        for row in result["constraints"]
    )


def test_intent_compiler_restores_missing_target_from_prompt_parser() -> None:
    compiler = _compiler_with_responses(
        [
            _response(
                '{"constraints": [{"relation": "flanking", '
                '"subjects": {"category": "nightstand", "count": 2}, '
                '"source": "explicit_prompt", '
                '"evidence_span": "two nightstands on either side of the bed"}]}'
            )
        ]
    )

    result = compiler.compile(
        "A bedroom with two nightstands on either side of the bed."
    )

    flanking = next(
        constraint
        for constraint in result["constraints"]
        if constraint["relation"] == "flanking"
    )
    assert flanking["targets"]["category"] == "bed"
    assert compiler.last_trace["restored_targets"] == [
        {
            "relation": "flanking",
            "subject_category": "nightstand",
            "target_category": "bed",
        }
    ]


def test_intent_compiler_rewrites_object_centering_mistaken_for_wall_centering() -> (
    None
):
    prompt = "Mount one mirror on the wall, centered directly above the dressing table."
    compiler = _compiler_with_responses(
        [
            _response(
                '{"constraints": [{"relation": "centered_on_wall", '
                '"subjects": {"category": "mirror", "count": 1}, '
                '"targets": {"category": "wall", "count": 1}, '
                '"source": "explicit_prompt", '
                '"evidence_span": "mirror centered directly above the dressing table"}]}'
            )
        ]
    )

    result = compiler.compile(prompt)

    alignment = next(
        row for row in result["constraints"] if row["relation"] == "centered_above"
    )
    assert alignment["subjects"]["category"] == "mirror"
    assert alignment["targets"]["category"] == "dressing_table"
    assert not any(
        row["relation"] == "centered_on_wall"
        and row["subjects"]["category"] == "mirror"
        for row in result["constraints"]
    )
    assert compiler.last_trace["normalized_centered_above"] == [
        {"subject_category": "mirror", "target_category": "dressing_table"}
    ]


def test_intent_compiler_rewrites_vertical_centering_mistaken_for_front_axis() -> None:
    prompt = (
        "Place one stool centered in front of and facing the dressing table. "
        "Mount one separate mirror on the wall, centered directly above the "
        "dressing table."
    )
    compiler = _compiler_with_responses(
        [
            _response(
                '{"constraints": ['
                '{"relation": "centered_above", '
                '"subjects": {"category": "stool", "count": 1}, '
                '"targets": {"category": "dressing_table", "count": 1}, '
                '"source": "explicit_prompt", '
                '"evidence_span": "one stool centered in front of and facing '
                'the dressing table"}, '
                '{"relation": "faces", '
                '"subjects": {"category": "stool", "count": 1}, '
                '"targets": {"category": "dressing_table", "count": 1}, '
                '"source": "explicit_prompt", '
                '"evidence_span": "stool facing the dressing table"}, '
                '{"relation": "centered_above", '
                '"subjects": {"category": "mirror", "count": 1}, '
                '"targets": {"category": "dressing_table", "count": 1}, '
                '"source": "explicit_prompt", '
                '"evidence_span": "mirror centered directly above the '
                'dressing table"}]}'
            )
        ]
    )

    result = compiler.compile(prompt)

    assert any(
        row["relation"] == "in_front_of"
        and row["subjects"]["category"] == "stool"
        and row["targets"]["category"] == "dressing_table"
        for row in result["constraints"]
    )
    assert any(
        row["relation"] == "centered_above"
        and row["subjects"]["category"] == "mirror"
        and row["targets"]["category"] == "dressing_table"
        for row in result["constraints"]
    )
    assert not any(
        row["relation"] == "centered_above" and row["subjects"]["category"] == "stool"
        for row in result["constraints"]
    )


def test_intent_compiler_restores_unique_edge_fields_omitted_by_llama() -> None:
    prompt = (
        "Arrange five dining chairs around one rectangular dining table: two "
        "evenly spaced along each long side and one centered on one short side, "
        "all facing the table."
    )
    compiler = _compiler_with_responses(
        [
            _response(
                '{"constraints": [{"relation": "edge_distribution", '
                '"subjects": {"category": "dining_chair", "count": 5}, '
                '"targets": {"category": "dining_table", "count": 1}, '
                '"evidence_span": "Arrange five dining chairs"}]}'
            )
        ]
    )

    result = compiler.compile(prompt)

    edge = next(
        row for row in result["constraints"] if row["relation"] == "edge_distribution"
    )
    assert edge["source"] == "explicit_prompt"
    assert edge["edge_frame"] == "target_local_rectangle"
    assert edge["orientation"] == "toward_target"
    assert [group["counts_per_edge"] for group in edge["groups"]] == [[2, 2], [1, 0]]
    assert compiler.last_trace["status"] == "ok"
    assert compiler.last_trace["restored_fields"] == [
        {
            "relation": "edge_distribution",
            "subject_category": "dining_chair",
            "fields": ["source", "edge_frame", "groups", "orientation"],
        }
    ]


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


def test_intent_compiler_restores_unique_media_target_without_fallback() -> None:
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

    assert compiler.last_trace["status"] == "ok"
    assert compiler.last_trace["restored_targets"]
    facing = next(row for row in result["constraints"] if row["relation"] == "faces")
    assert facing["targets"]["category"] == "tv_stand"


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
    assert result["constraints"][0]["relation"] == "against_wall"
    assert sum(row["relation"] == "against_wall" for row in result["constraints"]) == 1
    retry_message = compiler._test_calls[1]["messages"][1]["content"]
    assert "wall-relative directional relation" in retry_message
    assert "Do not convert 'X against the wall" in retry_message


def test_intent_compiler_restores_explicit_floor_support_omitted_by_llm() -> None:
    prompt = (
        "A living room with a two-seater sofa and two large potted plants on "
        "the floor near the sofa."
    )
    compiler = _compiler_with_responses(
        [
            _response(
                '{"constraints": [{"relation": "near", '
                '"subjects": {"category": "large_plant", "count": 2}, '
                '"targets": {"category": "two_seater_sofa", "count": 1}, '
                '"source": "explicit_prompt", '
                '"evidence_span": "plants near the sofa"}]}'
            )
        ]
    )

    result = compiler.compile(prompt)

    assert any(
        row["relation"] == "on_top_of"
        and row["subjects"]["category"] == "plant"
        and row["targets"]["category"] == "floor"
        for row in result["constraints"]
    )
    assert compiler.last_trace["enriched_constraints"]


def test_intent_compiler_restores_classroom_operation_zone_ontology() -> None:
    prompt = (
        "A classroom with a teacher's desk, six student desks with chairs, "
        "and a chalkboard."
    )
    compiler = _compiler_with_responses([_response('{"constraints": []}')])

    result = compiler.compile(prompt)

    assert {row["relation"] for row in result["constraints"]} >= {
        "operation_zone_at_wall",
        "instructional_surface_alignment",
    }


def test_intent_compiler_enriches_explicit_group_relations_omitted_by_llm() -> None:
    prompt = (
        "A practical office with four separate desks. Pair each desk with exactly "
        "one office chair. Place exactly one computer monitor on top of each desk, "
        "for four monitors in total. Place four large floor plants in four "
        "distinct room corners, exactly one plant per corner."
    )
    compiler = _compiler_with_responses([_response('{"constraints": []}')])

    result = compiler.compile(prompt)

    relations = {row["relation"] for row in result["constraints"]}
    assert {"paired_with", "one_per_support", "corner_distribution"} <= relations
    enriched_relations = {
        row["relation"] for row in compiler.last_trace["enriched_constraints"]
    }
    assert {"paired_with", "one_per_support", "corner_distribution"} <= (
        enriched_relations
    )


def test_intent_compiler_enriches_explicit_wall_anchor_and_access_omitted_by_llm() -> (
    None
):
    prompt = (
        "Place one storage cabinet against a wall without blocking circulation. "
        "Place one water dispenser against a wall and keep its front accessible."
    )
    compiler = _compiler_with_responses([_response('{"constraints": []}')])

    result = compiler.compile(prompt)

    wall_anchors = {
        row["subjects"]["category"]
        for row in result["constraints"]
        if row["relation"] == "against_wall"
    }
    access_pairs = {
        (row["subjects"]["category"], row["targets"]["category"])
        for row in result["constraints"]
        if row["relation"] == "clear_access"
    }
    assert {"storage_cabinet", "water_dispenser"} <= wall_anchors
    assert ("entrance", "storage_cabinet") in access_pairs
    assert ("water_dispenser", "room") in access_pairs
    assert {"against_wall", "clear_access"} <= {
        row["relation"] for row in compiler.last_trace["enriched_constraints"]
    }


def test_floor_supported_near_relation_rejects_containment() -> None:
    plant = _record("large_plant_0", "large_plant", (0.0, 0.0), (0.4, 0.4, 1.6))
    sofa = _record("two_seater_sofa_0", "two_seater_sofa", (0.0, 0.0), (2.0, 0.9, 1.0))
    floor = _record("floor_0", "floor", (0.0, 0.0), (6.0, 6.0, 0.05))
    case_pack = {
        "scene_geometry": {"objects": [plant, sofa, floor]},
        "intent_contract": validate_intent_contract(
            {
                "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
                "prompt": "plant on the floor near the sofa",
                "constraints": [
                    {
                        "relation": "on_top_of",
                        "subjects": {"category": "large_plant", "count": 1},
                        "targets": {"category": "floor", "count": 1},
                        "source": "explicit_prompt",
                        "evidence_span": "plant on the floor",
                    },
                    {
                        "relation": "near",
                        "subjects": {"category": "large_plant", "count": 1},
                        "targets": {"category": "two_seater_sofa", "count": 1},
                        "source": "explicit_prompt",
                        "evidence_span": "near the sofa",
                    },
                ],
            }
        ),
    }

    assert augment_contract_checks(case_pack)
    near_check = next(
        check
        for check in case_pack["checks"]
        if check["relation_type"] == "generic_near_relation"
    )
    assert near_check["evidence"]["dependency"]["requires_external_adjacency"]
    result = evaluate_functional_dependency(load_geometry(case_pack), near_check)

    assert result["label"] == "fail"
    assert "not external adjacency" in result["reason"]


def test_floor_supported_near_existing_check_gets_external_adjacency_guard() -> None:
    plant = _record("large_plant_0", "large_plant", (0.0, 0.0), (0.4, 0.4, 1.6))
    sofa = _record("two_seater_sofa_0", "two_seater_sofa", (0.0, 0.0), (2.0, 0.9, 1.0))
    floor = _record("floor_0", "floor", (0.0, 0.0), (6.0, 6.0, 0.05))
    near_constraint = {
        "relation": "near",
        "subjects": {"category": "large_plant", "count": 1},
        "targets": {"category": "two_seater_sofa", "count": 1},
        "source": "model_inferred",
        "evidence_span": "near the sofa",
        "inference_reason": "The model preserved a nearby plant relationship.",
    }
    case_pack = {
        "scene_geometry": {"objects": [plant, sofa, floor]},
        "intent_contract": validate_intent_contract(
            {
                "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
                "prompt": "plant on the floor near the sofa",
                "constraints": [
                    {
                        "relation": "on_top_of",
                        "subjects": {"category": "large_plant", "count": 1},
                        "targets": {"category": "floor", "count": 1},
                        "source": "explicit_prompt",
                        "evidence_span": "plant on the floor",
                    },
                    near_constraint,
                ],
            }
        ),
        "checks": [
            {
                "check_id": "existing_near_check",
                "metric": "functional_dependency",
                "subject_id": "large_plant_0",
                "target_ids": ["two_seater_sofa_0"],
                "relation_type": "generic_near_relation",
                "expected_use": "near",
                "check_source": "intent_contract",
                "scoring_tier": "core",
                "evidence": {"intent_constraint": near_constraint},
            }
        ],
    }

    assert augment_contract_checks(case_pack)
    near_check = case_pack["checks"][0]
    assert near_check["evidence"]["dependency"]["requires_external_adjacency"]


def test_floor_supported_near_wall_does_not_get_furniture_containment_guard() -> None:
    plant = _record("large_plant_0", "large_plant", (0.0, 0.0), (0.4, 0.4, 1.6))
    wall = _record("north_wall", "wall", (0.0, 2.5), (6.0, 0.1, 2.8))
    wall["object_type"] = "wall"
    floor = _record("floor_0", "floor", (0.0, 0.0), (6.0, 6.0, 0.05))
    floor["object_type"] = "floor"
    case_pack = {
        "scene_geometry": {"objects": [plant, wall, floor]},
        "intent_contract": validate_intent_contract(
            {
                "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
                "prompt": "plant on the floor near the north wall",
                "constraints": [
                    {
                        "relation": "on_top_of",
                        "subjects": {"category": "large_plant", "count": 1},
                        "targets": {"category": "floor", "count": 1},
                        "source": "explicit_prompt",
                        "evidence_span": "plant on the floor",
                    },
                    {
                        "relation": "near",
                        "subjects": {"category": "large_plant", "count": 1},
                        "targets": {"category": "wall", "count": 1},
                        "source": "explicit_prompt",
                        "evidence_span": "near the north wall",
                    },
                ],
            }
        ),
    }

    assert augment_contract_checks(case_pack)
    near_check = next(
        check
        for check in case_pack["checks"]
        if check["relation_type"] == "generic_near_relation"
    )
    assert not near_check["evidence"]["dependency"].get("requires_external_adjacency")


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


def test_intent_compiler_falls_back_after_unparseable_responses() -> None:
    compiler = _compiler_with_responses([_response("no json"), _response("still bad")])

    result = compiler.compile("A room with a desk.")

    assert result["warnings"]
    assert compiler.last_trace["status"] == "fallback"
    assert compiler.last_trace["retry_count"] == 1
    assert [item["status"] for item in compiler.last_trace["attempts"]] == [
        "error",
        "error",
        "deterministic_fallback",
    ]


def test_intent_compiler_falls_back_after_transport_errors() -> None:
    compiler = _compiler_with_responses(
        [ConnectionError("temporary outage"), ConnectionError("temporary outage")]
    )

    result = compiler.compile(
        "A living room with a sofa facing a TV stand and television on the opposite wall."
    )

    assert compiler.last_trace["status"] == "fallback"
    assert compiler.last_trace["failure_reason"] == "ConnectionError: temporary outage"
    assert [item["status"] for item in compiler.last_trace["attempts"]] == [
        "error",
        "error",
        "deterministic_fallback",
    ]
    assert any(
        row["relation"] == "on_top_of"
        and row["subjects"]["category"] == "television"
        and row["targets"]["category"] == "tv_stand"
        for row in result["constraints"]
    )


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


def test_new_scene_prompts_compile_complete_deterministic_contracts() -> None:
    bedroom = build_intent_contract(
        "A bedroom with one bed centered against the back wall. Place two "
        "nightstands, one on each side of the bed. Place one wardrobe against a "
        "side wall. Position one dressing table against a free wall, with one "
        "stool in front of and facing the dressing table. Mount one mirror on "
        "the wall, centered directly above the dressing table."
    )
    office = build_intent_contract(
        "An office with four desks. Pair each desk with exactly one office chair. "
        "Place exactly one computer monitor on top of each desk, for four monitors "
        "in total. Place one water dispenser against a wall and one wastebasket "
        "on the floor in one corner of the room."
    )
    living = build_intent_contract(
        "A long living room with one sofa facing one TV stand and one television "
        "on top of the TV stand. Place one dining table with five complete table "
        "settings, each including a plate, cutlery, and a drinking glass. Arrange "
        "five dining chairs: two evenly spaced along each long side and one on one "
        "short side, all facing the table. Place four floor plants in four distinct "
        "room corners, exactly one plant per corner. Place one storage cabinet "
        "against a wall without blocking circulation."
    )

    for contract in (bedroom, office, living):
        validate_intent_contract(contract)

    assert any(row["relation"] == "flanking" for row in bedroom["constraints"])
    assert any(
        row["relation"] == "on_wall" and row["subjects"]["category"] == "mirror"
        for row in bedroom["constraints"]
    )
    mirror_alignment = next(
        row
        for row in bedroom["constraints"]
        if row["relation"] == "centered_above"
        and row["subjects"]["category"] == "mirror"
    )
    assert mirror_alignment["targets"]["category"] == "dressing_table"
    assert any(
        row["relation"] == "paired_with"
        and row["subjects"]["count"] == row["targets"]["count"] == 4
        for row in office["constraints"]
    )
    assert any(
        row["relation"] == "one_per_support"
        and row["subjects"]["category"] == "monitor"
        and row["subjects"]["count"] == row["targets"]["count"] == 4
        for row in office["constraints"]
    )
    edge = next(
        row for row in living["constraints"] if row["relation"] == "edge_distribution"
    )
    assert edge["subjects"] == {
        "category": "dining_chair",
        "count": 5,
        "quantifier": "all",
    }
    assert edge["targets"]["category"] == "dining_table"
    assert [group["counts_per_edge"] for group in edge["groups"]] == [[2, 2], [1, 0]]
    assert edge["orientation"] == "toward_target"
    assert any(
        row["relation"] == "corner_distribution" for row in living["constraints"]
    )
    assert any(
        row["relation"] == "against_wall"
        and row["subjects"]["category"] == "storage_cabinet"
        for row in living["constraints"]
    )
    assert any(
        row["relation"] == "clear_access"
        and row["subjects"]["category"] == "entrance"
        and row["targets"]["category"] == "storage_cabinet"
        for row in living["constraints"]
    )
    component_counts = {
        row["subjects"]["category"]: row["subjects"]["count"]
        for row in living["constraints"]
        if row["relation"] == "required_count"
    }
    assert component_counts | {"plate": 5, "cutlery": 5, "glass": 5} == component_counts
    assert "floor" not in component_counts
    assert component_counts.get("table") is None


def test_schema_rejects_exclusive_relations_but_respects_roles() -> None:
    conflicting = [
        {
            "relation": "centered_in_room",
            "subjects": {"category": "bed", "count": 1},
            "targets": {"category": "room", "count": 1},
            "source": "explicit_prompt",
            "evidence_span": "bed centered in the room",
        },
        {
            "relation": "against_wall",
            "subjects": {"category": "bed", "count": 1},
            "targets": {"category": "wall", "count": 1},
            "source": "explicit_prompt",
            "evidence_span": "bed against the wall",
        },
    ]
    with pytest.raises(ValidationError, match="conflicting hard relations"):
        validate_intent_contract(
            {
                "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
                "constraints": conflicting,
            }
        )

    role_qualified = [dict(row) for row in conflicting]
    role_qualified[0] = {
        **role_qualified[0],
        "subjects": {"category": "chair", "role": "guest", "count": 1},
    }
    role_qualified[1] = {
        **role_qualified[1],
        "subjects": {"category": "chair", "role": "teacher", "count": 1},
    }
    validate_intent_contract(
        {
            "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
            "constraints": role_qualified,
        }
    )


def test_one_per_support_requires_distinct_support_owners() -> None:
    desks = [
        _record("desk_0", "desk", (-1.0, 0.0), (1.2, 0.7, 0.75)),
        _record("desk_1", "desk", (1.0, 0.0), (1.2, 0.7, 0.75)),
    ]
    for index, desk in enumerate(desks):
        desk["support_regions"] = [{"region_id": f"desk_{index}_top"}]
    monitors = [
        _record("monitor_0", "monitor", (-1.0, 0.0), (0.4, 0.2, 0.35)),
        _record("monitor_1", "monitor", (1.0, 0.0), (0.4, 0.2, 0.35)),
    ]
    monitors[0]["placement_info"] = {"parent_surface_id": "desk_0_top"}
    monitors[1]["placement_info"] = {"parent_surface_id": "desk_1_top"}
    relation = {
        "relation": "one_per_support",
        "subjects": {"category": "monitor", "count": 2},
        "targets": {"category": "desk", "count": 2},
        "source": "explicit_prompt",
        "evidence_span": "one monitor on each desk",
    }
    case_pack = {
        "stage": "final",
        "scene_geometry": {"objects": [*desks, *monitors]},
        "intent_contract": validate_intent_contract(_contract(relation)),
    }

    result = next(
        row
        for row in evaluate_intent_contract_extensions(case_pack)
        if row["relation_type"] == "one_per_support"
    )
    assert result["label"] == "pass"

    monitors[1]["placement_info"] = {"parent_surface_id": "desk_0_top"}
    result = next(
        row
        for row in evaluate_intent_contract_extensions(case_pack)
        if row["relation_type"] == "one_per_support"
    )
    assert result["label"] == "fail"
    assert result["diagnostics"]["missing_target_ids"] == ["desk_1"]
    assert result["diagnostics"]["duplicate_target_ids"] == ["desk_0"]


def test_corner_distribution_assigns_distinct_corners() -> None:
    floor = _record("floor_0", "floor", (0.0, 0.0), (10.0, 6.0, 0.05))
    plants = [
        _record("plant_0", "plant", (-4.7, -2.7), (0.4, 0.4, 1.2)),
        _record("plant_1", "plant", (-4.7, 2.7), (0.4, 0.4, 1.2)),
        _record("plant_2", "plant", (4.7, -2.7), (0.4, 0.4, 1.2)),
        _record("plant_3", "plant", (4.7, 2.7), (0.4, 0.4, 1.2)),
    ]
    relation = {
        "relation": "corner_distribution",
        "subjects": {"category": "plant", "count": 4},
        "targets": {"category": "room", "count": 1},
        "source": "explicit_prompt",
        "evidence_span": "four plants in four distinct room corners",
    }
    case_pack = {
        "stage": "final",
        "scene_geometry": {
            "objects": [floor, *plants],
            "rooms": [
                {
                    "bbox": {
                        "min": [-5.0, -3.0, 0.0],
                        "max": [5.0, 3.0, 3.0],
                    }
                }
            ],
        },
        "intent_contract": validate_intent_contract(_contract(relation)),
    }

    results = [
        row
        for row in evaluate_intent_contract_extensions(case_pack)
        if row["relation_type"] == "corner_distribution"
    ]
    assert len(results) == 4
    assert all(row["label"] == "pass" for row in results)
    assert (
        len({row["diagnostics"]["assignment"]["corner_index"] for row in results}) == 4
    )

    for plant in plants:
        plant["bbox_world"]["center"][:2] = [-4.7, -2.7]
        plant["bbox_world"]["min"][:2] = [-4.9, -2.9]
        plant["bbox_world"]["max"][:2] = [-4.5, -2.5]
    results = [
        row
        for row in evaluate_intent_contract_extensions(case_pack)
        if row["relation_type"] == "corner_distribution"
    ]
    assert any(row["label"] == "fail" for row in results)
