"""Focused coverage for the independent intent contract and edge geometry."""

from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace

import pytest

from pydantic import ValidationError

import scenesmith.scenebenchmark_critic.intent_compiler as intent_compiler_module
from scenesmith.scenebenchmark_critic.intent_schema import (
    INTENT_COMPILER_SPEC_VERSION,
    INTENT_COMPILER_SEMANTIC_IR_VERSION,
    INTENT_CONTRACT_SCHEMA_VERSION,
    canonical_selector_category,
    intent_compiler_wire_json_schema,
    intent_contract_json_schema,
    validate_intent_contract,
)
from scenesmith.utils.llm_json import json_response_format
from scenesmith.scenebenchmark_critic.intent_compiler import (
    IncompleteIntentContractError,
    IntentCompilationError,
    IntentCompiler,
    _attach_grounding_provenance,
    _grounding_catalog,
    _system_prompt,
    _validate_contract_completeness,
)
from scenesmith.scenebenchmark_critic.intent_contract import (
    augment_contract_checks,
    apply_contract_execution_states,
    bound_ids,
    build_intent_contract,
    intent_contract_required_counts,
    _nearest_wall_ids,
    selected_ids,
    selector_match_count,
    selector_for_phrase,
)
from scenesmith.scenebenchmark_critic.config import CriticConfig
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
    _binding_state_result,
    _evaluate_required_count,
    evaluate_intent_contract_extensions,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.edge_distribution import (
    evaluate_edge_distribution,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.orientation_contracts import (
    CONTRACT_ATTR,
    CONTRACT_CHECK_SOURCE,
    stabilize_orientation_contracts,
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


@pytest.mark.parametrize(
    ("wording", "expected_relation"),
    [
        ("next to", "next_to"),
        ("beside", "next_to"),
        ("adjacent to", "next_to"),
        ("near", "near"),
    ],
)
def test_deterministic_parser_preserves_window_adjacency_strength(
    wording: str, expected_relation: str
) -> None:
    contract = build_intent_contract(
        f"A bedroom with a bed positioned {wording} a window."
    )

    window_rows = [
        row
        for row in contract["constraints"]
        if row["relation"] in {"near", "next_to"}
        and row["subjects"]["category"] == "bed"
        and (row.get("targets") or {}).get("category") == "window"
    ]

    assert [row["relation"] for row in window_rows] == [expected_relation]
    assert window_rows[0]["stage"] == "furniture"


def test_structural_window_is_not_furniture_inventory() -> None:
    contract = build_intent_contract(
        "A bedroom with a bed next to a window.",
        task_spec=SceneTaskSpec(
            room_type="bedroom",
            style="standard",
            required_large_objects=["bed", "window"],
            required_wall_objects=["door"],
        ),
    )
    scene = SimpleNamespace(scenebenchmark_intent_contract=contract)

    assert not any(
        row["relation"] == "required_count"
        and row["subjects"]["category"] in {"door", "opening", "window"}
        for row in contract["constraints"]
    )
    assert "window" not in intent_contract_required_counts(scene)
    assert INTENT_COMPILER_SPEC_VERSION == "scenesmith.intent_compiler.v16"


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


def test_edge_distribution_can_describe_cohort_above_minimum_inventory() -> None:
    edge = _edge_relation(
        subject_count=5,
        groups=[
            {"edge_class": "long", "counts_per_edge": [2, 2]},
            {"edge_class": "short", "counts_per_edge": [1, 0]},
        ],
    )
    required = {
        "relation": "required_count",
        "subjects": {
            "category": "office_chair",
            "count": 4,
            "quantifier": "at_least",
        },
        "targets": None,
        "source": "task_compiler_inventory",
        "inference_reason": "SceneTaskSpec required_large_objects",
    }

    validated = validate_intent_contract(
        {
            **_contract(edge),
            "constraints": [required, edge],
        }
    )

    assert [
        row["subjects"]["count"]
        for row in validated["constraints"]
        if row["relation"] == "edge_distribution"
    ] == [5]


def test_edge_distribution_rejects_conflict_with_exact_inventory() -> None:
    edge = _edge_relation(
        subject_count=5,
        groups=[
            {"edge_class": "long", "counts_per_edge": [2, 2]},
            {"edge_class": "short", "counts_per_edge": [1, 0]},
        ],
    )
    required = {
        "relation": "required_count",
        "subjects": {
            "category": "office_chair",
            "count": 4,
            "quantifier": "exactly",
        },
        "targets": None,
        "source": "explicit_prompt",
        "evidence_span": "exactly four chairs",
    }

    with pytest.raises(ValidationError, match="conflicts with exact"):
        validate_intent_contract(
            {
                **_contract(edge),
                "constraints": [required, edge],
            }
        )


def test_bare_on_top_without_of_does_not_create_support_relation() -> None:
    contract = build_intent_contract(
        "A study with a monitor on top and a sofa chair in front of it."
    )

    assert not any(
        row["relation"] == "on_top_of" and row["subjects"]["category"] == "monitor"
        for row in contract["constraints"]
    )
    ordinary = build_intent_contract("A study with a bowl on table.")
    assert any(
        row["relation"] == "on_top_of"
        and row["subjects"]["category"] == "bowl"
        and row["targets"]["category"] == "table"
        for row in ordinary["constraints"]
    )
    explicit = build_intent_contract(
        "A study with a monitor on top of a desk and a sofa chair in front of it."
    )
    assert [
        (row["subjects"]["category"], row["targets"]["category"])
        for row in explicit["constraints"]
        if row["relation"] == "on_top_of"
    ] == [("monitor", "desk")]


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
        ("circular ceramic table", "table"),
        ("large rectangular rug", "rug"),
        ("rectangular coffee table", "coffee_table"),
        ("chest_of_drawer", "dresser"),
        ("chest of drawers", "dresser"),
        ("large abstract painting", "painting"),
        ("wall_cabinet", "wall_cabinet"),
        ("sofa_chair", "sofa_chair"),
        ("sofa chair", "sofa_chair"),
        ("glass bowls", "glass_bowl"),
        ("vase_flowers", "vase_flower"),
        ("unlisted modular console", "unlisted_modular_console"),
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


def test_role_qualified_dining_chairs_bind_with_real_edge_contract() -> None:
    table = _record("dining_table_0", "dining_table", (0.0, 0.0), (3.0, 1.0, 0.75))
    chairs = [
        _record("dining_chair_0", "chair", (-1.0, -0.8), (0.5, 0.5, 0.9)),
        _record("dining_chair_1", "chair", (1.0, -0.8), (0.5, 0.5, 0.9)),
        _record("dining_chair_2", "chair", (-1.0, 0.8), (0.5, 0.5, 0.9)),
        _record("dining_chair_3", "chair", (1.0, 0.8), (0.5, 0.5, 0.9)),
        _record("dining_chair_4", "chair", (1.8, 0.0), (0.5, 0.5, 0.9)),
    ]
    for chair in chairs:
        chair["metadata"] = {"semantic_name": "dining_chair"}

    subject_selector = {
        "category": "chair",
        "role": "dining_chair",
        "count": 5,
        "quantifier": "exactly",
    }
    assert selected_ids(subject_selector, chairs) == [
        "dining_chair_0",
        "dining_chair_1",
        "dining_chair_2",
        "dining_chair_3",
        "dining_chair_4",
    ]
    assert selector_match_count(subject_selector, chairs) == 5

    case_pack = {
        "intent_contract": {
            "constraints": [
                {
                    "relation": "edge_distribution",
                    "subjects": subject_selector,
                    "targets": {"category": "dining_table", "count": 1},
                    "edge_frame": "target_local_rectangle",
                    "groups": [
                        {"edge_class": "long", "counts_per_edge": [4, 0]},
                        {"edge_class": "short", "counts_per_edge": [1, 0]},
                    ],
                    "orientation": "unconstrained",
                    "source": "explicit_prompt",
                    "strength": "hard",
                }
            ]
        },
        "scene_geometry": {"objects": [table, *chairs]},
    }
    result = evaluate_edge_distribution(case_pack)[0]
    # This is the contract emitted for the replayed scene. Binding must be
    # complete even when the generated 2+2 layout does not satisfy its
    # one-long-edge topology.
    assert result["label"] == "fail"
    assert result["selected_related_objects"] == [
        "dining_chair_0",
        "dining_chair_1",
        "dining_chair_2",
        "dining_chair_3",
        "dining_chair_4",
    ]


def test_role_matching_normalizes_separators_and_rejects_wrong_role() -> None:
    chair = _record("seat_0", "chair", (0.0, 0.0), (0.5, 0.5, 0.9))
    chair["metadata"] = {"semantic_name": "dining-chair"}

    assert selected_ids(
        {"category": "chair", "role": "dining_chair", "count": 1}, [chair]
    ) == ["seat_0"]
    assert (
        selected_ids({"category": "chair", "role": "office_chair", "count": 1}, [chair])
        == []
    )


def test_display_zone_phrase_does_not_create_monitor_constraints() -> None:
    contract = build_intent_contract(
        "A living room with a sofa and a display zone against the wall."
    )

    assert not any(
        row.get("subjects", {}).get("category") == "monitor"
        for row in contract["constraints"]
    )
    assert not any(
        row.get("subjects", {}).get("category") == "monitor"
        for row in contract.get("coverage_requirements", [])
    )


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
        "furniture",
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


def test_required_count_rejects_excess_for_exact_quantifier() -> None:
    constraint = {
        "constraint_id": "intent_exact_coasters",
        "relation": "required_count",
        "subjects": {"category": "coaster", "count": 4, "quantifier": "exactly"},
    }
    objects = [
        _record(f"coaster_{index}", "coaster", (0.0, 0.0), (0.1, 0.1, 0.01))
        for index in range(8)
    ]

    result = _evaluate_required_count(constraint, objects, "core")

    assert result["label"] == "fail"
    assert result["diagnostics"]["observed_count"] == 8
    assert "exactly 4" in result["reason"]


def test_composite_flower_arrangement_binds_flower_and_its_vase() -> None:
    arrangement = _record(
        "flower_arrangement_0",
        "flower_arrangement",
        (0.0, 0.0),
        (0.3, 0.3, 0.6),
    )
    arrangement["metadata"] = {"semantic_name": "flower_arrangement"}
    arrangement["description"] = (
        "Tall elegant glass vase with floral arrangement of red roses and greenery"
    )
    objects = [arrangement]
    flower_selector = {"category": "flower", "count": 1}
    vase_selector = {"category": "vase", "count": 1}
    flower_count = {
        "constraint_id": "intent_flower_count",
        "relation": "required_count",
        "subjects": {**flower_selector, "quantifier": "exactly"},
    }
    arrangement_on_vase = {
        "constraint_id": "intent_flower_on_vase",
        "relation": "on_top_of",
        "stage": "manipuland",
        "subjects": flower_selector,
        "targets": vase_selector,
    }

    assert selected_ids(flower_selector, objects) == ["flower_arrangement_0"]
    assert selected_ids(vase_selector, objects) == ["flower_arrangement_0"]
    assert _evaluate_required_count(flower_count, objects, "core")["label"] == "pass"
    assert (
        _binding_state_result({"stage": "final"}, arrangement_on_vase, objects) is None
    )
    assert (
        evaluate_intent_contract_extensions(
            {
                "stage": "final",
                "intent_contract": {"constraints": [arrangement_on_vase]},
                "scene_geometry": {"objects": objects},
            }
        )
        == []
    )


def test_filled_vase_composite_binds_flower_and_its_vase() -> None:
    filled_vase = _record(
        "filled_container_0",
        "filled_vase",
        (0.0, 0.0),
        (0.3, 0.3, 0.6),
    )
    filled_vase["name"] = "filled_vase"
    filled_vase["description"] = "vase filled with flowers"
    filled_vase["metadata"] = {
        "composite_type": "filled_container",
        "container_asset": {"name": "vase"},
        "fill_assets": [{"name": "flowers"}],
    }
    objects = [filled_vase]
    flower_selector = {"category": "flower", "count": 1}
    vase_selector = {"category": "vase", "count": 1}

    assert selected_ids(flower_selector, objects) == ["filled_container_0"]
    assert selected_ids(vase_selector, objects) == ["filled_container_0"]


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


def _square_four_side_case(*, asymmetric: bool = False) -> dict:
    table = _record("dining_table_0", "dining_table", (0.0, 0.0), (1.0, 1.0, 0.75))
    chair_count = 3 if asymmetric else 4
    chairs = [
        _record("dining_chair_0", "dining_chair", (0.0, -0.8), (0.5, 0.5, 0.9)),
        _record("dining_chair_1", "dining_chair", (0.0, 0.8), (0.5, 0.5, 0.9)),
        _record("dining_chair_2", "dining_chair", (-0.8, 0.0), (0.5, 0.5, 0.9)),
        _record("dining_chair_3", "dining_chair", (0.8, 0.0), (0.5, 0.5, 0.9)),
    ][:chair_count]
    return {
        "intent_contract": {
            "constraints": [
                {
                    "relation": "edge_distribution",
                    "subjects": {
                        "category": "dining_chair",
                        "count": chair_count,
                        "quantifier": "exactly",
                    },
                    "targets": {
                        "category": "dining_table",
                        "count": 1,
                        "quantifier": "exactly",
                    },
                    "edge_frame": "target_local_rectangle",
                    "groups": [
                        {
                            "edge_class": "long",
                            "counts_per_edge": [1, 1],
                            "spacing": "equal_segments",
                        },
                        {
                            "edge_class": "short",
                            "counts_per_edge": [1, 0] if asymmetric else [1, 1],
                            "spacing": "equal_segments",
                        },
                    ],
                    "orientation": "unconstrained",
                    "source": "explicit_prompt",
                    "strength": "hard",
                }
            ]
        },
        "scene_geometry": {"objects": [table, *chairs]},
    }


def test_symmetric_four_edge_distribution_accepts_square_target() -> None:
    result = evaluate_edge_distribution(_square_four_side_case())[0]

    assert result["label"] == "pass"
    assert {slot["edge"] for slot in result["diagnostics"]["edge_slots"]} == {
        "front",
        "back",
        "left",
        "right",
    }


def test_asymmetric_edge_distribution_keeps_square_target_unresolved() -> None:
    result = evaluate_edge_distribution(_square_four_side_case(asymmetric=True))[0]

    assert result["label"] == "unresolved"
    assert "stable long/short edge frame" in result["reason"]


def test_table_seating_edge_distribution_rejects_unusable_table_gap() -> None:
    table = _record("dining_table_0", "dining_table", (0.0, 0.0), (2.0, 1.0, 0.75))
    chair = _record(
        "dining_chair_0",
        "dining_chair",
        (0.0, -1.75),
        (0.5, 0.5, 0.9),
        yaw_deg=180.0,
    )
    case_pack = {
        "intent_contract": {
            "constraints": [
                {
                    "relation": "edge_distribution",
                    "subjects": {
                        "category": "dining_chair",
                        "count": 1,
                        "quantifier": "exactly",
                    },
                    "targets": {
                        "category": "dining_table",
                        "count": 1,
                        "quantifier": "exactly",
                    },
                    "edge_frame": "target_local_rectangle",
                    "groups": [
                        {
                            "edge_class": "long",
                            "counts_per_edge": [1, 0],
                            "spacing": "equal_segments",
                        },
                        {
                            "edge_class": "short",
                            "counts_per_edge": [0, 0],
                            "spacing": "equal_segments",
                        },
                    ],
                    "orientation": "toward_target",
                    "source": "explicit_prompt",
                }
            ]
        },
        "scene_geometry": {"objects": [table, chair]},
    }

    result = evaluate_edge_distribution(case_pack)[0]

    assert result["label"] == "fail"
    slot = result["diagnostics"]["seat_slots"][0]
    assert slot["outside_gap_m"] > slot["max_outside_gap_m"]
    assert "maximum usable gap" in result["reason"]


def test_non_seating_edge_distribution_keeps_unbounded_gap_semantics() -> None:
    bed = _record("bed_0", "bed", (0.0, 0.0), (2.0, 1.0, 0.75))
    nightstand = _record("nightstand_0", "nightstand", (0.0, -1.75), (0.5, 0.5, 0.9))
    case_pack = {
        "intent_contract": {
            "constraints": [
                {
                    "relation": "edge_distribution",
                    "subjects": {
                        "category": "nightstand",
                        "count": 1,
                        "quantifier": "exactly",
                    },
                    "targets": {
                        "category": "bed",
                        "count": 1,
                        "quantifier": "exactly",
                    },
                    "edge_frame": "target_local_rectangle",
                    "groups": [
                        {
                            "edge_class": "long",
                            "counts_per_edge": [1, 0],
                            "spacing": "equal_segments",
                        },
                        {
                            "edge_class": "short",
                            "counts_per_edge": [0, 0],
                            "spacing": "equal_segments",
                        },
                    ],
                    "orientation": "unconstrained",
                    "source": "explicit_prompt",
                }
            ]
        },
        "scene_geometry": {"objects": [bed, nightstand]},
    }

    result = evaluate_edge_distribution(case_pack)[0]

    assert result["label"] == "pass"
    assert "max_outside_gap_m" not in result["diagnostics"]["seat_slots"][0]


def test_edge_distribution_owns_runtime_seat_orientation_contract() -> None:
    table = _record("dining_table_0", "dining_table", (0.0, 0.0), (2.0, 1.0, 0.75))
    chair = _record(
        "dining_chair_0",
        "dining_chair",
        (0.0, -0.8),
        (0.5, 0.5, 0.9),
        yaw_deg=180.0,
    )
    case_pack = {
        "task_instruction": "Place one dining chair facing the dining table.",
        "room_type": "dining room",
        "checks": [
            {
                "check_id": "stale_orientation_contract",
                "check_source": CONTRACT_CHECK_SOURCE,
                "subject_id": "dining_chair_0",
                "relation_type": "furniture_faces_furniture",
            }
        ],
        "intent_contract": {
            "constraints": [
                {
                    "relation": "edge_distribution",
                    "subjects": {
                        "category": "dining_chair",
                        "count": 1,
                        "quantifier": "exactly",
                    },
                    "targets": {
                        "category": "dining_table",
                        "count": 1,
                        "quantifier": "exactly",
                    },
                    "edge_frame": "target_local_rectangle",
                    "groups": [
                        {
                            "edge_class": "long",
                            "counts_per_edge": [1, 0],
                            "spacing": "equal_segments",
                        },
                        {
                            "edge_class": "short",
                            "counts_per_edge": [0, 0],
                            "spacing": "equal_segments",
                        },
                    ],
                    "orientation": "toward_target",
                    "source": "explicit_prompt",
                }
            ]
        },
        "scene_geometry": {"objects": [table, chair]},
    }
    scene = SimpleNamespace(
        **{
            CONTRACT_ATTR: {
                "dining_chair_0": {
                    "subject_id": "dining_chair_0",
                    "target_ids": ["dining_table_0"],
                    "relation_type": "furniture_faces_furniture",
                }
            }
        }
    )

    stabilize_orientation_contracts(
        case_pack,
        scene,
        CriticConfig(enabled=True, metrics=("functional_dependency",)),
        stage="furniture",
    )

    assert getattr(scene, CONTRACT_ATTR) == {}
    assert not [
        check
        for check in case_pack["checks"]
        if check.get("check_source") == CONTRACT_CHECK_SOURCE
        and check.get("subject_id") == "dining_chair_0"
    ]


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


def test_storage_cabinet_selector_binds_retrieved_generic_cabinet() -> None:
    cabinet = _record("cabinet_0", "cabinet", (0.0, -1.725), (0.9, 0.45, 1.1))
    cabinet["metadata"] = {"semantic_name": "cabinet"}
    selector = {"category": "storage_cabinet", "count": 1, "quantifier": "all"}

    assert selected_ids(selector, [cabinet]) == ["cabinet_0"]
    assert selector_match_count(selector, [cabinet]) == 1
    assert (
        _evaluate_required_count(
            {
                "constraint_id": "storage_cabinet_count",
                "relation": "required_count",
                "subjects": selector,
            },
            [cabinet],
            "core",
        )["label"]
        == "pass"
    )


def test_storage_cabinet_fallback_excludes_other_cabinet_variants() -> None:
    filing_cabinet = _record(
        "filing_cabinet_0", "filing_cabinet", (0.0, 0.0), (0.9, 0.45, 1.1)
    )
    wall_cabinet = _record(
        "wall_cabinet_0", "wall_cabinet", (1.0, 0.0), (0.9, 0.35, 0.7)
    )
    for obj in (filing_cabinet, wall_cabinet):
        obj["metadata"] = {"semantic_name": obj["category"]}

    assert (
        selected_ids(
            {"category": "storage_cabinet", "count": 1},
            [filing_cabinet, wall_cabinet],
        )
        == []
    )


def test_storage_cabinet_against_wall_uses_retrieved_generic_cabinet() -> None:
    cabinet = _record("cabinet_0", "cabinet", (0.0, -1.725), (0.9, 0.45, 1.1))
    cabinet["metadata"] = {"semantic_name": "cabinet"}
    south_wall = _record("south_wall", "wall", (0.0, -2.0), (4.0, 0.1, 2.8))
    constraint = {
        "constraint_id": "storage_cabinet_against_wall",
        "relation": "against_wall",
        "stage": "furniture",
        "strength": "hard",
        "subjects": {
            "category": "storage_cabinet",
            "count": 1,
            "quantifier": "all",
        },
        "targets": {"category": "wall", "count": 1, "quantifier": "all"},
        "source": "explicit_prompt",
        "evidence_span": "place one storage cabinet against a wall",
    }
    case_pack = {
        "stage": "furniture",
        "intent_contract": {"constraints": [constraint]},
        "scene_geometry": {"objects": [cabinet, south_wall]},
    }

    assert augment_contract_checks(case_pack)
    result = evaluate_functional_dependency(
        load_geometry(case_pack), case_pack["checks"][0]
    )

    assert result["relation_type"] == "back_against_wall"
    assert result["label"] == "pass"


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


def test_exact_category_precedence_excludes_specialized_parent_matches() -> None:
    tables = [
        _record("table_0", "table", (0.0, 0.0), (1.0, 0.6, 0.75)),
        _record("table_1", "table", (1.5, 0.0), (1.0, 0.6, 0.75)),
    ]
    side_table = _record("side_table_0", "side_table", (3.0, 0.0), (0.6, 0.5, 0.6))
    selector = {"category": "table", "count": 2, "quantifier": "all"}

    assert selected_ids(selector, [*tables, side_table]) == ["table_0", "table_1"]
    assert selector_match_count(selector, [*tables, side_table]) == 2
    assert bound_ids(selector, [*tables, side_table]) == ["table_0", "table_1"]


def test_parent_category_fallback_remains_available_without_exact_asset() -> None:
    side_table = _record("side_table_0", "side_table", (0.0, 0.0), (0.6, 0.5, 0.6))
    selector = {"category": "table", "count": 1, "quantifier": "all"}

    assert selected_ids(selector, [side_table]) == ["side_table_0"]
    assert selector_match_count(selector, [side_table]) == 1
    assert bound_ids(selector, [side_table]) == ["side_table_0"]


def test_exact_category_count_and_role_mismatch_stay_unresolved() -> None:
    table = _record("table_0", "table", (0.0, 0.0), (1.0, 0.6, 0.75))
    side_table = _record("side_table_0", "side_table", (1.5, 0.0), (0.6, 0.5, 0.6))
    assert (
        bound_ids(
            {"category": "table", "count": 2, "quantifier": "all"},
            [table, side_table],
        )
        == []
    )
    assert (
        selected_ids(
            {"category": "table", "role": "dining_table", "count": 1},
            [table],
        )
        == []
    )


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


def test_open_vocabulary_semantic_name_binds_manipuland_retrieval_category() -> None:
    mini_fridge = _record(
        "mini_fridge_0",
        "refrigerator",
        (0.0, 0.0),
        (0.3, 0.2, 0.35),
    )
    mini_fridge.update(
        {
            "name": "mini_fridge",
            "object_type": "manipuland",
            "metadata": {"semantic_name": "mini_fridge"},
        }
    )

    assert selected_ids({"category": "mini_fridge", "count": 1}, [mini_fridge]) == [
        "mini_fridge_0"
    ]


def test_generic_open_vocabulary_selector_matches_declared_specialization() -> None:
    bathtub = _record(
        "freestanding_bathtub_0",
        "freestanding_bathtub",
        (0.0, 0.0),
        (1.7, 0.8, 0.6),
    )
    bathtub["object_type"] = "furniture"

    assert selected_ids({"category": "bathtub", "count": 1}, [bathtub]) == [
        "freestanding_bathtub_0"
    ]


def test_composite_open_vocabulary_selector_requires_explicit_identity() -> None:
    fruit_bowl = _record(
        "filled_container_0",
        "bowl",
        (0.0, 0.0),
        (0.3, 0.3, 0.2),
    )
    fruit_bowl.update(
        {
            "name": "filled_bowl_of_fruit",
            "object_type": "manipuland",
            "metadata": {
                "composite_type": "filled_container",
                "container_asset": {"name": "bowl_of_fruit"},
                "fill_assets": [{"name": "apple"}, {"name": "pear"}],
            },
        }
    )
    empty_bowl = _record(
        "empty_bowl_0",
        "bowl",
        (1.0, 0.0),
        (0.3, 0.3, 0.2),
    )
    empty_bowl["object_type"] = "manipuland"

    selector = {"category": "bowl_of_fruit", "count": 1}
    assert selected_ids(selector, [fruit_bowl, empty_bowl]) == ["filled_container_0"]


@pytest.mark.parametrize("category", ["television", "media_cabinet"])
def test_canonical_adapter_category_binds_wall_mounted_open_category(
    category: str,
) -> None:
    mounted = _record("mounted_media_0", "decor", (0.0, 2.0), (1.2, 0.1, 0.7))
    mounted.update(
        {
            "category_norm": category,
            "object_type": "wall_mounted",
            "metadata": {"semantic_name": "mounted_media_endpoint"},
            "functional_hints": {"scene_object_type": "wall_mounted"},
        }
    )
    selector = {"category": category, "count": 1, "quantifier": "exactly"}

    assert selected_ids(selector, [mounted]) == ["mounted_media_0"]
    assert bound_ids(selector, [mounted]) == ["mounted_media_0"]


def test_open_vocabulary_canonical_adapter_category_binds_manipuland() -> None:
    plush = _record("plush_toy_0", "decor", (0.0, 0.0), (0.2, 0.2, 0.2))
    plush.update(
        {
            "category_norm": "plush_toy",
            "object_type": "manipuland",
            "metadata": {"semantic_name": "soft_bear_asset"},
            "functional_hints": {"scene_object_type": "manipuland"},
        }
    )
    selector = {"category": "plush_toy", "count": 1, "quantifier": "exactly"}

    assert selected_ids(selector, [plush]) == ["plush_toy_0"]
    assert bound_ids(selector, [plush]) == ["plush_toy_0"]


def test_ceiling_selector_binds_exact_open_vocabulary_semantic_name() -> None:
    string_light = _record(
        "string_light_0",
        "ceiling_object",
        (0.0, 0.0),
        (1.6, 0.4, 0.1),
    )
    string_light.update(
        {
            "object_type": "ceiling_mounted",
            "metadata": {"semantic_name": "string_light"},
            "functional_hints": {"scene_object_type": "ceiling_mounted"},
        }
    )

    assert selected_ids({"category": "string_light", "count": 1}, [string_light]) == [
        "string_light_0"
    ]


@pytest.mark.parametrize(
    "semantic_name", ["pendant_lamp", "pendant_light", "ceiling_light"]
)
def test_generic_lamp_selector_binds_mounted_lighting_specialization(
    semantic_name: str,
) -> None:
    pendant = _record(
        "pendant_lamp_0",
        "ceiling_object",
        (0.0, 0.0),
        (0.5, 0.5, 0.4),
    )
    pendant.update(
        {
            "object_type": "ceiling_mounted",
            "metadata": {"semantic_name": semantic_name},
            "functional_hints": {"scene_object_type": "ceiling_mounted"},
        }
    )

    assert selected_ids({"category": "lamp", "count": 1}, [pendant]) == [
        "pendant_lamp_0"
    ]


def test_stage_scoped_selector_count_uses_same_context_as_binding() -> None:
    table_lamp = _record(
        "table_lamp_0",
        "lamp",
        (0.0, 0.0),
        (0.2, 0.2, 0.4),
    )
    table_lamp["object_type"] = "manipuland"
    pendant = _record(
        "pendant_lamp_0",
        "ceiling_object",
        (0.0, 0.0),
        (0.5, 0.5, 0.4),
    )
    pendant.update(
        {
            "object_type": "ceiling_mounted",
            "metadata": {"semantic_name": "pendant_light"},
        }
    )
    selector = {"category": "lamp", "count": 1, "stage": "ceiling_mounted"}

    assert selected_ids(selector, [table_lamp]) == []
    assert selector_match_count(selector, [table_lamp]) == 0
    assert selected_ids(selector, [table_lamp, pendant]) == ["pendant_lamp_0"]
    assert selector_match_count(selector, [table_lamp, pendant]) == 1


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


def test_semantic_ir_validation_does_not_reinterpret_evidence_quantifier() -> None:
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
        },
        validate_prompt_semantics=False,
    )

    assert contract["constraints"][0]["targets"]["quantifier"] == "all"


def test_semantic_ir_validation_keeps_explicit_prompt_provenance_required() -> None:
    with pytest.raises(ValidationError, match="explicit_prompt relations require"):
        validate_intent_contract(
            {
                "schema_version": INTENT_CONTRACT_SCHEMA_VERSION,
                "constraints": [
                    {
                        "relation": "near",
                        "subjects": {"category": "floor_lamp", "count": 1},
                        "targets": {"category": "armchair", "count": 1},
                        "source": "explicit_prompt",
                    }
                ],
            },
            validate_prompt_semantics=False,
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


def test_directional_furniture_can_face_wall_mounted_instructional_surface() -> None:
    from scenesmith.scenebenchmark_critic.metrics.functional_dependency.relations import (
        _relation_target_is_valid,
    )

    desk = _record("student_desk_0", "student_desk", (0.0, 0.0), (1.2, 0.6, 0.7))
    desk["object_type"] = "furniture"
    desk["metadata"] = {"semantic_name": "student_desk"}
    chalkboard = _record("chalkboard_0", "chalkboard", (0.0, 2.0), (2.0, 0.1, 1.0))
    chalkboard["functional_hints"] = {"scene_object_type": "wall_mounted"}

    assert _relation_target_is_valid(desk, chalkboard, "furniture_faces_furniture")


def test_prompt_desk_facing_uses_view_direction_without_near_distance() -> None:
    desk = _record(
        "student_desk_0",
        "student_desk",
        (0.0, 0.0),
        (1.2, 0.6, 0.7),
        yaw_deg=180.0,
    )
    desk["object_type"] = "furniture"
    desk["metadata"] = {"semantic_name": "student_desk"}
    chalkboard = _record("chalkboard_0", "chalkboard", (0.0, 5.0), (2.0, 0.1, 1.0))
    chalkboard["functional_hints"] = {"scene_object_type": "wall_mounted"}
    constraint = {
        "constraint_id": "student_desks_face_board",
        "relation": "faces",
        "stage": "wall_mounted",
        "strength": "hard",
        "subjects": {"category": "student_desk", "count": 1},
        "targets": {"category": "instructional_surface", "count": 1},
        "source": "explicit_prompt",
        "evidence_span": "student desks face the chalkboard",
    }
    case_pack = {
        "stage": "wall_mounted",
        "intent_contract": {"constraints": [constraint]},
        "scene_geometry": {"objects": [desk, chalkboard]},
    }

    assert augment_contract_checks(case_pack)
    check = next(
        row
        for row in case_pack["checks"]
        if row["relation_type"] == "furniture_faces_furniture"
    )
    assert (
        check["evidence"]["dependency"]
        | {
            "subject_face": "back",
            "distance_required": False,
        }
        == check["evidence"]["dependency"]
    )
    result = evaluate_functional_dependency(load_geometry(case_pack), check)

    assert result["label"] == "pass"
    assert "gap" not in result["reason"]


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


def test_required_counts_can_be_filtered_by_generation_stage() -> None:
    scene = SimpleNamespace(
        scenebenchmark_intent_contract={
            "constraints": [
                {
                    "relation": "required_count",
                    "stage": "furniture",
                    "strength": "hard",
                    "subjects": {"category": "side_table", "count": 3},
                },
                {
                    "relation": "required_count",
                    "stage": "wall_mounted",
                    "strength": "hard",
                    "subjects": {"category": "storage_cabinet", "count": 2},
                },
            ]
        }
    )

    assert intent_contract_required_counts(scene) == {
        "side_table": 3,
        "storage_cabinet": 2,
    }
    assert intent_contract_required_counts(scene, stage="furniture") == {
        "side_table": 3
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


def _table_access_with_edge_seats(*, include_unrelated_blocker: bool = False) -> dict:
    table = _record(
        "conference_table_0",
        "conference_table",
        (0.0, 0.0),
        (3.0, 1.2, 0.75),
        yaw_deg=0.0,
    )
    chairs = [
        _record(
            f"office_chair_{index}",
            "office_chair",
            (x, 0.65),
            (0.5, 0.5, 1.0),
            yaw_deg=180.0,
        )
        for index, x in enumerate((-0.8, 0.0, 0.8))
    ]
    objects = [table, *chairs]
    if include_unrelated_blocker:
        objects.append(_record("cabinet_0", "cabinet", (1.25, 0.5), (0.4, 0.4, 1.0)))
    return {
        "stage": "furniture",
        "intent_contract": {
            "constraints": [
                {
                    "constraint_id": "table_circulation",
                    "relation": "clear_access",
                    "subjects": {"category": "conference_table", "count": 1},
                    "targets": {"category": "room", "count": 1},
                    "source": "explicit_prompt",
                    "strength": "hard",
                },
                {
                    "constraint_id": "table_edge_seats",
                    "relation": "edge_distribution",
                    "subjects": {"category": "office_chair", "count": 3},
                    "targets": {"category": "conference_table", "count": 1},
                    "source": "explicit_prompt",
                    "strength": "hard",
                },
            ]
        },
        "scene_geometry": {"objects": objects},
    }


def test_clear_access_allows_hard_edge_distribution_seats() -> None:
    result = next(
        row
        for row in evaluate_intent_contract_extensions(_table_access_with_edge_seats())
        if row["relation_type"] == "clear_access"
    )

    assert result["label"] == "pass"
    assert result["diagnostics"]["authorized_occupant_ids"] == [
        "office_chair_0",
        "office_chair_1",
        "office_chair_2",
    ]


def test_clear_access_keeps_non_edge_object_as_table_blocker() -> None:
    result = next(
        row
        for row in evaluate_intent_contract_extensions(
            _table_access_with_edge_seats(include_unrelated_blocker=True)
        )
        if row["relation_type"] == "clear_access"
    )

    assert result["label"] == "fail"
    assert result["diagnostics"]["blocking_ids"] == ["cabinet_0"]


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


@pytest.mark.parametrize("relation", ["edge_distribution", "surround"])
def test_spatial_accessibility_allows_contract_seating_cohort(
    relation: str,
) -> None:
    table = _record(
        "dining_table_0",
        "dining_table",
        (0.0, 0.0),
        (1.8, 0.9, 0.75),
    )
    chairs = [
        _record(
            f"dining_chair_{index}",
            "dining_chair",
            (x, 0.7),
            (0.45, 0.45, 0.9),
        )
        for index, x in enumerate((-0.5, 0.5))
    ]
    case_pack = {
        "checks": [
            {
                "check_id": "spatial_accessibility__dining_table_0",
                "metric": "spatial_accessibility",
                "subject_id": "dining_table_0",
            }
        ],
        "intent_contract": {
            "constraints": [
                {
                    "relation": relation,
                    "subjects": {"category": "dining_chair", "count": 2},
                    "targets": {"category": "dining_table", "count": 1},
                    "source": "explicit_prompt",
                    "strength": "hard",
                }
            ]
        },
    }

    attach_expected_access_companions(
        case_pack,
        {obj["id"]: obj for obj in [table, *chairs]},
    )

    assert case_pack["checks"][0]["expected_companion_ids"] == [
        "dining_chair_0",
        "dining_chair_1",
    ]


def test_spatial_accessibility_does_not_cross_pair_grouped_surfaces() -> None:
    desks = [
        _record(f"desk_{index}", "desk", (float(index), 0.0), (1.2, 0.6, 0.75))
        for index in range(2)
    ]
    chairs = [
        _record(
            f"chair_{index}",
            "chair",
            (float(index), 0.7),
            (0.45, 0.45, 0.9),
        )
        for index in range(2)
    ]
    case_pack = {
        "checks": [
            {
                "check_id": f"spatial_accessibility__desk_{index}",
                "metric": "spatial_accessibility",
                "subject_id": f"desk_{index}",
            }
            for index in range(2)
        ],
        "intent_contract": {
            "constraints": [
                {
                    "relation": "paired_with",
                    "subjects": {"category": "chair", "count": 2},
                    "targets": {"category": "desk", "count": 2},
                    "source": "explicit_prompt",
                    "strength": "hard",
                }
            ]
        },
    }

    attach_expected_access_companions(
        case_pack,
        {obj["id"]: obj for obj in [*desks, *chairs]},
    )

    assert all("expected_companion_ids" not in check for check in case_pack["checks"])


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


def _response(
    content: str,
    *,
    finish_reason: str = "stop",
    completion_tokens: int = 32,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=64,
            completion_tokens=completion_tokens,
            total_tokens=64 + completion_tokens,
        ),
    )


def _semantic_ir(requirements: list[dict]) -> str:
    return json.dumps(
        {
            "schema_version": INTENT_COMPILER_SEMANTIC_IR_VERSION,
            "requirements": [
                {"target_ref": None, **requirement} for requirement in requirements
            ],
        }
    )


def _semantic_scope(*groundings: str) -> str:
    return _semantic_ir(
        [
            {
                "requirement_id": f"scope_{index}",
                "kind": "soft_scope",
                "grounding": grounding,
                "reason": "non-geometric scope",
            }
            for index, grounding in enumerate(groundings)
        ]
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


# v16 deliberately removed deterministic prompt parsing/enrichment from the
# live IntentCompiler. The retained fixtures below document the v15 contract
# shape, but must never become requirements for the LLM-only admission path.
_LEGACY_LIVE_COMPILER_REASON = (
    "v15 legacy contract fixture; v16 accepts CompilerSemanticIR only"
)


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
            _response(
                _semantic_ir(
                    [
                        {
                            "requirement_id": "room_scope",
                            "kind": "soft_scope",
                            "grounding": "prompt:0",
                            "reason": "room inventory statement",
                        }
                    ]
                )
            ),
        ]
    )

    result = compiler.compile("A meeting room with a conference table.")

    assert result["schema_version"] == INTENT_CONTRACT_SCHEMA_VERSION
    assert result["retry_count"] == 1
    assert compiler.last_trace["retry_count"] == 1
    assert [item["status"] for item in compiler.last_trace["attempts"]] == [
        "error",
        "retry_ok",
    ]
    first_user_message = compiler._test_calls[0]["messages"][1]["content"]
    assert "Original scene prompt:" in first_user_message
    assert "Normalized SceneTaskSpec" not in first_user_message

    for call in compiler._test_calls:
        assert call["response_format"] == json_response_format(
            model=compiler._model,
            name="compiler_semantic_ir",
            schema=intent_compiler_wire_json_schema(),
        )


def test_intent_compiler_projects_grounded_catalog_relations_and_coverage() -> None:
    task_spec = SceneTaskSpec(
        room_type="living room",
        style="standard",
        required_large_objects=["display_shelf"] * 3,
    )
    compiler = _compiler_with_responses(
        [
            _response(
                _semantic_ir(
                    [
                        {
                            "requirement_id": "inventory",
                            "kind": "inventory",
                            "grounding": "prompt:0",
                            "subject_ref": "inventory:display_shelf",
                        },
                        {
                            "requirement_id": "wall_relation",
                            "kind": "relation",
                            "grounding": "prompt:0",
                            "relation": "against_wall",
                            "subject_ref": "inventory:display_shelf",
                            "target_ref": "anchor:wall",
                            "subject_count": 3,
                        },
                        {
                            "requirement_id": "no_tv",
                            "kind": "forbidden_inventory",
                            "grounding": "prompt:0",
                            "forbidden_category": "television",
                        },
                    ]
                )
            )
        ]
    )

    result = compiler.compile(
        "A living room with three display shelves against the wall and no TV.",
        task_spec=task_spec,
    )

    assert any(
        row["relation"] == "required_count"
        and row["subjects"]["category"] == "display_shelf"
        and row["subjects"]["count"] == 3
        for row in result["constraints"]
    )
    assert any(
        row["relation"] == "against_wall"
        and row["subjects"]["category"] == "display_shelf"
        and row["targets"]["category"] == "wall"
        for row in result["constraints"]
    )
    assert not any(
        row["subjects"]["category"] == "monitor" for row in result["constraints"]
    )
    assert result["coverage_requirements"] == [
        {
            "requirement_id": "no_tv",
            "kind": "forbidden_inventory",
            "disposition": "compiled",
            "normalized": "television",
            "earliest_stage": "furniture",
            "final_stage": "final",
            "source": "explicit_prompt",
            "evidence_span": "A living room with three display shelves against the wall and no TV.",
            "relation": "",
        }
    ]
    assert {row["disposition"] for row in result["coverage_ledger"]} == {"compiled"}
    assert (
        "inventory:display_shelf"
        in compiler.last_trace["attempts"][0]["accepted_entity_refs"]
    )


def test_intent_compiler_retries_hallucinated_ref_then_preserves_unresolved_coverage() -> (
    None
):
    task_spec = SceneTaskSpec(
        room_type="dining room",
        style="standard",
        required_large_objects=["dining_table"],
        required_small_objects=["bowl_of_fruit"],
    )
    compiler = _compiler_with_responses(
        [
            _response(
                _semantic_ir(
                    [
                        {
                            "requirement_id": "phantom",
                            "kind": "relation",
                            "grounding": "prompt:0",
                            "relation": "on_top_of",
                            "subject_ref": "inventory:phantom_object",
                            "target_ref": "inventory:dining_table",
                        }
                    ]
                )
            ),
            _response(
                _semantic_ir(
                    [
                        {
                            "requirement_id": "fruit_bowl",
                            "kind": "relation",
                            "grounding": "prompt:0",
                            "relation": "on_top_of",
                            "subject_ref": "inventory:bowl_of_fruit",
                            "target_ref": "inventory:dining_table",
                        },
                        {
                            "requirement_id": "ambiguous_detail",
                            "kind": "unresolved",
                            "grounding": "prompt:0",
                            "reason": "ambiguous decorative placement",
                        },
                    ]
                )
            ),
        ]
    )

    result = compiler.compile(
        "Put a bowl of fruit on the dining table with an ambiguous decorative detail.",
        task_spec=task_spec,
    )

    bowl_relation = next(
        row for row in result["constraints"] if row["relation"] == "on_top_of"
    )
    assert bowl_relation["subjects"]["category"] == "bowl_of_fruit"
    assert compiler.last_trace["status"] == "retry_ok"
    assert "unbound entity_ref" in compiler.last_trace["attempts"][0]["error"]
    assert compiler.last_trace["attempts"][0]["rejected_entity_refs"] == [
        {
            "requirement_id": "phantom",
            "grounding": "prompt:0",
            "entity_ref": "inventory:phantom_object",
            "reason": "unknown_or_unbound_entity_ref",
        }
    ]
    assert (
        compiler.last_trace["rejected_entity_refs"]
        == compiler.last_trace["attempts"][0]["rejected_entity_refs"]
    )
    assert result["coverage_requirements"][0]["disposition"] == "unresolved"


def test_intent_compiler_retry_teaches_catalog_target_ref_not_target_role() -> None:
    task_spec = SceneTaskSpec(
        room_type="bedroom", style="standard", required_large_objects=["bed"]
    )
    compiler = _compiler_with_responses(
        [
            _response(
                _semantic_ir(
                    [
                        {
                            "requirement_id": "bed_by_window",
                            "kind": "relation",
                            "grounding": "prompt:0",
                            "relation": "next_to",
                            "subject_ref": "inventory:bed",
                            "target_role": "window",
                            "target_cohort": "window",
                        }
                    ]
                )
            ),
            _response(
                _semantic_ir(
                    [
                        {
                            "requirement_id": "bed_by_window",
                            "kind": "relation",
                            "grounding": "prompt:0",
                            "relation": "next_to",
                            "subject_ref": "inventory:bed",
                            "target_ref": "anchor:window",
                            "target_role": "window",
                            "target_cohort": "window",
                        }
                    ]
                )
            ),
        ]
    )

    result = compiler.compile("Put the bed next to the window.", task_spec=task_spec)

    relation = next(
        row for row in result["constraints"] if row["relation"] == "next_to"
    )
    assert relation["targets"]["category"] == "window"
    assert compiler.last_trace["status"] == "retry_ok"
    correction = compiler._test_calls[1]["messages"][1]["content"]
    assert "target_role and target_cohort are optional metadata only" in correction
    assert 'target_ref="anchor:window"' in correction


def test_intent_compiler_retry_uses_semantic_ir_for_unary_secondary_ref() -> None:
    task_spec = SceneTaskSpec(
        room_type="bedroom", style="standard", required_large_objects=["bed"]
    )
    compiler = _compiler_with_responses(
        [
            _response(
                _semantic_ir(
                    [
                        {
                            "requirement_id": "bed_by_window",
                            "kind": "relation",
                            "grounding": "prompt:0",
                            "relation": "next_to",
                            "subject_ref": "inventory:bed",
                            "target_ref": "anchor:window",
                            "secondary_target_ref": "anchor:wall",
                        }
                    ]
                )
            ),
            _response(
                _semantic_ir(
                    [
                        {
                            "requirement_id": "bed_by_window",
                            "kind": "relation",
                            "grounding": "prompt:0",
                            "relation": "next_to",
                            "subject_ref": "inventory:bed",
                            "target_ref": "anchor:window",
                        }
                    ]
                )
            ),
        ]
    )

    result = compiler.compile("Put the bed next to the window.", task_spec=task_spec)

    assert result["retry_count"] == 1
    correction = compiler._test_calls[1]["messages"][1]["content"]
    assert "remove secondary_target_ref; keep exactly one target_ref" in correction
    assert "secondary_category" not in correction
    assert "targets object" not in correction


def test_intent_compiler_exhaustion_never_calls_deterministic_prompt_fallback(
    monkeypatch,
) -> None:
    def _unexpected_parser(*_args, **_kwargs):
        raise AssertionError(
            "live compiler must not invoke deterministic prompt parsing"
        )

    monkeypatch.setattr(
        intent_compiler_module, "build_intent_contract", _unexpected_parser
    )
    compiler = _compiler_with_responses([_response("not json"), _response("still bad")])

    with pytest.raises(IntentCompilationError, match="failed after two attempts"):
        compiler.compile("A room with a desk.")

    assert compiler.last_trace["status"] == "error"


def test_intent_compiler_ignores_refs_and_relation_metadata_on_coverage_rows() -> None:
    compiler = _compiler_with_responses(
        [
            _response(
                _semantic_ir(
                    [
                        {
                            "requirement_id": "no_tv",
                            "kind": "forbidden_inventory",
                            "grounding": "prompt:0",
                            "forbidden_category": "television",
                            "relation": "surround",
                            "subject_ref": "inventory:phantom_tv",
                            "target_ref": "anchor:room",
                            "subject_count": 2,
                            "target_count": 1,
                            "subject_role": "toy",
                            "target_role": "room_center",
                        }
                    ]
                )
            )
        ]
    )

    result = compiler.compile("A room with no TV.")

    assert result["retry_count"] == 0
    assert compiler.last_trace["status"] == "ok"
    assert compiler.last_trace["attempts"][0]["rejected_entity_refs"] == [
        {
            "requirement_id": "no_tv",
            "grounding": "prompt:0",
            "entity_ref": "inventory:phantom_tv",
            "reason": "ignored_non_endpoint_entity_ref",
        },
        {
            "requirement_id": "no_tv",
            "grounding": "prompt:0",
            "entity_ref": "anchor:room",
            "reason": "ignored_non_endpoint_entity_ref",
        },
    ]
    assert compiler.last_trace["attempts"][0]["accepted_entity_refs"] == []
    assert result["constraints"] == []
    assert result["coverage_requirements"][0]["kind"] == "forbidden_inventory"
    assert result["coverage_ledger"] == [
        {
            "requirement_id": "no_tv",
            "grounding": "prompt:0",
            "kind": "forbidden_inventory",
            "entity_refs": [],
            "surface_mentions": [],
            "reason": "",
            "ignored_semantic_fields": [
                "relation",
                "subject_count",
                "subject_ref",
                "subject_role",
                "target_count",
                "target_ref",
                "target_role",
            ],
            "disposition": "compiled",
        }
    ]


def test_intent_compiler_soft_scope_discards_scene_like_relation_payload() -> None:
    compiler = _compiler_with_responses(
        [
            _response(
                _semantic_ir(
                    [
                        {
                            "requirement_id": "playful_accents",
                            "kind": "soft_scope",
                            "grounding": "prompt:0",
                            "reason": "playful decorative mood",
                            "relation": "surround",
                            "subject_ref": "inventory:plush_toy",
                            "target_ref": "anchor:room",
                            "subject_count": 3,
                            "target_count": 1,
                            "subject_role": "accent",
                            "target_role": "center",
                        }
                    ]
                )
            )
        ]
    )

    result = compiler.compile("Use playful plush toys around the room.")

    assert result["retry_count"] == 0
    assert result["constraints"] == []
    assert result["coverage_requirements"] == [
        {
            "requirement_id": "playful_accents",
            "kind": "soft_scope",
            "disposition": "soft_scope",
            "normalized": "playful_decorative_mood",
            "earliest_stage": "floor_plan",
            "final_stage": "final",
            "source": "explicit_prompt",
            "evidence_span": "Use playful plush toys around the room.",
            "relation": "",
        }
    ]
    ledger = result["coverage_ledger"][0]
    assert ledger["entity_refs"] == []
    assert ledger["ignored_semantic_fields"] == [
        "relation",
        "subject_count",
        "subject_ref",
        "subject_role",
        "target_count",
        "target_ref",
        "target_role",
    ]
    assert [
        row["entity_ref"] for row in compiler.last_trace["rejected_entity_refs"]
    ] == ["inventory:plush_toy", "anchor:room"]


def test_intent_compiler_rejects_unbound_inventory_claim_before_coverage_admission() -> (
    None
):
    task_spec = SceneTaskSpec(
        room_type="office", style="standard", required_large_objects=["desk"]
    )
    compiler = _compiler_with_responses(
        [
            _response(
                _semantic_ir(
                    [
                        {
                            "requirement_id": "desk",
                            "kind": "inventory",
                            "grounding": "prompt:0",
                        }
                    ]
                )
            ),
            _response(
                _semantic_ir(
                    [
                        {
                            "requirement_id": "desk",
                            "kind": "inventory",
                            "grounding": "prompt:0",
                            "subject_ref": "inventory:desk",
                        }
                    ]
                )
            ),
        ]
    )

    result = compiler.compile("An office with a desk.", task_spec=task_spec)

    assert result["retry_count"] == 1
    assert compiler.last_trace["status"] == "retry_ok"
    assert (
        "inventory requires an inventory subject_ref"
        in compiler.last_trace["attempts"][0]["error"]
    )
    assert [row["relation"] for row in result["constraints"]] == ["required_count"]
    assert result["coverage_ledger"] == [
        {
            "requirement_id": "desk",
            "grounding": "prompt:0",
            "kind": "inventory",
            "entity_refs": ["inventory:desk"],
            "surface_mentions": [],
            "reason": "",
            "disposition": "compiled",
        }
    ]


def test_intent_compiler_failure_trace_keeps_all_rejected_entity_refs() -> None:
    task_spec = SceneTaskSpec(
        room_type="office", style="standard", required_large_objects=["desk"]
    )
    hallucinated_ir = _semantic_ir(
        [
            {
                "requirement_id": "monitor",
                "kind": "relation",
                "grounding": "prompt:0",
                "relation": "near",
                "subject_ref": "inventory:monitor",
                "target_ref": "inventory:desk",
            }
        ]
    )
    compiler = _compiler_with_responses(
        [_response(hallucinated_ir), _response(hallucinated_ir)]
    )

    with pytest.raises(IntentCompilationError, match="failed after two attempts"):
        compiler.compile("An office with a desk.", task_spec=task_spec)

    assert [
        row["entity_ref"] for row in compiler.last_trace["rejected_entity_refs"]
    ] == ["inventory:monitor", "inventory:monitor"]
    assert all(
        attempt["rejected_entity_refs"] for attempt in compiler.last_trace["attempts"]
    )


def test_intent_compiler_prompt_describes_semantic_ir_refs_not_legacy_selectors() -> (
    None
):
    prompt = _system_prompt()

    assert "subject_ref, target_ref" in prompt
    assert "do not emit targets, subjects, category" in prompt
    assert '"target_ref": "inventory:bed"' in prompt


def test_intent_compiler_keeps_relation_cohort_distinct_from_inventory_total() -> None:
    task_spec = SceneTaskSpec(
        room_type="dining room",
        style="standard",
        required_large_objects=["dining_table"] + ["dining_chair"] * 5,
    )
    compiler = _compiler_with_responses(
        [
            _response(
                _semantic_ir(
                    [
                        {
                            "requirement_id": "chairs_by_table_edges",
                            "kind": "relation",
                            "grounding": "prompt:0",
                            "relation": "edge_distribution",
                            "subject_ref": "inventory:dining_chair",
                            "target_ref": "inventory:dining_table",
                            "subject_count": 5,
                            "subject_cohort": "dining_seating",
                            "edge_frame": "target_local_rectangle",
                            "groups": [
                                {"edge_class": "long", "counts_per_edge": [2, 2]},
                                {"edge_class": "short", "counts_per_edge": [1, 0]},
                            ],
                            "orientation": "toward_target",
                        }
                    ]
                )
            )
        ]
    )

    result = compiler.compile(
        "Arrange four dining chairs along the long sides and one on a short side.",
        task_spec=task_spec,
    )

    edge = next(
        row for row in result["constraints"] if row["relation"] == "edge_distribution"
    )
    assert edge["subjects"]["category"] == "dining_chair"
    assert edge["subjects"]["count"] == 5
    assert edge["subjects"]["cohort"] == "dining_seating"
    assert [group["counts_per_edge"] for group in edge["groups"]] == [[2, 2], [1, 0]]


def test_intent_compiler_raises_after_provider_failures_without_fallback() -> None:
    compiler = _compiler_with_responses(
        [ConnectionError("temporary outage"), ConnectionError("temporary outage")]
    )

    with pytest.raises(IntentCompilationError, match="failed after two attempts"):
        compiler.compile("A room with a desk.")

    assert compiler.last_trace["status"] == "error"
    assert [row["status"] for row in compiler.last_trace["attempts"]] == [
        "error",
        "error",
    ]


def test_intent_compiler_without_task_spec_keeps_unbound_prompt_entity_unresolved() -> (
    None
):
    compiler = _compiler_with_responses(
        [
            _response(
                _semantic_ir(
                    [
                        {
                            "requirement_id": "desk_relation",
                            "kind": "unresolved",
                            "grounding": "prompt:0",
                            "reason": "desk is absent from SceneTaskSpec inventory",
                            "surface_mentions": ["desk", "monitor"],
                        }
                    ]
                )
            )
        ]
    )

    result = compiler.compile("Put a monitor on the desk.")

    assert result["constraints"] == []
    assert result["coverage_requirements"][0]["kind"] == "unresolved"
    assert result["coverage_requirements"][0]["disposition"] == "unresolved"
    assert not any(
        entry["entity_ref"] == "inventory:desk"
        for entry in compiler.last_trace["entity_catalog"]
    )


def test_intent_compiler_injects_both_task_compiler_constraint_channels() -> None:
    interaction = "nightstands should flank the bed and remain reachable"
    aesthetic = "keep the dining table centered in the room"
    compiler = _compiler_with_responses(
        [
            _response('{"constraints": [{"relation": "one_per_side"}]}'),
            _response(
                _semantic_scope(
                    "prompt:0", "interaction:0", "aesthetic:0", "aesthetic:1"
                )
            ),
        ]
    )

    result = compiler.compile(
        "A bedroom and dining area with a bed, two nightstands, and a dining table.",
        interaction_constraints=[interaction],
        aesthetic_constraints=[aesthetic, "use a modern material palette"],
    )

    assert result["constraints"] == []
    for call in compiler._test_calls:
        user_message = call["messages"][1]["content"]
        assert interaction in user_message
        assert aesthetic in user_message
        assert "use a modern material palette" in user_message
    assert {row["grounding"] for row in result["coverage_ledger"]} == {
        "prompt:0",
        "interaction:0",
        "aesthetic:0",
        "aesthetic:1",
    }


def test_intent_compiler_injects_complete_task_spec_and_owns_inventory() -> None:
    task_spec = SceneTaskSpec(
        room_type="bedroom",
        style="functional",
        required_large_objects=["bed", "nightstand", "nightstand"],
        required_wall_objects=["mirror"],
        required_ceiling_objects=["ceiling light"],
        required_small_objects=["book"],
        functional_zones=["sleeping_zone"],
        interaction_constraints=["nightstands should be reachable from the bed"],
        aesthetic_constraints=["functional layout with clear circulation paths"],
    )
    compiler = _compiler_with_responses(
        [_response(_semantic_scope("prompt:0", "interaction:0", "aesthetic:0"))]
    )

    result = compiler.compile(
        "A functional bedroom with one bed and one nightstand.", task_spec=task_spec
    )

    user_message = compiler._test_calls[0]["messages"][1]["content"]
    for value in (
        "bedroom",
        "functional",
        "required_large_objects",
        "required_wall_objects",
        "required_ceiling_objects",
        "required_small_objects",
        "functional_zones",
        "interaction_constraints",
        "aesthetic_constraints",
        "sleeping_zone",
    ):
        assert value in user_message
    required = {
        row["subjects"]["category"]: row
        for row in result["constraints"]
        if row["relation"] == "required_count"
    }
    assert required["nightstand"]["subjects"]["count"] == 2
    assert required["nightstand"]["source"] == "task_compiler_inventory"
    assert required["mirror"]["stage"] == "wall_mounted"
    assert required["ceiling_light"]["stage"] == "ceiling_mounted"
    assert required["book"]["stage"] == "manipuland"
    assert result["coverage_ledger"]


def test_task_spec_inventory_drops_overlapping_prompt_fragment_counts() -> None:
    task_spec = SceneTaskSpec(
        room_type="living room",
        style="standard",
        required_large_objects=["sofa_chair"] * 4,
        required_small_objects=["glass_bowl"] * 2,
    )
    compiler = _compiler_with_responses([_response(_semantic_scope("prompt:0"))])

    result = compiler.compile(
        "A room with four sofa chairs and two glass bowls.", task_spec=task_spec
    )

    required = {
        row["subjects"]["category"]: row["subjects"]["count"]
        for row in result["constraints"]
        if row["relation"] == "required_count"
    }
    assert required == {"glass_bowl": 2, "sofa_chair": 4}


def test_intent_compiler_retries_parseable_length_response() -> None:
    compiler = _compiler_with_responses(
        [
            _response('{"constraints": []}', finish_reason="length"),
            _response(_semantic_scope("prompt:0")),
        ]
    )

    result = compiler.compile("A room with a desk.")

    assert result["retry_count"] == 1
    assert compiler.last_trace["status"] == "retry_ok"
    assert [row["finish_reason"] for row in compiler.last_trace["attempts"]] == [
        "length",
        "stop",
    ]


def test_intent_compiler_retries_warnings_only_response() -> None:
    warning = "Small-object arrangements remain context."
    compiler = _compiler_with_responses(
        [
            _response('{"room_type": "living room", "warnings": ["' + warning + '"]}'),
            _response(_semantic_scope("prompt:0")),
        ]
    )

    result = compiler.compile("An empty living room.")

    assert compiler.last_trace["status"] == "retry_ok"
    assert [row["status"] for row in compiler.last_trace["attempts"]] == [
        "error",
        "retry_ok",
    ]
    assert "semantic IR schema_version" in compiler.last_trace["attempts"][0]["error"]
    assert result["warnings"] == []
    retry_message = compiler._test_calls[1]["messages"][1]["content"]
    assert "semantic IR schema_version" in retry_message


def test_intent_compiler_warnings_only_responses_are_runtime_failure() -> None:
    first_warning = "Table settings remain context."
    second_warning = "Circulation remains context."
    compiler = _compiler_with_responses(
        [
            _response('{"warnings": ["' + first_warning + '"]}'),
            _response('{"warnings": ["' + second_warning + '"]}'),
        ]
    )
    task_spec = SceneTaskSpec(
        room_type="living room",
        style="functional",
        required_large_objects=["sofa", "coffee table"],
    )

    with pytest.raises(IntentCompilationError, match="failed after two attempts"):
        compiler.compile("A living room.", task_spec=task_spec)

    assert compiler.last_trace["status"] == "error"
    assert [row["status"] for row in compiler.last_trace["attempts"]] == [
        "error",
        "error",
    ]


def test_contract_completeness_rejects_missing_task_inventory_count() -> None:
    with pytest.raises(
        IncompleteIntentContractError,
        match="missing authoritative required_count for 'desk'",
    ):
        _validate_contract_completeness(
            {"constraints": []}, {"required_large_objects": ["desk"]}
        )


def test_contract_completeness_rejects_unknown_relation_endpoint() -> None:
    contract = build_intent_contract(
        "A desk near a phantom object.",
        task_spec={"required_large_objects": ["desk"]},
    )
    contract["constraints"].append(
        {
            "relation": "near",
            "subjects": {"category": "desk", "count": 1},
            "targets": {"category": "phantom_object", "count": 1},
            "source": "model_inferred",
        }
    )

    with pytest.raises(
        IncompleteIntentContractError, match="endpoint 'phantom_object'"
    ):
        _validate_contract_completeness(contract, {"required_large_objects": ["desk"]})


def test_contract_completeness_accepts_specific_endpoint_for_generic_inventory() -> (
    None
):
    contract = build_intent_contract(
        "Six chairs.", task_spec={"required_large_objects": ["chair"] * 6}
    )
    contract["constraints"].append(
        {
            "relation": "paired_with",
            "subjects": {"category": "student_chair", "count": 6},
            "targets": {"category": "student_desk", "count": 6},
            "source": "room_ontology",
        }
    )
    contract["constraints"].append(
        {
            "relation": "required_count",
            "subjects": {"category": "student_desk", "count": 6},
            "targets": None,
            "source": "room_ontology",
        }
    )

    _validate_contract_completeness(contract, {"required_large_objects": ["chair"] * 6})


def test_contract_completeness_accepts_stable_noun_from_compound_inventory() -> None:
    contract = {
        "constraints": [
            {
                "relation": "required_count",
                "subjects": {"category": "bowl_of_fruit", "count": 1},
                "targets": None,
                "source": "task_compiler_inventory",
            },
            {
                "relation": "required_count",
                "subjects": {"category": "table", "count": 1},
                "targets": None,
                "source": "task_compiler_inventory",
            },
            {
                "relation": "on_top_of",
                "subjects": {"category": "bowl", "count": 1},
                "targets": {"category": "table", "count": 1},
                "source": "model_inferred",
                "inference_reason": "fruit bowl support",
            },
        ]
    }

    _validate_contract_completeness(
        contract,
        {
            "required_large_objects": ["table"],
            "required_small_objects": ["bowl_of_fruit"],
        },
    )


@pytest.mark.parametrize(
    "anchor",
    [
        "room",
        "wall",
        "floor",
        "ceiling",
        "entrance",
        "door",
        "opening",
        "window",
        "adjacent_wall",
    ],
)
def test_contract_completeness_accepts_environment_anchors(anchor: str) -> None:
    contract = build_intent_contract(
        "A desk.", task_spec={"required_large_objects": ["desk"]}
    )
    contract["constraints"].append(
        {
            "relation": "near",
            "subjects": {"category": "desk", "count": 1},
            "targets": {"category": anchor, "count": 1},
            "source": "model_inferred",
        }
    )

    _validate_contract_completeness(contract, {"required_large_objects": ["desk"]})


def test_intent_compiler_length_exhaustion_is_runtime_failure() -> None:
    compiler = _compiler_with_responses(
        [
            _response('{"constraints": []}', finish_reason="length"),
            _response('{"constraints": []}', finish_reason="length"),
        ]
    )
    task_spec = SceneTaskSpec(
        room_type="office",
        style="functional",
        required_large_objects=["desk", "desk", "wastebasket"],
        required_wall_objects=["clock"],
        required_small_objects=["monitor", "monitor"],
    )

    with pytest.raises(IntentCompilationError, match="failed after two attempts"):
        compiler.compile("An office.", task_spec=task_spec)

    assert compiler.last_trace["status"] == "error"


def test_task_spec_components_replace_composite_table_setting_count() -> None:
    contract = build_intent_contract(
        "A dining table with five complete table settings, each including a plate, "
        "cutlery, and a drinking glass.",
        task_spec={
            "required_large_objects": ["dining table"],
            "required_small_objects": [
                *(["plate"] * 5),
                *(["cutlery"] * 5),
                *(["glass"] * 5),
            ],
        },
    )

    counts = {
        row["subjects"]["category"]: row["subjects"]
        for row in contract["constraints"]
        if row["relation"] == "required_count"
    }
    assert {category: selector["count"] for category, selector in counts.items()} == {
        "cutlery": 5,
        "dining_table": 1,
        "glass": 5,
        "plate": 5,
    }
    assert counts["cutlery"]["quantifier"] == "at_least"
    assert counts["plate"]["quantifier"] == "at_least"
    assert counts["glass"]["quantifier"] == "at_least"


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
def test_intent_compiler_maps_reachability_to_near_not_flanking() -> None:
    interaction = "nightstands should be accessible from the bed"
    compiler = _compiler_with_responses(
        [
            _response(
                '{"constraints": [{"relation": "near", '
                '"subjects": {"category": "nightstand", "count": 2}, '
                '"targets": {"category": "bed", "count": 1}, '
                '"source": "model_inferred", "evidence_span": "", '
                '"inference_reason": "TaskCompiler interaction_constraints: '
                f'{interaction}"}}]}}'
            )
        ]
    )

    result = compiler.compile(
        "A bedroom with a bed and two nightstands.",
        interaction_constraints=[interaction],
    )

    inferred = [
        row for row in result["constraints"] if row["source"] == "model_inferred"
    ]
    assert [row["relation"] for row in inferred] == ["near"]
    assert inferred[0]["subjects"]["category"] == "nightstand"
    assert inferred[0]["targets"]["category"] == "bed"


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
def test_intent_compiler_uses_minimum_for_place_setting_cutlery_support() -> None:
    prompt = (
        "A dining room has table settings for four including plates, cutlery, and "
        "glasses on the dining table."
    )
    compiler = _compiler_with_responses(
        [
            _response(
                '{"constraints": [{"relation": "on_top_of", '
                '"subjects": {"category": "cutlery", "count": 4}, '
                '"targets": {"category": "dining_table", "count": 1}, '
                '"grounding": "prompt:0"}]}'
            )
        ]
    )
    task_spec = SceneTaskSpec(
        room_type="dining_room",
        style="functional",
        required_large_objects=["dining_table"],
        required_small_objects=[
            *(["plate"] * 4),
            *(["cutlery"] * 4),
            *(["glass"] * 4),
        ],
    )

    contract = compiler.compile(prompt, task_spec=task_spec)
    cutlery_support = next(
        row
        for row in contract["constraints"]
        if row["relation"] == "on_top_of" and row["subjects"]["category"] == "cutlery"
    )
    required_cutlery = next(
        row
        for row in contract["constraints"]
        if row["relation"] == "required_count"
        and row["subjects"]["category"] == "cutlery"
    )

    assert cutlery_support["subjects"]["quantifier"] == "minimum"
    assert required_cutlery["subjects"]["quantifier"] == "at_least"


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
def test_intent_compiler_prefers_explicit_prompt_over_task_constraint() -> None:
    inferred = "place the dining table against a wall"
    compiler = _compiler_with_responses(
        [
            _response(
                '{"constraints": ['
                '{"relation": "centered_in_room", '
                '"subjects": {"category": "dining_table", "count": 1}, '
                '"targets": {"category": "room", "count": 1}, '
                '"source": "explicit_prompt", '
                '"evidence_span": "center the dining table in the room"}, '
                '{"relation": "against_wall", '
                '"subjects": {"category": "dining_table", "count": 1}, '
                '"targets": {"category": "wall", "count": 1}, '
                '"source": "model_inferred", "evidence_span": "", '
                '"inference_reason": "TaskCompiler aesthetic_constraints: '
                f'{inferred}"}}]}}'
            )
        ]
    )

    result = compiler.compile(
        "Please center the dining table in the room.",
        aesthetic_constraints=[inferred],
    )

    assert any(row["relation"] == "centered_in_room" for row in result["constraints"])
    assert not any(row["relation"] == "against_wall" for row in result["constraints"])
    assert compiler.last_trace["rejected_task_compiler_constraints"] == [
        {
            "relation": "against_wall",
            "subject_category": "dining_table",
            "reason": "conflicts_with_explicit_prompt",
            "inference_reason": (
                "TaskCompiler aesthetic_constraints: "
                "place the dining table against a wall"
            ),
        }
    ]
    assert compiler.last_trace["unmapped_task_compiler_constraints"] == {
        "interaction_constraints": [],
        "aesthetic_constraints": [],
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


def test_intent_schema_preserves_supported_inventory_reconciliation_reason() -> None:
    relation = {
        "relation": "required_count",
        "subjects": {
            "category": "glass_bowl",
            "count": 3,
            "quantifier": "minimum",
        },
        "targets": None,
        "source": "task_compiler_inventory",
        "inference_reason": "SceneTaskSpec required_small_objects",
        "reconciliation_reason": "disjoint_support_cohort_minimum",
    }

    validated = validate_intent_contract(_contract(relation))

    assert validated["constraints"][0]["reconciliation_reason"] == (
        "disjoint_support_cohort_minimum"
    )


def test_intent_schema_rejects_unknown_reconciliation_reason() -> None:
    relation = {
        "relation": "required_count",
        "subjects": {"category": "glass_bowl", "count": 3},
        "targets": None,
        "source": "task_compiler_inventory",
        "inference_reason": "SceneTaskSpec required_small_objects",
        "reconciliation_reason": "unknown_reason",
    }

    with pytest.raises(ValidationError, match="reconciliation_reason"):
        validate_intent_contract(_contract(relation))


def test_generic_speaker_selector_matches_floor_speaker_assets() -> None:
    selector = {"category": "speaker", "count": 4, "quantifier": "exactly"}
    objects = [
        {
            "id": f"floor_speaker_{index}",
            "category": "floor_speaker",
            "category_norm": "floor_speaker",
            "object_type": "furniture",
            "metadata": {"semantic_name": "floor_speaker"},
        }
        for index in range(4)
    ]

    assert selected_ids(selector, objects) == [
        "floor_speaker_0",
        "floor_speaker_1",
        "floor_speaker_2",
        "floor_speaker_3",
    ]
    assert selector_match_count(selector, objects) == 4


def test_intent_compiler_wire_schema_excludes_free_text_provenance() -> None:
    schema = intent_compiler_wire_json_schema()
    requirement_schema = schema["properties"]["requirements"]["items"]

    assert schema["required"] == ["schema_version", "requirements"]
    assert {"subject_ref", "target_ref", "grounding", "kind"} <= set(
        requirement_schema["properties"]
    )
    assert {"source", "evidence_span", "inference_reason", "required_count"}.isdisjoint(
        requirement_schema["properties"]
    )
    assert "required_count" not in requirement_schema["properties"]["relation"]["enum"]


def test_intent_compiler_wire_schema_requires_nullable_target_ref_field() -> None:
    schema = intent_compiler_wire_json_schema()
    requirement_schema = schema["properties"]["requirements"]["items"]

    assert "target_ref" in requirement_schema["required"]
    assert requirement_schema["properties"]["target_ref"] == {
        "anyOf": [
            {
                "type": "string",
                "pattern": r"^(inventory|anchor):[a-z0-9_]+$",
            },
            {"type": "null"},
        ]
    }


def test_intent_compiler_retries_when_provider_omits_required_target_ref() -> None:
    missing_target_ref = json.dumps(
        {
            "schema_version": INTENT_COMPILER_SEMANTIC_IR_VERSION,
            "requirements": [
                {
                    "requirement_id": "prompt_scope",
                    "kind": "soft_scope",
                    "grounding": "prompt:0",
                    "reason": "non-geometric scope",
                }
            ],
        }
    )
    compiler = _compiler_with_responses(
        [_response(missing_target_ref), _response(_semantic_scope("prompt:0"))]
    )

    result = compiler.compile("A room with a desk.")

    assert result["retry_count"] == 1
    assert (
        "semantic IR requirement omitted target_ref"
        in compiler.last_trace["attempts"][0]["error"]
    )
    assert "Every SemanticIR requirement needs the target_ref field" in (
        compiler._test_calls[1]["messages"][1]["content"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subject_count", "1"),
        ("subject_count", True),
        ("surface_mentions", "desk and wall"),
    ],
)
def test_intent_compiler_retries_when_decoded_semantic_ir_violates_wire_types(
    field: str, value: object
) -> None:
    task_spec = SceneTaskSpec(
        room_type="office", style="standard", required_large_objects=["desk"]
    )
    malformed_requirement = {
        "requirement_id": "desk_by_wall",
        "kind": "relation",
        "grounding": "prompt:0",
        "relation": "near",
        "subject_ref": "inventory:desk",
        "target_ref": "anchor:wall",
        field: value,
    }
    compiler = _compiler_with_responses(
        [
            _response(_semantic_ir([malformed_requirement])),
            _response(_semantic_scope("prompt:0")),
        ]
    )

    result = compiler.compile("An office with a desk.", task_spec=task_spec)

    assert result["retry_count"] == 1
    assert compiler.last_trace["status"] == "retry_ok"
    assert (
        "semantic IR wire schema validation failed"
        in compiler.last_trace["attempts"][0]["error"]
    )


def test_intent_compiler_expands_grounding_ids_deterministically() -> None:
    prompt = "Place a desk near the wall. Keep the entrance clear."
    task_spec = {
        "interaction_constraints": ["desk should remain reachable"],
        "aesthetic_constraints": ["use a modern palette"],
    }
    catalog, rendered = _grounding_catalog(prompt, task_spec)

    assert catalog == {
        "prompt:0": "Place a desk near the wall.",
        "prompt:1": "Keep the entrance clear.",
        "interaction:0": "desk should remain reachable",
        "aesthetic:0": "use a modern palette",
    }
    assert "interaction:0" in rendered
    payload = _attach_grounding_provenance(
        {
            "constraints": [
                {"relation": "near", "grounding": "prompt:0"},
                {"relation": "near", "grounding": "interaction:0"},
            ]
        },
        catalog,
    )
    explicit, inferred = payload["constraints"]
    assert (
        explicit
        | {
            "source": "explicit_prompt",
            "evidence_span": "Place a desk near the wall.",
            "inference_reason": "",
        }
        == explicit
    )
    assert (
        inferred
        | {
            "source": "model_inferred",
            "evidence_span": "",
            "inference_reason": (
                "TaskCompiler interaction_constraints: desk should remain reachable"
            ),
        }
        == inferred
    )
    assert "grounding" not in explicit

    with pytest.raises(ValueError, match="unknown grounding id"):
        _attach_grounding_provenance(
            {"constraints": [{"relation": "near", "grounding": "prompt:9"}]},
            catalog,
        )


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
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


def test_intent_compiler_rejects_ungrounded_semantic_relations() -> None:
    invented = (
        '{"constraints": [{"relation": "flanking", '
        '"subjects": {"category": "nightstand", "count": 2}, '
        '"targets": {"category": "bed", "count": 1}, '
        '"source": "explicit_prompt", '
        '"evidence_span": "two nightstands"}]}'
    )
    compiler = _compiler_with_responses([_response(invented), _response(invented)])

    with pytest.raises(IntentCompilationError, match="failed after two attempts"):
        compiler.compile(
            "A bedroom with a bed, two nightstands, and a wardrobe in the corner."
        )

    assert compiler.last_trace["status"] == "error"
    assert [item["status"] for item in compiler.last_trace["attempts"]] == [
        "error",
        "error",
    ]


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
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


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
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


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
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


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
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
    assert compiler.last_trace["status"] == "ok_enriched"
    assert compiler.last_trace["restored_fields"] == [
        {
            "relation": "edge_distribution",
            "subject_category": "dining_chair",
            "fields": ["source", "edge_frame", "groups", "orientation"],
        }
    ]


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
def test_intent_compiler_corrects_unique_edge_shape_and_drops_duplicate_faces() -> None:
    prompt = (
        "Arrange seven office chairs around one rectangular conference table: "
        "three along each long side and one on one short side, all facing the table."
    )
    compiler = _compiler_with_responses(
        [
            _response(
                '{"constraints": ['
                '{"relation": "edge_distribution", '
                '"subjects": {"category": "office_chair", "count": 6}, '
                '"targets": {"category": "conference_table", "count": 1}, '
                '"edge_frame": "target_local_rectangle", '
                '"groups": [{"edge_class": "long", "counts_per_edge": [3, 3]}], '
                '"orientation": "toward_target", '
                '"grounding": "prompt:0"}, '
                '{"relation": "faces", '
                '"subjects": {"category": "office_chair", "count": 7}, '
                '"targets": {"category": "conference_table", "count": 1}, '
                '"grounding": "prompt:0"}]}'
            )
        ]
    )

    result = compiler.compile(prompt)

    edge = next(
        row for row in result["constraints"] if row["relation"] == "edge_distribution"
    )
    assert edge["subjects"]["count"] == 7
    assert [group["counts_per_edge"] for group in edge["groups"]] == [[3, 3], [1, 0]]
    assert not any(row["relation"] == "faces" for row in result["constraints"])
    restored = compiler.last_trace["restored_fields"]
    assert {"subjects.count", "groups"}.issubset(restored[0]["fields"])


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
def test_intent_compiler_rejects_wall_center_from_table_side_grounding() -> None:
    compiler = _compiler_with_responses(
        [
            _response(
                '{"constraints": [{"relation": "centered_on_wall", '
                '"subjects": {"category": "chair", "count": 1}, '
                '"targets": {"category": "wall", "count": 1}, '
                '"grounding": "prompt:0"}]}'
            ),
            _response('{"constraints": []}'),
        ]
    )

    result = compiler.compile("Place one chair centered along one table side.")

    assert compiler.last_trace["status"] == "retry_ok"
    assert "centered_on_wall hard intent" in compiler.last_trace["attempts"][0]["error"]
    assert not any(
        row["relation"] == "centered_on_wall" for row in result["constraints"]
    )


@pytest.mark.parametrize(
    ("prompt", "subject", "target"),
    [
        (
            "There is a utility cart near the table with three plates inside.",
            "plate",
            "utility_cart",
        ),
        (
            "A small cooler is holding utensils and napkins.",
            "utensil",
            "cooler",
        ),
    ],
)
@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
def test_intent_compiler_rejects_top_support_for_containment_wording(
    prompt: str,
    subject: str,
    target: str,
) -> None:
    invalid = (
        '{"constraints": [{"relation": "on_top_of", '
        f'"subjects": {{"category": "{subject}", "count": 1}}, '
        f'"targets": {{"category": "{target}", "count": 1}}, '
        '"grounding": "prompt:0"}]}'
    )
    compiler = _compiler_with_responses(
        [_response(invalid), _response('{"constraints": []}')]
    )

    result = compiler.compile(prompt)

    assert compiler.last_trace["status"] == "retry_ok"
    assert "containment wording" in compiler.last_trace["attempts"][0]["error"]
    assert not any(row["relation"] == "on_top_of" for row in result["constraints"])
    assert any(
        row["kind"] == "unsupported_relation" and row["normalized"] == "containment"
        for row in result["coverage_requirements"]
    )


def test_room_endpoint_is_not_reported_as_unsupported_object_containment() -> None:
    contract = build_intent_contract("A desk with two chairs inside the room.")

    assert not any(
        row["kind"] == "unsupported_relation"
        for row in contract["coverage_requirements"]
    )


def test_room_endpoint_does_not_hide_earlier_container_coverage() -> None:
    contract = build_intent_contract(
        "A cabinet holding books, with two chairs inside the room."
    )

    containment = [
        row
        for row in contract["coverage_requirements"]
        if row["kind"] == "unsupported_relation" and row["normalized"] == "containment"
    ]
    assert len(containment) == 1
    assert "cabinet holding books" in containment[0]["evidence_span"]


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
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


def test_intent_compiler_schema_exhaustion_is_runtime_failure() -> None:
    invalid = (
        '{"constraints": [{"relation": "centered_in_room", '
        '"subjects": {"category": "conference_table"}, '
        '"source": "explicit_prompt", '
        '"evidence_span": "table centered in the room"}]}'
    )
    compiler = _compiler_with_responses([_response(invalid), _response(invalid)])

    with pytest.raises(IntentCompilationError, match="failed after two attempts"):
        compiler.compile("A bedroom with a bed centered on the main wall.")

    assert compiler.last_trace["status"] == "error"


def test_intent_compiler_does_not_restore_missing_relation_targets() -> None:
    invalid = (
        '{"constraints": [{"relation": "faces", '
        '"subjects": {"category": "sofa"}, '
        '"source": "explicit_prompt", '
        '"evidence_span": "sofa faces the television"}]}'
    )
    compiler = _compiler_with_responses([_response(invalid), _response(invalid)])

    with pytest.raises(IntentCompilationError, match="failed after two attempts"):
        compiler.compile(
            "A living room with a sofa facing a TV stand and television on the opposite wall."
        )

    assert compiler.last_trace["status"] == "error"


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
    stages = {
        row["subjects"]["category"]: row["stage"]
        for row in validated["constraints"]
        if row["subjects"]["category"] in {"alarm_clock", "wastebasket"}
    }
    assert stages == {"alarm_clock": "manipuland", "wastebasket": "furniture"}


def test_room_center_with_table_side_seating_does_not_create_support() -> None:
    contract = build_intent_contract(
        "A table is placed in the middle of the room with two chairs on its long sides."
    )

    assert any(
        row["relation"] == "centered_in_room" and row["subjects"]["category"] == "table"
        for row in contract["constraints"]
    )
    assert not any(row["relation"] == "on_top_of" for row in contract["constraints"])


def test_wall_anchor_does_not_attach_to_structural_adjacency_target() -> None:
    contract = build_intent_contract(
        "A fridge is positioned near the door against the wall."
    )

    assert not any(
        row["relation"] == "against_wall"
        and row["subjects"]["category"] in {"door", "window", "opening"}
        for row in contract["constraints"]
    )


def test_display_cabinet_owns_window_adjacency_and_wall_anchor() -> None:
    contract = build_intent_contract(
        "A display cabinet is positioned next to a window against the wall."
    )

    assert any(
        row["relation"] == "next_to"
        and row["subjects"]["category"] == "display_cabinet"
        and row["targets"]["category"] == "window"
        for row in contract["constraints"]
    )
    assert any(
        row["relation"] == "against_wall"
        and row["subjects"]["category"] == "display_cabinet"
        for row in contract["constraints"]
    )
    assert not any(
        row["relation"] == "against_wall" and row["subjects"]["category"] == "window"
        for row in contract["constraints"]
    )


def test_display_shelf_phrase_does_not_create_monitor_constraints() -> None:
    prompt = "A living room with three display shelves against the wall and no TV."
    task_spec = SceneTaskSpec(
        room_type="living room",
        style="standard",
        required_large_objects=["display_shelf", "display_shelf", "display_shelf"],
    )

    selector = selector_for_phrase("three display shelves")
    assert selector == {"category": "display_shelf", "quantifier": "all", "count": 3}

    contract = build_intent_contract(prompt, task_spec=task_spec)
    display_shelf_rows = [
        row
        for row in contract["constraints"]
        if row.get("subjects", {}).get("category") == "display_shelf"
    ]
    assert any(
        row["relation"] == "required_count" and row["subjects"]["count"] == 3
        for row in display_shelf_rows
    )
    assert any(row["relation"] == "against_wall" for row in display_shelf_rows)
    assert not any(
        row.get("subjects", {}).get("category") == "monitor"
        for row in contract["constraints"]
    )
    assert any(
        row["kind"] == "forbidden_inventory" and row["normalized"] == "television"
        for row in contract["coverage_requirements"]
    )


def test_display_taxonomy_keeps_explicit_monitor_and_rejects_ambiguous_display() -> (
    None
):
    assert selector_for_phrase("a computer display") == {
        "category": "monitor",
        "quantifier": "all",
        "count": 1,
    }
    assert selector_for_phrase("a monitor") == {
        "category": "monitor",
        "quantifier": "all",
        "count": 1,
    }
    assert selector_for_phrase("a display") is None


def test_television_selector_prioritizes_structured_display_compound_identity() -> None:
    adapter_shelf = _record("display_shelf_0", "display", (0.0, 0.0), (1.0, 0.4, 1.2))
    adapter_shelf["metadata"] = {"semantic_name": "display_shelf"}
    true_television = _record("television_0", "television", (1.5, 0.0), (1.0, 0.1, 0.6))
    true_television["metadata"] = {"semantic_name": "television"}

    selector = {"category": "television", "count": 1, "quantifier": "all"}

    assert selected_ids(selector, [adapter_shelf]) == []
    assert selected_ids(selector, [true_television]) == ["television_0"]
    assert selected_ids(selector, [adapter_shelf, true_television]) == ["television_0"]


def test_display_shelf_invalid_semantic_ir_is_runtime_failure() -> None:
    compiler = _compiler_with_responses([_response("not json"), _response("still bad")])
    task_spec = SceneTaskSpec(
        room_type="living room",
        style="standard",
        required_large_objects=["display_shelf", "display_shelf", "display_shelf"],
    )

    with pytest.raises(IntentCompilationError, match="failed after two attempts"):
        compiler.compile(
            "A living room with three display shelves against the wall and no TV.",
            task_spec=task_spec,
        )

    assert compiler.last_trace["status"] == "error"


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


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
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


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
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


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
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


def test_classroom_chair_cardinality_binds_generic_chair_assets() -> None:
    prompt = (
        "A classroom with six student desks, each with a chair. A teacher's "
        "desk sits at the front near the chalkboard."
    )
    contract = build_intent_contract(prompt)
    paired = next(
        row
        for row in contract["constraints"]
        if row["relation"] == "paired_with"
        and row["subjects"]["category"] == "student_chair"
    )
    objects = [
        _record(f"chair_{index}", "chair", (float(index), 0.0), (0.5, 0.5, 0.9))
        for index in range(6)
    ]

    assert paired["subjects"]["count"] == 6
    assert bound_ids(paired["subjects"], objects) == [
        f"chair_{index}" for index in range(6)
    ]


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
def test_intent_compiler_normalizes_chair_one_per_support_to_pairing() -> None:
    prompt = "A classroom with six student desks, each with a chair."
    compiler = _compiler_with_responses(
        [
            _response(
                '{"constraints": [{"relation": "one_per_support", '
                '"subjects": {"category": "chair", "count": 6}, '
                '"targets": {"category": "student_desk", "count": 6}, '
                '"source": "explicit_prompt", '
                '"evidence_span": "six student desks, each with a chair"}]}'
            )
        ]
    )

    result = compiler.compile(prompt)

    assert not any(
        row["relation"] == "one_per_support"
        and row["subjects"]["category"] in {"chair", "student_chair"}
        for row in result["constraints"]
    )
    assert any(
        row["relation"] == "paired_with"
        and row["subjects"]["category"] in {"chair", "student_chair"}
        for row in result["constraints"]
    )


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
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


@pytest.mark.skip(reason=_LEGACY_LIVE_COMPILER_REASON)
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


def test_minimum_subject_relation_accepts_one_witness_among_extra_candidates() -> None:
    sofa = _record("sofa_0", "sofa", (0.0, 1.0), (2.0, 0.8, 0.9))
    near_table = _record("table_0", "table", (0.0, 0.0), (0.8, 0.6, 0.75))
    far_table = _record("table_1", "table", (-2.0, -1.0), (0.8, 0.6, 0.75))
    other_far_table = _record("table_2", "table", (2.0, -1.0), (0.8, 0.6, 0.75))
    constraint = {
        "constraint_id": "minimum_table_next_to_sofa",
        "relation": "next_to",
        "stage": "furniture",
        "strength": "hard",
        "subjects": {
            "category": "table",
            "count": 1,
            "quantifier": "minimum",
        },
        "targets": {"category": "sofa", "count": 1, "quantifier": "all"},
        "source": "explicit_prompt",
        "evidence_span": "another table right of the sofa",
    }
    case_pack = {
        "stage": "furniture",
        "intent_contract": {"constraints": [constraint]},
        "scene_geometry": {"objects": [sofa, near_table, far_table, other_far_table]},
    }

    assert augment_contract_checks(case_pack)
    store = load_geometry(case_pack)
    results = [
        {
            **evaluate_functional_dependency(store, check),
            "evidence": check["evidence"],
        }
        for check in case_pack["checks"]
    ]
    applied = apply_contract_execution_states(case_pack, results)

    relation_rows = [
        row for row in applied if row.get("relation_type") == "generic_near_relation"
    ]
    assert {row["primary_object"] for row in relation_rows} == {
        "table_0",
        "table_1",
        "table_2",
    }
    assert all(row["contract_state"] == "passed" for row in relation_rows)
    assert case_pack["intent_contract"]["execution"][0]["state"] == "passed"


@pytest.mark.parametrize(
    ("passing_count", "expected_state"),
    [(1, "failed"), (2, "passed"), (3, "failed")],
)
def test_bounded_table_relation_counts_parent_cohort_witnesses(
    passing_count: int, expected_state: str
) -> None:
    sofa = _record("sofa_0", "sofa", (0.0, 0.0), (2.0, 0.8, 0.9))
    table_rows = [
        _record("coffee_table_0", "coffee_table", (-0.1, 1.0), (1.0, 0.6, 0.45)),
        _record("coffee_table_1", "coffee_table", (0.1, 1.0), (1.0, 0.6, 0.45)),
        _record("end_table_0", "end_table", (1.5, 0.0), (0.5, 0.5, 0.55)),
    ]
    for index, table in enumerate(table_rows):
        if index >= passing_count:
            table["bbox_world"]["center"][1] = -1.0
            table["bbox_world"]["min"][1] = -1.3
            table["bbox_world"]["max"][1] = -0.7
    if passing_count == 3:
        # Force the right-of-sofa end table into the front cohort so the
        # cardinality guard sees three geometric witnesses for a two-table
        # contract.  The normal scene arrangement keeps this object lateral.
        table_rows[2]["bbox_world"]["center"][1] = 1.0
        table_rows[2]["bbox_world"]["min"][1] = 0.75
        table_rows[2]["bbox_world"]["max"][1] = 1.25
        table_rows[2]["bbox_world"]["center"][0] = 0.18
        table_rows[2]["bbox_world"]["min"][0] = -0.12
        table_rows[2]["bbox_world"]["max"][0] = 0.48
    constraint = {
        "constraint_id": "two_tables_in_front_of_sofa",
        "relation": "in_front_of",
        "stage": "furniture",
        "strength": "hard",
        "subjects": {"category": "table", "count": 2, "quantifier": "all"},
        "targets": {"category": "sofa", "count": 1, "quantifier": "all"},
        "source": "explicit_prompt",
        "evidence_span": "two tables in front of the sofa",
    }
    case_pack = {
        "stage": "furniture",
        "intent_contract": {"constraints": [constraint]},
        "scene_geometry": {"objects": [sofa, *table_rows]},
    }

    results = evaluate_intent_contract_extensions(case_pack)
    relation_rows = [
        row for row in results if row["relation_type"] == "front_axis_alignment"
    ]
    applied = apply_contract_execution_states(case_pack, results)

    assert {row["primary_object"] for row in relation_rows} == {
        "coffee_table_0",
        "coffee_table_1",
        "end_table_0",
    }
    assert sum(row["label"] == "pass" for row in relation_rows) == passing_count
    assert case_pack["intent_contract"]["execution"][0]["state"] == expected_state
    if expected_state == "passed":
        side_row = next(
            row for row in applied if row["primary_object"] == "end_table_0"
        )
        assert side_row["label"] == "unknown"
        assert side_row["scoring_tier"] == "ignored"
        assert side_row["diagnostics"]["relation_subject_candidate"] == "non_witness"


def test_in_front_of_complete_target_cohort_is_not_ambiguous() -> None:
    shelves = [
        _record(
            f"display_shelf_{index}",
            "display_shelf",
            (x, 2.0),
            (1.0, 0.4, 1.6),
            yaw_deg=180.0,
        )
        for index, x in enumerate((-2.0, 0.0, 2.0))
    ]
    sofa = _record("sofa_0", "sofa", (0.0, 1.0), (2.0, 0.8, 0.9))
    constraint = {
        "constraint_id": "sofa_in_front_of_display_shelf_cohort",
        "relation": "in_front_of",
        "stage": "furniture",
        "strength": "hard",
        "subjects": {"category": "sofa", "count": 1, "quantifier": "all"},
        "targets": {
            "category": "display_shelf",
            "count": 3,
            "quantifier": "all",
        },
        "source": "explicit_prompt",
        "evidence_span": "three display shelves with a sofa in front",
    }
    case_pack = {
        "stage": "furniture",
        "intent_contract": {"constraints": [constraint]},
        "scene_geometry": {"objects": [*shelves, sofa]},
    }

    results = evaluate_intent_contract_extensions(case_pack)

    assert not any(row["diagnostics"].get("binding_issue") for row in results)
    axial = [row for row in results if row["relation_type"] == "front_axis_alignment"]
    assert len(axial) == 1
    assert axial[0]["label"] == "pass"
    assert axial[0]["diagnostics"]["candidate_target_ids"] == [
        "display_shelf_0",
        "display_shelf_1",
        "display_shelf_2",
    ]
    assert axial[0]["related_objects"] == [
        "display_shelf_0",
        "display_shelf_1",
        "display_shelf_2",
    ]
    assert axial[0]["diagnostics"]["collective_target_count"] == 3
    assert axial[0]["diagnostics"]["collective_forward_distances_m"] == [1.0] * 3


def test_in_front_of_complete_cohort_does_not_choose_one_passing_member() -> None:
    shelves = [
        _record(
            f"display_shelf_{index}",
            "display_shelf",
            (x, 2.0),
            (1.0, 0.4, 1.6),
            yaw_deg=180.0,
        )
        for index, x in enumerate((-2.0, 0.0, 2.0))
    ]
    # This sofa is in front of the rightmost shelf, but not centered in front
    # of the complete shelf cohort. A per-target ``min(pass)`` implementation
    # would incorrectly accept that one member.
    sofa = _record("sofa_0", "sofa", (2.2, 1.0), (2.0, 0.8, 0.9))
    constraint = {
        "constraint_id": "sofa_in_front_of_offset_display_shelves",
        "relation": "in_front_of",
        "stage": "furniture",
        "strength": "hard",
        "subjects": {"category": "sofa", "count": 1, "quantifier": "all"},
        "targets": {
            "category": "display_shelf",
            "count": 3,
            "quantifier": "all",
        },
        "source": "explicit_prompt",
        "evidence_span": "three display shelves with a sofa in front",
    }
    case_pack = {
        "stage": "furniture",
        "intent_contract": {"constraints": [constraint]},
        "scene_geometry": {"objects": [*shelves, sofa]},
    }

    result = evaluate_intent_contract_extensions(case_pack)[0]

    assert result["label"] == "fail"
    assert result["diagnostics"]["collective_target_selection"] is True
    assert result["diagnostics"]["collective_forward_distances_m"] == [1.0] * 3


def test_in_front_of_unbounded_target_group_remains_ambiguous() -> None:
    shelves = [
        _record(
            f"display_shelf_{index}",
            "display_shelf",
            (x, 2.0),
            (1.0, 0.4, 1.6),
            yaw_deg=180.0,
        )
        for index, x in enumerate((-2.0, 0.0, 2.0))
    ]
    sofa = _record("sofa_0", "sofa", (0.0, 1.0), (2.0, 0.8, 0.9))
    constraint = {
        "constraint_id": "sofa_in_front_of_unbounded_shelves",
        "relation": "in_front_of",
        "stage": "furniture",
        "strength": "hard",
        "subjects": {"category": "sofa", "count": 1, "quantifier": "all"},
        "targets": {"category": "display_shelf", "quantifier": "all"},
        "source": "model_inferred",
        "inference_reason": "The sofa is in front of the shelves.",
    }
    case_pack = {
        "stage": "furniture",
        "intent_contract": {"constraints": [constraint]},
        "scene_geometry": {"objects": [*shelves, sofa]},
    }

    results = evaluate_intent_contract_extensions(case_pack)

    assert len(results) == 1
    assert results[0]["label"] == "fail"
    assert results[0]["diagnostics"]["binding_issue"] == "ambiguous"


def test_in_front_of_incomplete_target_cohort_remains_unresolved() -> None:
    shelves = [
        _record(
            f"display_shelf_{index}",
            "display_shelf",
            (x, 2.0),
            (1.0, 0.4, 1.6),
            yaw_deg=180.0,
        )
        for index, x in enumerate((-2.0, 0.0))
    ]
    sofa = _record("sofa_0", "sofa", (0.0, 1.0), (2.0, 0.8, 0.9))
    constraint = {
        "constraint_id": "sofa_in_front_of_incomplete_shelves",
        "relation": "in_front_of",
        "stage": "furniture",
        "strength": "hard",
        "subjects": {"category": "sofa", "count": 1, "quantifier": "all"},
        "targets": {
            "category": "display_shelf",
            "count": 3,
            "quantifier": "all",
        },
        "source": "explicit_prompt",
        "evidence_span": "three display shelves with a sofa in front",
    }
    case_pack = {
        "stage": "furniture",
        "intent_contract": {"constraints": [constraint]},
        "scene_geometry": {"objects": [*shelves, sofa]},
    }

    results = evaluate_intent_contract_extensions(case_pack)

    assert len(results) == 1
    assert results[0]["label"] == "fail"
    assert results[0]["diagnostics"]["binding_issue"] == "ambiguous"


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


def test_intent_compiler_raises_after_unparseable_responses() -> None:
    compiler = _compiler_with_responses([_response("no json"), _response("still bad")])

    with pytest.raises(IntentCompilationError, match="failed after two attempts"):
        compiler.compile("A room with a desk.")

    assert compiler.last_trace["status"] == "error"
    assert compiler.last_trace["retry_count"] == 1
    assert [item["status"] for item in compiler.last_trace["attempts"]] == [
        "error",
        "error",
    ]


def test_intent_compiler_raises_after_transport_errors() -> None:
    compiler = _compiler_with_responses(
        [ConnectionError("temporary outage"), ConnectionError("temporary outage")]
    )

    with pytest.raises(IntentCompilationError, match="failed after two attempts"):
        compiler.compile(
            "A living room with a sofa facing a TV stand and television on the opposite wall."
        )

    assert compiler.last_trace["status"] == "error"
    assert compiler.last_trace["failure_reason"] == "ConnectionError: temporary outage"
    assert [item["status"] for item in compiler.last_trace["attempts"]] == [
        "error",
        "error",
    ]


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


def test_intent_compiler_config_uses_deterministic_contract_without_llm(
    monkeypatch, tmp_path
) -> None:
    class UnexpectedCompiler:
        def __init__(self, **_kwargs):
            raise AssertionError("disabled intent compiler must not construct IntentCompiler")

    monkeypatch.setattr(hooks, "IntentCompiler", UnexpectedCompiler)

    contract, trace = hooks._compile_intent_contract_if_enabled(
        prompt="A living room with a sofa facing a TV stand.",
        scene_id=0,
        output_dir=tmp_path,
        cfg_dict={
            "scenebenchmark_critic": {
                "enabled": True,
                "intent_compiler": {"enabled": False},
            }
        },
    )

    assert trace["status"] == "disabled"
    assert trace["mode"] == "deterministic"
    assert contract["prompt"] == "A living room with a sofa facing a TV stand."


def test_intent_compiler_cache_uses_prompt_task_constraints_and_spec(
    monkeypatch, tmp_path
) -> None:
    calls: list[tuple[str, SceneTaskSpec | None]] = []

    class FakeCompiler:
        SPEC_VERSION = INTENT_COMPILER_SPEC_VERSION
        SCHEMA_VERSION = INTENT_CONTRACT_SCHEMA_VERSION

        def __init__(self, **_kwargs):
            self.last_trace = {}

        def compile(
            self,
            prompt: str,
            task_spec: SceneTaskSpec | None = None,
        ) -> dict:
            calls.append((prompt, task_spec))
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
    first_spec = SceneTaskSpec(
        room_type="room",
        style="standard",
        interaction_constraints=["keep the desk reachable"],
    )
    second_spec = first_spec.model_copy(
        update={"aesthetic_constraints": ["center the desk in the room"]}
    )
    hooks._compile_intent_contract_if_enabled(
        prompt="A room with a desk.",
        scene_id=0,
        output_dir=tmp_path,
        cfg_dict=cfg,
        task_spec=first_spec,
    )
    hooks._compile_intent_contract_if_enabled(
        prompt="A room with a desk.",
        scene_id=0,
        output_dir=tmp_path,
        cfg_dict=cfg,
        task_spec=first_spec,
    )
    hooks._compile_intent_contract_if_enabled(
        prompt="A room with a desk.",
        scene_id=0,
        output_dir=tmp_path,
        cfg_dict=cfg,
        task_spec=second_spec,
    )

    assert calls == [
        ("A room with a desk.", first_spec),
        ("A room with a desk.", second_spec),
    ]


def test_enabled_hook_runner_compiles_task_spec_before_intent(
    monkeypatch, tmp_path
) -> None:
    events: list[str] = []
    task_spec = SceneTaskSpec(
        room_type="bedroom",
        style="modern",
        interaction_constraints=["nightstands should flank the bed"],
        aesthetic_constraints=["keep the layout balanced"],
    )

    class FakeTaskCompiler:
        def __init__(self, **_kwargs):
            self.last_trace = {}

        def compile(self, prompt: str) -> SceneTaskSpec:
            events.append(f"task:{prompt}")
            return task_spec

    class IntentReached(RuntimeError):
        pass

    def fake_compile_intent(**kwargs):
        events.append("intent")
        assert kwargs["task_spec"] is task_spec
        raise IntentReached

    monkeypatch.setattr(hooks, "TaskCompiler", FakeTaskCompiler)
    monkeypatch.setattr(
        hooks, "_compile_intent_contract_if_enabled", fake_compile_intent
    )
    cfg = {
        "experiment": {"scene_expert": {"enabled": True, "mode": "harness_only"}},
        "furniture_agent": {"openai": {"model": "test-model"}},
        "scenebenchmark_critic": {"enabled": True},
    }

    with pytest.raises(IntentReached):
        hooks.build_hook_runner(
            prompt="A bedroom with a bed.",
            scene_id=0,
            output_dir=tmp_path,
            cfg_dict=cfg,
        )

    assert events == ["task:A bedroom with a bed.", "intent"]


def test_enabled_hook_runner_retains_normalized_critic_config(
    monkeypatch, tmp_path
) -> None:
    task_spec = SceneTaskSpec(room_type="bedroom", style="standard")

    class FakeTaskCompiler:
        def __init__(self, **_kwargs):
            self.last_trace = {}

        def compile(self, _prompt: str) -> SceneTaskSpec:
            return task_spec

    monkeypatch.setattr(hooks, "TaskCompiler", FakeTaskCompiler)
    monkeypatch.setattr(
        hooks,
        "apply_behavior_template",
        lambda *_args, **_kwargs: (task_spec, None),
    )
    monkeypatch.setattr(
        hooks,
        "_compile_intent_contract_if_enabled",
        lambda **_kwargs: ({}, {}),
    )
    monkeypatch.setattr(hooks, "GlobalPlanner", lambda **_kwargs: object())
    monkeypatch.setattr(
        hooks,
        "SceneExpertHookRunner",
        lambda **kwargs: kwargs,
    )

    runner_args = hooks.build_hook_runner(
        prompt="A bedroom with a bed.",
        scene_id=0,
        output_dir=tmp_path,
        cfg_dict={
            "experiment": {"scene_expert": {"enabled": True, "mode": "harness_only"}},
            "furniture_agent": {"openai": {"model": "test-model"}},
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["visual_clearance"],
            },
        },
    )

    assert isinstance(runner_args, dict)
    assert runner_args["critic_config"].enabled
    assert runner_args["critic_config"].metric_enabled("visual_clearance")


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
