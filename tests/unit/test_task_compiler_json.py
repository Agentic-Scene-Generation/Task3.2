"""Regression tests for TaskCompiler's resilient JSON parsing."""

from scenesmith.scene_expert.schemas import SceneTaskSpec
from scenesmith.scene_expert.task_compiler import (
    _SYSTEM_PROMPT,
    _extract_json_from_text,
    _fallback_spec_from_prompt,
    _normalize_stage_ownership,
    _repair_zero_target_relation_payloads,
)
from scenesmith.scenebenchmark_critic.intent_contract import build_intent_contract


def test_task_compiler_repairs_a_truncated_optional_constraint() -> None:
    """A complete task spec should survive a model cutoff in optional metadata."""
    raw = """{
      "room_type": "study",
      "style": "standard",
      "required_large_objects": ["desk"],
      "required_wall_objects": [],
      "required_ceiling_objects": [],
      "required_small_objects": [],
      "functional_zones": ["working_zone"],
      "interaction_constraints": [],
      "aesthetic_constraints": [],
      "intent_constraints": [
        {
          "relation": "against_wall",
          "subjects": {"category": "chair"},
          "targets": {"category": "wall"},
          "source": "explicit_prompt",
          "evidence_span": "chair against wall",
          "confidence": 1.0
        },
        {"relation"""

    spec = SceneTaskSpec.model_validate(_extract_json_from_text(raw))

    assert spec.room_type == "study"
    assert spec.required_large_objects == ["desk"]
    assert spec.intent_constraints[0].relation == "against_wall"


def test_task_compiler_requests_hard_typed_intent_constraints() -> None:
    assert '"intent_constraints"' in _SYSTEM_PROMPT
    assert '"source": "explicit_prompt | model_inferred"' in _SYSTEM_PROMPT
    assert "All emitted relations become hard constraints" in _SYSTEM_PROMPT
    assert "floor-standing objects as required_large_objects" in _SYSTEM_PROMPT
    assert 'explicit "X behind Y"' in _SYSTEM_PROMPT
    assert "`entrance` is a virtual" in _SYSTEM_PROMPT


def test_floor_supported_objects_are_owned_by_furniture_stage() -> None:
    spec = SceneTaskSpec.model_validate(
        {
            "room_type": "living room",
            "style": "standard",
            "required_large_objects": ["sofa", "plant"],
            "required_small_objects": ["plant", "plant"],
            "intent_constraints": [
                {
                    "relation": "on_top_of",
                    "subjects": {"category": "plant", "count": 2},
                    "targets": {"category": "floor"},
                    "source": "explicit_prompt",
                    "evidence_span": "two plants on the floor",
                }
            ],
        }
    )

    normalized = _normalize_stage_ownership(spec)

    assert normalized.required_large_objects.count("plant") == 2
    assert "plant" not in normalized.required_small_objects


def test_tv_stand_on_opposite_wall_stays_furniture() -> None:
    spec = SceneTaskSpec.model_validate(
        {
            "room_type": "living room",
            "style": "standard",
            "required_large_objects": ["sofa", "back_wall", "opposite_wall"],
            "required_wall_objects": ["tv_stand"],
            "intent_constraints": [
                {
                    "relation": "on_wall",
                    "subjects": {"category": "tv_stand", "count": 1},
                    "targets": {"category": "opposite_wall"},
                    "source": "explicit_prompt",
                    "evidence_span": "TV stand on the opposite wall",
                }
            ],
        }
    )

    normalized = _normalize_stage_ownership(spec)

    assert normalized.required_large_objects == ["sofa", "tv_stand"]
    assert normalized.required_wall_objects == []
    constraint = normalized.intent_constraints[0]
    assert constraint.relation == "against_wall"
    assert constraint.targets is not None
    assert constraint.targets.category == "opposite_wall"


def test_tv_stand_keeps_furniture_ownership_without_a_relation() -> None:
    spec = SceneTaskSpec.model_validate(
        {
            "room_type": "living room",
            "style": "standard",
            "required_wall_objects": ["tv_stand"],
        }
    )

    normalized = _normalize_stage_ownership(spec)

    assert normalized.required_large_objects == ["tv_stand"]
    assert normalized.required_wall_objects == []


def test_prompt_floor_relation_recovers_model_omission_for_stage_ownership() -> None:
    spec = SceneTaskSpec(
        room_type="living room",
        style="standard",
        required_small_objects=["plant"],
    )

    normalized = _normalize_stage_ownership(
        spec, prompt="A living room with two plants on the floor."
    )

    assert normalized.required_large_objects == ["plant", "plant"]
    assert normalized.required_small_objects == []


def test_room_centered_object_is_owned_by_furniture_stage() -> None:
    spec = SceneTaskSpec.model_validate(
        {
            "room_type": "living room",
            "style": "standard",
            "required_small_objects": ["rug"],
            "intent_constraints": [
                {
                    "relation": "centered_in_room",
                    "subjects": {"category": "rug", "count": 1},
                    "targets": {"category": "room"},
                    "source": "explicit_prompt",
                    "evidence_span": "rug in the middle",
                }
            ],
        }
    )

    normalized = _normalize_stage_ownership(spec)

    assert normalized.required_large_objects == ["rug"]
    assert normalized.required_small_objects == []


def test_fallback_assigns_explicit_floor_objects_to_furniture_stage() -> None:
    spec = _fallback_spec_from_prompt("A living room with two plants on the floor.")

    assert spec.compiler_status == "degraded"
    assert spec.required_large_objects.count("plant") == 2
    assert "plant" not in spec.required_small_objects
    assert any(
        constraint.relation == "on_top_of"
        and constraint.targets is not None
        and constraint.targets.category == "floor"
        for constraint in spec.intent_constraints
    )
    assert "floor" not in spec.required_large_objects


def test_fallback_classroom_keeps_chalkboard_at_wall_stage() -> None:
    spec = _fallback_spec_from_prompt(
        "A classroom with six student desks, each with a chair. A teacher's desk "
        "sits at the front near the chalkboard, which hangs on the wall."
    )

    assert spec.room_type == "classroom"
    assert spec.required_large_objects.count("student_desk") == 6
    assert spec.required_large_objects.count("student_chair") == 6
    assert "chalkboard" in spec.required_wall_objects
    assert "instructional_surface" not in spec.required_large_objects


def test_task_compiler_requests_two_anchor_between_relations() -> None:
    assert "centered_between" in _SYSTEM_PROMPT
    assert "secondary_category" in _SYSTEM_PROMPT
    assert 'ordinary\n  "X between A and B", emit "between"' in _SYSTEM_PROMPT
    assert "common-sense functional relations" in _SYSTEM_PROMPT
    assert "Do not invent coordinates" in _SYSTEM_PROMPT
    assert "Do not cap the number of constraints" in _SYSTEM_PROMPT


def test_fallback_propagates_each_anchor_count_to_distributed_objects() -> None:
    spec = _fallback_spec_from_prompt(
        "A collaborative office with two desks facing each other, an office "
        "chair and computer monitor at each desk, a filing cabinet, and a printer "
        "on the cabinet."
    )

    assert spec.required_large_objects.count("desk") == 2
    assert spec.required_large_objects.count("office_chair") == 2
    assert spec.required_small_objects.count("monitor") == 2
    assert any(
        row.relation == "across_from"
        and row.subjects.category == "desk"
        and row.subjects.count == 2
        and row.targets is not None
        and row.targets.category == "desk"
        and row.targets.count == 2
        for row in spec.intent_constraints
    )
    assert any(
        row.relation == "near"
        and row.subjects.category == "monitor"
        and row.subjects.count == 2
        and row.targets is not None
        and row.targets.category == "desk"
        and row.targets.count == 2
        for row in spec.intent_constraints
    )


def test_fallback_recovers_specific_reception_inventory_and_stage_ownership() -> None:
    spec = _fallback_spec_from_prompt(
        "A reception room with a reception desk against the back wall, an office "
        "chair behind it, two guest chairs facing the desk, a side table between "
        "the guest chairs, a filing cabinet in a corner, and a brochure holder on "
        "top of the reception desk. Keep an open route from the entrance to the "
        "desk and chairs."
    )

    assert spec.required_large_objects.count("reception_desk") == 1
    assert spec.required_large_objects.count("office_chair") == 1
    assert spec.required_large_objects.count("guest_chair") == 2
    assert spec.required_large_objects.count("side_table") == 1
    assert "entrance" not in spec.required_large_objects
    assert spec.required_small_objects == ["brochure_holder"]
    assert any(
        row.relation == "behind"
        and row.subjects.category == "office_chair"
        and row.targets is not None
        and row.targets.category == "reception_desk"
        for row in spec.intent_constraints
    )
    assert any(
        row.relation == "between"
        and row.subjects.category == "side_table"
        and row.targets is not None
        and row.targets.category == "guest_chair"
        and row.targets.count == 2
        and row.targets.secondary_category == "guest_chair"
        and row.targets.secondary_count == 2
        for row in spec.intent_constraints
    )
    assert {
        row.targets.category
        for row in spec.intent_constraints
        if row.relation == "clear_access"
        and row.subjects.category == "entrance"
        and row.targets is not None
    } == {"desk", "guest_chair"}
    assert any(
        row.relation == "on_top_of"
        and row.subjects.category == "brochure_holder"
        and row.targets is not None
        and row.targets.category == "reception_desk"
        for row in spec.intent_constraints
    )


def test_model_reception_contract_normalizes_repeated_anchors_and_virtual_entrance() -> (
    None
):
    spec = SceneTaskSpec.model_validate(
        {
            "room_type": "reception room",
            "style": "professional",
            "required_large_objects": [
                "reception desk",
                "office chair",
                "guest chair",
                "guest chair",
                "side table",
            ],
            "intent_constraints": [
                {
                    "relation": "behind",
                    "subjects": {"category": "office chair", "count": 1},
                    "targets": {"category": "reception desk"},
                    "source": "explicit_prompt",
                    "evidence_span": "office chair behind it",
                },
                {
                    "relation": "between",
                    "subjects": {"category": "side table", "count": 1},
                    "targets": {
                        "category": "guest chair",
                        "secondary_category": "guest chair",
                    },
                    "source": "explicit_prompt",
                    "evidence_span": "side table between the guest chairs",
                },
                {
                    "relation": "clear_access",
                    "subjects": {"category": "entrance", "count": 1},
                    "targets": {"category": "guest chair"},
                    "source": "explicit_prompt",
                    "evidence_span": "open route from the entrance to the chairs",
                },
            ],
        }
    )

    normalized = _normalize_stage_ownership(spec)

    assert "entrance" not in normalized.required_large_objects
    between = next(
        row for row in normalized.intent_constraints if row.relation == "between"
    )
    assert between.targets is not None
    assert between.targets.count == 2
    assert between.targets.secondary_count == 2
    route = next(
        row for row in normalized.intent_constraints if row.relation == "clear_access"
    )
    assert route.targets is not None
    assert route.targets.count == 2


def test_all_pairing_propagates_count_and_promotes_generic_family() -> None:
    prompt = (
        "A classroom with six student desks, each with a chair. A teacher's desk "
        "sits at the front near the chalkboard."
    )
    spec = SceneTaskSpec.model_validate(
        {
            "room_type": "classroom",
            "style": "standard",
            "required_large_objects": [
                *(["student_desk"] * 6),
                "student_chair",
                *(["chair"] * 5),
            ],
            "intent_constraints": [],
        }
    )

    normalized = _normalize_stage_ownership(spec, prompt=prompt)

    assert normalized.required_large_objects.count("student_desk") == 6
    assert normalized.required_large_objects.count("student_chair") == 6
    assert "chair" not in normalized.required_large_objects


def test_model_inventory_does_not_duplicate_rocking_chair_as_generic_chair() -> None:
    spec = SceneTaskSpec.model_validate(
        {
            "room_type": "nursery",
            "style": "standard",
            "required_large_objects": ["rocking_chair", "chair", "side_table"],
        }
    )

    normalized = _normalize_stage_ownership(
        spec,
        prompt="A nursery with a rocking chair near a small side table.",
    )

    assert normalized.required_large_objects == ["rocking_chair", "side_table"]


def test_model_inventory_keeps_explicit_generic_chair_beside_rocking_chair() -> None:
    spec = SceneTaskSpec.model_validate(
        {
            "room_type": "nursery",
            "style": "standard",
            "required_large_objects": ["rocking_chair", "chair"],
        }
    )

    normalized = _normalize_stage_ownership(
        spec,
        prompt="A nursery with a rocking chair and a chair.",
    )

    assert normalized.required_large_objects == ["rocking_chair", "chair"]


def test_model_inventory_does_not_duplicate_conference_table_from_anaphora() -> None:
    prompt = (
        "A meeting room with one rectangular conference table and three office chairs. "
        "Arrange all three chairs evenly spaced along one long side of the table."
    )
    spec = SceneTaskSpec.model_validate(
        {
            "room_type": "meeting room",
            "style": "professional",
            "required_large_objects": [
                "conference table",
                "office chair",
                "office chair",
                "office chair",
                "table",
            ],
        }
    )

    normalized = _normalize_stage_ownership(spec, prompt=prompt)

    assert normalized.required_large_objects.count("conference table") == 1
    assert normalized.required_large_objects.count("office chair") == 3
    assert "table" not in normalized.required_large_objects


def test_task_compiler_keeps_llm_meeting_inventory_when_required_count_has_target() -> (
    None
):
    """A redundant target must not discard an otherwise complete LLM response."""
    data = {
        "room_type": "meeting room",
        "style": "professional",
        "required_large_objects": [
            "conference table",
            *(["office chair"] * 7),
            "credenza",
        ],
        "required_wall_objects": ["presentation screen"],
        "intent_constraints": [
            {
                "relation": "required_count",
                "subjects": {"category": "office chair", "count": 7},
                "targets": {"category": "room"},
                "source": "explicit_prompt",
                "evidence_span": "seven office chairs",
            },
            {
                "relation": "one_per_side",
                "subjects": {"category": "office chair", "count": 6},
                "targets": {"category": "conference table"},
                "source": "explicit_prompt",
                "evidence_span": "six office chairs evenly spaced along the table's two long sides",
            },
            {
                "relation": "centered_in_room",
                "subjects": {"category": "conference table", "count": 1},
                "targets": {},
                "source": "explicit_prompt",
                "evidence_span": "conference table centered in the room",
            },
            {
                "relation": "against_wall",
                "subjects": {"category": "credenza", "count": 1},
                "targets": {"category": "conference table"},
                "source": "explicit_prompt",
                "evidence_span": "credenza against the opposite wall",
            },
        ],
    }

    repaired = _repair_zero_target_relation_payloads(data)
    spec = _normalize_stage_ownership(
        SceneTaskSpec.model_validate(repaired),
        prompt=(
            "A meeting room with one rectangular conference table and seven office "
            "chairs. Arrange six office chairs evenly spaced along the table's two "
            "long sides."
        ),
    )

    assert spec.required_large_objects.count("conference table") == 1
    assert spec.required_large_objects.count("office chair") == 7
    assert "credenza" in spec.required_large_objects
    assert spec.required_wall_objects == ["presentation screen"]
    assert spec.intent_constraints[0].targets is None
    assert [constraint.relation for constraint in spec.intent_constraints] == [
        "required_count",
        "one_per_side",
    ]


def test_task_compiler_preserves_llm_table_edge_topology_as_one_relation() -> None:
    prompt = (
        "A meeting room with one rectangular conference table and seven office chairs. "
        "Arrange six office chairs in two equal groups of three, evenly spaced along "
        "the table's two long sides. Place one remaining office chair centered along "
        "one short side, facing the table. Keep the opposite short side free of chairs."
    )
    data = {
        "room_type": "meeting room",
        "style": "professional",
        "required_large_objects": ["conference table", *("office chair",) * 7],
        "intent_constraints": [
            {
                "relation": "distributed_evenly",
                "subjects": {"category": "office chair", "count": 6},
                "targets": {"category": "conference table", "count": 1},
                "source": "explicit_prompt",
                "evidence_span": (
                    "six office chairs in two equal groups of three, evenly spaced "
                    "along the table's two long sides"
                ),
            },
            {
                "relation": "centered_on_wall",
                "subjects": {"category": "office chair", "count": 1},
                "targets": {"category": "conference table", "count": 1},
                "source": "explicit_prompt",
                "evidence_span": "one remaining office chair centered along one short side",
            },
            {
                "relation": "faces",
                "subjects": {"category": "office chair", "count": 7},
                "targets": {"category": "conference table", "count": 1},
                "source": "explicit_prompt",
                "evidence_span": "all chairs facing the table",
            },
        ],
    }

    spec = _normalize_stage_ownership(
        SceneTaskSpec.model_validate(_repair_zero_target_relation_payloads(data)),
        prompt=prompt,
    )
    contract = build_intent_contract(prompt, task_spec=spec)
    topology = [
        constraint
        for constraint in contract["constraints"]
        if constraint["relation"] == "one_per_side"
        and constraint["targets"]["category"] == "conference_table"
    ]

    assert len(topology) == 1
    assert "two long sides" in topology[0]["evidence_span"]
    assert "one short side" in topology[0]["evidence_span"]
    assert not any(
        constraint["relation"] in {"distributed_evenly", "faces"}
        and constraint["targets"].get("category") == "conference_table"
        for constraint in contract["constraints"]
    )


def test_task_compiler_moves_llm_presentation_screen_to_wall_stage() -> None:
    spec = _normalize_stage_ownership(
        SceneTaskSpec.model_validate(
            {
                "room_type": "meeting room",
                "style": "professional",
                "required_large_objects": ["presentation screen"],
            }
        )
    )

    assert spec.required_large_objects == []
    assert spec.required_wall_objects == ["instructional_surface"]
