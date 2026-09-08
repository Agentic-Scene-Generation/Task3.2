"""V5-shaped regressions: partial experience is not a whole-scene replay.

These are reduced CPU fixtures, not a model run or evidence of causal gain.
"""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from scenesmith.scene_expert.global_planner import (
    _SYSTEM_PROMPT,
    _format_memory_for_prompt,
)
from scenesmith.scene_expert.memory.adaptation import decision_applicability
from scenesmith.scene_expert.memory.contracts import selection_from_record
from scenesmith.scene_expert.memory.injection import build_memory_injection_bundle
from scenesmith.scene_expert.memory.schemas import (
    FailureCase,
    SpatialRelationMemory,
    SuccessCase,
)
from scenesmith.scene_expert.memory.usage import target_observations
from scenesmith.scene_expert.schemas import (
    MemoryAdaptation,
    MemoryPack,
    SceneTaskSpec,
    StageBrief,
    StageRelationContext,
)


def fixture(tmp_path, *, kind="success"):
    # The old case attaches wardrobe, room, and inventory rows to bed advice.
    relations = [
        SpatialRelationMemory(
            relation_type="against_wall", subject_role="bed", target_role="wall"
        ),
        SpatialRelationMemory(
            relation_type="corner_of_room", subject_role="wardrobe", target_role="room"
        ),
        SpatialRelationMemory(
            relation_type="required_count", subject_role="nightstand"
        ),
    ]
    if kind == "success":
        record = SuccessCase(
            case_id="bed-example",
            room_type="bedroom",
            stage="furniture",
            successful_pattern=[
                "Prioritize the opening-free wall for the bed headboard."
            ],
            spatial_relations=relations,
        )
    else:
        record = FailureCase(
            failure_id="access-example",
            room_type="bedroom",
            stage="furniture",
            object="nightstand",
            failure_type="access",
            bad_pattern="Blocked front access",
            failure_reason="Nightstand front access occluded",
            repair_action="Keep a clear approach",
            spatial_relations=relations,
        )
    source = selection_from_record(record, rank=1, memory_dir=tmp_path)
    state = {
        "room": {"width_m": 4.0},
        "objects": [
            {
                "object_id": "south_wall",
                "name": "south_wall",
                "category": "",
                "immutable": True,
            }
        ],
    }
    bindings = (
        [
            {"source_role": "bed", "current_role": "bed"},
            {
                "source_role": "wall",
                "current_role": "south_wall",
                "object_ids": ["south_wall"],
            },
        ]
        if kind == "success"
        else [{"source_role": "nightstand", "current_role": "nightstand"}]
    )
    choice = MemoryAdaptation(
        memory_type=source.memory_type,
        memory_id=source.memory_id,
        source_content_hash=source.content_hash,
        decision="adapted",
        source_relation_indices=[0] if kind == "success" else [2],
        bindings=bindings,
        actions=["Use the current anchor and keep access clear."],
        checks=["Recheck current intent and door clearance."],
        preconditions=["The current anchor must be usable."],
    )
    pack = MemoryPack(selections=[source], current_scene_state=state)
    task = SceneTaskSpec(
        room_type="bedroom",
        style="modern",
        required_large_objects=["bed", "nightstand"],
    )
    return source, choice, pack, task


def build(pack, task, choice, context=None):
    return build_memory_injection_bundle(
        stage="furniture",
        memory_pack=pack,
        task_spec=task,
        stage_brief=StageBrief(
            stage="furniture",
            stage_objective="Design the bedroom",
            memory_adaptations=[choice],
        ),
        relation_context=context,
    )


@pytest.mark.parametrize("kind", ["success", "failure"])
def test_scoped_advice_does_not_require_or_render_unrelated_source_inventory(
    tmp_path, kind
):
    source, choice, pack, task = fixture(tmp_path, kind=kind)
    original = deepcopy(source.model_dump())
    result = build(pack, task, choice)
    assert len(result.accepted_items) == 1
    assert "wardrobe" not in result.memory_text
    assert "corner_of_room" not in result.memory_text
    assert source.model_dump() == original
    assert result.accepted_items[0].source.model_dump() == original
    assert (
        result.accepted_items[0].adaptation.source_relation_indices
        == choice.source_relation_indices
    )
    assert "unverified" in result.memory_text  # Legacy geometry booleans are not proof.


def test_legacy_all_relations_scope_is_not_silently_reinterpreted(tmp_path):
    _, choice, pack, task = fixture(tmp_path)
    choice.source_relation_indices = None
    result = build(pack, task, choice)
    assert not result.accepted_items
    assert "missing_role_binding" in result.adaptation_decisions[0]["reasons"]


@pytest.mark.parametrize(
    "scope,reason",
    [
        ([], "empty_spatial_scope"),
        ([7], "unknown_source_relation_index"),
        ([0, 0], "duplicate_source_relation_index"),
    ],
)
def test_scope_cannot_erase_or_invent_evidence(tmp_path, scope, reason):
    _, choice, pack, task = fixture(tmp_path)
    choice.source_relation_indices = scope
    result = build(pack, task, choice)
    assert not result.accepted_items
    assert reason in result.adaptation_decisions[0]["reasons"]


@pytest.mark.parametrize("scope", [[-1], [True], [0.0], ["0"]])
def test_scope_schema_rejects_ambiguous_indices(tmp_path, scope):
    _, choice, _, _ = fixture(tmp_path)
    with pytest.raises(ValidationError):
        MemoryAdaptation.model_validate(
            choice.model_dump() | {"source_relation_indices": scope}
        )


def test_selected_relation_still_requires_real_anchor_binding(tmp_path):
    _, choice, pack, task = fixture(tmp_path)
    choice.bindings = choice.bindings[:1]
    result = build(pack, task, choice)
    assert not result.accepted_items
    assert "missing_role_binding" in result.adaptation_decisions[0]["reasons"]


def test_scope_cannot_override_explicit_rejection_or_changed_source(tmp_path):
    _, choice, pack, task = fixture(tmp_path)
    choice.decision = "rejected"
    assert build(pack, task, choice).adaptation_decisions[0]["reasons"] == [
        "planner_rejected"
    ]
    choice.decision = "adapted"
    choice.source_content_hash = "changed"
    assert (
        "source_hash_mismatch"
        in build(pack, task, choice).adaptation_decisions[0]["reasons"]
    )


def test_scoped_relation_is_the_only_target_that_can_get_outcome_credit(tmp_path):
    _, choice, pack, task = fixture(tmp_path)
    item = build(pack, task, choice).accepted_items[0]
    constraints = [
        {
            "constraint_id": "bed-wall",
            "relation": "against_wall",
            "subjects": {"category": "bed"},
            "targets": {"category": "wall"},
        },
        {
            "constraint_id": "wardrobe-corner",
            "relation": "corner_of_room",
            "subjects": {"category": "wardrobe"},
            "targets": {"category": "room"},
        },
    ]
    observations = target_observations(
        item.model_dump(),
        "furniture",
        {"relation_context": {"hard_constraints": constraints}},
    )
    assert [row["constraint_id"] for row in observations] == ["bed-wall"]


def test_delivery_still_checks_bound_wall_and_does_not_reuse_excluded_relations(
    tmp_path,
):
    _, choice, pack, task = fixture(tmp_path)
    item = build(pack, task, choice).accepted_items[0]

    def reasons(state, query, repair=True):
        return decision_applicability(
            item,
            stage="furniture",
            source_state=pack.current_scene_state,
            current_state=state,
            query=query,
            repair=repair,
            context=None,
        )

    assert not reasons(
        pack.current_scene_state, "Correct the bed against_wall placement"
    )
    assert "repair_issue_mismatch" in reasons(
        pack.current_scene_state, "Correct wardrobe corner_of_room"
    )
    changed = deepcopy(pack.current_scene_state)
    changed["objects"] = []
    assert "bound_object_missing" in reasons(changed, "Fix the bed")


def test_hard_contract_conflicts_are_not_bypassed_by_scope(tmp_path):
    source, choice, pack, task = fixture(tmp_path)
    source.spatial_relations[0]["relation_type"] = "faces"
    source.spatial_relations[0]["target_role"] = "table"
    # An excluded conflicting row is still blocked by the conservative source guard.
    choice.source_relation_indices = [2]
    context = StageRelationContext(
        stage="furniture",
        hard_constraints=[
            {
                "constraint_id": "current",
                "relation": "faces_away_from",
                "subjects": {"category": "bed"},
                "targets": {"category": "table"},
            }
        ],
    )
    assert not build(pack, task, choice, context).accepted_items


def test_planner_sees_original_relation_indices_and_explicit_scope_instructions(
    tmp_path,
):
    import json

    _, _, pack, _ = fixture(tmp_path)
    rows = json.loads(_format_memory_for_prompt(pack))[0]["spatial_relations"]
    assert [row["relation_index"] for row in rows] == [0, 1, 2]
    assert "source_relation_indices" in _SYSTEM_PROMPT
    assert "Bind EVERY" in _SYSTEM_PROMPT


def test_nonspatial_legacy_memory_still_has_an_explicit_text_only_path(tmp_path):
    source, choice, pack, task = fixture(tmp_path)
    source.spatial_relations = []
    choice.source_relation_indices = []
    assert build(pack, task, choice).accepted_items
