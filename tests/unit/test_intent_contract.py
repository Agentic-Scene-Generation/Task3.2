"""Regression tests for prompt-originated, geometry-grounded intent contracts."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pydrake.math import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.house import RoomGeometry
from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID
from scenesmith.scenebenchmark_critic.api import evaluate_room_scene
from scenesmith.scenebenchmark_critic import adapter, evaluator
from scenesmith.scenebenchmark_critic.config import CriticConfig
from scenesmith.scenebenchmark_critic.evaluator import run_case_pack_checks
from scenesmith.scenebenchmark_critic.furniture_relation_repair import (
    improve_furniture_relations,
)
from scenesmith.scenebenchmark_critic.intent_contract import (
    augment_contract_checks,
    bound_ids,
    build_intent_contract,
    contract_relation_requested,
    contract_seating_targets,
    is_hard_constraint,
    original_prompt_for_scene,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
    evaluate_intent_contract_extensions,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.builder import (
    build_checks,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.study_furniture_layout import (
    evaluate_study_furniture_layout,
)
from scenesmith.scenebenchmark_critic.prompt_context import format_agent_prompt_context
from scenesmith.scenebenchmark_critic.reports import (
    build_evaluation_payload,
    format_prompt_context,
)
from scenesmith.scene_expert.task_compiler import _fallback_spec_from_prompt


def _relations(contract: dict) -> set[tuple[str, str, str]]:
    return {
        (
            str(row.get("relation") or ""),
            str((row.get("subjects") or {}).get("category") or ""),
            str((row.get("targets") or {}).get("category") or ""),
        )
        for row in contract.get("constraints") or []
    }


def _explicit_constraint(
    constraint_id: str, relation: str, subject: str, target: str
) -> dict:
    return {
        "constraint_id": constraint_id,
        "relation": relation,
        "subjects": {"category": subject, "count": 1},
        "targets": {"category": target, "count": 1},
        "source": "explicit_prompt",
        "strength": "hard",
    }


def test_room_adapter_builds_v2_case_pack_without_mode_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = SimpleNamespace(
        room_id="training",
        room_type="classroom",
        text_description="An instructional room.",
    )
    monkeypatch.setattr(
        adapter,
        "_room_scene_geometry",
        lambda _scene: {"rooms": [], "objects": [], "relations": []},
    )
    monkeypatch.setattr(
        adapter,
        "attach_intent_contract_to_case_pack",
        lambda _scene, case_pack: case_pack.setdefault(
            "intent_contract", {"constraints": []}
        ),
    )

    def capture(case_pack: dict, metrics: object = None) -> list[dict]:
        assert "intent_contract_mode" not in case_pack
        assert "mode" not in case_pack["intent_contract"]
        return []

    monkeypatch.setattr(adapter, "build_all_checks", capture)

    case_pack = adapter.room_scene_to_case_pack(scene)

    assert case_pack["schema_version"] == "scenesmith.scenebenchmark_critic.v2"
    assert "_checks_built_intent_contract_mode" not in case_pack


def test_house_adapter_builds_v2_case_pack_without_mode_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    house = SimpleNamespace(
        rooms={},
        layout=SimpleNamespace(house_prompt="A shared instructional building."),
    )

    def capture(case_pack: dict, metrics: object = None) -> list[dict]:
        assert "intent_contract_mode" not in case_pack
        return []

    monkeypatch.setattr(adapter, "build_all_checks", capture)

    case_pack = adapter.house_scene_to_case_pack(house)

    assert case_pack["schema_version"] == "scenesmith.scenebenchmark_critic.v2"
    assert "_checks_built_intent_contract_mode" not in case_pack


@pytest.mark.parametrize("removed_field", ("constraint_mode", "intent_contract_mode"))
def test_old_constraint_mode_config_fails_fast(removed_field: str) -> None:
    with pytest.raises(ValueError, match=f"{removed_field} was removed"):
        evaluator._coerce_config({"scenebenchmark_critic": {removed_field: "shadow"}})


def test_old_constraint_mode_environment_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRITIC_CONSTRAINT_MODE", "shadow")

    with pytest.raises(ValueError, match="CRITIC_CONSTRAINT_MODE was removed"):
        evaluator._coerce_config({"scenebenchmark_critic": {"enabled": True}})


@pytest.mark.parametrize(
    ("prompt", "expected"),
    (
        (
            "A bedroom with a bed, two nightstands, and a wardrobe in the corner of the room.",
            {("required_count", "bed", ""), ("against_wall", "bed", "wall")},
        ),
        (
            "A living room with a two-seater sofa against the wall, a square rug in "
            "the middle in front of the sofa, and two large plants on the floor near "
            "the sofa.",
            {("against_wall", "sofa", "wall")},
        ),
        (
            "A classroom with six student desks, each with a chair. A teacher's desk "
            "sits at the front near the chalkboard, which hangs on the wall.",
            {
                ("paired_with", "student_chair", "student_desk"),
            },
        ),
        (
            "A bedroom featuring rustic farmhouse decor with exposed wooden beams.",
            {("required_count", "bed", ""), ("against_wall", "bed", "wall")},
        ),
        (
            "A living room with a sofa against the back wall facing a TV stand and "
            "television on the opposite wall, a coffee table centered between the sofa "
            "and TV stand, two armchairs flanking the coffee table near each end of the "
            "sofa, and a floor lamp beside one armchair.",
            {
                ("against_wall", "sofa", "wall"),
                ("centered_between", "coffee_table", "sofa"),
                ("faces", "sofa", "tv_stand"),
                ("flanking", "armchair", "coffee_table"),
            },
        ),
        (
            "A study with a desk centered against the back wall, an office chair tucked "
            "under the desk, a computer monitor on the desk, two guest chairs against "
            "the side wall facing the desk, and a bookshelf on the adjacent wall.",
            {
                ("centered_on_wall", "desk", "wall"),
                ("aligned_with", "office_chair", "desk"),
                ("against_wall", "guest_chair", "wall"),
                ("against_wall", "bookshelf", "wall"),
            },
        ),
        (
            "A bedroom with a bed centered on the main wall, a nightstand with a table "
            "lamp on each side of the bed, a dresser against the opposite wall directly "
            "facing the bed, and a wardrobe placed next to the dresser.",
            {
                ("centered_on_wall", "bed", "wall"),
                ("faces", "dresser", "bed"),
                ("flanking", "nightstand", "bed"),
                ("near", "wardrobe", "dresser"),
            },
        ),
        (
            "A dining room with a dining table in the center, four dining chairs arranged "
            "around it with one on each side, a sideboard against the wall behind the "
            "chairs on one side, and table settings for four.",
            {
                ("centered_in_room", "dining_table", "room"),
                ("one_per_side", "dining_chair", "dining_table"),
                ("against_wall", "sideboard", "wall"),
            },
        ),
    ),
)
def test_expectation_prompts_compile_to_generic_contracts(
    prompt: str, expected: set[tuple[str, str, str]]
) -> None:
    contract = build_intent_contract(prompt)

    assert expected <= _relations(contract)
    expected_rows = [
        row
        for row in contract["constraints"]
        if (
            row["relation"],
            row["subjects"].get("category", ""),
            row["targets"].get("category", ""),
        )
        in expected
    ]
    assert expected_rows
    assert all(is_hard_constraint(row) for row in expected_rows)


def test_parser_avoids_case_shaped_false_constraints() -> None:
    gallery = build_intent_contract(
        "A gallery with a cabinet in the center of the room."
    )
    dining = build_intent_contract(
        "A dining room with four dining chairs near a wall and a table by the window."
    )
    classroom = build_intent_contract(
        "A classroom with six student desks, each with a chair, and a projector screen."
    )
    bedside = build_intent_contract(
        "A bedroom with a nightstand with a table lamp on each side of the bed."
    )

    assert ("against_wall", "cabinet", "wall") not in _relations(gallery)
    assert ("one_per_side", "dining_chair", "dining_table") not in _relations(dining)
    assert ("against_wall", "teacher_desk", "wall") not in _relations(classroom)
    assert ("paired_with", "student_chair", "student_desk") in _relations(classroom)
    assert not any(
        relation == "distributed_evenly"
        for relation, _subject, _target in _relations(classroom)
    )
    assert ("on_top_of", "nightstand", "bed") not in _relations(bedside)
    assert ("flanking", "nightstand", "bed") in _relations(bedside)


def test_classroom_distribution_requires_explicit_layout_language() -> None:
    contract = build_intent_contract(
        "A classroom with six student desks, each with a chair. Arrange the "
        "student desks in rows facing the front."
    )

    assert ("paired_with", "student_chair", "student_desk") in _relations(contract)
    assert (
        "distributed_evenly",
        "student_desk",
        "student_chair",
    ) in _relations(contract)


def test_checkpoint_prompt_fallback_excludes_injected_layout_language() -> None:
    original_prompt = (
        "A classroom with six student desks, each with a chair. A teacher's desk "
        "sits at the front near the chalkboard."
    )
    scene = type(
        "CheckpointScene",
        (),
        {
            "text_description": original_prompt
            + "\n\n=== SceneExpert Stage Brief: furniture ===\n"
            + "Designer constraints:\n"
            + "  - Arrange the student desks in organized rows.\n"
            + "=== End Stage Brief ===",
            # Resumed scenes can retain this transient attribute. It must be
            # stripped with the same rule as text_description.
            "scene_expert_original_description": original_prompt
            + "\n\n=== SceneExpert Stage Brief: furniture ===\n"
            + "Designer constraints:\n"
            + "  - Arrange the student desks in organized rows.\n"
            + "=== End Stage Brief ===",
        },
    )()

    recovered_prompt = original_prompt_for_scene(scene)
    contract = build_intent_contract(recovered_prompt, room_type="classroom")

    assert recovered_prompt == original_prompt
    assert ("paired_with", "student_chair", "student_desk") in _relations(contract)
    assert not any(
        relation == "distributed_evenly"
        for relation, _subject, _target in _relations(contract)
    )


def test_rocking_chair_is_not_duplicated_as_a_generic_chair_relation() -> None:
    contract = build_intent_contract(
        "A nursery with a rocking chair near a small side table."
    )

    assert ("near", "rocking_chair", "side_table") in _relations(contract)
    assert ("near", "chair", "side_table") not in _relations(contract)


def test_classroom_each_with_builds_one_to_one_pair_checks_without_grid_gate() -> None:
    prompt = (
        "A classroom with six student desks, each with a chair. A teacher's desk "
        "sits at the front near the chalkboard."
    )
    task_spec = _fallback_spec_from_prompt(prompt)
    assert task_spec.required_large_objects.count("student_desk") == 6
    assert task_spec.required_large_objects.count("student_chair") == 6
    assert "chair" not in task_spec.required_large_objects
    desk_positions = [
        (-2.5, 2.0),
        (0.0, 2.0),
        (2.5, 2.0),
        (-2.5, 0.0),
        (0.0, 0.0),
        (2.5, 0.0),
    ]
    case_pack = {
        "task_instruction": prompt,
        "room_type": "classroom",
        "intent_contract": build_intent_contract(
            prompt,
            room_type="classroom",
            task_spec=task_spec,
        ),
        "scene_geometry": {
            "rooms": [
                {
                    "id": "classroom",
                    "bbox": {"min": [-4.0, -5.0, 0.0], "max": [4.0, 5.0, 2.7]},
                }
            ],
            "objects": [
                *[
                    _record(
                        f"student_desk_{index}",
                        "desk",
                        (x, y, 0.375),
                        (1.05, 0.5, 0.75),
                        name="student desk",
                        yaw_deg=180.0,
                    )
                    for index, (x, y) in enumerate(desk_positions)
                ],
                *[
                    _record(
                        f"student_chair_{index}",
                        "chair",
                        (x, y - 0.585, 0.45),
                        (0.5, 0.51, 0.9),
                        name="student chair",
                    )
                    for index, (x, y) in enumerate(desk_positions)
                ],
                _record(
                    "teacher_desk_0",
                    "desk",
                    (-2.5, 4.5, 0.38),
                    (1.4, 0.62, 0.76),
                    name="teacher desk",
                    yaw_deg=180.0,
                ),
            ],
        },
        "checks": [],
    }

    assert augment_contract_checks(case_pack)
    seat_checks = [
        check
        for check in case_pack["checks"]
        if check.get("check_source") == "intent_contract"
        and check.get("relation_type") == "seating_to_work_surface"
    ]
    surface_checks = [
        check
        for check in case_pack["checks"]
        if check.get("check_source") == "intent_contract"
        and check.get("relation_type") == "furniture_faces_furniture"
        and (check.get("evidence") or {}).get("paired_surface_facing")
    ]
    assert len(seat_checks) == 6
    assert len(surface_checks) == 6
    assert len({check["subject_id"] for check in seat_checks}) == 6
    assert len({check["target_ids"][0] for check in seat_checks}) == 6
    assert {
        (check["subject_id"], check["target_ids"][0]) for check in surface_checks
    } == {(check["target_ids"][0], check["subject_id"]) for check in seat_checks}
    assert not contract_relation_requested(case_pack, "distributed_evenly")

    results = run_case_pack_checks(
        case_pack,
        config=CriticConfig(
            enabled=True,
            metrics=("functional_dependency",),
        ),
    )
    assert not any(
        result.get("relation_type") == "classroom_workstation_distribution"
        for result in results
    )
    pair_results = [
        result
        for result in results
        if result.get("relation_type") == "seating_to_work_surface"
        and result.get("check_id") in {check["check_id"] for check in seat_checks}
    ]
    assert len(pair_results) == 6
    assert {result["label"] for result in pair_results} == {"pass"}
    surface_results = [
        result
        for result in results
        if result.get("check_id") in {check["check_id"] for check in surface_checks}
    ]
    assert len(surface_results) == 6
    assert {result["label"] for result in surface_results} == {"pass"}


def test_teacher_desk_at_front_does_not_infer_wall_constraint() -> None:
    """Front location is not an explicit wall-placement instruction."""
    prompt = (
        "A classroom with six student desks, each with a chair. A teacher's desk "
        "sits at the front near the chalkboard."
    )
    contract = build_intent_contract(prompt, room_type="classroom")
    assert not any(
        row["relation"] == "against_wall"
        and row["subjects"].get("category") == "teacher_desk"
        for row in contract["constraints"]
    )

    case_pack = {
        "task_instruction": prompt,
        "room_type": "classroom",
        "intent_contract": contract,
        "scene_geometry": {
            "objects": [
                _record(
                    f"teachers_desk_{index}",
                    "desk",
                    (float(index), 0.0, 0.4),
                    name="teacher desk",
                )
                for index in range(7)
            ]
        },
        "checks": [],
    }

    augment_contract_checks(case_pack)
    assert not any(
        check.get("relation_type") == "back_against_wall"
        for check in case_pack["checks"]
    )


def test_task_spec_instructional_topology_enables_controlled_functional_contract() -> (
    None
):
    prompt = "A flexible training space for a lesson."
    task_spec = {
        "required_large_objects": ["student desk", "chair", "instructor desk"],
        "required_wall_objects": ["whiteboard"],
        "functional_zones": ["teaching_zone", "student_seating_zone"],
    }

    contract = build_intent_contract(
        prompt, room_type="training_room", task_spec=task_spec
    )

    assert {
        ("operation_zone_at_wall", "teacher_desk", "wall"),
        (
            "instructional_surface_alignment",
            "instructional_surface",
            "teacher_desk",
        ),
    } <= _relations(contract)
    profile_rows = [
        row
        for row in contract["constraints"]
        if row["relation"]
        in {"operation_zone_at_wall", "instructional_surface_alignment"}
    ]
    assert profile_rows
    assert all(row["source"] == "room_ontology" for row in profile_rows)
    assert all(is_hard_constraint(row) for row in profile_rows)


def test_instructional_contract_requires_presenter_audience_and_focal_surface() -> None:
    contract = build_intent_contract(
        "A study with a desk and a whiteboard.",
        room_type="study",
        task_spec={
            "required_large_objects": ["desk"],
            "required_wall_objects": ["whiteboard"],
            "functional_zones": ["working_zone"],
        },
    )

    assert not any(
        row["relation"] in {"operation_zone_at_wall", "instructional_surface_alignment"}
        for row in contract["constraints"]
    )

    unbound_presenter = build_intent_contract(
        "A classroom where a teacher supervises students at six student desks "
        "with a whiteboard.",
        room_type="classroom",
        task_spec={
            "required_large_objects": ["student desk", "student chair"],
            "required_wall_objects": ["whiteboard"],
            "functional_zones": ["teaching_zone", "student_seating_zone"],
        },
    )
    assert not any(
        row["relation"] in {"operation_zone_at_wall", "instructional_surface_alignment"}
        for row in unbound_presenter["constraints"]
    )


def test_hard_operation_contract_overrides_conflicting_asset_facing_prior() -> None:
    prompt = (
        "A classroom with six student desks, each with a chair. A teacher's desk "
        "sits at the front near the chalkboard."
    )
    teacher = _record(
        "teacher_desk_0",
        "desk",
        (1.5, 1.5, 0.4),
        (1.4, 0.7, 0.8),
        name="teacher desk",
        yaw_deg=180.0,
    )
    teacher["functional_hints"] = {
        "orientation_dependencies": [
            {
                "relation_type": "front_faces",
                "target_category": "chair",
                "target_kind": "object_category",
                "source": "asset_annotations",
            }
        ]
    }
    case_pack = {
        "task_instruction": prompt,
        "room_type": "classroom",
        "intent_contract": build_intent_contract(prompt, room_type="classroom"),
        "scene_geometry": {
            "objects": [
                teacher,
                _record(
                    "student_chair_0",
                    "chair",
                    (1.5, 0.4, 0.45),
                    name="student chair",
                ),
            ]
        },
    }

    contract_checks = build_checks(case_pack, metrics=("functional_dependency",))

    assert not any(
        check.get("subject_id") == "teacher_desk_0"
        and check.get("check_source") == "asset_orientation_dependency"
        for check in contract_checks
    )


def test_ambiguous_teacher_desks_keep_asset_facing_priors() -> None:
    prompt = (
        "A classroom with six student desks, each with a chair. A teacher's desk "
        "sits at the front near the chalkboard."
    )
    teachers = []
    for index in range(2):
        teacher = _record(
            f"teacher_desk_{index}",
            "desk",
            (float(index), 1.5, 0.4),
            name="teacher desk",
        )
        teacher["functional_hints"] = {
            "orientation_dependencies": [
                {
                    "relation_type": "front_faces",
                    "target_category": "chair",
                    "target_kind": "object_category",
                    "source": "asset_annotations",
                }
            ]
        }
        teachers.append(teacher)
    case_pack = {
        "task_instruction": prompt,
        "room_type": "classroom",
        "intent_contract": build_intent_contract(prompt, room_type="classroom"),
        "scene_geometry": {
            "objects": [
                *teachers,
                _record("student_chair_0", "chair", name="student chair"),
            ]
        },
    }

    checks = build_checks(case_pack, metrics=("functional_dependency",))

    assert {
        check.get("subject_id")
        for check in checks
        if check.get("check_source") == "asset_orientation_dependency"
    } == {"teacher_desk_0", "teacher_desk_1"}


def test_role_specific_student_chair_satisfies_generic_chair_selector() -> None:
    assert bound_ids(
        {"category": "chair", "count": 1, "quantifier": "at_least"},
        [_record("student_chair_0", "student_chair", name="student chair")],
    ) == ["student_chair_0"]


@pytest.mark.parametrize(
    "asset_category",
    ("teachers_desk", "teacher's desk", "instructor_desk"),
)
def test_teacher_desk_selector_normalizes_asset_category_aliases(
    asset_category: str,
) -> None:
    teacher = _record("presenter_surface_0", asset_category, name="work surface")

    assert bound_ids(
        {"category": "teacher_desk", "count": 1, "quantifier": "all"},
        [teacher],
    ) == ["presenter_surface_0"]


def test_explicit_teacher_desk_against_wall_remains_a_core_constraint() -> None:
    contract = build_intent_contract(
        "A classroom has a teacher desk against the front wall."
    )
    assert ("against_wall", "teacher_desk", "wall") in _relations(contract)


def test_explicit_counts_preserve_role_specific_categories() -> None:
    contract = build_intent_contract(
        "A classroom with six student desks, each with a chair. "
        "A teacher's desk sits at the front near the chalkboard."
    )
    required = {
        (row["subjects"]["category"], row["subjects"]["count"])
        for row in contract["constraints"]
        if row["relation"] == "required_count"
    }

    assert ("student_desk", 6) in required
    assert ("chair", 6) in required
    assert ("desk", 6) not in required


def test_explicit_count_extension_rejects_broader_role_substitution() -> None:
    prompt = (
        "A classroom with six student desks, each with a chair. "
        "A teacher's desk sits at the front."
    )
    case_pack = {
        "intent_contract": build_intent_contract(prompt),
        "scene_geometry": {
            "objects": [
                *[
                    _record(f"student_desk_{index}", "desk", name="student desk")
                    for index in range(5)
                ],
                _record("teacher_desk_0", "desk", name="teacher desk"),
                *[
                    _record(f"student_chair_{index}", "chair", name="student chair")
                    for index in range(6)
                ],
            ]
        },
    }

    results = evaluate_intent_contract_extensions(case_pack)
    student_desks = next(
        row
        for row in results
        if row["relation_type"] == "required_count"
        and row["diagnostics"]["required_count"] == 6
        and row["evidence"]["intent_constraint"]["subjects"]["category"]
        == "student_desk"
    )

    assert student_desks["label"] == "fail"
    assert student_desks["diagnostics"]["observed_ids"] == [
        "student_desk_0",
        "student_desk_1",
        "student_desk_2",
        "student_desk_3",
        "student_desk_4",
    ]


def _bbox(center: tuple[float, float, float], size: tuple[float, float, float]) -> dict:
    lower = [center[index] - size[index] / 2.0 for index in range(3)]
    upper = [center[index] + size[index] / 2.0 for index in range(3)]
    return {"center": list(center), "size": list(size), "min": lower, "max": upper}


def _record(
    object_id: str,
    category: str,
    center: tuple[float, float, float] = (0.0, 0.0, 0.5),
    size: tuple[float, float, float] = (0.6, 0.6, 1.0),
    *,
    name: str | None = None,
    yaw_deg: float = 0.0,
) -> dict:
    return {
        "id": object_id,
        "category": category,
        "category_norm": category,
        "name": name or category,
        "description": name or category,
        "yaw_deg": yaw_deg,
        "bbox_world": _bbox(center, size),
    }


def test_contract_checks_are_core_without_shadow_metadata() -> None:
    case_pack = {
        "intent_contract": build_intent_contract("A sofa faces a TV stand."),
        "scene_geometry": {
            "objects": [
                _record("sofa_0", "sofa"),
                _record("tv_stand_0", "tv_stand", (0.0, 1.0, 0.4)),
            ]
        },
        "checks": [],
    }
    assert augment_contract_checks(case_pack)
    checks = [
        check
        for check in case_pack["checks"]
        if check.get("check_source") == "intent_contract"
    ]
    assert checks
    assert {check["scoring_tier"] for check in checks} == {"core"}
    assert "shadow_check_ids" not in case_pack["intent_contract"]


def test_model_inferred_relations_are_hard_and_unknown_sources_are_rejected() -> None:
    case_pack = {
        "intent_contract": build_intent_contract(
            "An office chair faces a desk.",
            task_spec={
                "intent_constraints": [
                    {
                        "relation": "faces",
                        "subjects": "armchair",
                        "targets": "television",
                        "source": "model_inferred",
                    },
                    {
                        "relation": "faces",
                        "subjects": "sofa",
                        "targets": "television",
                        "source": "vlm_observation",
                    },
                    {
                        "relation": "against_wall",
                        "subjects": "bookshelf",
                        "targets": "wall",
                        "source": "explicit_prompt",
                        "evidence_span": "An office chair faces a desk.",
                    },
                ]
            },
        ),
        "scene_geometry": {
            "objects": [
                _record("office_chair_0", "office_chair"),
                _record("desk_0", "desk", (0.0, 1.0, 0.4)),
                _record("armchair_0", "armchair", (1.0, 0.0, 0.5)),
                _record("sofa_0", "sofa", (-1.0, 0.0, 0.5)),
                _record("television_0", "television", (0.0, 2.0, 1.0)),
                _record("bookshelf_0", "bookshelf", (1.5, 0.0, 0.8)),
                _record("wall_0", "wall", (2.0, 0.0, 1.35), (0.1, 4.0, 2.7)),
            ]
        },
        "checks": [],
    }
    assert augment_contract_checks(case_pack)
    tiers_by_source = {
        check["evidence"]["intent_constraint"]["source"]: check["scoring_tier"]
        for check in case_pack["checks"]
        if check.get("check_source") == "intent_contract"
    }
    assert tiers_by_source["explicit_prompt"] == "core"
    assert tiers_by_source["model_inferred"] == "core"
    explicit_claim = next(
        row
        for row in case_pack["intent_contract"]["constraints"]
        if row["relation"] == "against_wall"
        and row["subjects"].get("category") == "bookshelf"
    )
    assert explicit_claim["source"] == "explicit_prompt"
    assert is_hard_constraint(explicit_claim)
    assert not any(
        row["source"] == "vlm_observation"
        for row in case_pack["intent_contract"]["constraints"]
    )
    assert contract_seating_targets(case_pack) == {
        "armchair_0": {"television_0"},
        "bookshelf_0": {"wall_0"},
        "office_chair_0": {"desk_0"},
    }


def test_contract_seating_targets_do_not_guess_unmentioned_chair_roles() -> None:
    case_pack = {
        "intent_contract": build_intent_contract(
            "A study with an office chair tucked under the desk, a guest chair against "
            "the wall, and an armchair."
        ),
        "scene_geometry": {
            "objects": [
                _record("office_chair_0", "office_chair"),
                _record("desk_0", "desk", (0.0, 1.0, 0.4)),
                _record("guest_chair_0", "chair", (1.4, 0.0, 0.5), name="guest chair"),
                _record("armchair_0", "armchair", (-1.4, 0.0, 0.5)),
                _record(
                    "side_wall",
                    "wall",
                    (2.0, 0.0, 1.35),
                    (0.1, 4.0, 2.7),
                ),
            ]
        },
    }

    assert contract_seating_targets(case_pack) == {
        "office_chair_0": {"desk_0"},
        "guest_chair_0": {"side_wall"},
    }


def test_single_prompt_selector_does_not_bind_multiple_candidate_assets() -> None:
    case_pack = {
        "intent_contract": build_intent_contract("A chair faces a desk."),
        "scene_geometry": {
            "objects": [
                _record("chair_0", "chair"),
                _record("armchair_0", "armchair", (1.0, 0.0, 0.5)),
                _record("desk_0", "desk", (0.0, 1.0, 0.4)),
            ]
        },
        "checks": [],
    }
    assert not augment_contract_checks(case_pack)
    assert contract_seating_targets(case_pack) == {}


def test_model_inferred_relation_enables_hard_topology_rule() -> None:
    case_pack = {
        "intent_contract": build_intent_contract(
            "A dining room with a dining table and four dining chairs.",
            task_spec={
                "intent_constraints": [
                    {
                        "relation": "one_per_side",
                        "subjects": "dining chair",
                        "targets": "dining table",
                        "source": "model_inferred",
                    }
                ]
            },
        )
    }

    assert contract_relation_requested(case_pack, "one_per_side")


def test_one_per_side_accepts_a_pair_on_opposite_bed_sides() -> None:
    case_pack = {
        "intent_contract": {
            "constraints": [
                {
                    **_explicit_constraint(
                        "bedside_pair", "one_per_side", "nightstand", "bed"
                    ),
                    "subjects": {"category": "nightstand", "count": 2},
                }
            ]
        },
        "scene_geometry": {
            "objects": [
                _record(
                    "bed_0", "bed", (0.0, 0.9, 0.4), (1.6, 2.05, 0.8), yaw_deg=180.0
                ),
                _record("nightstand_0", "nightstand", (1.07, 1.62, 0.3)),
                _record("nightstand_1", "nightstand", (-1.07, 1.62, 0.3)),
            ]
        },
    }

    result = next(
        item
        for item in evaluate_intent_contract_extensions(case_pack)
        if item["relation_type"] == "one_per_side"
    )

    assert result["label"] == "pass"
    assert result["diagnostics"]["occupied_sides"] == [[0, -1], [0, 1]]


def test_model_inferred_relation_enters_gate_and_critic_context() -> None:
    case_pack = {
        "intent_contract": build_intent_contract(
            "A room.",
            task_spec={
                "intent_constraints": [
                    {
                        "relation": "on_top_of",
                        "subjects": {"category": "book"},
                        "targets": {"category": "desk"},
                        "source": "model_inferred",
                    }
                ]
            },
        ),
        "scene_geometry": {
            "objects": [
                _record("book_0", "book", (1.0, 1.0, 0.1), (0.2, 0.2, 0.05)),
                _record("desk_0", "desk", (0.0, 0.0, 0.4), (1.2, 0.7, 0.8)),
            ]
        },
        "checks": [],
    }
    config = CriticConfig(
        enabled=True,
        hard_gate=True,
        metrics=("functional_dependency",),
    )

    results = run_case_pack_checks(case_pack, config=config)
    assert len(results) == 1
    assert results[0]["label"] == "fail"
    assert results[0]["scoring_tier"] == "core"

    payload = build_evaluation_payload(
        case_pack=case_pack,
        results=results,
        stage="furniture",
        scope="room:test",
        config=config,
    )
    scene_summary = payload["summary"]["scene_summary"]
    assert scene_summary["fail"] == 1
    assert scene_summary["excluded_auxiliary"] == 0
    assert payload["gate"]["blocked"]
    assert "object_on_support" in format_prompt_context(payload)
    assert "no degraded or failed checks" in format_agent_prompt_context(
        payload,
        agent_type="furniture",
    )


def test_generic_centered_on_wall_extension_uses_room_relative_geometry() -> None:
    case_pack = {
        "intent_contract": build_intent_contract(
            "A study with a desk centered against the back wall."
        ),
        "scene_geometry": {
            "rooms": [
                {
                    "id": "study",
                    "bbox": {"min": [-3.0, -2.0, 0.0], "max": [3.0, 2.0, 2.7]},
                }
            ],
            "objects": [
                _record("desk_0", "desk", (1.4, 1.1, 0.4), (1.2, 0.6, 0.8)),
                _record("west_wall", "wall", (-3.0, 0.0, 1.35), (0.1, 4.0, 2.7)),
                _record("east_wall", "wall", (3.0, 0.0, 1.35), (0.1, 4.0, 2.7)),
                _record("south_wall", "wall", (0.0, -2.0, 1.35), (6.0, 0.1, 2.7)),
                _record("north_wall", "wall", (0.0, 2.0, 1.35), (6.0, 0.1, 2.7)),
            ],
        },
    }

    results = evaluate_intent_contract_extensions(case_pack)
    result = next(
        item for item in results if item["relation_type"] == "centered_on_wall"
    )

    assert result["label"] == "fail"
    assert result["scoring_tier"] == "core"
    assert result["diagnostics"]["target_center_xy_m"] == [0.0, 1.62]


def test_room_relative_wall_target_binds_to_physical_walls() -> None:
    case_pack = {
        "stage": "furniture",
        "intent_contract": {
            "constraints": [
                _explicit_constraint("sofa_back", "against_wall", "sofa", "back_wall")
            ]
        },
        "scene_geometry": {
            "objects": [
                _record("sofa_0", "sofa", (0.0, -1.5, 0.4), (2.2, 0.8, 0.8)),
                _record("south_wall", "wall", (0.0, -2.0, 1.35), (5.0, 0.1, 2.7)),
                _record("north_wall", "wall", (0.0, 2.0, 1.35), (5.0, 0.1, 2.7)),
            ]
        },
    }

    results = evaluate_intent_contract_extensions(case_pack)

    assert results == []


def test_contract_study_does_not_invent_tangential_wall_slots() -> None:
    """Wall wording must not manufacture a bookshelf/chair coordinate profile."""
    prompt = (
        "A study with a desk centered against the back wall, an office chair tucked "
        "under the desk, two guest chairs against the side wall facing the desk, and "
        "a bookshelf on the adjacent wall."
    )
    case_pack = {
        "task_instruction": prompt,
        "intent_contract": build_intent_contract(prompt, room_type="study"),
        "scene_geometry": {
            "rooms": [
                {
                    "id": "study",
                    "bbox": {"min": [-2.5, -2.25, 0.0], "max": [2.5, 2.25, 2.7]},
                }
            ],
            "objects": [
                _record(
                    "study_desk_0",
                    "desk",
                    (0.0, 1.787, 0.4),
                    (1.6, 0.742, 0.8),
                    yaw_deg=180.0,
                ),
                _record(
                    "office_chair_0",
                    "office_chair",
                    (0.02, 1.038, 0.45),
                    (0.6, 0.664, 0.9),
                ),
                _record(
                    "bookshelf_0",
                    "bookshelf",
                    (-2.266, -1.2, 0.88),
                    (0.35, 0.955, 1.76),
                    yaw_deg=-90.0,
                ),
                _record(
                    "guest_chair_0",
                    "guest_chair",
                    (2.072, 0.56, 0.45),
                    (0.65, 0.696, 0.9),
                    yaw_deg=90.0,
                ),
                _record(
                    "guest_chair_1",
                    "guest_chair",
                    (2.072, -0.49, 0.45),
                    (0.65, 0.696, 0.9),
                    yaw_deg=90.0,
                ),
                _record("west_wall", "wall", (-2.475, 0.0, 1.35), (0.05, 4.5, 2.7)),
                _record("east_wall", "wall", (2.475, 0.0, 1.35), (0.05, 4.5, 2.7)),
                _record("south_wall", "wall", (0.0, -2.225, 1.35), (5.0, 0.05, 2.7)),
                _record("north_wall", "wall", (0.0, 2.225, 1.35), (5.0, 0.05, 2.7)),
            ],
        },
        "checks": [],
    }

    # These shelf/chair tangential coordinates are intentionally not supplied
    # by the prompt.  Contract mode must evaluate the four explicit relations,
    # not fail a legacy all-in-one wall-layout profile.
    assert evaluate_study_furniture_layout(case_pack) == []
    assert augment_contract_checks(case_pack)
    relation_types = {
        check["relation_type"]
        for check in case_pack["checks"]
        if check.get("check_source") == "intent_contract"
    }
    assert {"back_against_wall", "seating_to_work_surface"} <= relation_types


def _scene_object(
    object_id: str,
    name: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    object_type: ObjectType = ObjectType.FURNITURE,
    yaw_deg: float = 0.0,
) -> SceneObject:
    half = np.asarray(size, dtype=float) / 2.0
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=object_type,
        name=name,
        description=name.replace("_", " "),
        transform=RigidTransform(
            rpy=RollPitchYaw(0.0, 0.0, math.radians(yaw_deg)),
            p=np.asarray(position, dtype=float),
        ),
        bbox_min=-half,
        bbox_max=half,
    )


def _study_scene(tmp_path: Path) -> RoomScene:
    walls = [
        _scene_object(
            "west_wall",
            "wall",
            (-3.0, 0.0, 1.35),
            (0.1, 4.0, 2.7),
            object_type=ObjectType.WALL,
        ),
        _scene_object(
            "east_wall",
            "wall",
            (3.0, 0.0, 1.35),
            (0.1, 4.0, 2.7),
            object_type=ObjectType.WALL,
        ),
        _scene_object(
            "south_wall",
            "wall",
            (0.0, -2.0, 1.35),
            (6.0, 0.1, 2.7),
            object_type=ObjectType.WALL,
        ),
        _scene_object(
            "north_wall",
            "wall",
            (0.0, 2.0, 1.35),
            (6.0, 0.1, 2.7),
            object_type=ObjectType.WALL,
        ),
    ]
    geometry = RoomGeometry(
        sdf_tree=ET.ElementTree(ET.Element("sdf")),
        sdf_path=tmp_path / "study.sdf",
        walls=walls,
        width=4.0,
        length=6.0,
        wall_height=2.7,
        wall_thickness=0.1,
    )
    desk = _scene_object("desk_0", "desk", (1.4, 1.1, 0.4), (1.2, 0.6, 0.8))
    return RoomScene(
        room_geometry=geometry,
        scene_dir=tmp_path,
        room_id="study",
        room_type="study",
        text_description="A study with a desk centered against the back wall.",
        objects={desk.object_id: desk},
    )


def test_generic_centered_on_wall_repair_is_accepted_only_after_improvement(
    tmp_path: Path,
) -> None:
    scene = _study_scene(tmp_path)
    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency",),
    )

    before = evaluate_room_scene(scene, config=config, stage="before")
    assert (
        next(
            item
            for item in before["results"]
            if item.get("relation_type") == "centered_on_wall"
        )["label"]
        == "fail"
    )

    fixes = improve_furniture_relations(scene, config=config)

    assert [(fix.object_id, fix.relation_type) for fix in fixes] == [
        ("desk_0", "centered_on_wall")
    ]
    after = evaluate_room_scene(scene, config=config, stage="after")
    assert (
        next(
            item
            for item in after["results"]
            if item.get("relation_type") == "centered_on_wall"
        )["label"]
        == "pass"
    )


def test_in_front_of_reports_lateral_alignment_and_preserves_center_anchor() -> None:
    case_pack = {
        "intent_contract": {
            "constraints": [
                _explicit_constraint("rug_center", "centered_in_room", "rug", "room"),
                _explicit_constraint("rug_front", "in_front_of", "rug", "sofa"),
            ]
        },
        "scene_geometry": {
            "rooms": [
                {
                    "id": "living_room",
                    "bbox": {"min": [-2.5, -2.0, 0.0], "max": [2.5, 2.0, 2.7]},
                }
            ],
            "objects": [
                _record(
                    "sofa_0",
                    "sofa",
                    (-1.57, -1.445, 0.38),
                    (1.7, 0.95, 0.76),
                    yaw_deg=0.0,
                ),
                _record("rug_0", "rug", (0.0, 0.0, 0.02), (1.8, 1.8, 0.03)),
            ],
        },
    }

    result = next(
        item
        for item in evaluate_intent_contract_extensions(case_pack)
        if item.get("relation_type") == "front_axis_alignment"
    )

    assert result["label"] == "fail"
    assert result["diagnostics"]["repair_object_id"] == "sofa_0"
    np.testing.assert_allclose(
        result["diagnostics"]["repair_target_center_xy_m"], [0.0, -1.445]
    )
    assert result["diagnostics"]["forward_distance_m"] > 1.0
    assert result["diagnostics"]["lateral_offset_m"] > 1.0


def test_living_room_prompt_extracts_two_anchor_relations() -> None:
    prompt = (
        "A living room with a sofa against the back wall facing a TV stand and "
        "television on the opposite wall, a coffee table centered between the sofa "
        "and TV stand. A small rug lies between the coffee table and TV stand."
    )

    contract = build_intent_contract(prompt, room_type="living_room")
    constraints = {
        str(row["relation"]): row
        for row in contract["constraints"]
        if row["relation"] in {"centered_between", "between"}
    }

    assert constraints["centered_between"]["subjects"]["category"] == "coffee_table"
    assert constraints["centered_between"]["targets"]["category"] == "sofa"
    assert (
        constraints["centered_between"]["targets"]["secondary_category"] == "tv_stand"
    )
    assert constraints["between"]["subjects"]["category"] == "rug"
    assert constraints["between"]["targets"]["category"] == "coffee_table"
    assert constraints["between"]["targets"]["secondary_category"] == "tv_stand"
    assert all(is_hard_constraint(row) for row in constraints.values())


def test_media_furniture_ontology_hardens_inferred_television_support() -> None:
    prompt = (
        "A living room with a sofa facing a TV stand and television on the "
        "opposite wall."
    )

    contract = build_intent_contract(prompt, room_type="living_room")
    support = next(
        row
        for row in contract["constraints"]
        if row["relation"] == "on_top_of" and row["source"] == "model_inferred"
    )

    assert support["subjects"] == {
        "category": "television",
        "count": 1,
        "quantifier": "all",
    }
    assert support["targets"] == {
        "category": "tv_stand",
        "count": 1,
        "quantifier": "all",
    }
    assert support["source"] == "model_inferred"
    assert is_hard_constraint(support)

    objects = [
        _record("television_0", "television", (0.0, 0.0, 0.8)),
        _record("tv_stand_0", "tv_stand", (0.0, 0.0, 0.3)),
    ]
    assert bound_ids(support["subjects"], objects) == ["television_0"]
    assert bound_ids(support["targets"], objects) == ["tv_stand_0"]


def test_media_furniture_ontology_does_not_ground_wall_mounted_television() -> None:
    contract = build_intent_contract(
        "A living room with a TV stand and a wall-mounted television above it.",
        room_type="living_room",
    )

    assert not any(
        row["relation"] == "on_top_of" and row["source"] == "room_ontology"
        for row in contract["constraints"]
    )


def test_model_inferred_shared_target_uses_inventory_cardinality() -> None:
    task_spec = {
        "required_large_objects": ["bed", "nightstand", "nightstand"],
        "intent_constraints": [
            {
                "relation": "next_to",
                "subjects": {"category": "nightstand", "count": 1},
                "targets": {"category": "bed"},
                "source": "model_inferred",
                "inference_reason": "Nightstands are placed next to beds.",
            }
        ],
    }

    contract = build_intent_contract(
        "A bedroom with a bed and two nightstands.",
        room_type="bedroom",
        task_spec=task_spec,
    )
    relation = next(
        row
        for row in contract["constraints"]
        if row["relation"] == "next_to" and row["source"] == "model_inferred"
    )

    assert relation["subjects"] == {
        "category": "nightstand",
        "count": 2,
        "quantifier": "all",
    }
    assert relation["targets"] == {
        "category": "bed",
        "count": 1,
        "quantifier": "all",
    }
    objects = [
        _record("bed_0", "bed", (0.0, 0.0, 0.4)),
        _record("nightstand_0", "nightstand", (-1.0, 0.0, 0.3)),
        _record("nightstand_1", "nightstand", (1.0, 0.0, 0.3)),
    ]
    assert bound_ids(relation["subjects"], objects) == [
        "nightstand_0",
        "nightstand_1",
    ]


def test_between_and_flanking_evaluate_full_two_dimensional_alignment() -> None:
    case_pack = {
        "intent_contract": {
            "constraints": [
                {
                    **_explicit_constraint(
                        "coffee_between", "centered_between", "coffee_table", "sofa"
                    ),
                    "targets": {
                        "category": "sofa",
                        "count": 1,
                        "secondary_category": "tv_stand",
                        "secondary_count": 1,
                    },
                },
                {
                    **_explicit_constraint(
                        "rug_between", "between", "rug", "coffee_table"
                    ),
                    "targets": {
                        "category": "coffee_table",
                        "count": 1,
                        "secondary_category": "tv_stand",
                        "secondary_count": 1,
                    },
                },
                {
                    **_explicit_constraint(
                        "chairs_flank", "flanking", "armchair", "coffee_table"
                    ),
                    "subjects": {"category": "armchair", "count": 2},
                },
            ]
        },
        "scene_geometry": {
            "rooms": [
                {
                    "id": "living_room",
                    "bbox": {"min": [-2.5, -2.0, 0.0], "max": [2.5, 2.0, 2.7]},
                }
            ],
            "objects": [
                _record("sofa_0", "sofa", (0.0, -1.5, 0.4), (2.2, 0.8, 0.8)),
                _record("tv_stand_0", "tv_stand", (0.0, 1.7, 0.3), (1.4, 0.5, 0.6)),
                _record(
                    "coffee_table_0", "coffee_table", (-1.0, 0.0, 0.25), (1.0, 0.6, 0.5)
                ),
                _record("rug_0", "rug", (1.0, 0.85, 0.02), (1.2, 0.7, 0.03)),
                _record("armchair_0", "armchair", (-1.7, -0.5, 0.4), (0.65, 0.7, 0.8)),
                _record("armchair_1", "armchair", (-1.7, 0.7, 0.4), (0.65, 0.7, 0.8)),
            ],
        },
    }

    results = {
        str(item["relation_type"]): item
        for item in evaluate_intent_contract_extensions(case_pack)
    }

    assert results["centered_between_alignment"]["label"] == "fail"
    assert results["between_alignment"]["label"] == "fail"
    assert results["flanking"]["label"] == "fail"
    assert results["flanking"]["diagnostics"]["target_side_xy"] == [1.0, 0.0]
    slot_x = [
        row["target_center_xy_m"][0]
        for row in results["flanking"]["diagnostics"]["target_slots"]
    ]
    assert slot_x[0] < -1.0 < slot_x[1]


def test_in_front_of_does_not_move_two_independently_centered_objects() -> None:
    case_pack = {
        "intent_contract": {
            "constraints": [
                _explicit_constraint("sofa_center", "centered_on_wall", "sofa", "wall"),
                _explicit_constraint("rug_center", "centered_in_room", "rug", "room"),
                _explicit_constraint("rug_front", "in_front_of", "rug", "sofa"),
            ]
        },
        "scene_geometry": {
            "rooms": [
                {
                    "id": "living_room",
                    "bbox": {"min": [-2.5, -2.0, 0.0], "max": [2.5, 2.0, 2.7]},
                }
            ],
            "objects": [
                _record(
                    "sofa_0",
                    "sofa",
                    (-1.0, -1.445, 0.38),
                    (1.7, 0.95, 0.76),
                    yaw_deg=0.0,
                ),
                _record("rug_0", "rug", (0.0, 0.0, 0.02), (1.8, 1.8, 0.03)),
            ],
        },
    }

    result = next(
        item
        for item in evaluate_intent_contract_extensions(case_pack)
        if item.get("relation_type") == "front_axis_alignment"
    )

    assert result["label"] == "fail"
    assert "repair_object_id" not in result["diagnostics"]
