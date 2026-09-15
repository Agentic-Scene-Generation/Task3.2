from types import SimpleNamespace

from scenesmith.agent_utils.room import ObjectType
from scenesmith.manipuland_agents.cross_stage_inventory import (
    contract_manipuland_support_cohorts,
)
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)


class _Scene:
    def __init__(self, objects: dict[str, object], constraints: list[dict]) -> None:
        self.objects = objects
        self.scene_expert_task_spec = {"required_small_objects": []}
        self.scenebenchmark_intent_contract = {"constraints": constraints}

    def get_object(self, object_id):
        return next(
            (
                obj
                for candidate_id, obj in self.objects.items()
                if str(candidate_id) == str(object_id)
            ),
            None,
        )


def _object(object_id: str, category: str, *, support_surface: str | None = None):
    placement = (
        SimpleNamespace(parent_surface_id=support_surface)
        if support_surface is not None
        else None
    )
    return SimpleNamespace(
        object_id=object_id,
        object_type=(
            ObjectType.MANIPULAND
            if support_surface is not None
            else ObjectType.FURNITURE
        ),
        name=category,
        description=category.replace("_", " "),
        metadata={"semantic_name": category},
        support_surfaces=[],
        placement_info=placement,
        immutable=False,
    )


def _support_constraint(
    constraint_id: str,
    category: str,
    count: int,
    target_category: str,
    target_count: int,
) -> dict:
    return {
        "constraint_id": constraint_id,
        "relation": "on_top_of",
        "stage": "manipuland",
        "strength": "hard",
        "subjects": {
            "category": category,
            "count": count,
            "quantifier": "minimum",
        },
        "targets": {
            "category": target_category,
            "count": target_count,
            "quantifier": "all",
        },
    }


def test_disjoint_support_cohorts_allocate_distinct_plush_inventory() -> None:
    objects = {
        "bed_0": _object("bed_0", "bunk_bed"),
        "desk_0": _object("desk_0", "desk"),
        "desk_1": _object("desk_1", "desk"),
    }
    constraints = [
        _support_constraint("plush_on_desks", "plush_toy", 11, "desk", 2),
        _support_constraint("plush_on_bed", "plush_toy", 1, "bunk_bed", 1),
    ]
    scene = _Scene(objects, constraints)
    scene.scene_expert_task_spec["required_small_objects"] = ["plush_toy"] * 12

    cohorts = contract_manipuland_support_cohorts(scene)

    by_constraint = {}
    for cohort in cohorts:
        by_constraint.setdefault(cohort.constraint_id, []).append(cohort)
    assert sorted(
        cohort.required_count for cohort in by_constraint["plush_on_desks"]
    ) == [5, 6]
    assert (
        sum(cohort.required_count for cohort in by_constraint["plush_on_desks"]) == 11
    )
    assert by_constraint["plush_on_bed"][0].required_count == 1
    assert sum(cohort.required_count for cohort in cohorts) == 12


def test_open_vocabulary_small_objects_generate_contract_targets() -> None:
    objects = {
        "nightstand_0": _object("nightstand_0", "nightstand"),
        "bookshelf_0": _object("bookshelf_0", "bookshelf"),
    }
    constraints = [
        _support_constraint("clock_on_nightstand", "bedside_clock", 1, "nightstand", 1),
        _support_constraint(
            "candle_on_nightstand", "decorative_candle", 1, "nightstand", 1
        ),
        _support_constraint("novels_on_shelf", "novel", 3, "bookshelf", 1),
    ]
    scene = _Scene(objects, constraints)
    scene.scene_expert_task_spec["required_small_objects"] = [
        "bedside_clock",
        "decorative_candle",
        "novel",
        "novel",
        "novel",
    ]

    cohorts = contract_manipuland_support_cohorts(scene)

    assert {
        (cohort.category, cohort.target_id, cohort.required_count) for cohort in cohorts
    } == {
        ("alarm_clock", "nightstand_0", 1),
        ("decorative_candle", "nightstand_0", 1),
        ("novel", "bookshelf_0", 3),
    }


def test_contract_recovery_adds_every_omitted_required_support_target() -> None:
    objects = {
        "desk_0": _object("desk_0", "desk"),
        "desk_1": _object("desk_1", "desk"),
    }
    scene = _Scene(
        objects,
        [_support_constraint("plush_on_desks", "plush_toy", 11, "desk", 2)],
    )
    scene.scene_expert_task_spec["required_small_objects"] = ["plush_toy"] * 11
    agent = StatefulManipulandAgent.__new__(StatefulManipulandAgent)

    recovered = agent._recover_contract_required_manipuland_targets(
        scene=scene,
        furniture_data=[],
    )

    assert {str(selection.furniture_id) for selection in recovered} == {
        "desk_0",
        "desk_1",
    }
    assert all(selection.is_prompt_required for selection in recovered)
    assert sorted(
        6 if "place at least 6" in selection.suggested_items else 5
        for selection in recovered
    ) == [5, 6]
    assert all(
        "one reusable asset template" in selection.prompt_constraints
        for selection in recovered
    )


def test_target_cardinality_reports_wrong_support_as_capacity_failure(
    monkeypatch,
) -> None:
    desk_0 = _object("desk_0", "desk")
    desk_1 = _object("desk_1", "desk")
    desk_0.support_surfaces = [SimpleNamespace(surface_id="surface_desk_0")]
    desk_1.support_surfaces = [SimpleNamespace(surface_id="surface_desk_1")]
    plush = _object("plush_toy_0", "plush_toy", support_surface="surface_desk_1")
    scene = _Scene(
        {"desk_0": desk_0, "desk_1": desk_1, "plush_toy_0": plush},
        [_support_constraint("plush_on_desks", "plush_toy", 2, "desk", 2)],
    )
    scene.scene_expert_task_spec["required_small_objects"] = [
        "plush_toy",
        "plush_toy",
    ]
    agent = StatefulManipulandAgent.__new__(StatefulManipulandAgent)
    agent.scene = scene
    agent.current_furniture_id = "desk_0"
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.stateful_manipuland_agent.room_scene_to_case_pack",
        lambda _scene, stage: {
            "scene_geometry": {
                "objects": [
                    {
                        "id": object_id,
                        "category": obj.metadata["semantic_name"],
                        "category_norm": obj.metadata["semantic_name"],
                        "object_type": obj.object_type.value,
                    }
                    for object_id, obj in scene.objects.items()
                ]
            }
        },
    )

    failures = agent._current_target_cardinality_failures()

    assert len(failures) == 1
    assert failures[0].startswith("support_capacity_or_wrong_support:")
    assert "desk_0" in failures[0]
