"""External asset taxonomies must not hide the runtime role from predicates."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scenesmith.scenebenchmark_critic.adapter import ObjectType, _functional_hints
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions import (
    dining_place_setting as dining,
)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (ObjectType.FURNITURE, "furniture"),
        (ObjectType.MANIPULAND, "manipuland"),
        (ObjectType.THIN_COVERING, "manipuland"),
        (ObjectType.WALL_MOUNTED, "wall_mounted"),
        (ObjectType.CEILING_MOUNTED, "ceiling_mounted"),
    ],
)
@pytest.mark.parametrize("annotation", ["furniture_or_equipment", "manipuland"])
def test_runtime_role_outranks_external_annotation(
    kind: ObjectType, expected: str, annotation: str,
) -> None:
    obj = SimpleNamespace(
        object_type=kind, name="dining_chair", description="A dining chair.",
        metadata={"functional_hints": {"scene_object_type": annotation}},
    )
    before = deepcopy(obj.metadata)
    assert _functional_hints(obj, "dining_chair")["scene_object_type"] == expected
    assert obj.metadata == before  # Preserve the original asset annotation.


def test_structural_object_retains_its_annotation_fallback() -> None:
    obj = SimpleNamespace(
        object_type=ObjectType.WALL, name="wall", description="A wall.",
        metadata={"functional_hints": {"scene_object_type": "structural"}},
    )
    assert _functional_hints(obj, "wall")["scene_object_type"] == "structural"


def _box(center: tuple[float, float], size: tuple[float, float]) -> dict:
    return {
        "center": [*center, 0.8], "size": [*size, 0.1],
        "min": [center[0] - size[0] / 2, center[1] - size[1] / 2, 0.75],
        "max": [center[0] + size[0] / 2, center[1] + size[1] / 2, 0.85],
    }


def _eight_seat_case() -> dict:
    objects = [{
        "id": "dining_table_0", "name": "dining_table",
        "object_type": "furniture", "bbox_world": _box((0, 0), (2.1, 1.0)),
        "support_surfaces": [{"surface_id": "table.top"}],
    }]
    positions = [
        *((x, 0.9, 180) for x in [-0.7, 0.0, 0.7]),
        *((x, -0.9, 0) for x in [-0.7, 0.0, 0.7]),
        (-1.4, 0.0, -90), (1.4, 0.0, 90),
    ]
    for index, (x, y, yaw) in enumerate(positions):
        obj = SimpleNamespace(
            object_type=ObjectType.FURNITURE, name="dining_chair",
            description="A dining chair.",
            metadata={"functional_hints": {"scene_object_type": "furniture_or_equipment"}},
        )
        objects.append({
            "id": f"dining_chair_{index}", "name": "dining_chair",
            "object_type": "furniture", "category": "dining_chair",
            "bbox_world": _box((x, y), (0.45, 0.5)), "yaw_deg": yaw,
            "functional_hints": _functional_hints(obj, "dining_chair", yaw_deg=yaw),
        })
        objects.append({
            "id": f"plate_{index}", "name": "plate", "object_type": "manipuland",
            "bbox_world": _box((0.6 * x, 0.3 * y), (0.27, 0.27)),
            "placement_info": {"parent_surface_id": "table.top"},
        })
    return {
        "task_instruction": "A dining room with a table set for 8 people. Each setting has a plate.",
        "scene_geometry": {"objects": objects},
    }


def test_eight_annotated_chairs_produce_alignment_feedback_without_relaxing_geometry() -> None:
    pack = _eight_seat_case()
    result = dining.evaluate_dining_place_setting_alignment(pack)
    assert len(result) == 1
    assert result[0]["label"] == "fail"  # Deliberately misplaced plates still fail.
    assignments = result[0]["diagnostics"]["assignments"]
    assert len(assignments) == 8
    assert {row["seat_id"] for row in assignments} == {
        f"dining_chair_{i}" for i in range(8)
    }


def test_missing_seat_is_not_fabricated_to_match_eight_plates() -> None:
    pack = _eight_seat_case()
    pack["scene_geometry"]["objects"] = [
        obj for obj in pack["scene_geometry"]["objects"] if obj["id"] != "dining_chair_7"
    ]
    assert dining.evaluate_dining_place_setting_alignment(pack) == []
