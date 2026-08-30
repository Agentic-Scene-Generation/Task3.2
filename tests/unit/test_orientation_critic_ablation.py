from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from pydrake.math import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID
from scripts.run_orientation_critic_ablation import (
    VARIANTS,
    apply_variant,
    critic_llm_prompt,
)


def _scene_with_targets() -> RoomScene:
    objects = {}
    for object_id in (
        "dining_chair_0",
        "dining_chair_1",
        "dining_chair_2",
        "dining_chair_3",
        "sofa_0",
    ):
        objects[UniqueID(object_id)] = SceneObject(
            object_id=UniqueID(object_id),
            object_type=ObjectType.FURNITURE,
            name=object_id.rsplit("_", 1)[0],
            description=object_id,
            transform=RigidTransform(rpy=RollPitchYaw(0.0, 0.0, 0.25), p=np.zeros(3)),
        )
    return RoomScene(room_geometry=None, scene_dir=Path("."), objects=objects)


def test_orientation_variants_only_rotate_declared_targets() -> None:
    scene = _scene_with_targets()
    before = {
        str(object_id): RollPitchYaw(obj.transform.rotation()).yaw_angle()
        for object_id, obj in scene.objects.items()
    }
    variant = next(item for item in VARIANTS if item.name == "sofa_away_from_tv")

    changes = apply_variant(scene, variant)

    assert [change["object_id"] for change in changes] == ["sofa_0"]
    for object_id, obj in scene.objects.items():
        yaw = RollPitchYaw(obj.transform.rotation()).yaw_angle()
        delta = (yaw - before[str(object_id)]) % (2 * math.pi)
        expected = math.pi if str(object_id) == "sofa_0" else 0.0
        assert delta == pytest.approx(expected)


def test_sideways_variant_rotates_by_ninety_degrees() -> None:
    scene = _scene_with_targets()
    before = RollPitchYaw(
        scene.objects[UniqueID("sofa_0")].transform.rotation()
    ).yaw_angle()
    variant = next(item for item in VARIANTS if item.name == "sofa_sideways_to_tv")

    apply_variant(scene, variant)

    after = RollPitchYaw(
        scene.objects[UniqueID("sofa_0")].transform.rotation()
    ).yaw_angle()
    assert (after - before) % (2 * math.pi) == pytest.approx(math.pi / 2)


def test_critic_llm_prompt_only_injects_geometry_context_when_enabled() -> None:
    variant = next(item for item in VARIANTS if item.name == "chairs_sideways_to_table")
    context = "- fail: dining_chair_0 facing error is 91deg."

    on_prompt = critic_llm_prompt(variant=variant, benchmark_context=context)
    off_prompt = critic_llm_prompt(variant=variant, benchmark_context=None)

    assert variant.fragment_prompt in on_prompt
    assert variant.fragment_prompt in off_prompt
    assert variant.task_requirement in on_prompt
    assert variant.task_requirement in off_prompt
    assert variant.expected_orientation not in on_prompt
    assert variant.expected_orientation not in off_prompt
    assert "Additional SceneBenchmark geometry critic context" in on_prompt
    assert context in on_prompt
    assert "authoritative for this functional orientation" in on_prompt
    assert "Additional SceneBenchmark geometry critic context" not in off_prompt
    assert context not in off_prompt
