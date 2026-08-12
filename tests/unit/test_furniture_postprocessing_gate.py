from types import SimpleNamespace

import numpy as np
import pytest

from PIL import Image
from pydrake.all import RigidTransform

from scenesmith.agent_utils.furniture_postprocessing_gate import (
    repair_invalid_postprocessed_furniture,
    snapshot_furniture_transforms,
    validate_render_directory,
)
from scenesmith.agent_utils.room import ObjectType


class _Furniture:
    def __init__(self, object_id: str, xyz: tuple[float, float, float]) -> None:
        self.object_id = object_id
        self.object_type = ObjectType.FURNITURE
        self.bbox_min = np.array([-0.25, -0.25, 0.0])
        self.bbox_max = np.array([0.25, 0.25, 1.0])
        self.transform = RigidTransform(p=xyz)

    def compute_world_bounds(self):
        translation = self.transform.translation()
        return self.bbox_min + translation, self.bbox_max + translation


class _Scene:
    def __init__(self, furniture: _Furniture) -> None:
        self.objects = {furniture.object_id: furniture}
        self.room_geometry = SimpleNamespace(
            length=4.0,
            width=4.0,
            footprint_vertices=[
                (-2.0, -2.0),
                (2.0, -2.0),
                (2.0, 0.0),
                (0.0, 0.0),
                (0.0, 2.0),
                (-2.0, 2.0),
            ],
        )

    def remove_object(self, object_id):
        self.objects.pop(object_id, None)


def test_postphysics_gate_rolls_back_polygon_escape():
    furniture = _Furniture("chair_0", (-1.0, 1.0, 0.0))
    scene = _Scene(furniture)
    originals = snapshot_furniture_transforms(scene)
    furniture.transform = RigidTransform(p=[1.0, 1.0, 0.0])

    audit = repair_invalid_postprocessed_furniture(scene, originals)

    assert audit[0]["action"] == "rolled_back_to_pre_simulation_transform"
    assert "POLYGON_CONTAINMENT_CONFLICT" in audit[0]["reasons"]
    assert np.allclose(furniture.transform.translation(), [-1.0, 1.0, 0.0])


def test_render_gate_rejects_black_and_accepts_nonempty_image(tmp_path):
    Image.new("RGB", (64, 64), "black").save(tmp_path / "0_top.png")
    with pytest.raises(RuntimeError, match="almost entirely black"):
        validate_render_directory(tmp_path)

    image = Image.new("RGB", (64, 64), "black")
    for x in range(16, 48):
        for y in range(16, 48):
            image.putpixel((x, y), (160, 120, 80))
    image.save(tmp_path / "0_top.png")
    assert validate_render_directory(tmp_path) == [tmp_path / "0_top.png"]
