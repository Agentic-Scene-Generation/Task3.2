"""Cutlery must follow the actual collision-frame long axis on every table pose."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.manipuland_agents.tools.manipuland_tools import ManipulandTools


@pytest.mark.parametrize("size", [(0.19, 0.035), (0.035, 0.19)])
@pytest.mark.parametrize("surface_yaw", [0.0, math.pi / 2])
@pytest.mark.parametrize("forward", [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)])
def test_collision_long_axis_is_parallel_to_seat_approach(
    size: tuple[float, float], surface_yaw: float, forward: tuple[float, float],
) -> None:
    utensil = SimpleNamespace(
        name="dinner_fork", description="A four-tine fork.",
        bbox_min=np.array([-size[0] / 2, -size[1] / 2, 0.0]),
        bbox_max=np.array([size[0] / 2, size[1] / 2, 0.015]),
        placement_info=SimpleNamespace(rotation_2d=0.0),
    )
    surface = SimpleNamespace(
        transform=RigidTransform(RollPitchYaw(0.0, 0.0, surface_yaw), [0, 0, 0]),
    )
    degrees = ManipulandTools._dining_companion_rotation_degrees(utensil, surface, forward)
    rotation = RollPitchYaw(0.0, 0.0, math.radians(degrees)).ToRotationMatrix().matrix()
    intrinsic_axis = np.array([1.0, 0.0, 0.0]) if size[0] > size[1] else np.array([0.0, 1.0, 0.0])
    actual = (surface.transform.rotation().matrix() @ rotation @ intrinsic_axis)[:2]
    assert np.dot(actual, np.array(forward)) == pytest.approx(1.0, abs=1e-8)


def test_glass_keeps_designer_yaw_even_when_its_bbox_is_longer_in_x() -> None:
    glass = SimpleNamespace(
        name="water_glass", description="A water glass.",
        bbox_min=np.array([-0.08, -0.03, 0.0]), bbox_max=np.array([0.08, 0.03, 0.1]),
        placement_info=SimpleNamespace(rotation_2d=0.3),
    )
    surface = SimpleNamespace(transform=RigidTransform())
    assert ManipulandTools._dining_companion_rotation_degrees(glass, surface, (0, 1)) == pytest.approx(math.degrees(0.3))
