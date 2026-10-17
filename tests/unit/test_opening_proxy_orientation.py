"""Structural opening axes must preserve the room's cardinal wall frame."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scenesmith.agent_utils.house import WallDirection
from scenesmith.scenebenchmark_critic.adapter import _opening_geometry_proxy
from scenesmith.scenebenchmark_critic.core.geometry import front_vector
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
    _evaluate_axial_relation,
)
from scenesmith.scenebenchmark_critic.opening_geometry import opening_physical_bounds


@pytest.mark.parametrize("direction", list(WallDirection))
@pytest.mark.parametrize("opening_type", ["window", "door", "open"])
def test_opening_relation_axis_points_into_room(
    direction: WallDirection, opening_type: str
) -> None:
    """Interior objects are in front; exterior objects remain behind."""
    normal = np.array(direction.get_inward_normal())
    center = -2.0 * normal
    opening = SimpleNamespace(
        opening_id="opening_0",
        opening_type=opening_type,
        wall_direction=direction,
        center_world=[*center, 1.5],
        width=1.2,
        sill_height=0.9,
        height=1.2,
    )
    original = copy.deepcopy(vars(opening))
    scene = SimpleNamespace(
        room_id="room_0", room_geometry=SimpleNamespace(wall_thickness=0.05)
    )
    offset = np.array([7.0, -3.0, 0.0])
    proxy = _opening_geometry_proxy(opening, scene, offset)
    assert proxy is not None
    assert front_vector(proxy) == pytest.approx(normal)
    lower, upper = opening_physical_bounds(opening, offset=offset)
    assert proxy["bbox_world"]["min"] == pytest.approx(lower)
    assert proxy["bbox_world"]["max"] == pytest.approx(upper)
    assert vars(opening) == original

    category = "opening" if opening_type == "open" else opening_type
    constraint = {
        "constraint_id": "front_test",
        "relation": "in_front_of",
        "subjects": {"category": "rug", "count": 1},
        "targets": {"category": category, "count": 1},
        "source": "explicit_prompt",
        "strength": "hard",
    }
    for interior in [True, False]:
        point = center + offset[:2] + (normal if interior else -normal)
        subject = {
            "id": "rug_0",
            "category": "rug",
            "object_type": "furniture",
            "bbox_world": {
                "center": [*point, 0.01],
                "min": [*(point - 0.1), 0.0],
                "max": [*(point + 0.1), 0.02],
            },
        }
        objects = [subject, proxy]
        results = _evaluate_axial_relation(
            constraint,
            case_pack={"scene_geometry": {"objects": objects}},
            objects=objects,
            tier="core",
            behind=False,
        )
        assert len(results) == 1
        assert results[0]["label"] == ("pass" if interior else "fail")
        reverse = _evaluate_axial_relation(
            dict(constraint, relation="behind"),
            case_pack={"scene_geometry": {"objects": objects}},
            objects=objects,
            tier="core",
            behind=True,
        )
        assert reverse[0]["label"] == ("fail" if interior else "pass")
        if not interior:
            repair = np.array(results[0]["diagnostics"]["repair_target_center_xy_m"])
            assert np.dot(repair - (center + offset[:2]), normal) > 0.0


def test_non_cardinal_opening_keeps_existing_unavailable_proxy() -> None:
    """Do not infer an unsupported opening frame from its position or name."""
    opening = SimpleNamespace(
        opening_id="north_window",
        opening_type="window",
        wall_direction=None,
        center_world=[0.0, 2.0, 1.5],
        width=1.2,
        sill_height=0.9,
        height=1.2,
        wall_inward_normal=[0.6, -0.8],
    )
    scene = SimpleNamespace(
        room_id="room_0", room_geometry=SimpleNamespace(wall_thickness=0.05)
    )
    assert _opening_geometry_proxy(opening, scene, np.zeros(3)) is None
