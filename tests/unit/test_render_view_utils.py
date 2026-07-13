"""Tests for bpy-free render view construction."""

import math
import unittest

from scenesmith.agent_utils.render_view_utils import generate_angled_drawer_view


class TestRenderViewUtils(unittest.TestCase):
    """Verify drawer views remain serializable without mathutils or bpy."""

    def test_drawer_direction_is_normalized(self) -> None:
        view = generate_angled_drawer_view(
            surface={"surface_id": "drawer_surface"},
            joint_name="drawer_joint",
            drawer_direction=[3.0, 4.0, 0.0],
        )

        direction = view["direction"]
        magnitude = math.sqrt(sum(value * value for value in direction))
        self.assertAlmostEqual(magnitude, 1.0)
        self.assertEqual(view["name"], "drawer_drawer_joint_drawer_surface")
        self.assertTrue(view["is_drawer_view"])

    def test_vertical_motion_uses_stable_fallback(self) -> None:
        view = generate_angled_drawer_view(
            surface={},
            joint_name="vertical_joint",
            drawer_direction=[0.0, 0.0, 1.0],
            view_index=2,
        )

        self.assertEqual(view["name"], "drawer_vertical_joint_surface_2")
        self.assertEqual(len(view["direction"]), 3)


if __name__ == "__main__":
    unittest.main()
