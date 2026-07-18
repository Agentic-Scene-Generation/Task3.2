import unittest

import numpy as np

from scenesmith.agent_utils.mesh_frame import (
    gltf_y_up_bounds_to_scene_z_up,
    scene_dimensions_to_gltf_y_up,
    validate_uniform_dimension_fit,
)


class MeshFrameTest(unittest.TestCase):
    def test_scene_dimensions_are_reordered_for_gltf_scaling(self) -> None:
        self.assertEqual(
            scene_dimensions_to_gltf_y_up([1.6, 2.05, 0.8]),
            [1.6, 0.8, 2.05],
        )

    def test_y_up_bounds_become_grounded_z_up_bounds(self) -> None:
        bbox_min, bbox_max = gltf_y_up_bounds_to_scene_z_up(
            [[-2.05, 0.0, -2.25], [2.025, 2.5, 2.3]]
        )

        np.testing.assert_allclose(bbox_min, [-2.05, -2.3, 0.0])
        np.testing.assert_allclose(bbox_max, [2.025, 2.25, 2.5])

    def test_uniform_fit_accepts_normal_furniture_proportions(self) -> None:
        validate_uniform_dimension_fit(
            actual_dimensions=[1.6, 1.788, 0.982],
            requested_dimensions=[1.6, 2.05, 0.8],
        )

    def test_uniform_fit_rejects_freestanding_mesh_for_rug(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not fit requested proportions"):
            validate_uniform_dimension_fit(
                actual_dimensions=[1.8, 0.467, 1.203],
                requested_dimensions=[1.8, 1.8, 0.03],
            )


if __name__ == "__main__":
    unittest.main()
