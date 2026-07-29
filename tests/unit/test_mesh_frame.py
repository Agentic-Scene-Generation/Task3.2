import unittest

import numpy as np

from scenesmith.agent_utils.mesh_frame import (
    axis_agnostic_uniform_fit_exists,
    axis_agnostic_uniform_scale_shape_error,
    choose_uniform_scale_for_contract,
    gltf_y_up_bounds_to_scene_z_up,
    hssd_dimension_shape_error,
    scene_dimensions_to_gltf_y_up,
    uniform_scale_shape_error,
    validate_uniform_dimension_fit,
)


class MeshFrameTest(unittest.TestCase):
    def test_scale_invariant_shape_error_ignores_arbitrary_source_scale(self) -> None:
        target = [1.4, 0.3, 1.4]

        shape_match = uniform_scale_shape_error([14.0, 3.0, 14.0], target)
        wrong_shape = uniform_scale_shape_error([1.4, 1.2, 1.0], target)

        self.assertLess(shape_match, wrong_shape)

    def test_hssd_shape_error_accepts_scene_and_gltf_extent_orders(self) -> None:
        target = [0.9, 0.55, 2.0]

        scene_order = hssd_dimension_shape_error([0.9, 0.55, 2.0], target)
        gltf_order = hssd_dimension_shape_error([0.9, 2.0, 0.55], target)

        self.assertAlmostEqual(scene_order, 0.0)
        self.assertAlmostEqual(gltf_order, 0.0)

    def test_hssd_shape_error_does_not_treat_raw_axis_order_as_semantic(self) -> None:
        target = [1.2, 0.6, 0.75]

        error = axis_agnostic_uniform_scale_shape_error(
            [0.6276, 1.0963, 0.75],
            target,
        )

        self.assertLess(error, 0.2)
        self.assertTrue(
            axis_agnostic_uniform_fit_exists(
                [0.6276, 1.0963, 0.75],
                target,
                min_ratio=0.75,
                max_ratio=1.35,
            )
        )

    def test_uniform_scale_solver_uses_feasible_family_interval(self) -> None:
        source = np.array([1.6, 1.416, 0.944])

        scale, normalized = choose_uniform_scale_for_contract(
            source,
            [1.6, 2.05, 0.8],
            min_ratio=0.75,
            max_ratio=1.35,
            minimum_dimensions=[1.2, 1.6, 0.25],
            maximum_dimensions=[2.4, 2.4, 1.3],
        )
        actual = source * scale

        np.testing.assert_allclose(normalized, [1.6, 2.05, 0.8])
        self.assertGreaterEqual(actual[1], 1.6 - 1e-6)
        self.assertLessEqual(actual[2] / normalized[2], 1.35 + 1e-6)

    def test_uniform_scale_solver_rejects_incompatible_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot satisfy"):
            choose_uniform_scale_for_contract(
                [1.6, 2.0, 0.2],
                [1.7, 0.9, 0.85],
                min_ratio=0.75,
                max_ratio=1.35,
                minimum_dimensions=[1.1, 0.55, 0.55],
                maximum_dimensions=[3.5, 1.3, 1.3],
            )

    def test_family_envelope_makes_designer_dimensions_a_soft_target(self) -> None:
        scale, normalized = choose_uniform_scale_for_contract(
            [1.10, 0.63, 0.75],
            [0.60, 0.80, 0.75],
            min_ratio=0.75,
            max_ratio=1.35,
            minimum_dimensions=[0.90, 0.45, 0.60],
            maximum_dimensions=[1.30, 0.85, 0.95],
            enforce_requested_ratio=False,
        )

        actual = np.asarray([1.10, 0.63, 0.75]) * scale
        np.testing.assert_allclose(normalized, [0.90, 0.80, 0.75])
        self.assertTrue(np.all(actual >= [0.90, 0.45, 0.60]))
        self.assertTrue(np.all(actual <= [1.30, 0.85, 0.95]))

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

    def test_critical_fit_rejects_severely_short_bed(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "does not fit requested proportions",
        ):
            validate_uniform_dimension_fit(
                actual_dimensions=[1.6, 1.416, 0.944],
                requested_dimensions=[1.6, 2.05, 0.8],
                min_ratio=0.75,
                max_ratio=1.35,
            )

    def test_uniform_fit_accepts_float_roundoff_at_boundary(self) -> None:
        validate_uniform_dimension_fit(
            actual_dimensions=[0.7, 0.3 - 3e-8, 2.2],
            requested_dimensions=[0.7, 0.6, 2.2],
            min_ratio=0.5,
        )

    def test_uniform_fit_rejects_freestanding_mesh_for_rug(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not fit requested proportions"):
            validate_uniform_dimension_fit(
                actual_dimensions=[1.8, 0.467, 1.203],
                requested_dimensions=[1.8, 1.8, 0.03],
            )


if __name__ == "__main__":
    unittest.main()
