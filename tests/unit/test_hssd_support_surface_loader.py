import gzip
import json
import tempfile
import unittest

from pathlib import Path

import numpy as np

from pydrake.math import RigidTransform

from scenesmith.agent_utils.hssd_retrieval.support_surface_loader import (
    _filter_surfaces_by_layer_spacing,
    hssd_support_surface_path,
    load_hssd_support_surfaces,
)
from scenesmith.agent_utils.room import SupportSurface, UniqueID
from scenesmith.agent_utils.support_surface_extraction import (
    SupportSurfaceExtractionConfig,
)


class TestFilterSurfacesByLayerSpacing(unittest.TestCase):
    @staticmethod
    def _surface(height: float, width: float = 1.0) -> SupportSurface:
        return SupportSurface(
            surface_id=UniqueID.generate(),
            bounding_box_min=np.array([-width / 2, -0.5, 0.01]),
            bounding_box_max=np.array([width / 2, 0.5, 0.5]),
            transform=RigidTransform(p=[0.0, 0.0, height]),
        )

    def test_keeps_all_coplanar_top_surface_pieces(self):
        lower_left = self._surface(0.72000)
        lower_right = self._surface(0.72002)
        top_left = self._surface(0.75000, width=1.2)
        top_right = self._surface(0.75005, width=1.1)

        filtered = _filter_surfaces_by_layer_spacing(
            [lower_left, top_left, lower_right, top_right],
            min_spacing=0.05,
            top_clearance=0.5,
        )

        self.assertEqual(filtered, [top_left, top_right])


class TestHssdUpholsteredSeatPolicy(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.mesh_id = "test_upholstered_seat"
        self.config = SupportSurfaceExtractionConfig()
        self.surface_counter = 0

    def tearDown(self):
        self.temp_dir.cleanup()

    def _scene(self):
        class Scene:
            pass

        scene = Scene()

        def generate_surface_id():
            self.surface_counter += 1
            return UniqueID(f"S_{self.surface_counter}")

        scene.generate_surface_id = generate_surface_id
        return scene

    @staticmethod
    def _surface(
        *,
        index: int,
        height: float = 0.5,
        width: float = 1.6,
        depth: float = 0.7,
        clearances: tuple[float, ...] = (0.001,) + (0.2,) * 9,
        horizontal: bool = True,
    ) -> dict:
        return {
            "index": index,
            "area": width * depth,
            "modelNormal": {"x": 0.0, "y": 1.0, "z": 0.0},
            "isHorizontal": horizontal,
            "obb": {
                "centroid": [0.0, height, 0.0],
                "axesLengths": [width, 0.01, depth],
                "normalizedAxes": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
            "samples": [{"clearance": value} for value in clearances],
        }

    def _write_annotation(self, surfaces: list[dict]) -> None:
        path = hssd_support_surface_path(self.mesh_id, self.data_dir)
        path.parent.mkdir(parents=True)
        with gzip.open(path, "wt", encoding="utf-8") as annotation:
            json.dump({"supportSurfaces": surfaces}, annotation)

    def _load(self, policy: str):
        return load_hssd_support_surfaces(
            mesh_id=self.mesh_id,
            config=self.config,
            scene=self._scene(),
            data_dir=self.data_dir,
            surface_policy=policy,
            furniture_bbox_min=np.array([-1.0, -0.5, 0.0]),
            furniture_bbox_max=np.array([1.0, 0.5, 1.0]),
            furniture_scale_factor=1.0,
        )

    def test_general_policy_keeps_strict_minimum_clearance(self):
        self._write_annotation([self._surface(index=5)])

        surfaces = self._load("general")

        self.assertEqual(surfaces, [])

    def test_upholstered_policy_uses_p90_and_insets_seat_bounds(self):
        self._write_annotation(
            [
                self._surface(index=5),
                self._surface(index=6, width=0.18, depth=0.18),
                self._surface(index=7, height=0.9),
                self._surface(index=8, horizontal=False),
            ]
        )

        surfaces = self._load("upholstered_seat")

        self.assertEqual(len(surfaces), 1)
        surface = surfaces[0]
        np.testing.assert_allclose(surface.bounding_box_min[:2], [-0.72, -0.30])
        np.testing.assert_allclose(surface.bounding_box_max[:2], [0.72, 0.30])
        self.assertAlmostEqual(
            surface.bounding_box_max[2] - surface.bounding_box_min[2], 0.2
        )
        self.assertAlmostEqual(surface.transform.translation()[2], 0.51)
        self.assertFalse(surface.open_above)

    def test_annotation_without_finite_clearance_samples_is_open_above(self):
        self._write_annotation([self._surface(index=5, clearances=())])

        surfaces = self._load("general")

        self.assertEqual(len(surfaces), 1)
        self.assertTrue(surfaces[0].open_above)

    def test_annotation_with_finite_clearance_samples_is_bounded(self):
        self._write_annotation([self._surface(index=5, clearances=(0.2,) * 10)])

        surfaces = self._load("general")

        self.assertEqual(len(surfaces), 1)
        self.assertFalse(surfaces[0].open_above)

    def test_upholstered_policy_still_rejects_low_p90_clearance(self):
        self._write_annotation(
            [self._surface(index=5, clearances=(0.001,) + (0.04,) * 9)]
        )

        surfaces = self._load("upholstered_seat")

        self.assertEqual(surfaces, [])


if __name__ == "__main__":
    unittest.main()
