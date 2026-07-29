import unittest

import numpy as np

from pydrake.math import RigidTransform

from scenesmith.agent_utils.hssd_retrieval.support_surface_loader import (
    _filter_surfaces_by_layer_spacing,
)
from scenesmith.agent_utils.room import SupportSurface, UniqueID


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


if __name__ == "__main__":
    unittest.main()
