"""Tests for support surface mesh scaling."""

import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import trimesh

from pydrake.all import RigidTransform

from scenesmith.agent_utils.rescale_helpers import rescale_object_common
from scenesmith.agent_utils.room import (
    ObjectType,
    RoomScene,
    SceneObject,
    SupportSurface,
    UniqueID,
    extract_and_propagate_support_surfaces,
)
from scenesmith.agent_utils.sdf_generator import rescale_sdf
from scenesmith.agent_utils.support_surface_extraction import (
    SupportSurfaceExtractionConfig,
)
from scenesmith.utils.sdf_utils import extract_uniform_visual_mesh_scale


class TestSupportSurfaceMeshScaling(unittest.TestCase):
    """Tests for support surface mesh scaling with furniture scale_factor."""

    def setUp(self):
        """Create a test mesh file with known dimensions."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create a simple 1m x 1m table mesh with a flat top surface.
        # Top surface at Z=0.75m, dimensions 1m x 1m.
        vertices = np.array(
            [
                # Top surface (Z=0.75).
                [-0.5, -0.5, 0.75],
                [0.5, -0.5, 0.75],
                [0.5, 0.5, 0.75],
                [-0.5, 0.5, 0.75],
                # Bottom (Z=0).
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ]
        )
        faces = np.array(
            [
                # Top.
                [0, 1, 2],
                [0, 2, 3],
                # Bottom.
                [4, 6, 5],
                [4, 7, 6],
                # Sides.
                [0, 4, 1],
                [1, 4, 5],
                [1, 5, 2],
                [2, 5, 6],
                [2, 6, 3],
                [3, 6, 7],
                [3, 7, 0],
                [0, 7, 4],
            ]
        )
        self.test_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        self.mesh_path = self.temp_dir / "test_table.glb"
        self.test_mesh.export(str(self.mesh_path))

        # Create mock scene.
        self.scene = Mock(spec=RoomScene)
        self.scene.objects = {}
        self.scene.generate_surface_id = Mock(
            side_effect=lambda: UniqueID(
                f"S_{len(self.scene.generate_surface_id.call_args_list)}"
            )
        )

        # Create config.
        self.config = SupportSurfaceExtractionConfig()

    def tearDown(self):
        """Clean up temp files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extracted_surface_mesh_is_scaled_with_furniture(self):
        """Surface mesh should be scaled by furniture's scale_factor."""
        scale_factor = 0.8

        # Create furniture with scale_factor=0.8.
        furniture = SceneObject(
            object_id=UniqueID("table_001"),
            object_type=ObjectType.FURNITURE,
            name="table",
            description="Test table",
            transform=RigidTransform(),
            geometry_path=self.mesh_path,
            sdf_path=None,
            support_surfaces=[],
            metadata={},
            bbox_min=np.array([-0.5, -0.5, 0.0]),
            bbox_max=np.array([0.5, 0.5, 0.75]),
            scale_factor=scale_factor,
        )
        self.scene.objects = {furniture.object_id: furniture}

        # Extract surfaces.
        surfaces = extract_and_propagate_support_surfaces(
            scene=self.scene, furniture_object=furniture, config=self.config
        )

        self.assertGreater(len(surfaces), 0, "Should extract at least one surface")

        # Get the top surface (should be the largest).
        top_surface = surfaces[0]
        self.assertIsNotNone(top_surface.mesh, "Surface should have a mesh")

        # The original mesh had vertices at +/-0.5.
        # With scale_factor=0.8, mesh vertices should be at +/-0.4.
        expected_max = 0.5 * scale_factor  # 0.4
        actual_max = np.abs(top_surface.mesh.vertices[:, :2]).max()

        self.assertAlmostEqual(
            actual_max,
            expected_max,
            places=2,
            msg=f"Mesh vertices should be scaled by {scale_factor}. "
            f"Expected max ~{expected_max:.2f}, got {actual_max:.2f}. "
            f"This indicates mesh is not being scaled.",
        )

    def test_propagated_surface_mesh_is_scaled(self):
        """Propagated surfaces should have mesh scaled by target's scale_factor."""
        # Create two furniture pieces with same geometry but different scales.
        furniture_1 = SceneObject(
            object_id=UniqueID("table_001"),
            object_type=ObjectType.FURNITURE,
            name="table",
            description="Test table",
            transform=RigidTransform(),
            geometry_path=self.mesh_path,
            sdf_path=None,
            support_surfaces=[],
            metadata={},
            scale_factor=1.0,
        )

        furniture_2 = SceneObject(
            object_id=UniqueID("table_002"),
            object_type=ObjectType.FURNITURE,
            name="table",
            description="Test table",
            transform=RigidTransform(p=[2.0, 0.0, 0.0]),  # Different position.
            geometry_path=self.mesh_path,  # Same geometry.
            sdf_path=None,
            support_surfaces=[],
            metadata={},
            scale_factor=0.7,  # Different scale.
        )

        self.scene.objects = {
            furniture_1.object_id: furniture_1,
            furniture_2.object_id: furniture_2,
        }

        # Extract surfaces for furniture_1 (should propagate to furniture_2).
        surfaces_1 = extract_and_propagate_support_surfaces(
            scene=self.scene, furniture_object=furniture_1, config=self.config
        )

        self.assertGreater(
            len(surfaces_1), 0, "Should extract surfaces for furniture_1"
        )
        self.assertGreater(
            len(furniture_2.support_surfaces),
            0,
            "Surfaces should propagate to furniture_2",
        )

        # Get top surfaces.
        surface_1 = surfaces_1[0]
        surface_2 = furniture_2.support_surfaces[0]

        self.assertIsNotNone(surface_1.mesh, "Surface 1 should have mesh")
        self.assertIsNotNone(surface_2.mesh, "Surface 2 should have mesh")
        self.assertTrue(surface_1.open_above)
        self.assertEqual(surface_2.open_above, surface_1.open_above)

        # Furniture_1 scale=1.0, furniture_2 scale=0.7.
        # Mesh max coords should be proportionally different.
        max_1 = np.abs(surface_1.mesh.vertices[:, :2]).max()
        max_2 = np.abs(surface_2.mesh.vertices[:, :2]).max()

        expected_ratio = 0.7 / 1.0
        actual_ratio = max_2 / max_1

        self.assertAlmostEqual(
            actual_ratio,
            expected_ratio,
            places=2,
            msg=f"Propagated mesh should be scaled by target's scale_factor. "
            f"Expected ratio {expected_ratio:.2f}, got {actual_ratio:.2f}.",
        )

    def test_empty_hssd_annotation_does_not_recompute_high_poly_mesh(self):
        """An explicit empty annotation is left to the hard-target policy."""
        furniture = SceneObject(
            object_id=UniqueID("hssd_table_001"),
            object_type=ObjectType.FURNITURE,
            name="dining_table",
            description="Test HSSD dining table",
            transform=RigidTransform(),
            geometry_path=self.mesh_path,
            sdf_path=None,
            support_surfaces=[],
            metadata={
                "asset_source": "hssd",
                "hssd_mesh_id": "empty_annotation_mesh",
            },
            bbox_min=np.array([-0.5, -0.5, 0.0]),
            bbox_max=np.array([0.5, 0.5, 0.75]),
        )
        self.scene.objects = {furniture.object_id: furniture}

        with (
            patch(
                "scenesmith.agent_utils.hssd_retrieval.support_surface_loader."
                "load_hssd_support_surfaces",
                return_value=[],
            ) as load_hssd,
            patch(
                "scenesmith.agent_utils.room.extract_support_surfaces_from_mesh"
            ) as extract_from_mesh,
        ):
            surfaces = extract_and_propagate_support_surfaces(
                scene=self.scene, furniture_object=furniture, config=self.config
            )

        load_hssd.assert_called_once()
        extract_from_mesh.assert_not_called()
        self.assertEqual(surfaces, [])
        self.assertEqual(furniture.support_surfaces, surfaces)

    def test_baked_hssd_geometry_still_scales_source_annotations(self):
        """The baked-mesh flag must not skip source-space annotation scaling."""
        furniture = SceneObject(
            object_id=UniqueID("table_0"),
            object_type=ObjectType.FURNITURE,
            name="table",
            description="HSSD table",
            transform=RigidTransform(),
            geometry_path=self.mesh_path,
            scale_factor=2.0,
            metadata={
                "asset_source": "hssd",
                "hssd_mesh_id": "table",
                "geometry_scale_baked": True,
            },
        )
        self.scene.objects = {furniture.object_id: furniture}
        local = SupportSurface(
            surface_id=UniqueID("local"),
            transform=RigidTransform(p=[0, 0, 0.5]),
            bounding_box_min=np.array([-0.5, -0.5, 0.01]),
            bounding_box_max=np.array([0.5, 0.5, 0.51]),
        )
        with patch(
            "scenesmith.agent_utils.hssd_retrieval.support_surface_loader.load_hssd_support_surfaces",
            return_value=[local],
        ):
            surface = extract_and_propagate_support_surfaces(
                self.scene, furniture, self.config
            )[0]
        np.testing.assert_allclose(surface.bounding_box_max[:2], [1.0, 1.0])
        np.testing.assert_allclose(surface.transform.translation(), [0.0, 0.0, 1.0])

    def _baked_table_with_sdf(self, scale: str | None = None) -> SceneObject:
        """Create a baked Y-up mesh, matching the production HSM input convention."""
        mesh_path = self.temp_dir / "baked_table.glb"
        mesh = self.test_mesh.copy()
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        )
        mesh.export(mesh_path)
        sdf_path = self.temp_dir / "baked_table.sdf"
        root = ET.Element("sdf", version="1.7")
        model = ET.SubElement(root, "model", name="table")
        ET.SubElement(model, "pose").text = "4 5 6 0 0 0"
        link = ET.SubElement(model, "link", name="base_link")
        for tag in ["visual", "collision"]:
            mesh = ET.SubElement(
                ET.SubElement(ET.SubElement(link, tag, name=tag), "geometry"), "mesh"
            )
            ET.SubElement(mesh, "uri").text = mesh_path.name
            if scale is not None:
                ET.SubElement(mesh, "scale").text = scale
        ET.ElementTree(root).write(sdf_path, encoding="utf-8")
        return SceneObject(
            object_id=UniqueID("baked_table_0"),
            object_type=ObjectType.FURNITURE,
            name="table",
            description="Already baked table mesh",
            transform=RigidTransform(),
            geometry_path=mesh_path,
            sdf_path=sdf_path,
            bbox_min=np.array([-0.5, -0.5, 0.0]),
            bbox_max=np.array([0.5, 0.5, 0.75]),
            scale_factor=2.0,
            metadata={"geometry_scale_baked": True},
        )

    def test_baked_hsm_support_tracks_actual_sdf_rescale(self):
        """A real post-bake rescale must not insert the support plane into the table."""
        for factor in [1.05, 0.5, 1.0]:
            with self.subTest(factor=factor):
                table = self._baked_table_with_sdf()
                self.scene.objects = {table.object_id: table}
                self.scene.get_object.return_value = table
                if factor != 1.0:
                    result = rescale_object_common(
                        self.scene, str(table.object_id), factor, "furniture"
                    )
                    self.assertTrue(result.success)
                surface = extract_and_propagate_support_surfaces(
                    self.scene, table, self.config
                )[0]
                plane = float(surface.transform.translation()[2])
                self.assertGreater(plane, 0.75 * factor)
                self.assertAlmostEqual(plane, 0.76 * factor, places=5)
                self.assertAlmostEqual(
                    float(np.abs(surface.mesh.vertices[:, :2]).max()),
                    0.5 * factor,
                    places=5,
                )
                # The already-baked source scale2 must not be applied again.
                self.assertAlmostEqual(table.scale_factor, 2.0 * factor)
                self.assertEqual(
                    ET.parse(table.sdf_path).findtext(".//model/pose"), "4 5 6 0 0 0"
                )

    def test_baked_surface_propagation_uses_each_actual_sdf(self):
        """Shared geometry at different runtime scales keeps distinct physical surfaces."""
        first = self._baked_table_with_sdf("1.05 1.05 1.05")
        second_sdf = self.temp_dir / "second.sdf"
        tree = ET.parse(first.sdf_path)
        for node in tree.findall(".//mesh/scale"):
            node.text = "0.5 0.5 0.5"
        tree.write(second_sdf)
        second = SceneObject(
            object_id=UniqueID("baked_table_1"),
            object_type=ObjectType.FURNITURE,
            name="table",
            description="Other instance",
            transform=RigidTransform(p=[2, 0, 0]),
            geometry_path=first.geometry_path,
            sdf_path=second_sdf,
            scale_factor=1.0,
            metadata={"geometry_scale_baked": True},
        )
        self.scene.objects = {first.object_id: first, second.object_id: second}
        extract_and_propagate_support_surfaces(self.scene, first, self.config)
        self.assertAlmostEqual(
            first.support_surfaces[0].transform.translation()[2], 0.76 * 1.05, places=5
        )
        self.assertAlmostEqual(
            second.support_surfaces[0].transform.translation()[2], 0.76 * 0.5, places=5
        )
        self.assertAlmostEqual(
            float(np.abs(second.support_surfaces[0].mesh.vertices[:, :2]).max()),
            0.25,
            places=5,
        )

    def test_hssd_annotations_keep_source_scale_after_sdf_rescale(self):
        """Post-bake SDF scale is only for mesh-space extraction, not source annotations."""
        table = self._baked_table_with_sdf()
        table.metadata.update(asset_source="hssd", hssd_mesh_id="table")
        self.scene.objects = {table.object_id: table}
        self.scene.get_object.return_value = table
        self.assertTrue(
            rescale_object_common(
                self.scene, str(table.object_id), 1.05, "furniture"
            ).success
        )
        local = SupportSurface(
            surface_id=UniqueID("local"),
            transform=RigidTransform(p=[0, 0, 0.5]),
            bounding_box_min=np.array([-0.5, -0.5, 0.01]),
            bounding_box_max=np.array([0.5, 0.5, 0.51]),
        )
        with patch(
            "scenesmith.agent_utils.hssd_retrieval.support_surface_loader.load_hssd_support_surfaces",
            return_value=[local],
        ):
            surface = extract_and_propagate_support_surfaces(
                self.scene, table, self.config
            )[0]
        self.assertAlmostEqual(surface.transform.translation()[2], 0.5 * 2.1)

    def test_per_link_surfaces_do_not_apply_sdf_scale_twice(self):
        """Per-link extraction has already incorporated the runtime SDF scale."""
        table = self._baked_table_with_sdf("1.05 1.05 1.05")
        table.metadata.update(is_articulated=True, asset_source="hssd")
        self.scene.objects = {table.object_id: table}
        config = SupportSurfaceExtractionConfig(recompute_hssd_surfaces=True)
        local = SupportSurface(
            surface_id=UniqueID("link_surface"),
            transform=RigidTransform(p=[0, 0, 0.798]),
            bounding_box_min=np.array([-0.525, -0.525, 0.01]),
            bounding_box_max=np.array([0.525, 0.525, 0.798]),
        )
        with patch(
            "scenesmith.agent_utils.room.extract_support_surfaces_articulated",
            return_value=[local],
        ):
            surface = extract_and_propagate_support_surfaces(self.scene, table, config)[
                0
            ]
        self.assertAlmostEqual(surface.transform.translation()[2], 0.798)

    def test_sdf_visual_scale_matches_exact_mesh_and_accumulates_rescale(self):
        """Ignore other visuals and read the live scale after repeated SDF edits."""
        table = self._baked_table_with_sdf()
        tree = ET.parse(table.sdf_path)
        other = ET.SubElement(
            ET.SubElement(
                ET.SubElement(tree.find(".//link"), "visual", name="other"), "geometry"
            ),
            "mesh",
        )
        ET.SubElement(other, "uri").text = "other.obj"
        ET.SubElement(other, "scale").text = "3 3 3"
        tree.write(table.sdf_path)
        self.assertEqual(
            extract_uniform_visual_mesh_scale(table.sdf_path, table.geometry_path), 1.0
        )
        rescale_sdf(table.sdf_path, 0.5)
        rescale_sdf(table.sdf_path, 2.1)
        self.assertAlmostEqual(
            extract_uniform_visual_mesh_scale(table.sdf_path, table.geometry_path), 1.05
        )
        tree = ET.parse(table.sdf_path)
        tree.find(".//visual/geometry/mesh/uri").text = table.geometry_path.as_uri()
        tree.write(table.sdf_path)
        self.assertAlmostEqual(
            extract_uniform_visual_mesh_scale(table.sdf_path, table.geometry_path), 1.05
        )

    def test_sdf_visual_scale_rejects_invalid_or_ambiguous_geometry(self):
        """Reject unsupported physical scale bindings rather than guessing a surface."""
        for scale in ["1 1 2", "nan nan nan", "-1 -1 -1"]:
            with self.subTest(scale=scale):
                table = self._baked_table_with_sdf(scale)
                with self.assertRaises(ValueError):
                    extract_uniform_visual_mesh_scale(
                        table.sdf_path, table.geometry_path
                    )
        table = self._baked_table_with_sdf("1 1 1")
        tree = ET.parse(table.sdf_path)
        other = ET.SubElement(
            ET.SubElement(
                ET.SubElement(tree.find(".//link"), "visual", name="duplicate"),
                "geometry",
            ),
            "mesh",
        )
        ET.SubElement(other, "uri").text = table.geometry_path.name
        ET.SubElement(other, "scale").text = "2 2 2"
        tree.write(table.sdf_path)
        with self.assertRaises(ValueError):
            extract_uniform_visual_mesh_scale(table.sdf_path, table.geometry_path)
        with self.assertRaises(ValueError):
            extract_uniform_visual_mesh_scale(
                table.sdf_path, self.temp_dir / "missing.obj"
            )

    def test_upholstered_policy_passes_unscaled_bounds_without_propagation(self):
        """A relaxed seat surface remains scoped to its hard target instance."""
        geometry_path = self.mesh_path
        furniture = SceneObject(
            object_id=UniqueID("loveseat_0"),
            object_type=ObjectType.FURNITURE,
            name="loveseat",
            description="Test HSSD loveseat",
            transform=RigidTransform(),
            geometry_path=geometry_path,
            support_surfaces=[],
            metadata={
                "asset_source": "hssd",
                "hssd_mesh_id": "annotated_loveseat",
            },
            bbox_min=np.array([-1.0, -0.5, 0.0]),
            bbox_max=np.array([1.0, 0.5, 1.0]),
            scale_factor=2.0,
        )
        identical = SceneObject(
            object_id=UniqueID("loveseat_1"),
            object_type=ObjectType.FURNITURE,
            name="loveseat",
            description="Identical optional loveseat",
            transform=RigidTransform(p=[3.0, 0.0, 0.0]),
            geometry_path=geometry_path,
            support_surfaces=[],
            metadata={},
        )
        self.scene.objects = {
            furniture.object_id: furniture,
            identical.object_id: identical,
        }
        local_surface = SupportSurface(
            surface_id=UniqueID("local_seat"),
            bounding_box_min=np.array([-0.4, -0.2, 0.01]),
            bounding_box_max=np.array([0.4, 0.2, 0.2]),
            transform=RigidTransform(p=[0.0, 0.0, 0.5]),
        )

        with patch(
            "scenesmith.agent_utils.hssd_retrieval.support_surface_loader."
            "load_hssd_support_surfaces",
            return_value=[local_surface],
        ) as load_hssd:
            surfaces = extract_and_propagate_support_surfaces(
                scene=self.scene,
                furniture_object=furniture,
                config=self.config,
                surface_policy="upholstered_seat",
            )

        _, kwargs = load_hssd.call_args
        self.assertEqual(kwargs["surface_policy"], "upholstered_seat")
        np.testing.assert_allclose(kwargs["furniture_bbox_min"], [-0.5, -0.25, 0.0])
        np.testing.assert_allclose(kwargs["furniture_bbox_max"], [0.5, 0.25, 0.5])
        self.assertEqual(kwargs["furniture_scale_factor"], 2.0)
        self.assertEqual(len(surfaces), 1)
        self.assertEqual(
            furniture.metadata["support_surface_policy"], "upholstered_seat"
        )
        self.assertEqual(identical.support_surfaces, [])


class TestSupportSurfaceBehavior(unittest.TestCase):
    """Tests for SupportSurface behavior with scaled meshes."""

    def test_contains_point_2d_respects_scaled_mesh(self):
        """Point at edge of unscaled surface should be outside when mesh is scaled down."""
        vertices = np.array(
            [
                [-0.5, -0.5, 0],
                [0.5, -0.5, 0],
                [0.5, 0.5, 0],
                [-0.5, 0.5, 0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh_1x = trimesh.Trimesh(vertices=vertices, faces=faces)

        surface_1x = SupportSurface(
            surface_id=UniqueID("test_surface"),
            bounding_box_min=np.array([-0.5, -0.5, 0]),
            bounding_box_max=np.array([0.5, 0.5, 0.01]),
            transform=RigidTransform(),
            mesh=mesh_1x,
        )

        # Point at (0.4, 0.0) should be INSIDE 1x surface.
        self.assertTrue(surface_1x.contains_point_2d(np.array([0.4, 0.0])))

        # Scaled surface.
        mesh_0_5x = trimesh.Trimesh(vertices=vertices * 0.5, faces=faces)
        surface_0_5x = SupportSurface(
            surface_id=UniqueID("test_surface_scaled"),
            bounding_box_min=np.array([-0.25, -0.25, 0]),
            bounding_box_max=np.array([0.25, 0.25, 0.01]),
            transform=RigidTransform(),
            mesh=mesh_0_5x,
        )

        # Point at (0.4, 0.0) should be OUTSIDE 0.5x surface.
        self.assertFalse(surface_0_5x.contains_point_2d(np.array([0.4, 0.0])))

        # Point at (0.2, 0.0) should be INSIDE 0.5x surface.
        self.assertTrue(surface_0_5x.contains_point_2d(np.array([0.2, 0.0])))


if __name__ == "__main__":
    unittest.main()
