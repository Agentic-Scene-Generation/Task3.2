import tempfile
import unittest

from pathlib import Path

import trimesh

from scenesmith.agent_utils.asset_structure import (
    MeshComponentEvidence,
    evaluate_component_extents,
    inspect_hssd_candidate_structure,
)


class AssetStructureTest(unittest.TestCase):
    def test_rejects_bed_with_vertical_wall_sized_thin_component(self) -> None:
        result = evaluate_component_extents(
            family="bed",
            up_axis="+Z",
            components=[
                MeshComponentEvidence("bed", (1.65, 2.05, 0.75)),
                MeshComponentEvidence("Shader7", (1.747, 0.151, 2.105)),
            ],
        )

        self.assertTrue(result.rejected)
        self.assertIn("Shader7", result.reason)

    def test_accepts_mattress_and_normal_headboard(self) -> None:
        result = evaluate_component_extents(
            family="bed",
            up_axis="+Z",
            components=[
                MeshComponentEvidence("mattress", (1.60, 2.05, 0.25)),
                MeshComponentEvidence("headboard", (1.60, 0.12, 1.20)),
            ],
        )

        self.assertEqual("pass", result.status)

    def test_uses_declared_up_axis_instead_of_dimension_sorting(self) -> None:
        result = evaluate_component_extents(
            family="sofa",
            up_axis="+Y",
            components=[
                MeshComponentEvidence("backdrop", (1.80, 1.90, 0.14)),
            ],
        )

        self.assertTrue(result.rejected)

    def test_does_not_apply_backdrop_rule_to_panel_furniture(self) -> None:
        result = evaluate_component_extents(
            family="wardrobe",
            up_axis="+Z",
            components=[
                MeshComponentEvidence("door_panel", (1.80, 0.08, 2.10)),
            ],
        )

        self.assertEqual("pass", result.status)

    def test_inspection_binds_result_to_source_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "compound_bed.glb"
            scene = trimesh.Scene()
            scene.add_geometry(
                trimesh.creation.box(extents=(1.60, 2.05, 0.70)),
                node_name="bed",
                geom_name="bed",
            )
            scene.add_geometry(
                trimesh.creation.box(extents=(1.747, 0.151, 2.105)),
                node_name="Shader7",
                geom_name="Shader7",
            )
            path.write_bytes(scene.export(file_type="glb"))

            result = inspect_hssd_candidate_structure(
                mesh_path=path,
                family="bed",
                up_axis="+Z",
            )

            self.assertTrue(result.rejected)
            self.assertEqual(64, len(result.geometry_fingerprint))


if __name__ == "__main__":
    unittest.main()
