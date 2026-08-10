import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import trimesh

from scenesmith.agent_utils.asset_structure import inspect_hssd_candidate_structure
from scenesmith.agent_utils.hssd_retrieval_server.dataclasses import (
    HssdRetrievalServerRequest,
)
from scenesmith.agent_utils.hssd_retrieval_server.server_app import HssdRetrievalApp


class HssdRetrievalServerTest(unittest.TestCase):
    def test_scene_dimensions_are_preserved_for_hssd_size_ranking(self) -> None:
        app = HssdRetrievalApp(preload_retriever=False)
        retriever = MagicMock()
        retriever.config.object_type_mapping = {"FURNITURE": "large_objects"}
        mesh = MagicMock()
        mesh.extents = np.asarray([1.6, 2.05, 0.8], dtype=float)
        retriever.retrieve_multiple.return_value = [
            SimpleNamespace(mesh=mesh, mesh_id="bed_candidate", clip_score=0.9)
        ]
        app._retriever = retriever

        with tempfile.TemporaryDirectory() as output_dir:
            app._retrieve_internal(
                HssdRetrievalServerRequest(
                    object_description="upholstered bed",
                    object_type="FURNITURE",
                    desired_dimensions=(1.6, 2.05, 0.8),
                    output_dir=str(Path(output_dir)),
                )
            )

        ranked_dimensions = retriever.retrieve_multiple.call_args.kwargs[
            "desired_dimensions"
        ]
        np.testing.assert_allclose(ranked_dimensions, [1.6, 2.05, 0.8])

    def test_server_preserves_original_scene_graph_for_structural_admission(
        self,
    ) -> None:
        app = HssdRetrievalApp(preload_retriever=False)
        retriever = MagicMock()
        retriever.config.object_type_mapping = {"FURNITURE": "large_objects"}
        flattened = trimesh.creation.box(extents=(1.8, 0.9, 0.8))

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.glb"
            source_scene = trimesh.Scene()
            source_scene.add_geometry(
                trimesh.creation.box(extents=(1.8, 0.9, 0.8)),
                node_name="sofa_body",
                geom_name="sofa_body",
            )
            source_scene.add_geometry(
                trimesh.creation.box(extents=(2.01, 0.03, 0.80)),
                node_name="attached_panel",
                geom_name="attached_panel",
            )
            source_path.write_bytes(source_scene.export(file_type="glb"))
            retriever.retrieve_multiple.return_value = [
                SimpleNamespace(
                    mesh=flattened,
                    mesh_id="compound_sofa",
                    clip_score=0.9,
                    metadata=SimpleNamespace(up="", front=""),
                    source_mesh_path=source_path,
                )
            ]
            app._retriever = retriever

            response = app._retrieve_internal(
                HssdRetrievalServerRequest(
                    object_description="sofa",
                    object_type="FURNITURE",
                    desired_dimensions=(1.8, 0.9, 0.8),
                    output_dir=tmp,
                )
            )
            result = response.results[0]
            self.assertIsNotNone(result.structure_mesh_path)
            structural = inspect_hssd_candidate_structure(
                mesh_path=result.structure_mesh_path,
                family="sofa",
                up_axis="+Z",
            )

            self.assertGreater(
                structural.evidence.get("component_count", 0),
                1,
            )
            self.assertEqual("inconclusive", structural.status)


if __name__ == "__main__":
    unittest.main()
