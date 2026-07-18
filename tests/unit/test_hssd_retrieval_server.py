import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from scenesmith.agent_utils.hssd_retrieval_server.dataclasses import (
    HssdRetrievalServerRequest,
)
from scenesmith.agent_utils.hssd_retrieval_server.server_app import HssdRetrievalApp


class HssdRetrievalServerTest(unittest.TestCase):
    def test_scene_dimensions_are_converted_before_gltf_size_ranking(self) -> None:
        app = HssdRetrievalApp(preload_retriever=False)
        retriever = MagicMock()
        retriever.config.object_type_mapping = {"FURNITURE": "large_objects"}
        mesh = MagicMock()
        mesh.extents = np.asarray([1.6, 0.8, 2.05], dtype=float)
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
        np.testing.assert_allclose(ranked_dimensions, [1.6, 0.8, 2.05])


if __name__ == "__main__":
    unittest.main()
