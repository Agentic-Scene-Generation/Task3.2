"""Unit tests for all-source Zvec retrieval without a native Zvec index."""

import json
import sys
import tempfile
import types
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import trimesh

from scenesmith.agent_utils.hssd_retrieval.all_assets_zvec import (
    AllAssetsZvecRetriever,
)
from scenesmith.agent_utils.hssd_retrieval.config import HssdZvecConfig
from scenesmith.agent_utils.hssd_retrieval_server.server_app import HssdRetrievalApp


class _FakeDocument:
    def __init__(self, asset_id: str, asset_path: Path, score: float) -> None:
        self.fields = {"asset_id": asset_id, "asset_path": str(asset_path)}
        self.score = score


class _FakeCollection:
    def __init__(self, documents: list[_FakeDocument]) -> None:
        self.documents = documents
        self.query_calls: list[tuple[object, int, list[str]]] = []

    def query(self, queries: object, topk: int, output_fields: list[str]):
        self.query_calls.append((queries, topk, output_fields))
        return self.documents


class TestAllAssetsZvecRetriever(unittest.TestCase):
    def test_server_rejects_all_assets_mode_without_unit_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            collection_path = Path(temp_dir) / "collection"
            collection_path.mkdir()
            app = HssdRetrievalApp(
                preload_retriever=False,
                hssd_retrieval_backend="all_assets_embedding",
                hssd_zvec_collection_path=str(collection_path),
                hssd_embedding_base_url="http://127.0.0.1:8014",
            )

            with self.assertRaisesRegex(
                ValueError, "requires HSSD_ALL_ASSETS_MANIFEST_PATH"
            ):
                app._get_retriever()

    def test_uses_per_asset_manifest_scale_before_dimension_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            collection_path = root / "collection"
            collection_path.mkdir()
            asset_path = root / "chair.obj"
            trimesh.creation.box(extents=[100.0, 200.0, 50.0]).export(asset_path)
            asset_id = "3dfuture:chair-in-centimeters"
            manifest_path = root / "embedding_inputs.shared.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "asset_uid": asset_id,
                        "geometry_ref": {"unit_scale_to_m": 0.01},
                    }
                )
                + "\n"
            )
            config = HssdZvecConfig(
                collection_path=collection_path,
                base_url="http://127.0.0.1:8014",
                all_assets_manifest_path=manifest_path,
            )
            retriever = AllAssetsZvecRetriever(config, top_k=1)
            retriever._client = MagicMock()
            retriever._client.embed_text.return_value = [0.1, 0.2]
            retriever._collection = _FakeCollection(
                [_FakeDocument(asset_id, asset_path, score=0.9)]
            )
            fake_zvec = types.SimpleNamespace(
                Query=lambda **kwargs: kwargs,
            )

            with patch.dict(sys.modules, {"zvec": fake_zvec}):
                candidates = retriever.retrieve_multiple(
                    "a dining chair",
                    "FURNITURE",
                    desired_dimensions=np.asarray([1.0, 0.5, 2.0]),
                    max_candidates=1,
                )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        np.testing.assert_allclose(candidate.mesh.extents, [1.0, 2.0, 0.5])
        self.assertAlmostEqual(candidate.bbox_score, 0.0)
