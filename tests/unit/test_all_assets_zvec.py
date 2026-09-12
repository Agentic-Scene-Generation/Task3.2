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

    def test_server_passes_hssd_metadata_path_to_all_assets_retriever(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            collection_path = root / "collection"
            collection_path.mkdir()
            preprocessed_path = root / "preprocessed"
            preprocessed_path.mkdir()
            manifest_path = root / "embedding_inputs.shared.jsonl"
            manifest_path.write_text("")
            app = HssdRetrievalApp(
                preload_retriever=False,
                hssd_retrieval_backend="all_assets_embedding",
                hssd_preprocessed_path=str(preprocessed_path),
                hssd_zvec_collection_path=str(collection_path),
                hssd_embedding_base_url="http://127.0.0.1:8014",
                hssd_all_assets_manifest_path=str(manifest_path),
            )

            with patch(
                "scenesmith.agent_utils.hssd_retrieval_server.server_app."
                "AllAssetsZvecRetriever"
            ) as retriever_class:
                retriever = app._get_retriever()

        self.assertIs(retriever, retriever_class.return_value)
        retriever_class.assert_called_once()
        self.assertEqual(
            retriever_class.call_args.kwargs["hssd_preprocessed_path"],
            preprocessed_path,
        )

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

    def test_restores_hssd_orientation_before_exporting_all_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            collection_path = root / "collection"
            collection_path.mkdir()
            preprocessed_path = root / "preprocessed"
            preprocessed_path.mkdir()

            painting_id = "hssd:painting"
            mirror_id = "hssd:mirror"
            canonical_hssd_id = "hssd:already-canonical"
            other_id = "others:polyhaven:picture"
            painting_path = root / "painting.obj"
            mirror_path = root / "mirror.obj"
            canonical_hssd_path = root / "canonical-hssd.obj"
            other_path = root / "other.obj"
            trimesh.creation.box(extents=[0.8, 0.03, 0.6]).export(painting_path)
            trimesh.creation.box(extents=[0.5, 0.02, 0.7]).export(mirror_path)
            trimesh.creation.box(extents=[0.6, 0.8, 0.05]).export(canonical_hssd_path)
            trimesh.creation.box(extents=[0.4, 0.5, 0.01]).export(other_path)

            manifest_path = root / "embedding_inputs.shared.jsonl"
            manifest_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "asset_uid": asset_id,
                            "geometry_ref": {"unit_scale_to_m": 1.0},
                        }
                    )
                    for asset_id in (
                        painting_id,
                        mirror_id,
                        canonical_hssd_id,
                        other_id,
                    )
                )
                + "\n"
            )
            (preprocessed_path / "hssd_wnsynsetkey_index.json").write_text(
                json.dumps(
                    {
                        "wall_art.n.01": [
                            {
                                "id": "painting",
                                "name": "painting",
                                "up": "0,0,-1",
                                "front": "0,1,0",
                            },
                            {
                                "id": "mirror",
                                "name": "mirror",
                                "up": "0,0,-1",
                                "front": "0,1,0",
                            },
                            {
                                "id": "already-canonical",
                                "name": "picture frame",
                                "up": "",
                                "front": "",
                            },
                        ]
                    }
                )
            )
            config = HssdZvecConfig(
                collection_path=collection_path,
                base_url="http://127.0.0.1:8014",
                all_assets_manifest_path=manifest_path,
            )
            retriever = AllAssetsZvecRetriever(
                config,
                top_k=4,
                hssd_preprocessed_path=preprocessed_path,
            )
            retriever._client = MagicMock()
            retriever._client.embed_text.return_value = [0.1, 0.2]
            retriever._collection = _FakeCollection(
                [
                    _FakeDocument(painting_id, painting_path, score=0.9),
                    _FakeDocument(mirror_id, mirror_path, score=0.8),
                    _FakeDocument(canonical_hssd_id, canonical_hssd_path, score=0.75),
                    _FakeDocument(other_id, other_path, score=0.7),
                ]
            )
            fake_zvec = types.SimpleNamespace(Query=lambda **kwargs: kwargs)

            with patch.dict(sys.modules, {"zvec": fake_zvec}):
                candidates = retriever.retrieve_multiple(
                    "wall art, mirror, and picture frame",
                    "WALL_MOUNTED",
                    max_candidates=4,
                )

        by_id = {candidate.mesh_id: candidate for candidate in candidates}
        np.testing.assert_allclose(by_id[painting_id].mesh.extents, [0.8, 0.6, 0.03])
        np.testing.assert_allclose(by_id[mirror_id].mesh.extents, [0.5, 0.7, 0.02])
        np.testing.assert_allclose(
            by_id[canonical_hssd_id].mesh.extents, [0.6, 0.8, 0.05]
        )
        np.testing.assert_allclose(by_id[other_id].mesh.extents, [0.4, 0.5, 0.01])

    def test_rejects_hssd_candidate_without_orientation_index_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            collection_path = root / "collection"
            collection_path.mkdir()
            preprocessed_path = root / "preprocessed"
            preprocessed_path.mkdir()
            asset_path = root / "painting.obj"
            trimesh.creation.box(extents=[0.8, 0.03, 0.6]).export(asset_path)
            asset_id = "hssd:missing"
            manifest_path = root / "embedding_inputs.shared.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "asset_uid": asset_id,
                        "geometry_ref": {"unit_scale_to_m": 1.0},
                    }
                )
                + "\n"
            )
            (preprocessed_path / "hssd_wnsynsetkey_index.json").write_text("{}")
            config = HssdZvecConfig(
                collection_path=collection_path,
                base_url="http://127.0.0.1:8014",
                all_assets_manifest_path=manifest_path,
            )
            retriever = AllAssetsZvecRetriever(
                config,
                top_k=1,
                hssd_preprocessed_path=preprocessed_path,
            )
            retriever._client = MagicMock()
            retriever._client.embed_text.return_value = [0.1, 0.2]
            retriever._collection = _FakeCollection(
                [_FakeDocument(asset_id, asset_path, score=0.9)]
            )
            fake_zvec = types.SimpleNamespace(Query=lambda **kwargs: kwargs)

            with patch.dict(sys.modules, {"zvec": fake_zvec}):
                with self.assertRaisesRegex(ValueError, "No loadable all-assets"):
                    retriever.retrieve_multiple(
                        "a wall painting",
                        "WALL_MOUNTED",
                        max_candidates=1,
                    )

    def test_rejects_deep_semantic_match_for_planar_wall_art(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            collection_path = root / "collection"
            collection_path.mkdir()
            joystick_path = root / "joystick.obj"
            picture_path = root / "picture.obj"
            sconce_path = root / "sconce.obj"
            # glTF Y-up extents map to SceneSmith [width, depth, height].
            trimesh.creation.box(extents=[0.5, 0.16, 0.33]).export(joystick_path)
            trimesh.creation.box(extents=[0.5, 0.35, 0.03]).export(picture_path)
            trimesh.creation.box(extents=[0.2, 0.4, 0.3]).export(sconce_path)
            ids_and_paths = (
                ("others:objaverse:joystick", joystick_path),
                ("others:polyhaven:picture", picture_path),
                ("others:polyhaven:sconce", sconce_path),
            )
            manifest_path = root / "embedding_inputs.shared.jsonl"
            manifest_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "asset_uid": asset_id,
                            "geometry_ref": {"unit_scale_to_m": 1.0},
                        }
                    )
                    for asset_id, _ in ids_and_paths
                )
                + "\n"
            )
            config = HssdZvecConfig(
                collection_path=collection_path,
                base_url="http://127.0.0.1:8014",
                all_assets_manifest_path=manifest_path,
            )
            retriever = AllAssetsZvecRetriever(config, top_k=3)
            retriever._client = MagicMock()
            retriever._client.embed_text.return_value = [0.1, 0.2]
            retriever._collection = _FakeCollection(
                [
                    _FakeDocument(asset_id, asset_path, score=1.0 - index / 10)
                    for index, (asset_id, asset_path) in enumerate(ids_and_paths)
                ]
            )
            fake_zvec = types.SimpleNamespace(Query=lambda **kwargs: kwargs)

            with patch.dict(sys.modules, {"zvec": fake_zvec}):
                art_candidates = retriever.retrieve_multiple(
                    "gaming controller poster art",
                    "WALL_MOUNTED",
                    max_candidates=3,
                )
                light_candidates = retriever.retrieve_multiple(
                    "wall sconce with a globe shade",
                    "WALL_MOUNTED",
                    max_candidates=3,
                )

        self.assertNotIn(
            "others:objaverse:joystick",
            {candidate.mesh_id for candidate in art_candidates},
        )
        self.assertIn(
            "others:polyhaven:picture",
            {candidate.mesh_id for candidate in art_candidates},
        )
        self.assertIn(
            "others:polyhaven:sconce",
            {candidate.mesh_id for candidate in light_candidates},
        )
