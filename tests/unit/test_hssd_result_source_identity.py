"""Keep retrieved identity separate from requested labels at the API boundary."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scenesmith.agent_utils.hssd_retrieval.data_loader import HssdMeshMetadata
from scenesmith.agent_utils.hssd_retrieval_server.dataclasses import (
    HssdRetrievalServerRequest,
    HssdRetrievalServerResponse,
)
from scenesmith.agent_utils.hssd_retrieval_server.server_app import HssdRetrievalApp


@pytest.fixture
def annotation_module(monkeypatch):
    # This lookup is a pure file reader. Load the actual module without the
    # critic package's unrelated geometry/Drake initialization.
    path = (
        Path(__file__).parents[2]
        / "scenesmith/scenebenchmark_critic/current_asset_annotations.py"
    )
    name = "scenesmith.scenebenchmark_critic.current_asset_annotations"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def retrieve(tmp_path, metadata, asset_id="spoon_mesh"):
    mesh = MagicMock()
    mesh.extents = np.array([0.05, 0.01, 0.25])
    fields = dict(mesh_id=asset_id, mesh=mesh, clip_score=0.7)
    if metadata is not None:
        fields["metadata"] = metadata
    candidate = SimpleNamespace(**fields)
    retriever = MagicMock()
    retriever.retrieve_multiple.return_value = [candidate]
    app = SimpleNamespace(_fatal_error=None, _get_retriever=lambda: retriever)
    request = HssdRetrievalServerRequest(
        object_description="Stainless steel dinner fork, four tines",
        object_type="MANIPULAND",
        output_dir=str(tmp_path),
        num_candidates=4,
    )
    response = HssdRetrievalApp._retrieve_internal(app, request)
    mesh.export.assert_called_once_with(str(tmp_path / f"{asset_id}.glb"))
    assert response.query_description == request.object_description
    return HssdRetrievalServerResponse.from_dict(response.to_dict())


def test_wrong_geometry_does_not_inherit_query_identity(tmp_path):
    metadata = HssdMeshMetadata(
        mesh_id="spoon_mesh",
        name="Silver spoon",
        wordnet_key="spoon.n.01",
        up="",
        front="",
    )
    response = retrieve(tmp_path, metadata)
    result = response.results[0]
    assert result.object_name == "Silver spoon"
    assert result.category == "spoon.n.01"
    assert result.hssd_id == "spoon_mesh"
    assert result.object_name != response.query_description
    assert result.category != "small_objects"


def test_namespaced_candidate_uses_its_own_annotation(tmp_path, annotation_module):
    with patch.object(
        annotation_module,
        "get_current_asset_annotation",
        return_value={"category_key": "reamer", "name": "Citrus reamer"},
    ) as lookup:
        response = retrieve(tmp_path, None, "3dfuture:fixture")
    lookup.assert_called_once_with("3dfuture:fixture")
    assert response.results[0].object_name == "Citrus reamer"
    assert response.results[0].category == "reamer"


@pytest.mark.parametrize("record", [{}, {"category": "spoon"}])
def test_missing_namespaced_name_is_unknown_or_source_category(
    tmp_path, record, annotation_module
):
    with patch.object(
        annotation_module,
        "get_current_asset_annotation",
        return_value=record,
    ):
        response = retrieve(tmp_path, None, "hssd:fixture")
    result = response.results[0]
    assert result.object_name == record.get("category", "")
    assert result.category == record.get("category", "unknown")
    assert result.object_name != response.query_description


def test_missing_annotation_never_falls_back_to_the_request(
    tmp_path, annotation_module
):
    with patch.object(
        annotation_module,
        "get_current_asset_annotation",
        side_effect=KeyError("missing"),
    ):
        response = retrieve(tmp_path, None, "hssd:missing")
    assert response.results[0].object_name == ""
    assert response.results[0].category == "unknown"


def test_missing_hssd_name_preserves_synset_without_query_substitution(tmp_path):
    metadata = HssdMeshMetadata(
        mesh_id="spoon_mesh",
        name="",
        wordnet_key="spoon.n.01",
        up="",
        front="",
    )
    result = retrieve(tmp_path, metadata).results[0]
    assert result.object_name == ""
    assert result.category == "spoon.n.01"
