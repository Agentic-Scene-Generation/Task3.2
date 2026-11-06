"""Deployment mount changes preserve mesh and image asset identity."""

import json

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import trimesh

from scenesmith.agent_utils.hssd_retrieval.all_assets_paths import (
    resolve_all_assets_path,
)
from scenesmith.agent_utils.hssd_retrieval.all_assets_zvec import (
    AllAssetsZvecRetriever,
)
from scenesmith.agent_utils.hssd_retrieval.config import HssdZvecConfig


def test_default_paths_and_nonmatching_prefix_are_preserved(monkeypatch):
    monkeypatch.delenv("HSSD_ALL_ASSETS_PATH_PREFIX_MAP", raising=False)
    original = Path("published/model.glb")
    assert resolve_all_assets_path(original) == original
    monkeypatch.setenv(
        "HSSD_ALL_ASSETS_PATH_PREFIX_MAP",
        json.dumps({str(Path.cwd() / "release"): str(Path.cwd() / "mount")}),
    )
    sibling = Path.cwd() / "release-other/model.glb"
    assert resolve_all_assets_path(sibling) == sibling


@pytest.mark.parametrize("raw", ['[]', '{', '{"relative":"also-relative"}'])
def test_invalid_mount_configuration_fails_clearly(monkeypatch, raw):
    monkeypatch.setenv("HSSD_ALL_ASSETS_PATH_PREFIX_MAP", raw)
    with pytest.raises(ValueError):
        resolve_all_assets_path("model.glb")


def test_longest_component_prefix_wins_and_escape_is_rejected(tmp_path, monkeypatch):
    source, target = tmp_path / "release", tmp_path / "mount"
    specific = tmp_path / "specific"
    monkeypatch.setenv("HSSD_ALL_ASSETS_PATH_PREFIX_MAP", json.dumps({
        str(source): str(target), str(source / "meshes"): str(specific),
    }))
    assert resolve_all_assets_path(source / "meshes/model.glb") == specific / "model.glb"
    with pytest.raises(ValueError, match="escape"):
        resolve_all_assets_path(source / "../outside.glb")


def test_retrieval_and_render_evidence_share_explicit_mount(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from scenesmith.agent_utils.asset_router.rendered_asset_choice import (
        _all_assets_manifest_image_paths,
    )

    published, mounted = tmp_path / "published", tmp_path / "mounted"
    mounted.mkdir()
    mesh_path = mounted / "fork.glb"
    trimesh.creation.box(extents=[0.20, 0.01, 0.03]).export(mesh_path)
    image = mounted / "fork.png"
    image.write_bytes(b"fixture-image")
    uid = "others:ycb:030_fork"
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "asset_uid": uid, "geometry_ref": {"unit_scale_to_m": 1.0},
        "image_paths": [str(published / "fork.png")],
    }) + "\n")
    monkeypatch.setenv("HSSD_ALL_ASSETS_PATH_PREFIX_MAP", json.dumps({
        str(published): str(mounted),
    }))
    monkeypatch.setenv("HSSD_ALL_ASSETS_MANIFEST_PATH", str(manifest))
    config = HssdZvecConfig(collection_path=tmp_path, base_url="http://unused",
                            all_assets_manifest_path=manifest)
    retriever = AllAssetsZvecRetriever(config, top_k=1)
    retriever._client = MagicMock()
    retriever._client.embed_text.return_value = [0.1, 0.2]
    retriever._collection = MagicMock()
    retriever._collection.query.return_value = [SimpleNamespace(
        fields={"asset_id": uid, "asset_path": str(published / "fork.glb")},
        score=0.9,
    )]
    with patch.dict(sys.modules, {"zvec": SimpleNamespace(Query=lambda **kw: kw)}):
        candidates = retriever.retrieve_multiple("dinner fork", "MANIPULAND")
    assert [c.mesh_id for c in candidates] == [uid]
    assert candidates[0].mesh.extents == pytest.approx([0.20, 0.01, 0.03])
    assert _all_assets_manifest_image_paths(uid) == (image,)
    assert str(published / "fork.png") in manifest.read_text()


@pytest.mark.parametrize("source_axis,expected", [
    ("z_up", [0.20, 0.01, 0.03]),
    ("y_up", [0.20, 0.03, 0.01]),
    (None, [0.20, 0.03, 0.01]),
])
def test_external_source_axis_normalizes_geometry_once(tmp_path, source_axis, expected):
    import sys
    from types import SimpleNamespace

    uid = "others:ycb:fixture_fork"
    mesh_path = tmp_path / "fork.glb"
    trimesh.creation.box(extents=[0.20, 0.03, 0.01]).export(mesh_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "asset_uid": uid, "geometry_ref": {"unit_scale_to_m": 1.0,
            "source_axis": source_axis, "normalized_axis_frame": "asset_local_y_up"},
    }) + "\n")
    retriever = AllAssetsZvecRetriever(HssdZvecConfig(
        collection_path=tmp_path, base_url="http://unused", all_assets_manifest_path=manifest,
    ), top_k=1)
    retriever._client = MagicMock()
    retriever._client.embed_text.return_value = [0.1, 0.2]
    retriever._collection = MagicMock()
    retriever._collection.query.return_value = [SimpleNamespace(
        fields={"asset_id": uid, "asset_path": str(mesh_path)}, score=0.9,
    )]
    with patch.dict(sys.modules, {"zvec": SimpleNamespace(Query=lambda **kw: kw)}):
        first = retriever.retrieve_multiple("dinner fork", "MANIPULAND")
        second = retriever.retrieve_multiple("dinner fork", "MANIPULAND")
    assert first[0].mesh.extents == pytest.approx(expected)
    assert second[0].mesh.bounds == pytest.approx(first[0].mesh.bounds)
    # Every request reloads the original; no transform is written back to it.
    assert trimesh.load(mesh_path, force="mesh").extents == pytest.approx([0.20, 0.03, 0.01])


def test_source_policy_excludes_unapproved_generated_meshes_before_loading(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setenv("HSSD_ALL_ASSETS_ALLOWED_SOURCES", "hssd,3dfuture,others")
    mesh_path = tmp_path / "fork.glb"
    trimesh.creation.box(extents=[0.20, 0.01, 0.03]).export(mesh_path)
    retriever = AllAssetsZvecRetriever(
        HssdZvecConfig(collection_path=tmp_path, base_url="http://unused"), top_k=2,
    )
    retriever._client = MagicMock()
    retriever._client.embed_text.return_value = [0.1, 0.2]
    retriever._collection = MagicMock()
    retriever._collection.query.return_value = [SimpleNamespace(
        fields={"asset_id": uid, "asset_path": str(mesh_path)}, score=0.9,
    ) for uid in ["generated:unapproved_fork", "others:ycb:030_fork"]]
    load = trimesh.load
    with (
        patch.dict(sys.modules, {"zvec": SimpleNamespace(Query=lambda **kw: kw)}),
        patch("scenesmith.agent_utils.hssd_retrieval.all_assets_zvec.trimesh.load", wraps=load) as loader,
    ):
        candidates = retriever.retrieve_multiple("dinner fork", "MANIPULAND")
    assert [item.mesh_id for item in candidates] == ["others:ycb:030_fork"]
    assert loader.call_count == 1


def test_unknown_source_policy_fails_before_retrieval(tmp_path, monkeypatch):
    monkeypatch.setenv("HSSD_ALL_ASSETS_ALLOWED_SOURCES", "typo")
    with pytest.raises(ValueError, match="namespace"):
        AllAssetsZvecRetriever(HssdZvecConfig(
            collection_path=tmp_path, base_url="http://unused",
        ), top_k=1)
