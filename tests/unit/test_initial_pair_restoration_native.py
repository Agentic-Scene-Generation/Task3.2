"""Real Linux Drake restoration and scoring, without LLMs or asset services."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytest.importorskip("pydrake")
import numpy as np
from omegaconf import OmegaConf
from pydrake.all import RigidTransform

from scenesmith.agent_utils.house import RoomGeometry
from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID
from scenesmith.scene_expert.slow_memory.paired import (
    copy_scene_tree,
    digest,
    read_json,
    tree_hashes,
    write_json,
)
from scenesmith.scene_expert.slow_memory.paired_rescore import restore_raw_scene
from scenesmith.scene_expert.slow_memory.paired_scoring import (
    SCORING_PROTOCOL,
    save_scoring_proof,
    score_raw_candidate,
    validate_scoring_proof,
)


def _native_state(tmp_path: Path, *, collision: bool) -> tuple[Path, dict, dict]:
    source = tmp_path / "canonical"
    room = source / "room_bedroom"
    room.mkdir(parents=True)
    sdf = source / "room.sdf"
    sdf.write_text(
        '<sdf version="1.7"><model name="room"><link name="room_geometry_body_link">'
        '<pose>0 0 -0.05 0 0 0</pose><collision name="floor_collision"><geometry><box>'
        "<size>4 4 0.1</size></box></geometry></collision></link></model></sdf>"
    )
    box = room / "box.sdf"
    box.write_text(
        '<sdf version="1.7"><model name="box"><link name="box_link"><collision name="collision">'
        "<geometry><box><size>0.4 0.4 0.4</size></box></geometry></collision></link></model></sdf>"
    )
    scene = RoomScene(
        RoomGeometry(ET.parse(sdf), sdf, width=4, length=4), room, room_id="bedroom"
    )
    for name, x in (("nightstand_0", 0.0), ("nightstand_1", 0.1 if collision else 1.0)):
        scene.add_object(
            SceneObject(
                UniqueID(name),
                ObjectType.FURNITURE,
                name,
                "nightstand",
                RigidTransform([x, 0.0, 0.5]),
                sdf_path=box,
                image_path=room / "unused_legacy_preview.png",
                bbox_min=np.array([-0.2] * 3),
                bbox_max=np.array([0.2] * 3),
            )
        )
    state = scene.to_state_dict()
    # Exact quaternion serialized in real 008/B. Its Drake matrix round trip
    # changes the final component by 3.469446951953614e-18.
    state["objects"]["nightstand_0"]["transform"]["rotation_wxyz"] = [
        0.9996803707709186,
        0.0,
        0.0,
        -0.025281540604135816,
    ]
    snapshot = {
        "room_relative": "room_bedroom",
        "room_id": "bedroom",
        "source_scene_root": str(source),
        "scene_attributes": {},
    }
    destination = tmp_path / "group_000/B"
    copy_scene_tree(source, destination / "raw_scene")
    write_json(destination / "raw_state.json", state)
    # Original assets deliberately unavailable: evaluation must use copies.
    sdf.unlink()
    box.unlink()
    return destination, state, snapshot


@pytest.mark.parametrize("collision", [False, True])
def test_real_drake_restores_and_scores_private_assets(
    tmp_path: Path, collision: bool
) -> None:
    directory, source, snapshot = _native_state(tmp_path, collision=collision)
    before = tree_hashes(directory / "raw_scene")
    scene = restore_raw_scene(
        directory, snapshot, source_candidate=tmp_path / "legacy/B"
    )
    restoration = read_json(directory / "restoration_proof.json")
    assert restoration["status"] == "verified"
    assert len(restoration["rotation_roundoff"]) == 1
    assert restoration["asset_bindings"]
    assert len(restoration["unavailable_reference_images"]) == 2
    assert scene.room_geometry.sdf_path.is_relative_to(directory / "raw_scene")
    restored_before = copy.deepcopy(scene.to_state_dict())
    cfg = OmegaConf.create(
        {
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["physics_collision"],
            },
            "physics_validation": {
                "object_penetration_threshold_m": 0.001,
                "floor_penetration_tolerance_m": 0.05,
                "manipuland_furniture_tolerance_m": 0.02,
            },
        }
    )
    report, proof = score_raw_candidate(scene, cfg)
    assert (report["summary"]["scene_summary"]["fail"] > 0) is collision
    assert report["summary"]["scene_summary"]["unknown"] == 0
    assert scene.to_state_dict() == restored_before
    assert tree_hashes(directory / "raw_scene") == before
    proof["raw_state_sha256"] = digest(source)
    proof["restoration_proof_sha256"] = digest(restoration)
    write_json(directory / "report.json", report)
    save_scoring_proof(directory, report, proof, before)
    candidate = {
        "raw_state_hash": digest(source),
        "evaluation_state_hash": digest(restored_before),
        "scoring_protocol": SCORING_PROTOCOL,
        "raw_files": before,
    }
    validate_scoring_proof(directory, candidate)
    assert read_json(directory / "raw_state.json") == source
    # A subsequent real geometry change must invalidate the exported proof.
    corrupted = read_json(directory / "restored_state.json")
    corrupted["objects"]["nightstand_0"]["transform"]["translation"][0] += 0.01
    write_json(directory / "restored_state.json", corrupted)
    with pytest.raises(ValueError):
        validate_scoring_proof(directory, candidate)


@pytest.mark.parametrize("failure", ["missing_geometry", "external_geometry"])
def test_restoration_requires_captured_physical_assets(
    tmp_path: Path, failure: str
) -> None:
    directory, state, snapshot = _native_state(tmp_path, collision=False)
    if failure == "missing_geometry":
        (directory / "raw_scene/room_bedroom/box.sdf").unlink()
    else:
        state["objects"]["nightstand_0"]["sdf_path"] = "/uncaptured/other.sdf"
        write_json(directory / "raw_state.json", state)
    with pytest.raises(ValueError, match="retained asset missing|outside captured"):
        restore_raw_scene(directory, snapshot, source_candidate=tmp_path / "legacy/B")
