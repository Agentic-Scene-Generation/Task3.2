"""Native floor evidence must precede immutable initial-candidate capture."""

from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import trimesh

from omegaconf import OmegaConf
from pydrake.all import RigidTransform

from scenesmith.agent_utils.house import RoomGeometry
from scenesmith.agent_utils.room import (
    ObjectType,
    RoomScene,
    SceneObject,
    SupportSurface,
    UniqueID,
)
from scenesmith.scene_expert.support_readiness import prepare_floor_support_evidence
from scenesmith.scenebenchmark_critic.adapter import _object_to_geometry
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
    evaluate_intent_contract_extensions,
)


@pytest.fixture
def support_config() -> dict:
    cfg = OmegaConf.load("configurations/manipuland_agent/base_manipuland_agent.yaml")
    root = OmegaConf.create({"paths": {"hssd_data_dir": "data"}, "agent": cfg})
    return OmegaConf.to_container(root.agent.support_surface_extraction, resolve=True)


def _floor(tmp_path: Path) -> tuple[RoomScene, SceneObject]:
    # GLTF geometry uses Y-up; the native room/collision state uses Z-up.
    mesh = trimesh.creation.box(extents=[4.2, 0.1, 4.0])
    mesh.apply_translation([0.0, -0.05, 0.0])
    geometry = tmp_path / "floor.glb"
    mesh.export(geometry)
    sdf = tmp_path / "room.sdf"
    sdf.write_text(
        '<sdf version="1.7"><model name="room"><link name="room_geometry_body_link">'
        '<collision name="floor"><pose>0 0 -0.05 0 0 0</pose><geometry><box>'
        "<size>4.2 4.0 0.1</size></box></geometry></collision></link></model></sdf>"
    )
    floor = SceneObject(
        object_id=UniqueID("floor_nursery"),
        object_type=ObjectType.FLOOR,
        name="Floor",
        description="Floor surface",
        transform=RigidTransform(),
        geometry_path=geometry,
        bbox_min=np.array([-2.1, -2.0, -0.1]),
        bbox_max=np.array([2.1, 2.0, 0.0]),
        immutable=True,
    )
    room = RoomGeometry(ET.parse(sdf), sdf, floor=floor, length=4.2, width=4.0)
    scene = RoomScene(
        room, tmp_path, room_id="nursery", text_description="Blocks on the floor"
    )
    return scene, floor


def _readiness(scene: RoomScene, floor: SceneObject) -> list[dict]:
    return evaluate_intent_contract_extensions(
        {
            "stage": "furniture",
            "scene_geometry": {
                "objects": [_object_to_geometry(floor, scene, np.zeros(3))]
            },
            "intent_contract": {
                "constraints": [
                    {
                        "constraint_id": "blocks_on_floor",
                        "relation": "on_top_of",
                        "subjects": {
                            "category": "wooden_play_block",
                            "stage": "manipuland",
                        },
                        "targets": {"category": "floor"},
                        "stage": "manipuland",
                        "strength": "hard",
                        "source": "explicit_prompt",
                    }
                ]
            },
        }
    )


def test_native_floor_evidence_survives_capture_and_fresh_scene_restore(
    tmp_path: Path, support_config: dict
) -> None:
    scene, floor = _floor(tmp_path)
    assert scene.objects == {}  # Floor must not create duplicate collision geometry.
    assert _readiness(scene, floor)[0]["label"] == "fail"
    geometry_bytes = floor.geometry_path.read_bytes()
    collision_bytes = scene.room_geometry.sdf_path.read_bytes()

    evidence = prepare_floor_support_evidence(scene, support_config)

    assert evidence[0]["status"] == "extracted"
    surface = floor.support_surfaces[0]
    assert surface.area == pytest.approx(16.8, abs=0.01)
    assert surface.transform.translation()[2] == pytest.approx(0.0, abs=0.01)
    assert surface.contains_point_2d(np.array([0.0, 0.0]))
    assert not surface.contains_point_2d(np.array([2.2, 0.0]))
    assert _readiness(scene, floor)[0]["label"] == "pass"
    assert floor.geometry_path.read_bytes() == geometry_bytes
    assert scene.room_geometry.sdf_path.read_bytes() == collision_bytes
    assert scene.objects == {}

    # Every candidate receives serialized native evidence before its read-only
    # scoring boundary. Restoring a new scene must preserve the same surface.
    captured = copy.deepcopy(scene.to_state_dict())
    restored = RoomScene(None, tmp_path, room_id="nursery")
    restored.restore_from_state_dict(captured)
    restored_floor = restored.get_object(UniqueID("floor_nursery"))
    assert restored_floor is not None
    before = copy.deepcopy(restored.to_state_dict())
    assert _readiness(restored, restored_floor)[0]["label"] == "pass"
    assert restored.to_state_dict() == before == captured
    with patch(
        "scenesmith.scene_expert.support_readiness.extract_and_propagate_support_surfaces"
    ) as extract:
        assert prepare_floor_support_evidence(restored, support_config) == []
    extract.assert_not_called()
    # Preparing a floor before capture must not consume the uncheckpointed
    # sequential furniture-surface counter or collide with its next allocation.
    original_next = scene.generate_surface_id()
    restored_next = restored.generate_surface_id()
    assert original_next == restored_next
    assert restored_next not in {s.surface_id for s in restored_floor.support_surfaces}


def test_missing_floor_geometry_remains_a_failure(
    tmp_path: Path, support_config: dict
) -> None:
    scene, floor = _floor(tmp_path)
    floor.geometry_path.unlink()

    evidence = prepare_floor_support_evidence(scene, support_config)

    assert evidence[0]["status"] == "error"
    assert evidence[0]["surfaces"] == 0
    assert floor.support_surfaces == []
    assert _readiness(scene, floor)[0]["label"] == "fail"


def test_existing_concave_floor_surface_is_preserved(
    tmp_path: Path, support_config: dict
) -> None:
    scene, floor = _floor(tmp_path)
    surface = SupportSurface(
        UniqueID("floor_surface_nursery"),
        np.array([-2.0, -2.0, 0.0]),
        np.array([2.0, 2.0, 2.5]),
        RigidTransform(),
        exact_boundary_vertices=[(-2, -2), (2, -2), (2, 0), (0, 0), (0, 2), (-2, 2)],
    )
    floor.support_surfaces = [surface]
    before = copy.deepcopy(scene.to_state_dict())
    with patch(
        "scenesmith.scene_expert.support_readiness.extract_and_propagate_support_surfaces"
    ) as extract:
        assert prepare_floor_support_evidence(scene, support_config) == []
    extract.assert_not_called()
    assert floor.support_surfaces[0] is surface
    assert not surface.contains_point_2d(np.array([1.0, 1.0]))
    assert scene.to_state_dict() == before


def test_hook_prepares_floor_before_stage_context_and_agent_creation(
    tmp_path: Path, support_config: dict
) -> None:
    from scenesmith.scene_expert.hooks import SceneExpertHookRunner
    from scenesmith.scene_expert.schemas import SceneTaskSpec, StageRelationContext
    from scenesmith.scenebenchmark_critic.config import CriticConfig

    scene, floor = _floor(tmp_path)
    projector = Mock()

    def project(**kwargs) -> StageRelationContext:
        assert floor.support_surfaces
        assert kwargs["scene"] is scene
        assert scene.text_description == "Blocks on the floor"
        return StageRelationContext(stage="furniture")

    projector.project.side_effect = project
    runner = SceneExpertHookRunner(
        prompt=scene.text_description,
        scene_id=0,
        output_dir=tmp_path,
        mode="harness_only",
        component_flags={},
        task_spec=SceneTaskSpec(room_type="nursery", style="standard"),
        harness=Mock(),
        global_planner=Mock(),
        relation_projector=projector,
        retriever=None,
        stage_verifier=Mock(),
        full_verifier=Mock(),
        repair_controller=Mock(),
        trace_logger=None,
        memory_writer=None,
        memory_store=None,
        qwen_model="test",
        critic_config=CriticConfig(enabled=True),
        support_surface_config=support_config,
    )
    runner._save_context_bundle = Mock()
    runner._write_stage_policy_audit = Mock()
    runner._record_memory_pre_stage_activity = Mock()
    runner.pre_stage("furniture", scene)

    assert _readiness(scene, floor)[0]["label"] == "pass"
    assert scene.scene_expert_original_description == "Blocks on the floor"
    row = json.loads(
        (tmp_path / "scene_000/scene_expert/timing/support_readiness.jsonl").read_text()
    )
    assert row["phase"] == "pre_stage" and row["targets"][0]["status"] == "extracted"
