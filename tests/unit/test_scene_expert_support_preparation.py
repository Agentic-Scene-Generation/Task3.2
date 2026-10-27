"""Regression coverage for furniture evidence produced after its old gate."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import trimesh
from omegaconf import OmegaConf
from pydrake.all import RigidTransform

from scenesmith.agent_utils.room import ObjectType, SceneObject, UniqueID
from scenesmith.scene_expert.support_readiness import prepare_stage_support_evidence
from scenesmith.scenebenchmark_critic.adapter import _object_to_geometry
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
    evaluate_intent_contract_extensions,
)


@pytest.fixture
def support_config() -> dict:
    cfg = OmegaConf.load("configurations/manipuland_agent/base_manipuland_agent.yaml")
    root = OmegaConf.create({"paths": {"hssd_data_dir": "data"}, "agent": cfg})
    return OmegaConf.to_container(root.agent.support_surface_extraction, resolve=True)


def _rug(tmp_path):
    # The file is already at its final size, independent of the source scale.
    mesh = trimesh.creation.box(extents=[2.0, 0.02, 1.0])
    mesh.apply_translation([0.0, 0.01, 0.0])
    geometry = tmp_path / "rug.glb"
    mesh.export(geometry)
    obj = SceneObject(
        object_id=UniqueID("rug_0"),
        object_type=ObjectType.FURNITURE,
        name="rug",
        description="Play rug",
        transform=RigidTransform(),
        geometry_path=geometry,
        metadata={"asset_source": "thin_covering", "geometry_scale_baked": True},
        bbox_min=np.array([-1.0, -0.5, 0.0]),
        bbox_max=np.array([1.0, 0.5, 0.02]),
        scale_factor=1.4,
    )
    scene = SimpleNamespace(
        room_id="playroom",
        objects={obj.object_id: obj},
        generate_surface_id=lambda: UniqueID("S_0"),
    )
    return scene, obj


def _readiness(scene, obj):
    target = _object_to_geometry(obj, scene, np.zeros(3))
    return evaluate_intent_contract_extensions(
        {
            "stage": "furniture",
            "scene_geometry": {"objects": [target]},
            "intent_contract": {
                "constraints": [
                    {
                        "constraint_id": "block_on_rug",
                        "relation": "on_top_of",
                        "stage": "manipuland",
                        "subjects": {
                            "category": "building_block",
                            "stage": "manipuland",
                        },
                        "targets": {"category": "rug"},
                        "strength": "hard",
                        "source": "explicit_prompt",
                    }
                ]
            },
        }
    )


def test_native_mesh_evidence_resolves_readiness_without_second_scale(
    tmp_path, support_config
):
    scene, obj = _rug(tmp_path)
    before = _readiness(scene, obj)
    assert before[0]["label"] == "fail"

    evidence = prepare_stage_support_evidence(
        scene, {"results": before}, support_config
    )

    assert evidence[0]["status"] == "extracted"
    assert _readiness(scene, obj)[0]["label"] == "pass"
    surface = obj.support_surfaces[0]
    # A second 1.4x scale would allow invalid placements outside the actual rug.
    np.testing.assert_allclose(surface.bounding_box_max[:2], [1.0, 0.5], atol=0.01)
    assert not surface.contains_point_2d(np.array([1.2, 0.0]))


@pytest.mark.parametrize("failure", [None, FileNotFoundError("mesh missing")])
def test_unavailable_surfaces_remain_a_failure(tmp_path, support_config, failure):
    scene, obj = _rug(tmp_path)
    rows = _readiness(scene, obj)
    with patch(
        "scenesmith.scene_expert.support_readiness.extract_and_propagate_support_surfaces",
        return_value=[],
        side_effect=failure,
    ):
        evidence = prepare_stage_support_evidence(
            scene, {"results": rows}, support_config
        )
    assert evidence[0]["status"] == ("error" if failure else "empty")
    assert obj.support_surfaces == []
    assert _readiness(scene, obj)[0]["label"] == "fail"


def test_unknown_evidence_does_not_trigger_geometry_work(tmp_path, support_config):
    scene, obj = _rug(tmp_path)
    rows = _readiness(scene, obj)
    rows[0]["label"] = "unknown"
    with patch(
        "scenesmith.scene_expert.support_readiness.extract_and_propagate_support_surfaces"
    ) as extract:
        assert (
            prepare_stage_support_evidence(scene, {"results": rows}, support_config)
            == []
        )
    extract.assert_not_called()


def test_asset_factory_marks_final_geometry_baked(tmp_path):
    from scenesmith.agent_utils.asset_manager import AssetManager

    manager = object.__new__(AssetManager)
    manager.registry = Mock(generate_unique_id=Mock(return_value=UniqueID("rug_0")))
    obj = manager._create_scene_object(
        config=SimpleNamespace(
            short_name="rug", description="Play rug", image_path=None
        ),
        object_type=ObjectType.FURNITURE,
        sdf_path=tmp_path / "rug.sdf",
        final_gltf_path=tmp_path / "rug.gltf",
        bbox_min=np.array([-1.0, -0.5, 0.0]),
        bbox_max=np.array([1.0, 0.5, 0.02]),
        additional_metadata={"asset_source": "thin_covering"},
        scale_factor=1.4,
    )
    assert obj.metadata["geometry_scale_baked"] is True
    assert obj.scale_factor == 1.4  # Retained for source HSSD annotations.
