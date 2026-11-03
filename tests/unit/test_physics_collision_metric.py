from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from pydrake.all import RigidTransform

from scenesmith.agent_utils.house import RoomGeometry
from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID
from scenesmith.scenebenchmark_critic import evaluate_room_scene
from scenesmith.scenebenchmark_critic.adapter import room_scene_to_case_pack
from scenesmith.scenebenchmark_critic.config import CriticConfig
from scenesmith.scenebenchmark_critic.evaluator import (
    evaluate_case_pack,
    run_case_pack_checks,
)


def _case_pack(evidence, *, objects=None):
    return {
        "scene_geometry": {"objects": list(objects or [])},
        "physics_evidence": evidence,
        "checks": [],
    }


def test_hard_physics_penetration_is_canonical_scoring_failure() -> None:
    result = evaluate_case_pack(
        _case_pack(
            {
                "available": True,
                "collisions": [
                    {
                        "object_a_id": "stool_0",
                        "object_b_id": "stool_1",
                        "penetration_depth_m": 0.1764,
                        "classification": "hard",
                    }
                ],
            }
        ),
        config={
            "scenebenchmark_critic": {"enabled": True, "metrics": ["physics_collision"]}
        },
    )

    assert result["results"][0]["label"] == "fail"
    assert result["summary"]["scene_summary"]["total_checks"] == 1
    assert result["summary"]["scene_summary"]["fail"] == 1


def test_support_contact_and_shallow_contact_do_not_fail_collision_metric() -> None:
    result = evaluate_case_pack(
        _case_pack(
            {
                "available": True,
                "penetration_tolerance_m": 0.01,
                "collisions": [
                    {
                        "object_a_id": "book_0",
                        "object_b_id": "shelf_0",
                        "penetration_depth_m": 0.02,
                        "classification": "expected_support",
                    },
                    {
                        "object_a_id": "pillow_0",
                        "object_b_id": "sofa_0",
                        "penetration_depth_m": 0.02,
                        "classification": "soft_furnishing",
                    },
                    {
                        "object_a_id": "tv_0",
                        "object_b_id": "stand_0",
                        "penetration_depth_m": 0.005,
                        "classification": "hard",
                    },
                ],
            }
        ),
        config={
            "scenebenchmark_critic": {"enabled": True, "metrics": ["physics_collision"]}
        },
    )

    assert result["results"][0]["label"] == "pass"


def test_unavailable_physics_is_unknown_not_pass() -> None:
    result = evaluate_case_pack(
        _case_pack({"available": False, "error": "validator unavailable"}),
        config={
            "scenebenchmark_critic": {"enabled": True, "metrics": ["physics_collision"]}
        },
    )

    assert result["results"][0]["label"] == "unknown"


def _collision_case(*, first, second):
    return _case_pack(
        {
            "available": True,
            "collisions": [
                {
                    "object_a_id": first["id"],
                    "object_b_id": second["id"],
                    "penetration_depth_m": 0.05,
                    "classification": "hard",
                }
            ],
        },
        objects=[first, second],
    )


def _evaluate_collision_owner(*, first, second):
    result = evaluate_case_pack(
        _collision_case(first=first, second=second),
        config={
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["physics_collision"],
            }
        },
    )
    return result["results"][0]["diagnostics"]


def test_furniture_collision_is_owned_by_furniture_stage() -> None:
    diagnostics = _evaluate_collision_owner(
        first={"id": "sofa_0", "category_norm": "sofa", "object_type": "furniture"},
        second={
            "id": "table_0",
            "category_norm": "coffee_table",
            "object_type": "furniture",
        },
    )

    assert diagnostics["earliest_stage"] == "furniture"
    assert diagnostics["endpoint_generation_owners"] == {
        "sofa_0": "furniture",
        "table_0": "furniture",
    }


def test_cross_stage_collision_is_owned_when_latest_movable_endpoint_exists() -> None:
    furniture = {
        "id": "sofa_0",
        "category_norm": "sofa",
        "object_type": "furniture",
    }
    wall_mounted = {
        "id": "mirror_0",
        "category_norm": "mirror",
        "object_type": "wall_mounted",
    }
    manipuland = {
        "id": "vase_0",
        "category_norm": "vase",
        "object_type": "manipuland",
    }

    wall_diagnostics = _evaluate_collision_owner(first=furniture, second=wall_mounted)
    manipuland_diagnostics = _evaluate_collision_owner(
        first=furniture, second=manipuland
    )

    assert wall_diagnostics["earliest_stage"] == "wall_mounted"
    assert manipuland_diagnostics["earliest_stage"] == "manipuland"


def test_structural_collision_anchor_does_not_become_repair_owner() -> None:
    diagnostics = _evaluate_collision_owner(
        first={
            "id": "cabinet_0",
            "category_norm": "storage_cabinet",
            "object_type": "furniture",
        },
        second={
            "id": "north_wall",
            "category_norm": "wall",
            "object_type": "structural",
        },
    )

    assert diagnostics["earliest_stage"] == "furniture"
    assert diagnostics["endpoint_generation_owners"] == {
        "cabinet_0": "furniture",
        "north_wall": "",
    }


@pytest.fixture
def bound_scene(tmp_path: Path) -> Callable[[bool], RoomScene]:
    """Build fresh native state with constructed evidence, without measuring physics."""

    def create(colliding: bool = False) -> RoomScene:
        directory = tmp_path / ("overlap" if colliding else "separated")
        directory.mkdir()
        room_sdf = directory / "room.sdf"
        room_sdf.write_text(
            '<sdf version="1.7"><model name="room"><link '
            'name="room_geometry_body_link"/></model></sdf>',
            encoding="utf-8",
        )
        cube_sdf = directory / "cube.sdf"
        cube_sdf.write_text(
            '<sdf version="1.7"><model name="cube"><link name="body">'
            '<collision name="cube_collision"><geometry><box><size>1 1 1</size>'
            "</box></geometry></collision></link></model></sdf>",
            encoding="utf-8",
        )
        geometry = RoomGeometry(
            ET.parse(room_sdf), room_sdf, length=6.0, width=6.0
        )
        scene = RoomScene(
            geometry,
            directory,
            text_description="Synthetic room for physics evidence freshness tests.",
        )
        for index, x in enumerate([0.0, 0.96 if colliding else 1.1]):
            scene.add_object(
                SceneObject(
                    UniqueID(f"cube_{index}"),
                    ObjectType.FURNITURE,
                    "fixture_cube",
                    "Synthetic cube for evidence binding tests",
                    RigidTransform([x, 0.0, 0.5]),
                    sdf_path=cube_sdf,
                    bbox_min=np.array([-0.5, -0.5, -0.5]),
                    bbox_max=np.array([0.5, 0.5, 0.5]),
                    metadata={"semantic_name": "box"},
                )
            )
        # Deliberately constructed unit data; this fixture makes no Drake query
        # and does not claim that its collision values are measured evidence.
        scene.metadata["scenebenchmark_physics_evidence"] = {
            "schema_version": "scenesmith.physics_evidence.v1",
            "available": True,
            "source_phase": "unit_test_binding_only",
            "scene_hash": scene.content_hash(),
            "penetration_tolerance_m": 0.001,
            "collisions": (
                [
                    {
                        "object_a_id": "cube_0",
                        "object_b_id": "cube_1",
                        "penetration_depth_m": 0.04,
                        "classification": "hard",
                    }
                ]
                if colliding
                else []
            ),
        }
        return scene

    return create


def _evaluate_bound_scene(scene: RoomScene) -> tuple[dict, list[str]]:
    pack = room_scene_to_case_pack(
        scene, stage="final", metrics=("physics_collision",)
    )
    results = run_case_pack_checks(
        pack, CriticConfig(enabled=True, metrics=("physics_collision",))
    )
    labels = [row["label"] for row in results if row["metric"] == "physics_collision"]
    assert len(labels) == 1
    return pack, labels


def test_current_clean_evidence_is_preserved(bound_scene) -> None:
    scene = bound_scene(False)
    evidence = scene.metadata["scenebenchmark_physics_evidence"]
    raw = deepcopy(evidence)
    pack, labels = _evaluate_bound_scene(scene)
    assert labels == ["pass"]
    assert pack["physics_evidence"] == raw
    assert pack["physics_evidence"] is not evidence
    assert scene.metadata["scenebenchmark_physics_evidence"] == raw


def test_current_collision_evidence_remains_a_failure(bound_scene) -> None:
    scene = bound_scene(True)
    raw = deepcopy(scene.metadata["scenebenchmark_physics_evidence"])
    pack, labels = _evaluate_bound_scene(scene)
    assert labels == ["fail"]
    assert pack["physics_evidence"] == raw
    assert scene.metadata["scenebenchmark_physics_evidence"] == raw


def test_native_move_invalidates_old_clean_evidence(bound_scene) -> None:
    scene = bound_scene(False)
    raw = deepcopy(scene.metadata["scenebenchmark_physics_evidence"])
    assert scene.move_object(UniqueID("cube_1"), RigidTransform([0.96, 0.0, 0.5]))
    assert scene.content_hash() != raw["scene_hash"]
    pack, labels = _evaluate_bound_scene(scene)
    assert labels == ["unknown"]
    assert pack["physics_evidence"]["available"] is False
    assert pack["physics_evidence"]["scene_hash"] == raw["scene_hash"]
    assert pack["physics_evidence"]["current_scene_hash"] == scene.content_hash()
    assert scene.metadata["scenebenchmark_physics_evidence"] == raw


def test_available_physics_without_state_binding_is_unknown(bound_scene) -> None:
    scene = bound_scene(False)
    del scene.metadata["scenebenchmark_physics_evidence"]["scene_hash"]
    raw = deepcopy(scene.metadata["scenebenchmark_physics_evidence"])
    pack, labels = _evaluate_bound_scene(scene)
    assert labels == ["unknown"] and pack["physics_evidence"]["available"] is False
    assert "scene_hash" not in pack["physics_evidence"]
    assert scene.metadata["scenebenchmark_physics_evidence"] == raw


def test_attribute_evidence_fallback_checks_the_current_state(bound_scene) -> None:
    scene = bound_scene(False)
    raw = deepcopy(scene.metadata.pop("scenebenchmark_physics_evidence"))
    scene.scenebenchmark_physics_evidence = deepcopy(raw)
    pack, labels = _evaluate_bound_scene(scene)
    assert labels == ["pass"] and pack["physics_evidence"] == raw
    assert scene.move_object(UniqueID("cube_1"), RigidTransform([0.96, 0.0, 0.5]))
    stale_pack, stale_labels = _evaluate_bound_scene(scene)
    assert stale_labels == ["unknown"]
    assert stale_pack["physics_evidence"]["available"] is False
    assert "scenebenchmark_physics_evidence" not in scene.metadata
    assert scene.scenebenchmark_physics_evidence == raw


def test_unavailable_evidence_preserves_error_without_hashing(
    bound_scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = bound_scene(False)
    raw = {"available": False, "error": "validator unavailable", "collisions": []}
    scene.metadata["scenebenchmark_physics_evidence"] = deepcopy(raw)

    def should_not_hash() -> str:
        raise AssertionError("Unavailable evidence must not trigger hashing")

    monkeypatch.setattr(scene, "content_hash", should_not_hash)
    pack, labels = _evaluate_bound_scene(scene)
    assert labels == ["unknown"] and pack["physics_evidence"] == raw
    assert scene.metadata["scenebenchmark_physics_evidence"] == raw


def test_absent_physics_evidence_is_unknown(bound_scene) -> None:
    scene = bound_scene(False)
    scene.metadata.pop("scenebenchmark_physics_evidence")
    assert not hasattr(scene, "scenebenchmark_physics_evidence")
    pack, labels = _evaluate_bound_scene(scene)
    assert labels == ["unknown"] and pack["physics_evidence"] is None


def test_metadata_physics_evidence_precedes_attribute_fallback(bound_scene) -> None:
    scene = bound_scene(False)
    raw = deepcopy(scene.metadata["scenebenchmark_physics_evidence"])
    fallback = {"available": False, "error": "old fallback"}
    scene.scenebenchmark_physics_evidence = deepcopy(fallback)
    pack, labels = _evaluate_bound_scene(scene)
    assert labels == ["pass"] and pack["physics_evidence"] == raw
    assert scene.metadata["scenebenchmark_physics_evidence"] == raw
    assert scene.scenebenchmark_physics_evidence == fallback


def test_uncomputable_physics_state_identity_is_unknown(
    bound_scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = bound_scene(False)
    raw = deepcopy(scene.metadata["scenebenchmark_physics_evidence"])

    def hash_unavailable() -> str:
        raise ValueError("Deliberately unavailable state identity")

    monkeypatch.setattr(scene, "content_hash", hash_unavailable)
    pack, labels = _evaluate_bound_scene(scene)
    assert labels == ["unknown"] and pack["physics_evidence"]["available"] is False
    assert pack["physics_evidence"]["error"].endswith("ValueError")
    assert scene.metadata["scenebenchmark_physics_evidence"] == raw


def test_public_room_critic_rejects_stale_physics_after_native_move(bound_scene) -> None:
    scene = bound_scene(False)
    config = CriticConfig(enabled=True, metrics=("physics_collision",))
    raw = deepcopy(scene.metadata["scenebenchmark_physics_evidence"])
    fresh = evaluate_room_scene(scene, config=config, stage="final")
    assert fresh["results"][0]["label"] == "pass"
    assert scene.move_object(UniqueID("cube_1"), RigidTransform([0.96, 0.0, 0.5]))
    stale = evaluate_room_scene(scene, config=config, stage="final")
    assert stale["results"][0]["label"] == "unknown"
    assert scene.metadata["scenebenchmark_physics_evidence"] == raw
