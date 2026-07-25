import math

from pathlib import Path

import numpy as np

from pydrake.math import RigidTransform, RollPitchYaw

from scenesmith.agent_utils import furniture_accessibility_guard
from scenesmith.agent_utils.furniture_accessibility_guard import (
    _best_candidate,
    improve_storage_front_access,
)
from scenesmith.agent_utils.house import RoomGeometry
from scenesmith.agent_utils.room import (
    ObjectType,
    PlacementInfo,
    RoomScene,
    SceneObject,
    SupportSurface,
    UniqueID,
)
from scenesmith.scenebenchmark_critic.config import CriticConfig


class _NonCopyableMesh:
    def __deepcopy__(self, _memo):
        raise AssertionError("accessibility candidates must not deepcopy meshes")


def _object(
    object_id: str,
    name: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    object_type: ObjectType = ObjectType.FURNITURE,
) -> SceneObject:
    half = np.asarray(size, dtype=float) / 2.0
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=object_type,
        name=name,
        description=name,
        transform=RigidTransform(
            rpy=RollPitchYaw(0.0, 0.0, 0.0),
            p=np.asarray(position, dtype=float),
        ),
        bbox_min=-half,
        bbox_max=half,
    )


def _scene(
    tmp_path: Path,
) -> tuple[RoomScene, SceneObject, SceneObject, SupportSurface]:
    storage = _object("storage_piece", "bookshelf", (0.0, 0.0, 0.9), (0.8, 0.5, 1.8))
    surface = SupportSurface(
        surface_id=UniqueID("storage_surface"),
        bounding_box_min=np.array([-0.2, -0.15, 0.0]),
        bounding_box_max=np.array([0.2, 0.15, 0.0]),
        transform=RigidTransform(p=[0.0, 0.0, 1.8]),
        mesh=_NonCopyableMesh(),
    )
    storage.support_surfaces = [surface]
    child = _object(
        "stored_item",
        "book",
        (0.0, 0.0, 1.9),
        (0.1, 0.1, 0.1),
        object_type=ObjectType.MANIPULAND,
    )
    child.placement_info = PlacementInfo(
        parent_surface_id=surface.surface_id,
        position_2d=np.zeros(2),
        rotation_2d=0.0,
    )
    geometry = RoomGeometry(
        sdf_tree=None,
        sdf_path=tmp_path / "room.sdf",
        walls=[],
        width=4.0,
        length=4.0,
        wall_height=2.7,
        wall_thickness=0.1,
    )
    scene = RoomScene(
        room_geometry=geometry,
        scene_dir=tmp_path,
        room_id="accessibility_room",
        room_type="study",
        objects={storage.object_id: storage, child.object_id: child},
    )
    return scene, storage, child, surface


def _payload(scene: RoomScene) -> dict:
    moved = float(scene.objects[UniqueID("storage_piece")].transform.translation()[0])
    label = "pass" if moved > 0.1 else "fail"
    return {
        "summary": {
            "scene_summary": {
                "fail": 0 if label == "pass" else 1,
                "degraded": 0,
                "score": 1.0 if label == "pass" else 0.0,
            }
        },
        "results": [
            {
                "metric": "spatial_accessibility",
                "primary_object": "storage_piece",
                "label": label,
                "diagnostics": {"access_ratio": 1.0 if label == "pass" else 0.0},
            }
        ],
    }


def _degraded_payload(scene: RoomScene) -> dict:
    moved = float(scene.objects[UniqueID("storage_piece")].transform.translation()[0])
    label = "pass" if moved > 0.1 else "degraded"
    return {
        "summary": {
            "scene_summary": {
                "fail": 0,
                "degraded": 0 if label == "pass" else 1,
                "score": 1.0 if label == "pass" else 0.8,
            }
        },
        "results": [
            {
                "metric": "spatial_accessibility",
                "primary_object": "storage_piece",
                "label": label,
                "diagnostics": {"access_ratio": 1.0 if label == "pass" else 0.5},
            }
        ],
    }


def test_rejected_candidate_restores_pose_without_copying_mesh(
    tmp_path: Path, monkeypatch
) -> None:
    scene, storage, child, surface = _scene(tmp_path)
    original_surface_list = storage.support_surfaces
    old_storage = storage.transform.GetAsMatrix4().copy()
    old_child = child.transform.GetAsMatrix4().copy()
    old_surface = surface.transform.GetAsMatrix4().copy()

    def fake_evaluate(current_scene, _config):
        payload = _payload(current_scene)
        current_scene.objects[storage.object_id].support_surfaces.clear()
        return payload

    monkeypatch.setattr(
        furniture_accessibility_guard,
        "_evaluate",
        fake_evaluate,
    )

    candidate = _best_candidate(
        scene,
        storage,
        current_score=(1, 0, 0.0),
        config=CriticConfig(enabled=True),
        max_translation_m=0.2,
        step_m=0.2,
        max_candidate_evaluations=2,
    )

    assert candidate is None
    np.testing.assert_allclose(storage.transform.GetAsMatrix4(), old_storage)
    np.testing.assert_allclose(child.transform.GetAsMatrix4(), old_child)
    np.testing.assert_allclose(surface.transform.GetAsMatrix4(), old_surface)
    assert storage.support_surfaces is original_surface_list
    assert storage.support_surfaces == [surface]
    assert surface.mesh is storage.support_surfaces[0].mesh


def test_accepted_candidate_moves_storage_surface_and_child(
    tmp_path: Path, monkeypatch
) -> None:
    scene, storage, child, surface = _scene(tmp_path)
    monkeypatch.setattr(
        furniture_accessibility_guard,
        "_candidate_directions",
        lambda _scene, _obj: [np.array([1.0, 0.0])],
    )
    monkeypatch.setattr(
        furniture_accessibility_guard,
        "_candidate_distances",
        lambda _max_translation, _step: [0.2],
    )
    monkeypatch.setattr(
        furniture_accessibility_guard,
        "_evaluate",
        lambda current_scene, _config: _payload(current_scene),
    )

    fixes = improve_storage_front_access(
        scene,
        config=CriticConfig(enabled=True),
        max_translation_m=0.2,
        step_m=0.2,
    )

    assert [fix.subject_id for fix in fixes] == ["storage_piece"]
    assert math.isclose(float(storage.transform.translation()[0]), 0.2, abs_tol=1e-8)
    assert math.isclose(float(child.transform.translation()[0]), 0.2, abs_tol=1e-8)
    assert math.isclose(float(surface.transform.translation()[0]), 0.2, abs_tol=1e-8)
    assert surface.mesh is storage.support_surfaces[0].mesh


def test_degraded_storage_is_only_repaired_when_requested(
    tmp_path: Path, monkeypatch
) -> None:
    scene, storage, _, _ = _scene(tmp_path)
    monkeypatch.setattr(
        furniture_accessibility_guard,
        "_candidate_directions",
        lambda _scene, _obj: [np.array([1.0, 0.0])],
    )
    monkeypatch.setattr(
        furniture_accessibility_guard,
        "_candidate_distances",
        lambda _max_translation, _step: [0.2],
    )
    monkeypatch.setattr(
        furniture_accessibility_guard,
        "_evaluate",
        lambda current_scene, _config: _degraded_payload(current_scene),
    )

    assert improve_storage_front_access(scene, config=CriticConfig(enabled=True)) == []
    assert math.isclose(float(storage.transform.translation()[0]), 0.0, abs_tol=1e-8)

    fixes = improve_storage_front_access(
        scene,
        config=CriticConfig(enabled=True),
        repair_degraded=True,
    )

    assert [fix.subject_id for fix in fixes] == ["storage_piece"]
    assert math.isclose(float(storage.transform.translation()[0]), 0.2, abs_tol=1e-8)
