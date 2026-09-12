"""Focused coverage for the read-only SceneEval geometry adapter."""

from __future__ import annotations

import json

from pathlib import Path

import trimesh

from scenesmith.scene_expert.sceneeval_geometry import (
    evaluate_output_root,
    evaluate_scene_state,
    write_output_root_evaluation,
)


def _object(object_id: str, path: Path, translation: list[float]) -> dict:
    return {
        "object_id": object_id,
        "object_type": "furniture",
        "geometry_path": str(path),
        "transform": {"translation": translation, "rotation_wxyz": [1, 0, 0, 0]},
    }


def _state(objects: dict) -> dict:
    return {
        "room_geometry": {
            "footprint_vertices": [[-2, -2], [2, -2], [2, 2], [-2, 2]],
            "openings": [
                {
                    "opening_id": "door_1",
                    "opening_type": "door",
                    "clearance_bbox_min": [-0.5, -2.0, 0.0],
                    "clearance_bbox_max": [0.5, -1.0, 2.0],
                }
            ],
        },
        "objects": objects,
    }


def _write_box(path: Path) -> None:
    trimesh.creation.box(extents=[1, 1, 1]).export(path)


def test_evaluate_scene_state_reports_geometry_metrics(tmp_path: Path) -> None:
    mesh_path = tmp_path / "cube.ply"
    _write_box(mesh_path)
    state_path = tmp_path / "scene_state.json"
    state_path.write_text(
        json.dumps(
            _state(
                {
                    "first": _object("first", mesh_path, [0, -1.5, 0.5]),
                    "second": _object("second", mesh_path, [0.2, -1.5, 0.5]),
                    "outside": _object("outside", mesh_path, [3, 0, 0.5]),
                }
            )
        ),
        encoding="utf-8",
    )

    result = evaluate_scene_state(state_path)

    assert result["results"]["CollisionMetric"]["scene_in_collision"] is True
    assert result["results"]["OutOfBoundMetric"]["out_of_bound_object_count"] == 1
    assert result["results"]["OpeningClearanceMetric"]["blocked_opening_count"] == 1
    assert 0 <= result["results"]["NavigabilityMetric"]["navigability"] <= 1
    assert "ObjCountMetric" in result["not_run"]["vlm_metrics"]


def test_output_root_discovers_final_scene_states_and_writes_bundle(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "cube.ply"
    _write_box(mesh_path)
    state_path = (
        tmp_path
        / "critic_on"
        / "batch_001"
        / "hydra"
        / "scene_000"
        / "room_bedroom"
        / "scene_states"
        / "final_scene"
        / "scene_state.json"
    )
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(_state({"bed": _object("bed", mesh_path, [0, 0, 0.5])})),
        encoding="utf-8",
    )

    result = evaluate_output_root(tmp_path)
    destination = write_output_root_evaluation(tmp_path)

    assert result["summary"]["scene_room_count"] == 1
    assert result["scenes"][0]["scene_id"] == "scene_000"
    assert destination.is_file()
    assert (
        json.loads(destination.read_text(encoding="utf-8"))["summary"][
            "scene_room_count"
        ]
        == 1
    )


def test_missing_geometry_is_explicit_unavailable_evidence(tmp_path: Path) -> None:
    state_path = tmp_path / "scene_state.json"
    state_path.write_text(
        json.dumps(
            _state({"missing": _object("missing", tmp_path / "none.glb", [0, 0, 0])})
        ),
        encoding="utf-8",
    )

    result = evaluate_scene_state(state_path)

    assert result["available_object_count"] == 0
    assert result["unavailable_objects"] == [
        {"object_id": "missing", "reason": "missing_geometry"}
    ]


def test_dimension_fallback_uses_length_for_x_and_width_for_y(tmp_path: Path) -> None:
    mesh_path = tmp_path / "cube.ply"
    _write_box(mesh_path)
    state_path = tmp_path / "scene_state.json"
    payload = _state({"inside": _object("inside", mesh_path, [1.8, 1.8, 0.5])})
    payload["room_geometry"].pop("footprint_vertices")
    payload["room_geometry"].update({"length": 4.0, "width": 2.0})
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate_scene_state(state_path)

    assert result["floor_boundary_source"] == "room_dimensions"
    assert result["results"]["OutOfBoundMetric"]["out_of_bound_object_count"] == 1


def test_gltf_assets_are_converted_from_y_up_to_scene_state_z_up(
    tmp_path: Path,
) -> None:
    floor_path = tmp_path / "floor" / "floor.gltf"
    mesh_path = tmp_path / "object" / "cube.gltf"
    floor_path.parent.mkdir()
    mesh_path.parent.mkdir()
    trimesh.creation.box(extents=[4, 0.1, 2]).export(floor_path)
    _write_box(mesh_path)
    state_path = tmp_path / "scene_state.json"
    payload = _state({"inside": _object("inside", mesh_path, [0, 0.5, 0.5])})
    payload["room_geometry"].pop("footprint_vertices")
    payload["room_geometry"].update(
        {
            "length": 4.0,
            "width": 2.0,
            "floor": {
                "geometry_path": str(floor_path),
                "transform": {"translation": [0, 0, 0], "rotation_wxyz": [1, 0, 0, 0]},
            },
        }
    )
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate_scene_state(state_path)

    assert result["floor_boundary_source"] == "floor_mesh_convex_hull"
    assert result["results"]["OutOfBoundMetric"]["out_of_bound_object_count"] == 0
