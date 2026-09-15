"""Read-only SceneEval-compatible geometry report generation.

This consumes SceneSmith's final ``scene_state.json`` directly. SceneEval's
public HSSD loader cannot load generated per-run mesh paths, so results are
explicitly labelled ``scenesmith_adapted`` instead of an official full score.
"""

from __future__ import annotations

import argparse
import json

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from scipy import ndimage
from scipy.spatial.transform import Rotation
from shapely import contains_xy
from shapely.geometry import MultiPoint, Point, Polygon


SCHEMA_VERSION = "scenesmith.sceneeval_geometry.v1"
REFERENCE_PLAN = "SceneEval v1.1 no_llm_plan (scenesmith_adapted)"
METRICS = (
    "CollisionMetric",
    "NavigabilityMetric",
    "OutOfBoundMetric",
    "OpeningClearanceMetric",
)
GLTF_Y_UP_TO_SCENESMITH_Z_UP = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@dataclass(frozen=True)
class LoadedObject:
    object_id: str
    mesh: trimesh.Trimesh
    bounds: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolve_path(raw_path: Any, state_path: Path) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for parent in (state_path.parent, *state_path.parents):
        resolved = parent / candidate
        if resolved.is_file():
            return resolved
    return None


def _matrix_from_transform(value: dict[str, Any]) -> np.ndarray:
    translation = np.asarray(value.get("translation") or [0.0, 0.0, 0.0], dtype=float)
    quaternion = np.asarray(
        value.get("rotation_wxyz") or [1.0, 0.0, 0.0, 0.0], dtype=float
    )
    norm = np.linalg.norm(quaternion)
    if (
        translation.shape != (3,)
        or quaternion.shape != (4,)
        or not np.isfinite(translation).all()
        or not np.isfinite(quaternion).all()
        or norm == 0
    ):
        raise ValueError("invalid SceneSmith rigid transform")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]] / norm).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry()
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError("asset contains no meshes")
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise ValueError("asset is not a non-empty mesh")
    if path.suffix.lower() in {".glb", ".gltf"}:
        loaded.apply_transform(GLTF_Y_UP_TO_SCENESMITH_Z_UP)
    return loaded


def _load_objects(
    payload: dict[str, Any], state_path: Path
) -> tuple[list[LoadedObject], list[dict[str, str]]]:
    raw_objects = payload.get("objects") or {}
    if not isinstance(raw_objects, dict):
        return [], [{"reason": "invalid_objects_payload"}]
    objects: list[LoadedObject] = []
    unavailable: list[dict[str, str]] = []
    for object_id, value in raw_objects.items():
        if not isinstance(value, dict) or value.get("object_type") in {"wall", "floor"}:
            continue
        path = _resolve_path(value.get("geometry_path"), state_path)
        if path is None:
            unavailable.append(
                {"object_id": str(object_id), "reason": "missing_geometry"}
            )
            continue
        try:
            mesh = _load_mesh(path)
            mesh.apply_transform(_matrix_from_transform(value.get("transform") or {}))
        except (OSError, ValueError, TypeError) as exc:
            unavailable.append(
                {
                    "object_id": str(object_id),
                    "reason": f"unreadable_geometry:{type(exc).__name__}",
                }
            )
            continue
        objects.append(LoadedObject(str(object_id), mesh, np.asarray(mesh.bounds)))
    return objects, unavailable


def _floor_polygon(
    payload: dict[str, Any], room_geometry: dict[str, Any], state_path: Path
) -> tuple[Polygon, str]:
    points = room_geometry.get("footprint_vertices")
    if isinstance(points, list) and len(points) >= 3:
        polygon = Polygon([(float(point[0]), float(point[1])) for point in points])
        if polygon.is_valid and polygon.area > 0:
            return polygon, "footprint_vertices"

    floor = room_geometry.get("floor")
    if not isinstance(floor, dict):
        objects = payload.get("objects") or {}
        if isinstance(objects, dict):
            floor = next(
                (
                    value
                    for value in objects.values()
                    if isinstance(value, dict) and value.get("object_type") == "floor"
                ),
                None,
            )
    if isinstance(floor, dict):
        floor_path = _resolve_path(floor.get("geometry_path"), state_path)
        if floor_path is not None:
            try:
                mesh = _load_mesh(floor_path)
                mesh.apply_transform(
                    _matrix_from_transform(floor.get("transform") or {})
                )
                polygon = MultiPoint(mesh.vertices[:, :2]).convex_hull
                if (
                    isinstance(polygon, Polygon)
                    and polygon.is_valid
                    and polygon.area > 0
                ):
                    return polygon, "floor_mesh_convex_hull"
            except (OSError, ValueError, TypeError):
                pass

    width = float(room_geometry.get("width") or 0.0)
    length = float(room_geometry.get("length") or 0.0)
    if width <= 0 or length <= 0:
        raise ValueError("room geometry has no usable footprint")
    # RoomGeometry defines length as the X axis and width as the Y axis.
    return (
        Polygon(
            [
                (-length / 2, -width / 2),
                (length / 2, -width / 2),
                (length / 2, width / 2),
                (-length / 2, width / 2),
            ]
        ),
        "room_dimensions",
    )


def _aabb_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(np.all(first[0] < second[1]) and np.all(second[0] < first[1]))


def _collision(objects: list[LoadedObject]) -> dict[str, Any]:
    pairs: list[list[str]] = []
    try:
        manager = trimesh.collision.CollisionManager()
    except ValueError as exc:
        return {"status": "unknown", "reason": f"collision_backend_unavailable:{exc}"}
    for index, first in enumerate(objects):
        for second in objects[index + 1 :]:
            if not _aabb_overlap(first.bounds, second.bounds):
                continue
            manager.add_object(first.object_id, first.mesh)
            collides = bool(manager.in_collision_single(second.mesh))
            manager.remove_object(first.object_id)
            if collides:
                pairs.append([first.object_id, second.object_id])
    return {
        "status": "ok",
        "scene_in_collision": bool(pairs),
        "collision_pair_count": len(pairs),
        "collision_pairs": pairs,
    }


def _navigability(
    objects: list[LoadedObject], floor: Polygon, resolution: int = 256
) -> dict[str, Any]:
    minimum_x, minimum_y, maximum_x, maximum_y = floor.bounds
    extent = max(maximum_x - minimum_x, maximum_y - minimum_y)
    if extent <= 0:
        return {"status": "unknown", "reason": "empty_floor_polygon"}
    origin = np.asarray([minimum_x - 0.2, minimum_y - 0.2])
    scale = (resolution - 1) / (extent + 0.4)
    coordinates = np.arange(resolution, dtype=float) / scale + origin[0]
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    canvas = contains_xy(floor, x_grid, y_grid)
    kernel_size = max(1, int(round(0.2 * scale)))
    canvas = ndimage.binary_erosion(
        canvas,
        structure=np.ones((kernel_size, kernel_size), dtype=bool),
        border_value=0,
    )
    for obj in objects:
        hull = MultiPoint(obj.mesh.vertices[:, :2]).convex_hull
        if isinstance(hull, Polygon) and hull.area > 0:
            canvas[contains_xy(hull, x_grid, y_grid)] = False
    labels, component_count = ndimage.label(
        canvas, structure=np.ones((3, 3), dtype=bool)
    )
    walkable = int(np.count_nonzero(canvas))
    largest = int(np.bincount(labels.ravel())[1:].max()) if component_count else 0
    return {
        "status": "ok",
        "navigability": round(largest / walkable, 6) if walkable else 0.0,
        "connected_components": component_count,
        "walkable_pixels": walkable,
    }


def _out_of_bounds(objects: list[LoadedObject], floor: Polygon) -> dict[str, Any]:
    evaluations: dict[str, dict[str, Any]] = {}
    for obj in objects:
        points = obj.mesh.vertices[:, :2]
        if len(points) == 0:
            continue
        in_bound = sum(floor.covers(Point(point)) for point in points)
        ratio = in_bound / len(points)
        evaluations[obj.object_id] = {
            "ratio_in_bound": round(ratio, 6),
            "out_of_bound": ratio < 0.99,
            "sample_count": int(len(points)),
        }
    return {
        "status": "ok",
        "out_of_bound_object_count": sum(
            item["out_of_bound"] for item in evaluations.values()
        ),
        "objects": evaluations,
    }


def _opening_clearance(
    objects: list[LoadedObject], room_geometry: dict[str, Any]
) -> dict[str, Any]:
    evaluations: dict[str, dict[str, Any]] = {}
    for opening in room_geometry.get("openings") or []:
        if not isinstance(opening, dict):
            continue
        opening_id = str(opening.get("opening_id") or "")
        minimum = np.asarray(opening.get("clearance_bbox_min") or [], dtype=float)
        maximum = np.asarray(opening.get("clearance_bbox_max") or [], dtype=float)
        if not opening_id or minimum.shape != (3,) or maximum.shape != (3,):
            continue
        clearance_bounds = np.asarray([minimum, maximum])
        blockers = [
            obj.object_id
            for obj in objects
            if _aabb_overlap(clearance_bounds, obj.bounds)
        ]
        evaluations[opening_id] = {
            "opening_type": str(opening.get("opening_type") or "unknown"),
            "blocked": bool(blockers),
            "blockers": blockers,
        }
    return {
        "status": "ok" if evaluations else "unknown",
        "reason": None if evaluations else "no_openings_with_clearance_bounds",
        "blocked_opening_count": sum(item["blocked"] for item in evaluations.values()),
        "openings": evaluations,
    }


def evaluate_scene_state(state_path: Path) -> dict[str, Any]:
    """Evaluate one final SceneSmith state without mutating its artifacts."""
    payload = _read_json(state_path)
    room_geometry = payload.get("room_geometry") or {}
    if not isinstance(room_geometry, dict):
        raise ValueError(f"invalid room geometry: {state_path}")
    objects, unavailable = _load_objects(payload, state_path)
    try:
        floor, floor_boundary_source = _floor_polygon(
            payload, room_geometry, state_path
        )
    except ValueError as exc:
        unknown = {"status": "unknown", "reason": str(exc)}
        return {
            "schema_version": SCHEMA_VERSION,
            "reference": REFERENCE_PLAN,
            "state_path": str(state_path),
            "available_object_count": len(objects),
            "unavailable_objects": unavailable,
            "results": {metric: unknown for metric in METRICS},
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "reference": REFERENCE_PLAN,
        "state_path": str(state_path),
        "available_object_count": len(objects),
        "unavailable_objects": unavailable,
        "floor_boundary_source": floor_boundary_source,
        "results": {
            "CollisionMetric": _collision(objects),
            "NavigabilityMetric": _navigability(objects, floor),
            "OutOfBoundMetric": _out_of_bounds(objects, floor),
            "OpeningClearanceMetric": _opening_clearance(objects, room_geometry),
        },
        "not_run": {
            "vlm_metrics": [
                "ObjCountMetric",
                "ObjAttributeMetric",
                "ObjObjRelationshipMetric",
                "ObjArchRelationshipMetric",
                "SupportMetric",
                "AccessibilityMetric",
            ],
            "reason": "requires a separately configured VLM; not part of no-VLM evaluation",
        },
    }


def _final_state_paths(output_root: Path) -> list[Path]:
    return sorted(
        output_root.glob(
            "critic_on/batch_*/hydra/scene_*/room_*/scene_states/final_scene/scene_state.json"
        )
    )


def _scene_directory(state_path: Path) -> Path:
    return next(
        parent
        for parent in state_path.parents
        if parent.name.startswith("scene_") and parent.name[6:].isdigit()
    )


def _mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return round(float(np.mean(present)), 6) if present else None


def _rate(values: Iterable[bool]) -> float | None:
    present = list(values)
    return round(sum(present) / len(present), 6) if present else None


def evaluate_output_root(output_root: Path) -> dict[str, Any]:
    """Evaluate every complete room state below a critic-probe output root."""
    root = output_root.resolve()
    scenes: list[dict[str, Any]] = []
    for state_path in _final_state_paths(root):
        try:
            result = evaluate_scene_state(state_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            result = {
                "schema_version": SCHEMA_VERSION,
                "reference": REFERENCE_PLAN,
                "state_path": str(state_path),
                "status": "unknown",
                "error": f"{type(exc).__name__}: {exc}",
            }
        result.update(
            {
                "scene_id": _scene_directory(state_path).name,
                "room_id": state_path.parents[2].name,
            }
        )
        scenes.append(result)
    return {
        "schema_version": SCHEMA_VERSION,
        "reference": REFERENCE_PLAN,
        "output_root": str(root),
        "summary": {
            "scene_room_count": len(scenes),
            "evaluated_scene_room_count": sum("results" in scene for scene in scenes),
            "collision_free_scene_room_rate": _rate(
                [
                    not scene["results"]["CollisionMetric"]["scene_in_collision"]
                    for scene in scenes
                    if "results" in scene
                    and scene["results"]["CollisionMetric"].get("status") == "ok"
                ]
            ),
            "mean_navigability": _mean(
                scene["results"]["NavigabilityMetric"].get("navigability")
                for scene in scenes
                if "results" in scene
                and scene["results"]["NavigabilityMetric"].get("status") == "ok"
            ),
        },
        "scenes": scenes,
    }


def write_output_root_evaluation(output_root: Path) -> Path:
    destination = output_root / "sceneeval" / "sceneeval_geometry.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(evaluate_output_root(output_root), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        f"SceneEval no-VLM geometry results: {write_output_root_evaluation(args.output_root)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
