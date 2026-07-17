"""Adapters from SceneSmith scene objects to SceneBenchmark-style case packs."""

from __future__ import annotations

import math
import re

from typing import Any

import numpy as np

from pydrake.math import RollPitchYaw

from scenesmith.agent_utils.house import HouseScene
from scenesmith.agent_utils.room import (
    ObjectType,
    RoomScene,
    SceneObject,
    SupportSurface,
)
from scenesmith.scenebenchmark_critic.asset_library_annotations import (
    get_hssd_asset_annotations,
)
from scenesmith.scenebenchmark_critic.evaluator import build_all_checks
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.constants import (
    BEDS,
    MEDIA,
    NIGHTSTANDS,
    SEATING,
    SUPPORT_CATEGORY_GROUPS,
    SUPPORT_INTERNAL_AFFORDANCES,
    SUPPORT_TOP_SURFACE_AFFORDANCES,
    SUPPORTED_SMALL,
    SUPPORTS,
    WORK_SURFACE_CATEGORY_GROUPS,
    WORK_SURFACES,
)

MEDIA_TARGET_CATEGORIES = MEDIA | {"monitor", "screen", "tv_stand"}
WORK_SURFACE_EXPOSURE_CATEGORIES = (
    WORK_SURFACES
    | NIGHTSTANDS
    | {
        "bookshelf",
        "buffet",
        "console",
        "counter",
        "credenza",
        "end_table",
        "island",
        "media_console",
        "shelf",
        "sideboard",
        "tv_stand",
        "wall_shelf",
    }
)
WORK_SURFACE_GROUPS = WORK_SURFACE_CATEGORY_GROUPS | {"storage_surface"}
CATEGORY_ALIASES = {
    "arm chair": "armchair",
    "armchair": "armchair",
    "bar stool": "bar_stool",
    "barstool": "bar_stool",
    "bean bag": "bean_bag",
    "beanbag": "bean_bag",
    "beanbag chair": "beanbag_chair",
    "book shelf": "bookshelf",
    "bookcase": "bookshelf",
    "bookshelf": "bookshelf",
    "cabinetdrawer": "drawer",
    "ceiling light": "ceiling_light",
    "coffee table": "coffee_table",
    "coffeetable": "coffee_table",
    "computer display": "monitor",
    "computer monitor": "monitor",
    "computer screen": "monitor",
    "couch": "sofa",
    "craft activity table": "table",
    "craft table": "table",
    "cubby storage shelf": "shelf",
    "desk lamp": "desk_lamp",
    "desklamp": "desk_lamp",
    "dining chair": "dining_chair",
    "dining table": "dining_table",
    "end table": "side_table",
    "floor lamp": "floor_lamp",
    "floorlamp": "floor_lamp",
    "fridge": "refrigerator",
    "lcd monitor": "monitor",
    "media console": "media_console",
    "night stand": "nightstand",
    "notebook computer": "laptop",
    "notebookcomputer": "laptop",
    "office chair": "office_chair",
    "officechair": "office_chair",
    "pendant light": "pendant_light",
    "range oven": "range_oven",
    "side table": "side_table",
    "sidetable": "nightstand",
    "sidetabledesk": "nightstand",
    "simplebookcase": "bookshelf",
    "table lamp": "table_lamp",
    "tablelamp": "table_lamp",
    "tablet": "tablet_computer",
    "tablet computer": "tablet_computer",
    "television": "television",
    "toy chest storage bench": "bench",
    "toy storage bin": "storage_bin",
    "tv stand": "tv_stand",
    "tv": "television",
    "tvstand": "tv_stand",
    "wall cabinet": "wall_cabinet",
    "wall shelf": "wall_shelf",
    "wine cabinet": "wine_cabinet",
}
HEURISTIC_AFFORDANCE_MAP = {
    "armchair": {"sittable"},
    "bar_stool": {"sittable"},
    "bar_table": {"supportable"},
    "bean_bag": {"sittable"},
    "beanbag_chair": {"sittable"},
    "bed": {"sittable", "sleepable", "supportable"},
    "bench": {"sittable"},
    "book": {"graspable"},
    "bookshelf": {"supportable"},
    "bottle": {"graspable"},
    "bowl": {"containable", "graspable"},
    "cabinet": {"containable", "openable", "supportable"},
    "chair": {"sittable"},
    "coffee_table": {"supportable"},
    "cup": {"graspable"},
    "desk": {"supportable"},
    "desk_lamp": {"toggleable"},
    "dining_chair": {"sittable"},
    "dining_table": {"supportable"},
    "door": {"openable"},
    "drawer": {"containable", "openable"},
    "dresser": {"containable", "openable", "supportable"},
    "floor_lamp": {"toggleable"},
    "glass": {"graspable"},
    "lamp": {"toggleable"},
    "laptop": {"graspable"},
    "keyboard": {"graspable"},
    "monitor": {"graspable"},
    "mouse": {"graspable"},
    "loveseat": {"sittable"},
    "microwave": {"containable", "openable"},
    "mug": {"graspable"},
    "nightstand": {"containable", "openable", "supportable"},
    "office_chair": {"sittable"},
    "plate": {"graspable"},
    "range_oven": {"containable", "openable", "supportable"},
    "refrigerator": {"containable", "openable"},
    "remote": {"graspable"},
    "shelf": {"supportable"},
    "side_table": {"supportable"},
    "sofa": {"sittable"},
    "stool": {"sittable"},
    "table": {"supportable"},
    "table_lamp": {"toggleable"},
    "tablet_computer": {"graspable"},
    "tray": {"graspable", "supportable"},
    "tv_stand": {"containable", "openable", "supportable"},
    "vase": {"containable", "graspable"},
    "wall_cabinet": {"containable", "openable", "supportable"},
    "wall_shelf": {"supportable"},
    "wardrobe": {"containable", "openable"},
    "window": {"openable"},
    "wine_cabinet": {"containable", "openable", "supportable"},
}
ROOM_KEYWORD_SET = {
    "bathroom",
    "bedroom",
    "cafe",
    "children_room",
    "classroom",
    "dining_room",
    "entryway",
    "gym",
    "hotel_room",
    "kitchen",
    "library",
    "living_room",
    "museum_gallery",
    "music_room",
    "nursery",
    "office",
    "playroom",
    "study",
    "study_room",
}
CATEGORY_KEYWORDS = {
    "armchair": ["armchair", "lounge chair", "easy chair"],
    "bed": ["bed", "bedframe", "mattress", "sleeping area"],
    "bench": ["bench", "seating bench"],
    "book": ["book", "notebook", "stack of books"],
    "bookshelf": ["bookshelf", "bookcase", "shelving unit"],
    "bottle": ["bottle", "water bottle"],
    "cabinet": ["cabinet", "storage cabinet", "cupboard"],
    "chair": ["chair", "side chair", "dining chair", "seating"],
    "coffee_table": ["coffee table", "center table", "low table"],
    "cup": ["cup", "drinking cup"],
    "desk": ["desk", "work desk", "study desk", "writing desk"],
    "desk_lamp": ["desk lamp", "table lamp", "reading lamp"],
    "dresser": ["dresser", "chest of drawers", "bureau"],
    "drawer": ["drawer", "pull-out drawer", "storage drawer"],
    "floor_lamp": ["floor lamp", "standing lamp"],
    "laptop": ["laptop", "notebook computer"],
    "keyboard": ["keyboard", "computer keyboard"],
    "loveseat": ["loveseat", "two-seat sofa", "small couch"],
    "microwave": ["microwave", "microwave oven"],
    "monitor": ["monitor", "computer monitor", "screen", "display"],
    "mug": ["mug", "coffee mug", "cup"],
    "nightstand": ["nightstand", "bedside table", "side table"],
    "office_chair": ["office chair", "desk chair", "task chair", "swivel chair"],
    "range_oven": ["oven", "range oven", "stove", "cooker"],
    "refrigerator": ["refrigerator", "fridge", "cooling cabinet"],
    "remote": ["remote", "remote control"],
    "rug": ["rug", "carpet", "floor mat"],
    "shelf": ["shelf", "wall shelf", "storage shelf"],
    "sofa": ["sofa", "couch", "settee", "three-seat sofa"],
    "table": ["table", "dining table", "work table"],
    "table_lamp": ["table lamp", "desk lamp", "reading lamp"],
    "tablet_computer": ["tablet computer", "tablet", "touchscreen"],
    "tv_stand": ["tv stand", "media console", "television console"],
    "vase": ["vase", "flower vase", "decorative vase"],
    "wall_cabinet": ["wall cabinet", "storage cabinet", "cupboard"],
    "wall_shelf": ["wall shelf", "shelf", "storage shelf"],
    "wardrobe": ["wardrobe", "closet", "armoire"],
}
CATEGORY_GROUPS = {
    "armchair": "seating",
    "bed": "sleeping",
    "bench": "seating",
    "book": "small_object",
    "bookshelf": "storage",
    "bottle": "small_object",
    "cabinet": "storage",
    "chair": "seating",
    "coffee_table": "work_surface",
    "cup": "small_object",
    "desk": "work_surface",
    "desk_lamp": "lighting",
    "dresser": "storage",
    "drawer": "storage",
    "floor_lamp": "lighting",
    "laptop": "small_object",
    "keyboard": "small_object",
    "loveseat": "seating",
    "microwave": "appliance",
    "monitor": "media",
    "mouse": "small_object",
    "mug": "small_object",
    "nightstand": "storage_surface",
    "office_chair": "seating",
    "range_oven": "appliance",
    "refrigerator": "appliance_storage",
    "remote": "small_object",
    "rug": "soft_furnishing",
    "shelf": "storage",
    "sofa": "seating",
    "table": "work_surface",
    "table_lamp": "lighting",
    "tablet_computer": "small_object",
    "tv_stand": "storage_surface",
    "vase": "decor",
    "wall_cabinet": "storage",
    "wall_shelf": "storage_surface",
    "wardrobe": "storage",
}
KNOWN_CATEGORY_TOKENS = (
    set(CATEGORY_ALIASES.values())
    | set(HEURISTIC_AFFORDANCE_MAP)
    | set(CATEGORY_KEYWORDS)
    | set(SUPPORTS)
    | set(SUPPORTED_SMALL)
    | set(SEATING)
    | set(BEDS)
    | set(NIGHTSTANDS)
    | set(WORK_SURFACE_EXPOSURE_CATEGORIES)
    | set(MEDIA_TARGET_CATEGORIES)
    | {
        "bar_stool",
        "ceiling_light",
        "counter",
        "door",
        "island",
        "keyboard",
        "monitor",
        "mouse",
        "notebook_computer",
        "plate",
        "plant",
        "screen",
        "table_lamp",
        "tablet_computer",
        "television",
        "tray",
        "wall_cabinet",
        "window",
    }
)
GENERIC_CATEGORY_WORDS = {
    "black",
    "comfortable",
    "large",
    "modern",
    "simple",
    "small",
    "white",
    "wooden",
}


def room_scene_to_case_pack(
    scene: RoomScene,
    *,
    stage: str = "adhoc",
    metrics: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    scene_geometry = _room_scene_geometry(scene)
    case_pack = {
        "schema_version": "scenesmith.scenebenchmark_critic.v1",
        "scene_id": f"{scene.room_id}:{stage}",
        "source_method": "scenesmith_online",
        "task_instruction": scene.text_description,
        "room_type": scene.room_type,
        "scene_geometry": scene_geometry,
        "checks": [],
    }
    case_pack["checks"] = build_all_checks(case_pack, metrics=metrics)
    return case_pack


def house_scene_to_case_pack(
    house: HouseScene,
    *,
    stage: str = "adhoc",
    metrics: tuple[str, ...] | list[str] | None = None,
    include_object_types: list[ObjectType] | tuple[ObjectType, ...] | None = None,
) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    rooms: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []

    for room_id, room in house.rooms.items():
        ox, oy = house._get_room_position(room_id)
        room_offset = np.array([ox, oy, 0.0])
        room_geom = _room_scene_geometry(
            room, room_offset=room_offset, include_object_types=include_object_types
        )
        rooms.extend(room_geom["rooms"])
        objects.extend(room_geom["objects"])
        relations.extend(room_geom["relations"])

    case_pack = {
        "schema_version": "scenesmith.scenebenchmark_critic.v1",
        "scene_id": f"house:{stage}",
        "source_method": "scenesmith_online",
        "task_instruction": (
            getattr(house.layout, "house_prompt", "")
            or getattr(house.layout, "prompt", "")
            or ""
        ),
        "scene_geometry": {
            "unit": "m",
            "rooms": rooms,
            "objects": objects,
            "relations": relations,
            "task_relation_graph": {},
        },
        "checks": [],
    }
    case_pack["checks"] = build_all_checks(case_pack, metrics=metrics)
    return case_pack


def _room_scene_geometry(
    scene: RoomScene,
    *,
    room_offset: np.ndarray | None = None,
    include_object_types: list[ObjectType] | tuple[ObjectType, ...] | None = None,
) -> dict[str, Any]:
    offset = room_offset if room_offset is not None else np.zeros(3)
    objects: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    surface_owner_ids = _surface_owner_ids(scene)

    added_object_ids: set[str] = set()
    if scene.room_geometry and scene.room_geometry.floor:
        objects.append(_object_to_geometry(scene.room_geometry.floor, scene, offset))
        added_object_ids.add(str(scene.room_geometry.floor.object_id))
    if scene.room_geometry:
        for wall in scene.room_geometry.walls:
            wall_id = str(wall.object_id)
            if wall_id in added_object_ids:
                continue
            objects.append(_object_to_geometry(wall, scene, offset))
            added_object_ids.add(wall_id)

    included_ids: set[str] = set()
    allowed_types = (
        set(include_object_types) if include_object_types is not None else None
    )
    for obj in scene.objects.values():
        if str(obj.object_id) in added_object_ids:
            continue
        if allowed_types is not None and obj.object_type not in allowed_types:
            continue
        objects.append(_object_to_geometry(obj, scene, offset))
        added_object_ids.add(str(obj.object_id))
        included_ids.add(str(obj.object_id))
        if obj.placement_info:
            target_object_id = surface_owner_ids.get(
                str(obj.placement_info.parent_surface_id)
            )
            relations.append(
                {
                    "relation_type": "placed_on_surface",
                    "subject": str(obj.object_id),
                    "object": target_object_id,
                    "subject_id": str(obj.object_id),
                    "target_ids": [target_object_id] if target_object_id else [],
                    "target_surface_id": str(obj.placement_info.parent_surface_id),
                    "placement_method": obj.placement_info.placement_method,
                }
            )
        relations.extend(_metadata_dependency_relations(obj, surface_owner_ids))

    return {
        "unit": "m",
        "rooms": [_room_geometry_record(scene, offset)],
        "objects": objects,
        "relations": [
            relation
            for relation in relations
            if str(relation.get("subject_id")) in included_ids
        ],
        "task_relation_graph": {},
        "scene_shell": _scene_shell_record(scene, offset),
    }


def _surface_owner_ids(scene: RoomScene) -> dict[str, str]:
    owners: dict[str, str] = {}
    candidates: list[SceneObject] = []
    if scene.room_geometry and scene.room_geometry.floor:
        candidates.append(scene.room_geometry.floor)
    if scene.room_geometry:
        candidates.extend(scene.room_geometry.walls)
    candidates.extend(scene.objects.values())
    for obj in candidates:
        for surface in obj.support_surfaces:
            owners[str(surface.surface_id)] = str(obj.object_id)
    return owners


def _room_geometry_record(scene: RoomScene, offset: np.ndarray) -> dict[str, Any]:
    geom = scene.room_geometry
    length = float(getattr(geom, "length", 0.0) or 0.0)
    width = float(getattr(geom, "width", 0.0) or 0.0)
    height = float(getattr(geom, "wall_height", 2.5) or 2.5)
    x0, x1 = -length / 2.0 + offset[0], length / 2.0 + offset[0]
    y0, y1 = -width / 2.0 + offset[1], width / 2.0 + offset[1]
    return {
        "id": scene.room_id,
        "room_type": scene.room_type,
        "bbox": {"min": [x0, y0, 0.0], "max": [x1, y1, height]},
        "floor_polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
    }


def _object_to_geometry(
    obj: SceneObject, scene: RoomScene, offset: np.ndarray
) -> dict[str, Any]:
    bounds = obj.compute_world_bounds()
    if bounds is None:
        world_min = np.array(obj.transform.translation(), dtype=float) + offset
        world_max = world_min.copy()
    else:
        world_min, world_max = bounds
        world_min = np.array(world_min, dtype=float) + offset
        world_max = np.array(world_max, dtype=float) + offset

    center = (world_min + world_max) / 2.0
    size = np.maximum(world_max - world_min, 0.0)
    local_size = _local_bbox_size(obj, fallback_size=size)
    category = _category_for_object(obj)
    yaw = _semantic_yaw_deg(obj, scene=scene)
    functional_hints = _functional_hints(obj, category, yaw_deg=yaw)

    support_regions = _metadata_support_regions(obj, offset)
    existing_region_ids = {
        str(region.get("region_id") or "")
        for region in support_regions
        if isinstance(region, dict)
    }
    for surface in obj.support_surfaces:
        region = _support_surface_to_region(surface, offset)
        if region["region_id"] in existing_region_ids:
            continue
        support_regions.append(region)
        existing_region_ids.add(region["region_id"])
    if support_regions:
        functional_hints["support_region_summary"] = _support_region_summary(
            support_regions
        )

    record: dict[str, Any] = {
        "id": str(obj.object_id),
        "room": scene.room_id,
        "name": obj.name,
        "description": obj.description,
        "object_type": obj.object_type.value,
        "category": category,
        "category_norm": category,
        "yaw_deg": yaw,
        "bbox_world": {
            "center": center.tolist(),
            "size": size.tolist(),
            "min": world_min.tolist(),
            "max": world_max.tolist(),
        },
        "footprint_world": _footprint_world(obj, offset),
        "interaction_faces": _interaction_faces(
            center, local_size, yaw, category, functional_hints
        ),
        "functional_hints": functional_hints,
        "object_function_profile": _object_function_profile(obj, category),
        "metadata": dict(obj.metadata),
    }
    record["nav_obstacle_class"] = _nav_obstacle_class(record)
    interaction_height = record["functional_hints"].get("interaction_height_m")
    if isinstance(interaction_height, dict):
        record["interaction_height_m"] = dict(interaction_height)
    if support_regions:
        record["support_regions"] = support_regions
    if obj.placement_info:
        record["placement_info"] = {
            "parent_surface_id": str(obj.placement_info.parent_surface_id),
            "position_2d": obj.placement_info.position_2d.tolist(),
            "rotation_2d": float(obj.placement_info.rotation_2d),
            "placement_method": obj.placement_info.placement_method,
        }
    return record


def _scene_shell_record(scene: RoomScene, offset: np.ndarray) -> dict[str, Any]:
    doors: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    geom = scene.room_geometry
    for opening in getattr(geom, "openings", []) or []:
        record = _opening_record(opening, offset)
        opening_type = str(getattr(opening, "opening_type", "") or "").lower()
        if opening_type == "window":
            windows.append(record)
        elif opening_type in {"door", "open"}:
            doors.append(record)
    return {"doors": doors, "windows": windows}


def _opening_record(opening: Any, offset: np.ndarray) -> dict[str, Any]:
    center = np.array(getattr(opening, "center_world", [0.0, 0.0, 0.0]), dtype=float)
    center = center + offset
    record = {
        "id": str(getattr(opening, "opening_id", "")),
        "opening_id": str(getattr(opening, "opening_id", "")),
        "opening_type": str(getattr(opening, "opening_type", "")),
        "center": center.tolist(),
        "position": center.tolist(),
        "width": float(getattr(opening, "width", 0.0) or 0.0),
        "height": float(getattr(opening, "height", 0.0) or 0.0),
        "sill_height": float(getattr(opening, "sill_height", 0.0) or 0.0),
        "wall_direction": str(getattr(opening, "wall_direction", "")),
    }
    bbox_min = getattr(opening, "clearance_bbox_min", None)
    bbox_max = getattr(opening, "clearance_bbox_max", None)
    if bbox_min is not None and bbox_max is not None:
        record["bbox"] = {
            "min": (np.array(bbox_min, dtype=float) + offset).tolist(),
            "max": (np.array(bbox_max, dtype=float) + offset).tolist(),
        }
    return record


def _local_bbox_size(obj: SceneObject, *, fallback_size: np.ndarray) -> np.ndarray:
    if obj.bbox_min is None or obj.bbox_max is None:
        return fallback_size
    bbox_min = np.array(obj.bbox_min, dtype=float)
    bbox_max = np.array(obj.bbox_max, dtype=float)
    return np.maximum(bbox_max - bbox_min, 0.0)


def _footprint_world(obj: SceneObject, offset: np.ndarray) -> list[list[float]]:
    if obj.bbox_min is None or obj.bbox_max is None:
        center = np.array(obj.transform.translation(), dtype=float) + offset
        return [[float(center[0]), float(center[1])]] * 4
    bbox_min = np.array(obj.bbox_min, dtype=float)
    bbox_max = np.array(obj.bbox_max, dtype=float)
    z = float((bbox_min[2] + bbox_max[2]) / 2.0) if len(bbox_min) >= 3 else 0.0
    corners_local = np.array(
        [
            [bbox_min[0], bbox_min[1], z],
            [bbox_max[0], bbox_min[1], z],
            [bbox_max[0], bbox_max[1], z],
            [bbox_min[0], bbox_max[1], z],
        ]
    )
    corners_world = np.array([obj.transform @ corner for corner in corners_local])
    corners_world = corners_world + offset
    return corners_world[:, :2].tolist()


def _interaction_faces(
    center: np.ndarray,
    size: np.ndarray,
    yaw_deg: float,
    category: str,
    functional_hints: dict[str, Any],
) -> list[dict[str, Any]]:
    cx, cy, cz = [float(value) for value in center[:3]]
    sx, sy, sz = [max(float(value), 0.0) for value in size[:3]]
    z_min = cz - sz / 2.0
    z_max = cz + sz / 2.0
    yaw = math.radians(yaw_deg)
    fx, fy = _front_vector_from_hint(yaw, functional_hints)
    px, py = -fy, fx

    def face(
        name: str,
        dx: float,
        dy: float,
        affordances: list[str],
        z: float,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "center": [cx + dx, cy + dy, z],
            "normal_xy": _normalize_xy(dx, dy),
            "affordances": affordances,
        }

    support_z = z_max
    seat_z = min(max(z_min + sz * 0.45, 0.35), 0.65)
    open_z = min(max((z_min + z_max) * 0.5, 0.7), 1.4)
    affordances = {
        str(item).strip()
        for item in (
            functional_hints.get("functional_categories")
            or functional_hints.get("candidate_affordances")
            or []
        )
        if str(item).strip()
    }

    faces: list[dict[str, Any]] = []
    if (
        category
        in {
            "chair",
            "office_chair",
            "dining_chair",
            "stool",
            "armchair",
            "sofa",
            "loveseat",
            "bench",
        }
        or "sittable" in affordances
    ):
        faces.append(face("front", fx * sy / 2.0, fy * sy / 2.0, ["sittable"], seat_z))
    elif category == "bed" or "sleepable" in affordances:
        faces.extend(
            [
                face("left", px * sx / 2.0, py * sx / 2.0, ["sleepable"], support_z),
                face("right", -px * sx / 2.0, -py * sx / 2.0, ["sleepable"], support_z),
                face("front", fx * sy / 2.0, fy * sy / 2.0, ["sleepable"], support_z),
            ]
        )
    elif (
        category in {"cabinet", "dresser", "wardrobe", "drawer"}
        or "openable" in affordances
    ):
        faces.append(face("front", fx * sy / 2.0, fy * sy / 2.0, ["openable"], open_z))
    elif "graspable" in affordances and not {"supportable", "sittable"} & affordances:
        faces.append(face("front", fx * sy / 2.0, fy * sy / 2.0, ["graspable"], cz))
    else:
        face_affordances = sorted(affordances) or ["supportable"]
        faces.append(
            face("front", fx * sy / 2.0, fy * sy / 2.0, face_affordances, support_z)
        )
        if "supportable" in affordances:
            faces.append(
                face("left", px * sx / 2.0, py * sx / 2.0, face_affordances, support_z)
            )
            faces.append(
                face(
                    "right", -px * sx / 2.0, -py * sx / 2.0, face_affordances, support_z
                )
            )
    return faces


def _normalize_xy(dx: float, dy: float) -> list[float]:
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return [0.0, 0.0]
    return [dx / norm, dy / norm]


def _front_vector_from_hint(
    yaw_rad: float, functional_hints: dict[str, Any]
) -> tuple[float, float]:
    # SceneSmith canonical convention uses +Y as front at yaw=0.
    fx, fy = -math.sin(yaw_rad), math.cos(yaw_rad)
    front_hint = str(functional_hints.get("front_hint") or "").strip().lower()
    if front_hint == "back":
        return -fx, -fy
    if front_hint == "left":
        return -fy, fx
    if front_hint == "right":
        return fy, -fx
    return fx, fy


def _semantic_yaw_deg(obj: SceneObject, *, scene: RoomScene | None = None) -> float:
    # 2026-07-17 修改原因：PlacementInfo.rotation_2d 是父支撑面的局部角度，
    # 不能直接当成 world yaw；桌面自身经常带有 180° 朝向，直接使用会把
    # monitor 等 surface manipuland 的 front 判反。合并父 surface yaw 后再供
    # SceneBenchmark 规则使用，同时继续避免物理投影带来的微小姿态噪声。
    placement_info = getattr(obj, "placement_info", None)
    if obj.object_type == ObjectType.MANIPULAND and placement_info is not None:
        local_yaw = float(placement_info.rotation_2d)
        world_yaw = _parent_surface_world_yaw(
            obj, scene=scene, local_yaw=local_yaw
        )
        if world_yaw is not None:
            return math.degrees(world_yaw)
        # 2026-07-17 修改原因：父 surface 可能来自被裁剪的 scene 或铰接 link，
        # 找不到 metadata 时不能把 surface-local 角度静默当成 world 角度；退回
        # 当前对象实际世界姿态，避免泛化场景中再次固定误判为 0°。
        return math.degrees(RollPitchYaw(obj.transform.rotation()).yaw_angle())
    return math.degrees(RollPitchYaw(obj.transform.rotation()).yaw_angle())


def _parent_surface_world_yaw(
    obj: SceneObject, *, scene: RoomScene | None, local_yaw: float
) -> float | None:
    """Return composed world yaw for a manipuland's parent support surface."""
    placement_info = getattr(obj, "placement_info", None)
    if placement_info is None or scene is None:
        return None

    parent_surface_id = str(placement_info.parent_surface_id)
    for owner in getattr(scene, "objects", {}).values():
        for surface in getattr(owner, "support_surfaces", []):
            if str(surface.surface_id) != parent_surface_id:
                continue
            local_rotation = RollPitchYaw(0.0, 0.0, float(local_yaw)).ToRotationMatrix()
            world_rotation = surface.transform.rotation() @ local_rotation
            return float(RollPitchYaw(world_rotation).yaw_angle())
    return None


def _front_hint_from_access_direction(raw: Any, *, yaw_deg: float) -> str | None:
    # 2026-07-10 修改原因：23f21a8 将 SceneBenchmark front 轴统一为 yaw=0
    # 指向 +Y；访问方向推导也必须使用同一基准，否则交互面会旋转 90 度。
    direction = _access_direction_xy(raw)
    if direction is None:
        return None
    dx, dy = direction
    yaw = math.radians(yaw_deg)
    fx, fy = -math.sin(yaw), math.cos(yaw)
    candidates = {
        "front": (fx, fy),
        "back": (-fx, -fy),
        "left": (-fy, fx),
        "right": (fy, -fx),
    }
    best_face, best_dot = max(
        ((face, dx * axis[0] + dy * axis[1]) for face, axis in candidates.items()),
        key=lambda item: item[1],
    )
    return best_face if best_dot >= 0.5 else None


def _access_direction_xy(raw: Any) -> tuple[float, float] | None:
    if isinstance(raw, dict):
        if "primary" in raw:
            return _access_direction_xy(raw.get("primary"))
        values = [raw.get("x"), raw.get("y")]
    elif (
        isinstance(raw, list)
        and raw
        and all(isinstance(item, (list, tuple, dict)) for item in raw)
    ):
        return _access_direction_xy(raw[0])
    else:
        values = raw
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        return None
    try:
        dx, dy = float(values[0]), float(values[1])
    except (TypeError, ValueError):
        return None
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return None
    return dx / norm, dy / norm


def _support_surface_to_region(
    surface: SupportSurface, offset: np.ndarray
) -> dict[str, Any]:
    bbox_min = surface.bounding_box_min
    bbox_max = surface.bounding_box_max
    corners_local = np.array(
        [
            [bbox_min[0], bbox_min[1], 0.0],
            [bbox_max[0], bbox_min[1], 0.0],
            [bbox_max[0], bbox_max[1], 0.0],
            [bbox_min[0], bbox_max[1], 0.0],
        ]
    )
    corners_world = np.array([surface.transform @ corner for corner in corners_local])
    corners_world = corners_world + offset
    z_world = float((surface.transform @ np.array([0.0, 0.0, 0.0]))[2] + offset[2])
    return {
        "region_id": str(surface.surface_id),
        "support_kind": "top_surface",
        "height_world_z": z_world,
        "clearance_above_m": None,
        "access_type": "top",
        "area_m2": surface.area,
        "polygon_world_xy": corners_world[:, :2].tolist(),
        "source": "scenesmith_support_surface",
        "bbox_local": {
            "min": surface.bounding_box_min.tolist(),
            "max": surface.bounding_box_max.tolist(),
        },
    }


def _metadata_support_regions(
    obj: SceneObject, offset: np.ndarray
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for item in _metadata_support_region_items(obj):
        if not isinstance(item, dict):
            continue
        region = dict(item)
        if "polygon_world_xy" in region:
            region["polygon_world_xy"] = _offset_xy_polygon(
                region.get("polygon_world_xy"), offset
            )
        regions.append(region)
    return regions


def _metadata_support_region_items(obj: SceneObject) -> list[Any]:
    raw = obj.metadata.get("support_regions")
    if raw is None:
        raw = obj.metadata.get("support_region")
    nested_hints = obj.metadata.get("functional_hints")
    if raw is None and isinstance(nested_hints, dict):
        raw = nested_hints.get("support_regions")
        if raw is None:
            raw = nested_hints.get("support_region")
    if raw is None:
        return []
    return raw if isinstance(raw, list) else [raw]


def _support_region_summary(regions: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = sorted(
        {
            str(region.get("support_kind") or "")
            for region in regions
            if region.get("support_kind")
        }
    )
    sources = sorted(
        {str(region.get("source") or "") for region in regions if region.get("source")}
    )
    return {
        "count": len(regions),
        "support_kinds": kinds,
        "source": sources[0] if len(sources) == 1 else ("mixed" if sources else None),
    }


def _offset_xy_polygon(raw: Any, offset: np.ndarray) -> Any:
    if not isinstance(raw, list):
        return raw
    points: list[Any] = []
    for point in raw:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                points.append(
                    [
                        float(point[0]) + float(offset[0]),
                        float(point[1]) + float(offset[1]),
                    ]
                )
                continue
            except Exception:
                pass
        points.append(point)
    return points


def _category_for_object(obj: SceneObject) -> str:
    for key in ("category_norm", "category", "asset_category"):
        raw = obj.metadata.get(key)
        if raw:
            return _canonical_category(str(raw))
    return _canonical_category(f"{obj.object_id} {obj.name} {obj.description}")


def _functional_hints(
    obj: SceneObject, category: str, *, yaw_deg: float = 0.0
) -> dict[str, Any]:
    text = " ".join(
        [
            obj.name.lower(),
            obj.description.lower(),
            " ".join(str(v).lower() for v in obj.metadata.values()),
        ]
    )
    categories: set[str] = set()
    if _contains_any(text, ("chair", "stool", "sofa", "couch", "bench", "seat")):
        categories.add("sittable")
    if _contains_any(text, ("bed", "mattress")):
        categories.add("sleepable")
    if _contains_any(
        text, ("table", "desk", "shelf", "nightstand", "counter", "cabinet")
    ):
        categories.add("supportable")
    if _contains_any(text, ("cabinet", "drawer", "wardrobe", "door")):
        categories.add("openable")
    if _contains_any(text, ("lamp", "light", "switch")):
        categories.add("toggleable")
    if obj.object_type in {ObjectType.MANIPULAND, ObjectType.THIN_COVERING}:
        categories.add("graspable")
    if category in SUPPORTS or category in WORK_SURFACE_EXPOSURE_CATEGORIES:
        categories.add("supportable")
    if category in {"cabinet", "dresser", "drawer", "wardrobe"}:
        categories.add("openable")
    categories.update(HEURISTIC_AFFORDANCE_MAP.get(category, set()))

    metadata_hints = _metadata_functional_hints(obj)
    explicit = metadata_hints.get("functional_categories")
    if isinstance(explicit, list):
        categories.update(str(item).strip().lower() for item in explicit)
    candidate_affordances = metadata_hints.get("candidate_affordances")
    if isinstance(candidate_affordances, list):
        categories.update(
            str(item).strip().lower()
            for item in candidate_affordances
            if str(item).strip()
        )
    annotation_affordances = metadata_hints.get("affordances")
    if isinstance(annotation_affordances, list):
        categories.update(
            str(item).strip().lower()
            for item in annotation_affordances
            if str(item).strip()
        )
    if _asset_annotation_is_not_functional(metadata_hints):
        categories.clear()
    anchor_type = metadata_hints.get("anchor_type") or _anchor_type(categories)

    hints: dict[str, Any] = dict(metadata_hints)
    hints.update(
        {
            "functional_categories": sorted(categories),
            "candidate_affordances": sorted(categories),
            "anchor_type": str(anchor_type) if anchor_type is not None else None,
            "category_group": str(
                metadata_hints.get("category_group")
                or _category_group(obj, category, categories)
            ),
            "category_keywords": list(
                metadata_hints.get("category_keywords") or _category_keywords(category)
            ),
            "scene_object_type": str(
                metadata_hints.get("scene_object_type") or _scene_object_type(obj)
            ),
        }
    )
    if (
        not str(hints.get("front_hint") or "").strip()
        and str(hints.get("front_face") or "").strip()
    ):
        hints["front_hint"] = str(hints["front_face"]).strip()
    if not str(hints.get("front_hint") or "").strip():
        front_hint = _front_hint_from_access_direction(
            hints.get("access_direction") or hints.get("access_directions"),
            yaw_deg=yaw_deg,
        )
        if front_hint:
            hints["front_hint"] = front_hint
    if not str(hints.get("front_hint") or "").strip():
        front_hint = _front_hint(category, categories)
        if front_hint:
            hints["front_hint"] = front_hint
    if not hints.get("target_relation"):
        target_relation = _target_relation(category, categories)
        if target_relation:
            hints["target_relation"] = target_relation
    if not str(hints.get("mobility_class") or "").strip():
        hints["mobility_class"] = _mobility_class(obj, category)
    if not str(hints.get("accessibility_policy") or "").strip():
        hints["accessibility_policy"] = _accessibility_policy(
            obj, category, categories, str(hints.get("mobility_class") or "")
        )
    elif obj.object_type == ObjectType.MANIPULAND and categories:
        hints["accessibility_policy"] = "required"
    if not hints.get("access_sides"):
        hints["access_sides"] = _access_sides(
            category, categories, hints.get("front_hint")
        )
    if not isinstance(hints.get("metric_relevance"), dict):
        hints["metric_relevance"] = _metric_relevance(
            categories, _as_string_list(hints.get("explicit_target_relation"))
        )
    return hints


def _anchor_type(categories: set[str]) -> str | None:
    if "sittable" in categories:
        return "seat_surface"
    if "supportable" in categories:
        return "top_surface"
    if "openable" in categories:
        return "front_access"
    if "graspable" in categories:
        return "grasp_region"
    return None


def _front_hint(category: str, categories: set[str]) -> str | None:
    if "supportable" in categories:
        return "top"
    if category in {"mug", "cup"}:
        return "side"
    if "openable" in categories or "sittable" in categories:
        return "front"
    if "graspable" in categories:
        return "reachable_side"
    return None


def _target_relation(category: str, categories: set[str]) -> list[str]:
    if "sittable" in categories:
        if "sofa" in category or "loveseat" in category:
            return ["coffee_table", "tv_stand"]
        return ["desk", "table"]
    if "graspable" in categories:
        return ["table", "desk", "nightstand", "coffee_table", "shelf", "cabinet"]
    if "openable" in categories:
        return ["wall", "clear_space"]
    if "supportable" in categories:
        return ["graspable_object"]
    return []


def _mobility_class(obj: SceneObject, category: str) -> str:
    if obj.object_type in {ObjectType.WALL_MOUNTED, ObjectType.CEILING_MOUNTED}:
        return "mounted"
    if category in {
        "bar_stool",
        "bean_bag",
        "beanbag_chair",
        "chair",
        "dining_chair",
        "office_chair",
        "stool",
    }:
        return "movable"
    if category in {
        "armchair",
        "bench",
        "coffee_table",
        "desk",
        "dining_table",
        "loveseat",
        "nightstand",
        "side_table",
        "sofa",
        "table",
        "tv_stand",
    }:
        return "semi_movable"
    if category in {
        "bed",
        "bookshelf",
        "cabinet",
        "dresser",
        "refrigerator",
        "shelf",
        "wardrobe",
        "wine_cabinet",
    }:
        return "fixed"
    return "unknown"


def _accessibility_policy(
    obj: SceneObject, category: str, categories: set[str], mobility_class: str
) -> str:
    if obj.object_type in {ObjectType.WALL_MOUNTED, ObjectType.CEILING_MOUNTED}:
        return "ignored"
    if obj.object_type == ObjectType.MANIPULAND:
        return "required" if categories else "ignored"
    if mobility_class == "movable" and "sittable" in categories:
        return "optional"
    if category in {"book", "bottle", "bowl", "cup", "mug", "plate", "remote"}:
        return "ignored"
    return "required" if categories else "ignored"


def _access_sides(category: str, categories: set[str], front_hint: Any) -> list[str]:
    face = str(front_hint or "").strip().lower()
    if face not in {"front", "back", "left", "right", "top", "bottom"}:
        face = "front"
    sides: list[str] = []
    if "openable" in categories or "sittable" in categories:
        sides.append(face)
    if "supportable" in categories:
        sides.append("top")
    if "sleepable" in categories:
        sides.extend(["left", "right", face])
    if category in {"bed", "bookshelf", "cabinet", "dresser", "wardrobe"}:
        sides.append("front")
    return list(dict.fromkeys(sides))


def _metric_relevance(
    categories: set[str], explicit_target_relations: list[str]
) -> dict[str, float]:
    values = {
        "interaction_clearance": 0.0,
        "spatial_accessibility": 0.0,
        "functional_dependency": 0.0,
    }
    priority = {
        "interaction_clearance": ["openable", "sittable", "graspable", "supportable"],
        "spatial_accessibility": ["openable", "sittable", "graspable", "supportable"],
    }
    for metric, preferred in priority.items():
        overlap = [item for item in preferred if item in categories]
        if overlap:
            values[metric] = 1.0 if overlap[0] == preferred[0] else 0.8
    if "graspable" in categories and explicit_target_relations:
        values["spatial_accessibility"] = max(values["spatial_accessibility"], 0.7)
    if explicit_target_relations:
        values["functional_dependency"] = 1.0
    return values


def _as_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    return [str(value).strip() for value in values if str(value).strip()]


def _metadata_functional_hints(obj: SceneObject) -> dict[str, Any]:
    hints: dict[str, Any] = _hssd_annotation_functional_hints(obj)
    nested = obj.metadata.get("functional_hints")
    if isinstance(nested, dict):
        hints.update(nested)

    passthrough_keys = (
        "functional_categories",
        "candidate_affordances",
        "affordances",
        "category_group",
        "category_keywords",
        "access_type",
        "affordance_confidence",
        "affordance_source",
        "asset_local_front_axis",
        "interaction_surface_map",
        "interaction_height_m",
        "placement_class",
        "benchmark_relevance",
        "classification_source",
        "canonical_orientation_is_semantic_front",
        "front_face",
        "front_confidence",
        "front_hint",
        "scene_object_type",
        "access_direction",
        "access_directions",
        "anchor_type",
        "asset_annotation_source",
        "access_sides",
        "accessibility_policy",
        "attachment_dependencies",
        "classification_confidence",
        "classification_reason",
        "functional_dependency",
        "functional_dependencies",
        "explicit_target_relation",
        "low_confidence_candidates",
        "metric_relevance",
        "mobility_class",
        "operation_space",
        "operation_spaces",
        "orientation_dependencies",
        "part_of_furniture",
        "part_of_forniture",
        "target_relation",
    )
    for key in passthrough_keys:
        if key in obj.metadata:
            hints[key] = obj.metadata[key]
    return hints


_HSSD_METADATA_ID_KEYS = ("hssd_mesh_id", "asset_id", "object_id")


def _hssd_annotation_functional_hints(obj: SceneObject) -> dict[str, Any]:
    asset_id = _hssd_asset_id_from_metadata(obj.metadata)
    if not asset_id:
        return {}
    try:
        record = get_hssd_asset_annotations(asset_id)
    except Exception:
        return {}
    if not isinstance(record, dict):
        return {}
    hints = record.get("scenebenchmark_functional_hints")
    if not isinstance(hints, dict):
        hints = (record.get("scenebenchmark_fd_sa") or {}).get("functional_hints")
    if not isinstance(hints, dict):
        return {}
    out = dict(hints)
    out.setdefault("asset_annotation_source", "hssd_annotations")
    out.setdefault("classification_source", "hssd_annotations")
    out["hssd_annotation_asset_id"] = normalize_hssd_id_for_adapter(asset_id)
    return out


def _hssd_asset_id_from_metadata(metadata: Any) -> str | None:
    if not isinstance(metadata, dict):
        return None
    for key in _HSSD_METADATA_ID_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize_hssd_id_for_adapter(value: str) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("hssd:"):
        text = text.split(":", 1)[1]
    return text


def _metadata_dependency_relations(
    obj: SceneObject, surface_owner_ids: dict[str, str]
) -> list[dict[str, Any]]:
    raw = obj.metadata.get("functional_dependencies")
    if raw is None:
        raw = obj.metadata.get("functional_dependency")
    nested_hints = obj.metadata.get("functional_hints")
    if raw is None and isinstance(nested_hints, dict):
        raw = nested_hints.get("functional_dependencies")
        if raw is None:
            raw = nested_hints.get("functional_dependency")
    items = raw if isinstance(raw, list) else [raw]
    relations: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        relation_type = str(
            item.get("relation_type") or item.get("type") or "functional_dependency"
        )
        target_ids = _dependency_target_ids(item)
        target_surface_id = item.get("target_surface_id") or item.get(
            "parent_surface_id"
        )
        if target_surface_id and not target_ids:
            owner_id = surface_owner_ids.get(str(target_surface_id))
            if owner_id:
                target_ids = [owner_id]
        relation: dict[str, Any] = {
            "relation_type": relation_type,
            "subject": str(
                item.get("subject") or item.get("subject_id") or obj.object_id
            ),
            "subject_id": str(
                item.get("subject_id") or item.get("subject") or obj.object_id
            ),
            "target_ids": target_ids,
            "annotation_source": str(item.get("source") or "metadata"),
        }
        if target_ids:
            relation["object"] = target_ids[0]
        if target_surface_id:
            relation["target_surface_id"] = str(target_surface_id)
        for key in ("confidence", "reason", "evidence", "scoring_tier"):
            if key in item:
                relation[key] = item[key]
        relations.append(relation)
    return relations


def _dependency_target_ids(item: dict[str, Any]) -> list[str]:
    raw = (
        item.get("target_ids")
        or item.get("targets")
        or item.get("target_id")
        or item.get("target")
        or item.get("object")
        or item.get("object_id")
        or item.get("parent_object_id")
    )
    values = raw if isinstance(raw, list) else [raw]
    target_ids: list[str] = []
    for value in values:
        target_id = str(value or "")
        if target_id and target_id not in target_ids:
            target_ids.append(target_id)
    return target_ids


def _scene_object_type(obj: SceneObject) -> str:
    if obj.object_type == ObjectType.WALL_MOUNTED:
        return "wall_mounted"
    if obj.object_type == ObjectType.CEILING_MOUNTED:
        return "ceiling_mounted"
    if obj.object_type == ObjectType.MANIPULAND:
        return "manipuland"
    if obj.object_type == ObjectType.THIN_COVERING:
        return "manipuland"
    if obj.object_type == ObjectType.FURNITURE:
        return "furniture"
    return "unknown"


def _category_group(obj: SceneObject, category: str, categories: set[str]) -> str:
    matched_group = _category_group_for_key(category)
    if matched_group:
        return matched_group
    if obj.object_type == ObjectType.WALL_MOUNTED:
        return "decor"
    if obj.object_type == ObjectType.CEILING_MOUNTED:
        return "ceiling"
    if obj.object_type in {ObjectType.MANIPULAND, ObjectType.THIN_COVERING}:
        return "small_object"
    if category in {
        "chair",
        "office_chair",
        "dining_chair",
        "stool",
        "bar_stool",
        "armchair",
        "sofa",
        "loveseat",
        "bench",
        "bean_bag",
        "beanbag_chair",
    }:
        return "seating"
    if category in {"bed"}:
        return "sleeping"
    if category in {"television", "tv", "monitor", "screen"}:
        return "media"
    if category in {
        "cabinet",
        "dresser",
        "drawer",
        "microwave",
        "range_oven",
        "refrigerator",
        "wall_cabinet",
        "wardrobe",
        "wine_cabinet",
    }:
        return "storage"
    if category in {
        "bookshelf",
        "buffet",
        "console",
        "credenza",
        "media_console",
        "shelf",
        "sideboard",
        "tv_stand",
        "wall_shelf",
    }:
        return "storage_surface"
    if "supportable" in categories:
        return "work_surface"
    return obj.object_type.value


def _category_group_for_key(category: str | None) -> str | None:
    if not category:
        return None
    if category in CATEGORY_GROUPS:
        return CATEGORY_GROUPS[category]
    for key, value in CATEGORY_GROUPS.items():
        if _category_matches_key(category, key):
            return value
    return None


def _category_matches_key(category: str | None, key: str | None) -> bool:
    category_tokens = _tokenize_category_name(category)
    key_tokens = _tokenize_category_name(key)
    if not category_tokens or not key_tokens:
        return False
    if category_tokens == key_tokens:
        return True
    window = len(key_tokens)
    for index in range(len(category_tokens) - window + 1):
        if category_tokens[index : index + window] == key_tokens:
            return True
    return False


def _category_keywords(category: str | None) -> list[str]:
    if not category:
        return []
    if category in CATEGORY_KEYWORDS:
        return list(CATEGORY_KEYWORDS[category])
    for key, values in CATEGORY_KEYWORDS.items():
        if _category_matches_key(category, key):
            return list(values)
    return [category.replace("_", " ")]


def _tokenize_category_name(value: str | None) -> list[str]:
    if not value:
        return []
    return [token for token in str(value).strip().lower().split("_") if token]


def _asset_annotation_is_not_functional(hints: dict[str, Any]) -> bool:
    source = str(hints.get("classification_source") or "").strip().lower()
    annotation_source = str(hints.get("asset_annotation_source") or "").strip()
    relevance = str(hints.get("benchmark_relevance") or "").strip().lower()
    return (
        (source == "asset_annotation" or bool(annotation_source))
        and bool(relevance)
        and relevance != "functional"
    )


def _nav_obstacle_class(record: dict[str, Any]) -> str:
    hints = record.get("functional_hints") or {}
    group = str(hints.get("category_group") or "")
    category = str(record.get("category_norm") or record.get("category") or "")
    if group == "decor":
        return "decor_ignore"
    bbox = record.get("bbox_world") or {}
    bmin = bbox.get("min") or []
    bmax = bbox.get("max") or []
    size = bbox.get("size") or []
    if len(bmin) < 3 or len(bmax) < 3:
        return "small_ignore"
    z_min, z_max = float(bmin[2]), float(bmax[2])
    area = 0.0
    if len(size) >= 2:
        area = max(float(size[0]), 0.0) * max(float(size[1]), 0.0)
    height = z_max - z_min
    if any(token in category for token in ("rug", "mat", "carpet")):
        return "low_ignore"
    if z_min > 0.25:
        return "mounted_ignore"
    if area <= 0.12 and height <= 0.5:
        return "small_ignore"
    if z_max < 0.45:
        return "low_ignore"
    return "blocking"


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _strip_room_prefix_words(text: str) -> str:
    text = text.strip()
    while True:
        trimmed = False
        for room_keyword in sorted(ROOM_KEYWORD_SET, key=len, reverse=True):
            room_prefix = room_keyword.replace("_", " ")
            if text.startswith(room_prefix + " "):
                text = text[len(room_prefix) + 1 :].strip()
                trimmed = True
                break
        if not trimmed:
            return text


def _strip_trailing_instance_tokens(tokens: list[str]) -> list[str]:
    trimmed = list(tokens)
    while trimmed:
        tail = trimmed[-1]
        if re.fullmatch(r"(?:[a-z]\d*|f\d+|\d+)", tail):
            trimmed.pop()
            continue
        break
    return trimmed


def _category_token_windows(tokens: list[str]) -> list[str]:
    windows: list[str] = []
    max_width = min(4, len(tokens))
    for width in range(max_width, 0, -1):
        for start in range(0, len(tokens) - width + 1):
            windows.append("_".join(tokens[start : start + width]))
    return windows


def _canonical_category(raw: str) -> str:
    compact = _normalize_category_text(raw)
    if not compact:
        return "unknown"
    compact = _strip_room_prefix_words(compact)
    if compact in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[compact]
    compact_no_space = compact.replace(" ", "")
    if compact_no_space in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[compact_no_space]

    tokens_list = _strip_trailing_instance_tokens(
        [token for token in compact.replace(" ", "_").split("_") if token]
    )
    tokens = set(tokens_list)
    for window in _category_token_windows(tokens_list):
        alias = CATEGORY_ALIASES.get(window) or CATEGORY_ALIASES.get(
            window.replace("_", " ")
        )
        if alias:
            return alias
        if window in KNOWN_CATEGORY_TOKENS:
            return window

    for alias, category in sorted(
        CATEGORY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        phrase = alias.replace("_", " ").strip()
        phrase_no_space = phrase.replace(" ", "")
        if (
            (" " in phrase and phrase in compact)
            or phrase in tokens
            or (
                len(phrase_no_space) > 3
                and phrase_no_space != phrase
                and phrase_no_space in compact_no_space
            )
        ):
            return category

    filtered_tokens = [
        token
        for token in tokens_list
        if token not in GENERIC_CATEGORY_WORDS and not token.isdigit()
    ]
    tokens = set(filtered_tokens)
    token_map = {
        "floor": "floor",
        "wall": "wall",
        "window": "window",
        "door": "door",
        "chair": "chair",
        "stool": "stool",
        "bench": "bench",
        "sofa": "sofa",
        "couch": "sofa",
        "loveseat": "loveseat",
        "desk": "desk",
        "table": "table",
        "counter": "counter",
        "island": "island",
        "sideboard": "sideboard",
        "buffet": "buffet",
        "credenza": "credenza",
        "console": "console",
        "cabinet": "cabinet",
        "dresser": "dresser",
        "wardrobe": "wardrobe",
        "shelf": "shelf",
        "bed": "bed",
        "lamp": "lamp",
        "light": "lamp",
        "monitor": "monitor",
        "notebook": "book",
        "notebook_computer": "laptop",
        "keyboard": "keyboard",
        "mouse": "mouse",
        "remote": "remote",
        "screen": "monitor",
        "display": "monitor",
        "tablet": "tablet_computer",
        "laptop": "laptop",
        "mug": "mug",
        "cup": "cup",
        "bottle": "bottle",
        "bowl": "bowl",
        "plate": "plate",
        "tray": "tray",
        "book": "book",
        "plant": "plant",
        "vase": "vase",
        "rug": "rug",
    }
    for token, category in token_map.items():
        if token in tokens:
            return category
    for window in _category_token_windows(filtered_tokens):
        if window in KNOWN_CATEGORY_TOKENS:
            return window
    if not filtered_tokens:
        return "unknown"
    return "_".join(filtered_tokens[: min(3, len(filtered_tokens))])


def _normalize_category_text(raw: str) -> str:
    cleaned = raw.strip().lower()
    cleaned = re.sub(r"factory$", "", cleaned)
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _object_function_profile(obj: SceneObject, category: str) -> dict[str, bool]:
    hints = _functional_hints(obj, category)
    group = str(hints["category_group"]).strip().lower()
    categories = {
        str(item).strip().lower()
        for item in (
            hints.get("functional_categories")
            or hints.get("candidate_affordances")
            or []
        )
        if str(item).strip()
    }
    support_regions = [
        region
        for region in _metadata_support_region_items(obj)
        if isinstance(region, dict)
    ]
    region_kinds = {
        str(region.get("support_kind") or "").strip().lower()
        for region in support_regions
    }
    if obj.support_surfaces:
        region_kinds.add("top_surface")
    access_type = hints.get("access_type") or {}
    primary_access = (
        str(access_type.get("primary") or "").strip().lower()
        if isinstance(access_type, dict)
        else str(access_type or "").strip().lower()
    )
    if not primary_access:
        access_types = {
            str(region.get("access_type") or "").strip().lower()
            for region in support_regions
            if isinstance(region, dict)
        }
        primary_access = next((value for value in access_types if value), "")
    scene_object_type = str(hints.get("scene_object_type") or "").strip().lower()
    text_tokens = {
        token
        for token in " ".join(
            [
                str(obj.object_id),
                category,
                obj.name,
                obj.description,
            ]
        )
        .replace("-", "_")
        .replace(" ", "_")
        .split("_")
        if token
    }

    can_support_top = bool(
        categories & SUPPORT_TOP_SURFACE_AFFORDANCES
        or "top_surface" in region_kinds
        or group in WORK_SURFACE_GROUPS
        or category in SUPPORTS
        or category in WORK_SURFACE_EXPOSURE_CATEGORIES
    )
    has_internal_shelf = bool(
        categories & SUPPORT_INTERNAL_AFFORDANCES
        or any(
            any(term in kind for term in ("shelf", "drawer", "cabinet", "storage"))
            for kind in region_kinds
        )
        or primary_access
        in {
            "front_open",
            "front-open",
            "open_shelf",
            "openable_storage",
            "internal_storage",
        }
        or group in SUPPORT_CATEGORY_GROUPS
    )
    is_media_target = bool(
        category in MEDIA_TARGET_CATEGORIES
        and not (text_tokens & {"remote", "controller"})
    )
    inferred = {
        "can_support_top": bool(
            can_support_top
            or category in WORK_SURFACES
            or category in NIGHTSTANDS
            or group in WORK_SURFACE_GROUPS
        ),
        "has_internal_shelf": has_internal_shelf,
        "is_small_placeable": bool(
            obj.object_type in {ObjectType.MANIPULAND, ObjectType.THIN_COVERING}
            or (
                category in SUPPORTED_SMALL
                and scene_object_type not in {"wall_mounted", "ceiling_mounted"}
            )
        ),
        "is_seating": category in SEATING or group == "seating",
        "is_work_surface": category in WORK_SURFACE_EXPOSURE_CATEGORIES
        or group in WORK_SURFACE_GROUPS,
        "is_media_target": is_media_target or group == "media",
        "is_bedside_surface": category in NIGHTSTANDS or category == "end_table",
        "is_sleeping_surface": category in BEDS or group == "sleeping",
    }
    explicit = obj.metadata.get("object_function_profile")
    if not isinstance(explicit, dict):
        return inferred
    merged = dict(inferred)
    for key in tuple(merged):
        if key in explicit:
            merged[key] = bool(explicit[key])
    return merged
