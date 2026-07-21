"""Deterministic functional-zone guidance and validation for furniture layouts.

The furniture VLM is good at selecting related objects but can lose the spatial
grammar that makes a room usable.  This module covers two high-value layouts
whose relations are explicit in the task: a living-room conversation group and
a front-facing classroom.  It intentionally produces compact guidance and
geometry-only reports so the same contract can drive both prompting and repair.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


WALLS = ("north", "south", "east", "west")
INWARD_NORMALS: dict[str, np.ndarray] = {
    "north": np.asarray([0.0, -1.0]),
    "south": np.asarray([0.0, 1.0]),
    "east": np.asarray([-1.0, 0.0]),
    "west": np.asarray([1.0, 0.0]),
}


@dataclass(frozen=True)
class FunctionalLayoutReport:
    """Rule-based report for a room's functional furniture relationships."""

    layout_family: str
    anchor_wall: str | None
    score: float
    penalty: float
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_family": self.layout_family,
            "anchor_wall": self.anchor_wall,
            "score": float(self.score),
            "penalty": float(self.penalty),
            "issues": list(self.issues),
            "metrics": dict(self.metrics),
        }


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    try:
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
    except Exception:
        pass
    return getattr(cfg, key, default)


def _cfg_float(cfg: Any, key: str, default: float) -> float:
    try:
        return float(_cfg_get(cfg, key, default))
    except (TypeError, ValueError):
        return default


def _original_scene_text(scene: Any) -> str:
    original = getattr(
        scene,
        "scene_expert_original_description",
        getattr(scene, "text_description", ""),
    )
    return (
        f"{getattr(scene, 'room_type', '')} {original}"
        .lower()
        .replace("_", " ")
    )


def functional_layout_family(scene: Any) -> str | None:
    """Return the active functional layout without trusting injected memory."""
    text = _original_scene_text(scene)
    if "classroom" in text or "student desk" in text or "teacher's desk" in text:
        return "classroom"
    if "living room" in text or "sofa" in text or "couch" in text:
        return "living_room"
    return None


def _opening_value(opening: Any, key: str, default: str = "") -> str:
    raw = opening.get(key, default) if isinstance(opening, dict) else getattr(
        opening, key, default
    )
    raw = getattr(raw, "value", raw)
    return str(raw or default).lower()


def wall_opening_types(scene: Any) -> dict[str, list[str]]:
    result = {wall: [] for wall in WALLS}
    room_geometry = getattr(scene, "room_geometry", None)
    for opening in list(getattr(room_geometry, "openings", []) or []):
        wall = _opening_value(opening, "wall_direction")
        if wall in result:
            result[wall].append(_opening_value(opening, "opening_type", "opening"))
    return result


def choose_functional_anchor_wall(scene: Any, layout_family: str) -> str | None:
    """Choose a long, uninterrupted wall for a sofa back or classroom front."""
    room_geometry = getattr(scene, "room_geometry", None)
    if room_geometry is None:
        return None
    length = float(getattr(room_geometry, "length", 0.0) or 0.0)
    width = float(getattr(room_geometry, "width", 0.0) or 0.0)
    if length <= 0.0 or width <= 0.0:
        return None

    openings = wall_opening_types(scene)
    best_wall: str | None = None
    best_score = -float("inf")
    for wall in WALLS:
        wall_length = length if wall in ("north", "south") else width
        types = openings[wall]
        score = wall_length * (0.30 if layout_family == "classroom" else 0.15)
        if not types:
            score += 8.0
        score -= 12.0 * sum(t in ("door", "open") for t in types)
        score -= 7.0 * sum(t == "window" for t in types)
        if score > best_score:
            best_wall = wall
            best_score = score
    return best_wall


def _wall_summary(scene: Any) -> str:
    return ", ".join(
        f"{wall}={'/'.join(types) if types else 'solid'}"
        for wall, types in wall_opening_types(scene).items()
    )


def format_functional_layout_guidance(scene: Any, cfg: Any | None = None) -> str:
    """Return room-aware expert constraints for the first designer call."""
    if not bool(_cfg_get(cfg, "enabled", True)):
        return ""
    family = functional_layout_family(scene)
    if family is None:
        return ""
    wall = choose_functional_anchor_wall(scene, family)
    if wall is None:
        return ""
    wall_summary = _wall_summary(scene)

    if family == "living_room":
        return (
            "Living-room functional-zone plan:\n"
            f"- Use {wall}_wall as the sofa-back anchor because it is the best "
            "uninterrupted wall; rotate the sofa to face into the room.\n"
            "- Center the rug directly in front of the sofa as part of one "
            "conversation group, not as a separate zone elsewhere in the room.\n"
            "- Put one large floor plant near each sofa end. Do not line both "
            "plants up on the same side or between the sofa and rug.\n"
            "- Preserve the door approach and do not put the sofa under a window "
            "when a solid wall is available.\n"
            f"- Opening summary: {wall_summary}."
        )

    classroom_cfg = _cfg_get(cfg, "classroom", {})
    columns = max(1, int(_cfg_get(classroom_cfg, "preferred_columns", 3)))
    return (
        "Classroom functional-zone plan:\n"
        f"- Treat {wall}_wall as the front/chalkboard wall.\n"
        f"- Arrange student desk-chair pairs in a regular grid (prefer {columns} "
        "columns), with every student desk and chair facing the front wall.\n"
        "- Place exactly one chair behind each student desk at a usable sitting "
        "distance; preserve row aisles and consistent orientation.\n"
        "- Put the teacher desk centered ahead of every student row near the "
        "front wall, facing back toward the students. Keep the chalkboard zone "
        "behind the teacher desk clear for the wall-mounted stage.\n"
        f"- Opening summary: {wall_summary}."
    )


def _is_furniture(obj: Any) -> bool:
    object_type = getattr(obj, "object_type", "")
    value = getattr(object_type, "value", object_type)
    return str(value).lower() == "furniture" and not getattr(obj, "immutable", False)


def _categorized_objects(
    scene: Any, category_resolver: Callable[[str], str | None]
) -> dict[str, list[tuple[str, Any]]]:
    result: dict[str, list[tuple[str, Any]]] = {}
    for object_id, obj in getattr(scene, "objects", {}).items():
        if not _is_furniture(obj):
            continue
        category = category_resolver(
            f"{object_id} {getattr(obj, 'name', '')} {getattr(obj, 'description', '')}"
        )
        if category:
            result.setdefault(category, []).append((str(object_id), obj))
    return result


def _position(obj: Any) -> np.ndarray | None:
    try:
        value = np.asarray(obj.transform.translation(), dtype=float)
    except Exception:
        return None
    return value[:2] if value.size >= 2 and np.all(np.isfinite(value[:2])) else None


def _basis(obj: Any) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        rotation = np.asarray(obj.transform.rotation().matrix(), dtype=float)
        lateral = rotation[:2, 0]
        forward = rotation[:2, 1]
    except Exception:
        return None
    lateral_norm = float(np.linalg.norm(lateral))
    forward_norm = float(np.linalg.norm(forward))
    if lateral_norm <= 1e-6 or forward_norm <= 1e-6:
        return None
    return lateral / lateral_norm, forward / forward_norm


def _world_bounds(obj: Any) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        bounds = obj.compute_world_bounds()
    except Exception:
        return None
    if bounds is None:
        return None
    return np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)


def furnishable_room_bounds_xy(
    scene: Any,
) -> tuple[float, float, float, float] | None:
    """Return wall-inner-face bounds, not the floor plan's outer envelope."""
    room_geometry = getattr(scene, "room_geometry", None)
    if room_geometry is None:
        return None
    length = float(getattr(room_geometry, "length", 0.0) or 0.0)
    width = float(getattr(room_geometry, "width", 0.0) or 0.0)
    wall_thickness = max(
        0.0, float(getattr(room_geometry, "wall_thickness", 0.0) or 0.0)
    )
    half_length = length / 2.0 - wall_thickness
    half_width = width / 2.0 - wall_thickness
    if half_length <= 0.0 or half_width <= 0.0:
        return None
    return -half_length, -half_width, half_length, half_width


def _room_bounds(scene: Any) -> tuple[float, float, float, float] | None:
    return furnishable_room_bounds_xy(scene)


def _nearest_wall(scene: Any, obj: Any) -> tuple[str, float] | None:
    room_bounds = _room_bounds(scene)
    bounds = _world_bounds(obj)
    if room_bounds is None or bounds is None:
        return None
    min_x, min_y, max_x, max_y = room_bounds
    obj_min, obj_max = bounds
    distances = {
        "north": abs(max_y - float(obj_max[1])),
        "south": abs(float(obj_min[1]) - min_y),
        "east": abs(max_x - float(obj_max[0])),
        "west": abs(float(obj_min[0]) - min_x),
    }
    wall = min(distances, key=distances.get)
    return wall, distances[wall]


def _report(
    family: str,
    wall: str | None,
    issues: list[str],
    metrics: dict[str, float | int | str],
) -> FunctionalLayoutReport:
    penalty = min(0.85, 0.12 * len(issues))
    return FunctionalLayoutReport(
        layout_family=family,
        anchor_wall=wall,
        score=max(0.0, 1.0 - penalty),
        penalty=penalty,
        issues=issues,
        metrics=metrics,
    )


def _evaluate_living_room(
    scene: Any,
    categorized: dict[str, list[tuple[str, Any]]],
    cfg: Any,
) -> FunctionalLayoutReport:
    issues: list[str] = []
    metrics: dict[str, float | int | str] = {}
    anchor_wall = choose_functional_anchor_wall(scene, "living_room")
    sofas = categorized.get("sofa", [])
    rugs = categorized.get("rug", [])
    plants = categorized.get("plant", [])
    if not sofas:
        return _report("living_room", anchor_wall, issues, metrics)

    sofa_id, sofa = sofas[0]
    sofa_position = _position(sofa)
    sofa_basis = _basis(sofa)
    nearest = _nearest_wall(scene, sofa)
    if nearest is not None:
        nearest_wall, wall_gap = nearest
        metrics["sofa_wall"] = nearest_wall
        metrics["sofa_wall_gap_m"] = round(wall_gap, 3)
        max_gap = _cfg_float(cfg, "sofa_wall_max_gap_m", 0.30)
        if wall_gap > max_gap:
            issues.append(
                f"functional layout: {sofa_id} is {wall_gap:.2f}m from its nearest "
                f"wall, above sofa-back limit {max_gap:.2f}m"
            )
        if sofa_basis is not None:
            inward_dot = float(np.dot(sofa_basis[1], INWARD_NORMALS[nearest_wall]))
            metrics["sofa_inward_dot"] = round(inward_dot, 3)
            min_dot = _cfg_float(cfg, "sofa_inward_min_dot", 0.70)
            if inward_dot < min_dot:
                issues.append(
                    f"functional layout: {sofa_id} does not face inward from "
                    f"{nearest_wall}_wall (alignment={inward_dot:.2f})"
                )

    if sofa_position is None or sofa_basis is None:
        return _report("living_room", anchor_wall, issues, metrics)
    lateral, forward = sofa_basis

    if rugs:
        rug_id, rug = rugs[0]
        rug_position = _position(rug)
        if rug_position is not None:
            delta = rug_position - sofa_position
            front_distance = float(np.dot(delta, forward))
            lateral_offset = abs(float(np.dot(delta, lateral)))
            center_distance = float(np.linalg.norm(delta))
            metrics.update(
                {
                    "rug_front_distance_m": round(front_distance, 3),
                    "rug_lateral_offset_m": round(lateral_offset, 3),
                    "rug_center_distance_m": round(center_distance, 3),
                }
            )
            min_front = _cfg_float(cfg, "rug_min_front_distance_m", 0.35)
            max_distance = _cfg_float(cfg, "rug_max_center_distance_m", 2.20)
            max_lateral = _cfg_float(cfg, "rug_max_lateral_offset_m", 0.75)
            if (
                front_distance < min_front
                or center_distance > max_distance
                or lateral_offset > max_lateral
            ):
                issues.append(
                    f"functional layout: {rug_id} is not centered in front of "
                    f"{sofa_id} (front={front_distance:.2f}m, lateral="
                    f"{lateral_offset:.2f}m, distance={center_distance:.2f}m)"
                )

    if len(plants) >= 2:
        side_values: list[float] = []
        max_distance = _cfg_float(cfg, "plant_max_sofa_distance_m", 1.80)
        max_forward = _cfg_float(cfg, "plant_max_forward_offset_m", 1.00)
        for plant_id, plant in plants[:2]:
            plant_position = _position(plant)
            if plant_position is None:
                continue
            delta = plant_position - sofa_position
            side_values.append(float(np.dot(delta, lateral)))
            distance = float(np.linalg.norm(delta))
            forward_offset = abs(float(np.dot(delta, forward)))
            if distance > max_distance or forward_offset > max_forward:
                issues.append(
                    f"functional layout: {plant_id} is not near a sofa end "
                    f"(distance={distance:.2f}m, forward offset={forward_offset:.2f}m)"
                )
        min_side = _cfg_float(cfg, "plant_min_flank_offset_m", 0.20)
        if len(side_values) == 2 and not (
            min(side_values) < -min_side and max(side_values) > min_side
        ):
            issues.append(
                f"functional layout: the two plants are not flanking opposite ends "
                f"of {sofa_id}"
            )

    return _report("living_room", anchor_wall, issues, metrics)


def _evaluate_classroom(
    scene: Any,
    categorized: dict[str, list[tuple[str, Any]]],
    cfg: Any,
) -> FunctionalLayoutReport:
    issues: list[str] = []
    metrics: dict[str, float | int | str] = {}
    anchor_wall = choose_functional_anchor_wall(scene, "classroom")
    desks = categorized.get("student_desk", [])
    chairs = categorized.get("chair", [])
    teacher_desks = categorized.get("teacher_desk", [])
    if not desks:
        return _report("classroom", anchor_wall, issues, metrics)

    desk_data: list[tuple[str, Any, np.ndarray, np.ndarray]] = []
    for desk_id, desk in desks:
        position = _position(desk)
        basis = _basis(desk)
        if position is not None and basis is not None:
            desk_data.append((desk_id, desk, position, basis[1]))
    if not desk_data:
        return _report("classroom", anchor_wall, issues, metrics)

    mean_forward = np.sum([item[3] for item in desk_data], axis=0)
    norm = float(np.linalg.norm(mean_forward))
    if norm <= 1e-6:
        issues.append("functional layout: student desks do not share one orientation")
        return _report("classroom", anchor_wall, issues, metrics)
    mean_forward /= norm

    tolerance_deg = _cfg_float(cfg, "orientation_tolerance_degrees", 15.0)
    min_alignment = math.cos(math.radians(tolerance_deg))
    inconsistent = [
        desk_id
        for desk_id, _, _, forward in desk_data
        if float(np.dot(forward, mean_forward)) < min_alignment
    ]
    if inconsistent:
        issues.append(
            "functional layout: student desks have inconsistent orientation: "
            + ", ".join(inconsistent)
        )

    if anchor_wall is not None:
        front_alignment = float(np.dot(mean_forward, -INWARD_NORMALS[anchor_wall]))
        metrics["student_front_alignment"] = round(front_alignment, 3)
        if front_alignment < min_alignment:
            issues.append(
                f"functional layout: student desks do not face the selected "
                f"{anchor_wall}_wall classroom front"
            )

    unassigned = {
        chair_id: (chair, _position(chair), _basis(chair))
        for chair_id, chair in chairs
    }
    pair_distance = _cfg_float(cfg, "desk_chair_center_distance_m", 0.68)
    max_pair_error = _cfg_float(cfg, "desk_chair_pair_error_m", 0.48)
    unpaired: list[str] = []
    for desk_id, _, desk_position, desk_forward in desk_data:
        expected = desk_position - desk_forward * pair_distance
        candidates = [
            (
                float(np.linalg.norm(chair_position - expected)),
                chair_id,
                chair_basis,
            )
            for chair_id, (_, chair_position, chair_basis) in unassigned.items()
            if chair_position is not None
        ]
        if not candidates:
            unpaired.append(desk_id)
            continue
        error, chair_id, chair_basis = min(candidates, key=lambda item: item[0])
        if error > max_pair_error:
            unpaired.append(desk_id)
            continue
        if chair_basis is not None and float(
            np.dot(chair_basis[1], desk_forward)
        ) < min_alignment:
            unpaired.append(desk_id)
            continue
        unassigned.pop(chair_id, None)
    metrics["paired_student_desks"] = len(desk_data) - len(unpaired)
    if unpaired:
        issues.append(
            "functional layout: student desks lack a correctly aligned chair behind "
            "them: " + ", ".join(unpaired)
        )

    if teacher_desks:
        teacher_id, teacher = teacher_desks[0]
        teacher_position = _position(teacher)
        teacher_basis = _basis(teacher)
        if teacher_position is not None:
            teacher_front = float(np.dot(teacher_position, mean_forward))
            student_front = max(
                float(np.dot(position, mean_forward))
                for _, _, position, _ in desk_data
            )
            lead = teacher_front - student_front
            metrics["teacher_lead_distance_m"] = round(lead, 3)
            min_lead = _cfg_float(cfg, "teacher_front_min_lead_m", 0.55)
            if lead < min_lead:
                issues.append(
                    f"functional layout: {teacher_id} is not ahead of all student "
                    f"rows (lead={lead:.2f}m)"
                )
        if teacher_basis is not None:
            opposition = float(np.dot(teacher_basis[1], mean_forward))
            metrics["teacher_student_facing_dot"] = round(opposition, 3)
            if opposition > -min_alignment:
                issues.append(
                    f"functional layout: {teacher_id} does not face the students"
                )

    return _report("classroom", anchor_wall, issues, metrics)


def evaluate_functional_layout(
    scene: Any,
    category_resolver: Callable[[str], str | None],
    cfg: Any | None = None,
) -> FunctionalLayoutReport | None:
    """Evaluate high-value spatial relations for supported room families."""
    if not bool(_cfg_get(cfg, "enabled", True)):
        return None
    family = functional_layout_family(scene)
    if family is None:
        return None
    categorized = _categorized_objects(scene, category_resolver)
    family_cfg = _cfg_get(cfg, family, {})
    if family == "living_room":
        return _evaluate_living_room(scene, categorized, family_cfg)
    return _evaluate_classroom(scene, categorized, family_cfg)
