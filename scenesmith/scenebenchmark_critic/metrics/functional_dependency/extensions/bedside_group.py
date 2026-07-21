"""Bed-local topology and door-clearance checks for bedside furniture groups."""

from __future__ import annotations

import math
import re

from typing import Any

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
    bbox_gap_xy,
    front_vector,
    object_footprint_polygon,
    polygon_bounds_xy,
    side_vector,
)

RELATION_TYPE = "bedside_group_alignment"


def evaluate_bedside_group_alignment(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    """Require bedside tables at the head end and on distinct bed sides."""
    geometry = case_pack.get("scene_geometry") or {}
    objects = [
        obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    ]
    beds = [obj for obj in objects if _is_bed(obj)]
    nightstands = [obj for obj in objects if _is_nightstand(obj)]
    if not beds or not nightstands:
        return []

    associated = _associate_nightstands(beds, nightstands)
    rooms = [room for room in geometry.get("rooms") or [] if isinstance(room, dict)]
    doors = [
        door
        for door in ((geometry.get("scene_shell") or {}).get("doors") or [])
        if isinstance(door, dict)
    ]
    results: list[dict[str, Any]] = []
    for bed in beds:
        members = associated.get(str(bed["id"]), [])
        if not members:
            continue
        result = _evaluate_group(bed, members, rooms=rooms, doors=doors)
        if result is not None:
            results.append(result)
    return results


def _evaluate_group(
    bed: dict[str, Any],
    nightstands: list[dict[str, Any]],
    *,
    rooms: list[dict[str, Any]],
    doors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    bed_center = bbox_center_xy(bed)
    bed_polygon = object_footprint_polygon(bed)
    if bed_center is None or not bed_polygon:
        return None

    front = front_vector(bed)
    side = side_vector(bed)
    head = (-front[0], -front[1])
    half_length = _span_along_axis(bed_polygon, front) / 2.0
    half_width = _span_along_axis(bed_polygon, side) / 2.0
    if min(half_length, half_width) <= 1e-6:
        return None

    diagnostics: list[dict[str, Any]] = []
    failures: list[str] = []
    side_signs: list[int] = []
    for nightstand in nightstands:
        center = bbox_center_xy(nightstand)
        polygon = object_footprint_polygon(nightstand)
        if center is None or not polygon:
            continue
        dx = center[0] - bed_center[0]
        dy = center[1] - bed_center[1]
        head_offset = dx * head[0] + dy * head[1]
        side_offset = dx * side[0] + dy * side[1]
        stand_half_head = _span_along_axis(polygon, head) / 2.0
        stand_half_side = _span_along_axis(polygon, side) / 2.0
        minimum_head_offset = max(0.15, 0.25 * half_length)
        maximum_head_offset = half_length + stand_half_head + 0.45
        minimum_side_offset = max(0.18, 0.40 * half_width)
        maximum_side_offset = half_width + stand_half_side + 0.60
        gap = bbox_gap_xy(bed, nightstand)
        maximum_gap = max(0.45, 0.25 * min(2.0 * half_length, 2.0 * half_width))
        at_head = minimum_head_offset <= head_offset <= maximum_head_offset
        beside = minimum_side_offset <= abs(side_offset) <= maximum_side_offset
        adjacent = gap is not None and gap <= maximum_gap
        side_sign = 1 if side_offset > 0.0 else -1 if side_offset < 0.0 else 0
        if side_sign:
            side_signs.append(side_sign)
        diagnostics.append(
            {
                "nightstand_id": str(nightstand["id"]),
                "head_offset_m": round(head_offset, 4),
                "minimum_head_offset_m": round(minimum_head_offset, 4),
                "maximum_head_offset_m": round(maximum_head_offset, 4),
                "side_offset_m": round(side_offset, 4),
                "minimum_side_offset_m": round(minimum_side_offset, 4),
                "maximum_side_offset_m": round(maximum_side_offset, 4),
                "bbox_gap_m": round(gap, 4) if gap is not None else None,
                "maximum_bbox_gap_m": round(maximum_gap, 4),
                "at_head_end": at_head,
                "beside_bed": beside,
                "adjacent": adjacent,
                "side": (
                    "left" if side_sign > 0 else "right" if side_sign < 0 else "center"
                ),
            }
        )
        if not at_head:
            failures.append(
                f"`{nightstand['id']}` is not at the bed head end "
                f"(head-local offset {head_offset:+.2f}m)"
            )
        if not beside:
            failures.append(
                f"`{nightstand['id']}` is not in a bed-side slot "
                f"(side-local offset {side_offset:+.2f}m)"
            )
        if not adjacent:
            gap_text = "unknown" if gap is None else f"{gap:.2f}m"
            failures.append(
                f"`{nightstand['id']}` is not reachable from the bed "
                f"(bbox gap {gap_text}, allowed {maximum_gap:.2f}m)"
            )

    if not diagnostics:
        return None
    # 2026-07-15 修改原因：两个床头柜即使都与床相邻，也可能被物理净空修复
    # 搬到床的同一侧；成对场景必须在 bed-local 左右轴上异号分列。
    opposite_sides = len(nightstands) < 2 or (-1 in side_signs and 1 in side_signs)
    if not opposite_sides:
        failures.append("multiple nightstands occupy the same side of the bed")

    room = _room_for_bed(bed, bed_center, rooms)
    head_wall = _headboard_wall(bed_center, head, half_length, room)
    group_objects = [bed, *nightstands]
    actual_door_conflicts = _actual_door_conflicts(group_objects, doors)
    target_slot_door_conflicts = _target_slot_door_conflicts(
        bed,
        nightstands,
        bed_center=bed_center,
        head=head,
        side=side,
        half_length=half_length,
        half_width=half_width,
        head_wall=head_wall,
        doors=doors,
    )
    door_conflicts = sorted(actual_door_conflicts | target_slot_door_conflicts)
    if actual_door_conflicts:
        failures.append(
            "the current bed group intersects door clearance "
            + ", ".join(f"`{item}`" for item in sorted(actual_door_conflicts))
        )
    elif target_slot_door_conflicts:
        failures.append(
            "rebuilding the missing head-side slots on this wall would intersect "
            "door clearance "
            + ", ".join(f"`{item}`" for item in sorted(target_slot_door_conflicts))
        )

    failed = bool(failures)
    bed_id = str(bed["id"])
    member_ids = sorted(str(item["id"]) for item in nightstands)
    if failed:
        reason = (
            "Bedside tables must stay at the bed's head end (opposite the bed front), "
            "in reachable bed-local side slots, with paired tables on opposite sides. "
            + "; ".join(failures)
            + "."
        )
    else:
        reason = (
            f"All {len(member_ids)} bedside table(s) occupy reachable head-side "
            "slots; paired tables are split across the bed's left and right sides, "
            "and the group does not intersect door clearance."
        )

    repair_advice = _repair_advice(
        bed_id,
        member_ids,
        door_conflicts=door_conflicts,
        head_wall=head_wall,
    )
    return {
        "check_id": f"fd_{bed_id}_{RELATION_TYPE}",
        "metric": "functional_dependency",
        "label": "fail" if failed else "pass",
        "confidence": 0.96 if failed else 0.92,
        "primary_object": bed_id,
        "related_objects": member_ids,
        "selected_related_objects": member_ids,
        "blocking_objects": door_conflicts,
        "relation_type": RELATION_TYPE,
        "reason": reason,
        "repair_advice": repair_advice,
        "diagnostics": {
            "bed_front_xy": [round(front[0], 6), round(front[1], 6)],
            "bed_head_xy": [round(head[0], 6), round(head[1], 6)],
            "bed_half_length_m": round(half_length, 4),
            "bed_half_width_m": round(half_width, 4),
            "headboard_wall": head_wall,
            "nightstand_slots": diagnostics,
            "opposite_sides": opposite_sides,
            "actual_door_conflicts": sorted(actual_door_conflicts),
            "target_slot_door_conflicts": sorted(target_slot_door_conflicts),
        },
        "evidence": {
            "constraint": "bed_local_head_side_slots_and_door_clearance",
            "coordinate_frame": "bed_local_xy",
        },
        "evaluation_source": "scenesmith_bedside_group_alignment",
        "scoring_tier": "core",
    }


def _associate_nightstands(
    beds: list[dict[str, Any]], nightstands: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    associated = {str(bed["id"]): [] for bed in beds}
    for nightstand in nightstands:
        candidates: list[tuple[float, str]] = []
        for bed in beds:
            polygon = object_footprint_polygon(bed)
            if not polygon:
                continue
            front = front_vector(bed)
            side = side_vector(bed)
            bed_length = _span_along_axis(polygon, front)
            bed_width = _span_along_axis(polygon, side)
            gap = bbox_gap_xy(bed, nightstand)
            if gap is None or gap > max(1.2, 0.6 * max(bed_length, bed_width)):
                continue
            candidates.append((gap, str(bed["id"])))
        if candidates:
            _, bed_id = min(candidates)
            associated[bed_id].append(nightstand)
    for members in associated.values():
        members.sort(key=lambda item: str(item.get("id") or ""))
    return associated


def _target_slot_door_conflicts(
    bed: dict[str, Any],
    nightstands: list[dict[str, Any]],
    *,
    bed_center: tuple[float, float],
    head: tuple[float, float],
    side: tuple[float, float],
    half_length: float,
    half_width: float,
    head_wall: str,
    doors: list[dict[str, Any]],
) -> set[str]:
    if not head_wall:
        return set()
    matching_doors = [
        door
        for door in doors
        if str(door.get("wall_direction") or "").strip().lower() == head_wall
    ]
    if not matching_doors:
        return set()

    assignments = _target_side_assignments(nightstands, bed_center, side)
    conflicts: set[str] = set()
    for nightstand, sign in assignments:
        polygon = object_footprint_polygon(nightstand)
        center = bbox_center_xy(nightstand)
        if not polygon or center is None:
            continue
        stand_half_side = _span_along_axis(polygon, side) / 2.0
        target_center = (
            bed_center[0]
            + head[0] * half_length
            + side[0] * sign * (half_width + stand_half_side + 0.08),
            bed_center[1]
            + head[1] * half_length
            + side[1] * sign * (half_width + stand_half_side + 0.08),
        )
        translated = [
            (x + target_center[0] - center[0], y + target_center[1] - center[1])
            for x, y in polygon
        ]
        bounds = polygon_bounds_xy(translated)
        for door in matching_doors:
            if _bounds_overlap(bounds, _door_bounds(door)):
                door_id = str(door.get("id") or door.get("opening_id") or "")
                if door_id:
                    conflicts.add(door_id)
    return conflicts


def _target_side_assignments(
    nightstands: list[dict[str, Any]],
    bed_center: tuple[float, float],
    side: tuple[float, float],
) -> list[tuple[dict[str, Any], int]]:
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for nightstand in nightstands:
        center = bbox_center_xy(nightstand)
        if center is None:
            continue
        offset = (center[0] - bed_center[0]) * side[0] + (
            center[1] - bed_center[1]
        ) * side[1]
        ranked.append((offset, str(nightstand["id"]), nightstand))
    ranked.sort(key=lambda row: (row[0], row[1]))
    if len(ranked) == 1:
        offset, _, nightstand = ranked[0]
        return [(nightstand, 1 if offset >= 0.0 else -1)]
    assignments: list[tuple[dict[str, Any], int]] = []
    for index, (_, _, nightstand) in enumerate(ranked):
        assignments.append((nightstand, -1 if index < len(ranked) / 2.0 else 1))
    return assignments


def _actual_door_conflicts(
    objects: list[dict[str, Any]], doors: list[dict[str, Any]]
) -> set[str]:
    conflicts: set[str] = set()
    for obj in objects:
        polygon = object_footprint_polygon(obj)
        if not polygon:
            continue
        bounds = polygon_bounds_xy(polygon)
        for door in doors:
            if _bounds_overlap(bounds, _door_bounds(door)):
                door_id = str(door.get("id") or door.get("opening_id") or "")
                if door_id:
                    conflicts.add(door_id)
    return conflicts


def _room_for_bed(
    bed: dict[str, Any],
    center: tuple[float, float],
    rooms: list[dict[str, Any]],
) -> dict[str, Any] | None:
    bed_room = str(bed.get("room") or "")
    for room in rooms:
        if bed_room and str(room.get("id") or "") == bed_room:
            return room
    for room in rooms:
        bounds = _room_bounds(room)
        if (
            bounds
            and bounds[0] <= center[0] <= bounds[2]
            and bounds[1] <= center[1] <= bounds[3]
        ):
            return room
    return rooms[0] if len(rooms) == 1 else None


def _headboard_wall(
    center: tuple[float, float],
    head: tuple[float, float],
    half_length: float,
    room: dict[str, Any] | None,
) -> str:
    bounds = _room_bounds(room)
    if bounds is None:
        return ""
    head_point = (
        center[0] + head[0] * half_length,
        center[1] + head[1] * half_length,
    )
    x0, y0, x1, y1 = bounds
    distances = {
        "west": abs(head_point[0] - x0),
        "east": abs(head_point[0] - x1),
        "south": abs(head_point[1] - y0),
        "north": abs(head_point[1] - y1),
    }
    direction, distance = min(distances.items(), key=lambda item: (item[1], item[0]))
    room_scale = min(abs(x1 - x0), abs(y1 - y0))
    # 2026-07-15 修改原因：物理避障可能已把床头从墙面拉开约 0.5m；仍需识别
    # 原 headboard wall 才能判断恢复床头槽位是否撞门，但不把房间中央床误绑墙。
    return direction if distance <= max(0.55, 0.12 * room_scale) else ""


def _repair_advice(
    bed_id: str,
    member_ids: list[str],
    *,
    door_conflicts: list[str],
    head_wall: str,
) -> str:
    members = ", ".join(f"`{item}`" for item in member_ids)
    if door_conflicts:
        doors = ", ".join(f"`{item}`" for item in door_conflicts)
        wall = f" the current {head_wall} headboard wall" if head_wall else " this wall"
        return (
            f"Do not repair {doors} by moving `{bed_id}` or {members} independently along{wall}. "
            f"Move `{bed_id}` and all bedside tables ({members}) as one coordinated group "
            "to a wall whose complete headboard-and-side-slot envelope is clear of doors. "
            "Place the bed first with its head end against the new wall, rebuild one "
            "bed-local nightstand slot on each requested side at the head end, preserve "
            "parallel yaw, then recheck physics and door clearance."
        )
    return (
        f"Keep `{bed_id}` fixed and rebuild {members} from bed-local coordinates: the "
        "head direction is opposite the bed front; put every nightstand at that head "
        "end and, when there are two or more, split them across the left and right "
        "sides. Do not place either table at the foot or move group members independently."
    )


def _span_along_axis(
    polygon: list[tuple[float, float]], axis: tuple[float, float]
) -> float:
    projections = [x * axis[0] + y * axis[1] for x, y in polygon]
    return max(projections) - min(projections)


def _door_bounds(door: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = door.get("bbox") or {}
    minimum = bbox.get("min") or []
    maximum = bbox.get("max") or []
    if len(minimum) < 2 or len(maximum) < 2:
        return None
    return (
        float(minimum[0]),
        float(minimum[1]),
        float(maximum[0]),
        float(maximum[1]),
    )


def _room_bounds(
    room: dict[str, Any] | None,
) -> tuple[float, float, float, float] | None:
    bbox = (room or {}).get("bbox") or {}
    minimum = bbox.get("min") or []
    maximum = bbox.get("max") or []
    if len(minimum) < 2 or len(maximum) < 2:
        return None
    return (
        float(minimum[0]),
        float(minimum[1]),
        float(maximum[0]),
        float(maximum[1]),
    )


def _bounds_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float] | None,
) -> bool:
    if second is None:
        return False
    tolerance = 1e-4
    return (
        min(first[2], second[2]) - max(first[0], second[0]) > tolerance
        and min(first[3], second[3]) - max(first[1], second[1]) > tolerance
    )


def _object_text(obj: dict[str, Any]) -> str:
    return " ".join(
        str(obj.get(key) or "").strip().lower().replace("-", "_")
        for key in ("id", "name", "description", "category", "category_norm")
    )


def _is_bed(obj: dict[str, Any]) -> bool:
    category = str(obj.get("category_norm") or obj.get("category") or "").lower()
    object_id = str(obj.get("id") or "").strip().lower()
    descriptive_text = " ".join(
        str(obj.get(key) or "").strip().lower() for key in ("name", "description")
    )
    # 2026-07-15 修改原因：房间 floor ID 通常是 ``floor_bedroom``；按任意
    # ``_bed`` 子串识别会把整块地板当床，并让床头几何完全失真。类别优先，
    # 仅允许 ID 以 bed 开头或名称/描述出现独立单词 bed。
    return (
        category in {"bed", "bunk_bed", "double_bed", "single_bed"}
        or object_id == "bed"
        or object_id.startswith(("bed_", "bed-"))
        or bool(re.search(r"\bbed\b", descriptive_text))
    )


def _is_nightstand(obj: dict[str, Any]) -> bool:
    category = str(obj.get("category_norm") or obj.get("category") or "").lower()
    text = _object_text(obj)
    return category in {"bedside_table", "nightstand", "night_stand"} or any(
        token in text for token in ("nightstand", "night_stand", "bedside table")
    )
