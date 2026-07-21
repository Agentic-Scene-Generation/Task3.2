"""General dining-chair distribution checks for rectangular tables."""

from __future__ import annotations

import math
from typing import Any

from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.manipuland_completeness import (
    _bbox_gap_xy,
    _footprint_short_side,
    _is_dining_seat,
    _is_dining_table,
    _object_identity_text,
)
from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
    front_vector,
)

RELATION_TYPE = "dining_seat_distribution"


def evaluate_dining_seat_distribution(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check that chairs are centered or evenly spaced along each table edge."""
    objects = [
        obj
        for obj in ((case_pack.get("scene_geometry") or {}).get("objects") or [])
        if isinstance(obj, dict) and obj.get("id")
    ]
    objects_by_id = {str(obj["id"]): obj for obj in objects}
    tables = [
        obj for obj in objects if _is_dining_table(obj) and not _is_round_table(obj)
    ]
    seats_by_table = _positionally_associated_seats(tables, objects_by_id)
    results: list[dict[str, Any]] = []
    for table in tables:
        result = _evaluate_table(
            table, seats_by_table.get(str(table["id"]), [])
        )
        if result is not None:
            results.append(result)
    return results


def _evaluate_table(
    table: dict[str, Any], seats: list[dict[str, Any]]
) -> dict[str, Any] | None:
    center = bbox_center_xy(table)
    size = (table.get("bbox_world") or {}).get("size") or []
    seats = [seat for seat in seats if "bench" not in _object_identity_text(seat)]
    if center is None or len(size) < 2 or not seats:
        return None
    width, depth = float(size[0]), float(size[1])
    if min(width, depth) <= 1e-6:
        return None
    yaw = math.radians(float(table.get("yaw_deg") or 0.0))
    tangent_x = (math.cos(yaw), math.sin(yaw))
    tangent_y = (-math.sin(yaw), math.cos(yaw))
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = {
        "left": [], "right": [], "front": [], "back": []
    }
    for seat in seats:
        seat_center = bbox_center_xy(seat)
        if seat_center is None:
            continue
        dx, dy = seat_center[0] - center[0], seat_center[1] - center[1]
        local_x = dx * tangent_x[0] + dy * tangent_x[1]
        local_y = dx * tangent_y[0] + dy * tangent_y[1]
        # 2026-07-15 修改原因：座椅因碰撞或净空向外拉开后，按“到无限延长
        # 桌边直线的距离”会把短边椅误归到长边。改用有限桌边线段距离，确保
        # 桌角之外的座椅仍由其实际相邻桌边负责，且不依赖某个场景的绝对尺寸。
        edge, tangent_position = _nearest_table_edge(
            local_x, local_y, width=width, depth=depth
        )
        grouped[edge].append((seat, tangent_position))

    diagnostics: list[dict[str, Any]] = []
    failures: list[str] = []
    for edge, members in grouped.items():
        if not members:
            continue
        edge_length = depth if edge in {"left", "right"} else width
        chair_spans = [_seat_tangent_span(seat, edge, yaw) for seat, _ in members]
        perpendicular_length = width if edge in {"left", "right"} else depth
        # 2026-07-13 修改原因：只扣半个椅宽会把长边端部槽位推到桌角，导致
        # 座椅同时侵占相邻短边。至少扣除半个垂直边长，使槽位明确属于当前边。
        margin = max(max(chair_spans) / 2, perpendicular_length / 2)
        usable_span = max(0.0, edge_length - 2 * margin)
        count = len(members)
        # 2026-07-13 修改原因：槽位由桌边长度和该边实际座椅数推导；单椅取
        # 中点，多椅在扣除椅宽边距后等距分布，避免固定四人桌坐标。
        slots = (
            [0.0]
            if count == 1
            else [-usable_span / 2 + i * usable_span / (count - 1) for i in range(count)]
        )
        actual = sorted(members, key=lambda row: (row[1], str(row[0]["id"])))
        for (seat, position), slot, chair_span in zip(actual, slots, sorted(chair_spans)):
            deviation = abs(position - slot)
            allowed = max(0.08, min(0.35 * chair_span, 0.08 * edge_length))
            passed = deviation <= allowed
            # 2026-07-14 修改原因：多椅同边可以平行朝向桌边而不必都斜指桌心；
            # 单椅边位才要求严格正对。2026-07-15 的有限边归类会保证拉远后的
            # 短边椅仍各自落在单椅边位，不再因错误分组漏掉 180° 翻转。
            facing_error = _seat_facing_error_deg(seat, center) if count == 1 else None
            facing_passed = facing_error is None or facing_error <= 10.0
            diagnostics.append({
                "seat_id": str(seat["id"]), "edge": edge,
                "tangent_position_m": round(position, 4),
                "target_position_m": round(slot, 4),
                "deviation_m": round(deviation, 4),
                "allowed_deviation_m": round(allowed, 4), "aligned": passed,
                "facing_error_deg": round(facing_error, 2) if facing_error is not None else None,
                "facing_allowed_error_deg": 10.0,
                "facing_aligned": facing_passed,
            })
            if not passed:
                direction = "positive" if slot > position else "negative"
                failures.append(
                    f"`{seat['id']}` on the {edge} edge is {deviation:.2f}m from "
                    f"its evenly distributed slot; move it in the {direction} edge direction"
                )
            if not facing_passed:
                failures.append(
                    f"`{seat['id']}` on the {edge} edge is rotated {facing_error:.1f}° "
                    "away from the table center; rotate it in place so its front normal "
                    "faces the table, preserving its table-edge slot and clearance"
                )
    if not diagnostics:
        return None
    table_id = str(table["id"])
    related = sorted(str(seat["id"]) for seat in seats)
    failed = bool(failures)
    reason = (
        "Dining chairs on each rectangular table edge must be centered when alone "
        "and evenly distributed when multiple chairs share the edge. "
        "For a dining chair, use an exact table-local slot and do not use generic "
        "center snapping or shift the chair along the edge normal to resolve a "
        "door conflict; move the table or door-compatible layout instead. "
        + "; ".join(failures)
        if failed else
        "Dining chairs are centered or evenly distributed along their respective table edges."
    )
    return {
        "check_id": f"fd_{table_id}_{RELATION_TYPE}",
        "metric": "functional_dependency", "label": "fail" if failed else "pass",
        "confidence": 0.93 if failed else 0.89, "primary_object": table_id,
        "related_objects": related, "selected_related_objects": related,
        "blocking_objects": [], "relation_type": RELATION_TYPE, "reason": reason,
        "diagnostics": {"seat_slots": diagnostics},
        "evidence": {"distribution": "table_local_edge_slots"},
        "evaluation_source": "scenesmith_dining_seat_distribution", "scoring_tier": "core",
    }


def _seat_tangent_span(seat: dict[str, Any], edge: str, table_yaw: float) -> float:
    size = (seat.get("bbox_world") or {}).get("size") or []
    if len(size) < 2:
        return 0.45
    axis = (math.cos(table_yaw), math.sin(table_yaw)) if edge in {"front", "back"} else (-math.sin(table_yaw), math.cos(table_yaw))
    return max(0.2, abs(axis[0]) * float(size[0]) + abs(axis[1]) * float(size[1]))


def _positionally_associated_seats(
    tables: list[dict[str, Any]],
    objects_by_id: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Associate nearby dining seats without using their current facing."""
    associated = {str(table["id"]): [] for table in tables}
    for seat in objects_by_id.values():
        if not _is_dining_seat(seat) or "bench" in _object_identity_text(seat):
            continue
        seat_scale = _footprint_short_side(seat)
        if seat_scale is None:
            continue
        candidates: list[tuple[float, float, str]] = []
        for table in tables:
            table_scale = _footprint_short_side(table)
            gap = _bbox_gap_xy(table, seat)
            if table_scale is None or gap is None:
                continue
            association_gap = max(seat_scale, 0.25 * table_scale)
            if gap <= association_gap:
                # 2026-07-15 修改原因：朝向本身正是本检查要发现的问题，不能再
                # 用“必须已朝桌子”作为关联前提。多桌场景按归一化间隙分配给
                # 最近桌组，既覆盖拉椅净空，也避免同一椅被相邻桌重复认领。
                candidates.append(
                    (gap, gap / max(association_gap, 1e-6), str(table["id"]))
                )
        if candidates:
            _, _, table_id = min(candidates)
            associated[table_id].append(seat)
    for seats in associated.values():
        seats.sort(key=lambda item: str(item.get("id") or ""))
    return associated


def _nearest_table_edge(
    local_x: float,
    local_y: float,
    *,
    width: float,
    depth: float,
) -> tuple[str, float]:
    """Return the nearest finite rectangular edge and its tangent coordinate."""
    half_width = width / 2.0
    half_depth = depth / 2.0
    clamped_x = min(max(local_x, -half_width), half_width)
    clamped_y = min(max(local_y, -half_depth), half_depth)
    x_scale = max(half_width, 1e-6)
    y_scale = max(half_depth, 1e-6)
    candidates = (
        (
            math.hypot(local_x + half_width, local_y - clamped_y),
            -(abs(local_x) / x_scale),
            "left",
            local_y,
        ),
        (
            math.hypot(local_x - half_width, local_y - clamped_y),
            -(abs(local_x) / x_scale),
            "right",
            local_y,
        ),
        (
            math.hypot(local_x - clamped_x, local_y + half_depth),
            -(abs(local_y) / y_scale),
            "front",
            local_x,
        ),
        (
            math.hypot(local_x - clamped_x, local_y - half_depth),
            -(abs(local_y) / y_scale),
            "back",
            local_x,
        ),
    )
    _, _, edge, tangent_position = min(
        candidates, key=lambda row: (row[0], row[1], row[2])
    )
    return edge, tangent_position


def _seat_facing_error_deg(
    seat: dict[str, Any], table_center: tuple[float, float] | None
) -> float | None:
    """Return angular error between the annotated chair front and table center."""
    # 2026-07-14 修改原因：check_facing_tool 的宽松通过阈值会把约 13° 的
    # dining_chair_2 偏角判为正确；餐桌座位检查需要更严格的 10° 误差。
    if table_center is None:
        return None
    center = bbox_center_xy(seat)
    if center is None:
        return None
    dx = float(table_center[0]) - float(center[0])
    dy = float(table_center[1]) - float(center[1])
    if abs(dx) + abs(dy) <= 1e-6:
        return None
    # 2026-07-15 修改原因：优先复用 adapter 已按资产 front_hint 生成的交互面，
    # 避免再次硬编码本地 +Y；没有交互面时再使用统一 geometry.front_vector。
    front = next(
        (
            face.get("normal_xy")
            for face in (seat.get("interaction_faces") or [])
            if isinstance(face, dict)
            and face.get("name") == "front"
            and isinstance(face.get("normal_xy"), list)
            and len(face["normal_xy"]) >= 2
        ),
        None,
    )
    if front is None and "yaw_deg" not in seat:
        return None
    if front is None:
        fx, fy = front_vector(seat)
    else:
        fx, fy = float(front[0]), float(front[1])
    front_norm = math.hypot(fx, fy)
    target_norm = math.hypot(dx, dy)
    if front_norm <= 1e-6 or target_norm <= 1e-6:
        return None
    cosine = (fx * dx + fy * dy) / (front_norm * target_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _is_round_table(table: dict[str, Any]) -> bool:
    text = _object_identity_text(table)
    return any(token in text for token in ("round", "circular", "oval", "ellipse"))
