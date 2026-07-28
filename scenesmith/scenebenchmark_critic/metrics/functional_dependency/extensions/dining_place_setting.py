"""Dining place-setting alignment checks for the embedded critic."""

from __future__ import annotations

import functools
import math
import re

from typing import Any

from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.manipuland_completeness import (
    CUTLERY_GROUPS,
    _bbox_gap_xy,
    _footprint_short_side,
    _is_dining_seat,
    _is_dining_table,
    _matches_item_group,
    _object_identity_text,
    _object_text,
    _placement_surface_id,
    _required_groups,
    _scene_object_type,
    _surface_owner_map,
)
from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
    front_vector,
)

RELATION_TYPE = "dining_place_setting_alignment"


def evaluate_dining_place_setting_alignment(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check that each place setting lies in its assigned dining seat's front lane."""
    geometry = case_pack.get("scene_geometry") or {}
    objects = [
        obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    ]
    if not objects or not _prompt_requests_place_settings(case_pack):
        return []

    objects_by_id = {str(obj["id"]): obj for obj in objects}
    surface_owner = _surface_owner_map(objects)
    results: list[dict[str, Any]] = []
    for table in objects:
        if not _is_dining_table(table):
            continue
        table_id = str(table["id"])
        surface_ids = {
            surface_id
            for surface_id, owner_id in surface_owner.items()
            if owner_id == table_id
        }
        if not surface_ids:
            continue
        surface_items = [
            obj
            for obj in objects
            if _scene_object_type(obj) == "manipuland"
            and _placement_surface_id(obj) in surface_ids
        ]
        anchors = [obj for obj in surface_items if _is_place_anchor(obj)]
        seats = _associated_discrete_seats(table, objects_by_id)
        # 2026-07-13 修改原因：只有离散座位与餐位锚点可一对一对应时，
        # “椅子正前方”才有唯一、可执行的几何含义。长凳或数量不一致场景交给
        # completeness/视觉 critic，避免强行套用四人餐桌拓扑。
        if len(anchors) < 2 or len(anchors) != len(seats):
            continue
        result = _evaluate_table_alignment(
            table=table,
            seats=seats,
            anchors=anchors,
            companions=[obj for obj in surface_items if _is_place_companion(obj)],
        )
        if result is not None:
            results.append(result)
    return results


def _evaluate_table_alignment(
    *,
    table: dict[str, Any],
    seats: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    companions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    table_center = bbox_center_xy(table)
    table_short_side = _short_side(table)
    if table_center is None or table_short_side is None:
        return None
    assignment = _seat_lane_assignment(table, seats, anchors, table_center)
    if assignment is None:
        return None

    anchor_to_seat: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    failures: list[str] = []
    for seat, anchor in assignment:
        seat_center = bbox_center_xy(seat)
        anchor_center = bbox_center_xy(anchor)
        if seat_center is None or anchor_center is None:
            return None
        forward = _usable_seat_front(seat, seat_center, table_center)
        lateral_axis = (-forward[1], forward[0])
        recommended_center, recommended_surface_id = _recommended_anchor_center(
            table, seat, anchor, seat_center, forward
        )
        signed_lateral_offset = (
            (anchor_center[0] - seat_center[0]) * lateral_axis[0]
            + (anchor_center[1] - seat_center[1]) * lateral_axis[1]
        )
        lateral_offset = abs(signed_lateral_offset)
        longitudinal = (
            (anchor_center[0] - seat_center[0]) * forward[0]
            + (anchor_center[1] - seat_center[1]) * forward[1]
        )
        allowed = _anchor_centerline_tolerance(seat, anchor, lateral_axis)
        if recommended_center is None:
            target_center = (
                anchor_center[0] - signed_lateral_offset * lateral_axis[0],
                anchor_center[1] - signed_lateral_offset * lateral_axis[1],
            )
            longitudinal_slot_offset = 0.0
            longitudinal_allowed = math.inf
        else:
            target_center = recommended_center
            longitudinal_slot_offset = abs(
                (anchor_center[0] - target_center[0]) * forward[0]
                + (anchor_center[1] - target_center[1]) * forward[1]
            )
            longitudinal_allowed = _anchor_longitudinal_tolerance(anchor, forward)
        correction = (
            target_center[0] - anchor_center[0],
            target_center[1] - anchor_center[1],
        )
        seat_id = str(seat["id"])
        anchor_id = str(anchor["id"])
        anchor_to_seat[anchor_id] = seat
        aligned = (
            longitudinal > 0.0
            and lateral_offset <= allowed
            and longitudinal_slot_offset <= longitudinal_allowed
        )
        diagnostics.append(
            {
                "seat_id": seat_id,
                "anchor_id": anchor_id,
                "lateral_offset_m": round(lateral_offset, 4),
                "signed_lateral_offset_m": round(signed_lateral_offset, 4),
                "allowed_lateral_offset_m": round(allowed, 4),
                "longitudinal_offset_m": round(longitudinal, 4),
                "longitudinal_slot_offset_m": round(longitudinal_slot_offset, 4),
                "allowed_longitudinal_slot_offset_m": (
                    None
                    if math.isinf(longitudinal_allowed)
                    else round(longitudinal_allowed, 4)
                ),
                "recommended_translation_xy_m": [
                    round(correction[0], 4),
                    round(correction[1], 4),
                ],
                "recommended_anchor_center_xy_m": [
                    round(target_center[0], 4),
                    round(target_center[1], 4),
                ],
                "recommended_support_surface_id": recommended_surface_id,
                "aligned": aligned,
                "companion_ids": [],
                "misaligned_companion_ids": [],
            }
        )
        if not aligned:
            failures.append(
                f"`{anchor_id}` is not centered in front of `{seat_id}` "
                f"(lateral {lateral_offset:.2f}m, allowed {allowed:.2f}m; "
                f"edge-slot offset {longitudinal_slot_offset:.2f}m, allowed "
                f"{longitudinal_allowed:.2f}m); move its "
                f"whole place-setting cluster by ({correction[0]:+.2f}, "
                f"{correction[1]:+.2f})m in world XY so the plate/bowl center "
                f"reaches approximately ({target_center[0]:.2f}, "
                f"{target_center[1]:.2f})m"
                + (
                    f" on support surface `{recommended_surface_id}`"
                    if recommended_surface_id
                    else ""
                )
            )

    rows_by_anchor = {row["anchor_id"]: row for row in diagnostics}
    for companion in companions:
        anchor = _nearest_cluster_anchor(companion, anchors, table_short_side)
        if anchor is None:
            continue
        anchor_id = str(anchor["id"])
        seat = anchor_to_seat.get(anchor_id)
        if seat is None:
            continue
        seat_center = bbox_center_xy(seat)
        companion_center = bbox_center_xy(companion)
        if seat_center is None or companion_center is None:
            continue
        forward = _usable_seat_front(seat, seat_center, table_center)
        lateral_axis = (-forward[1], forward[0])
        lateral_offset = abs(
            (companion_center[0] - seat_center[0]) * lateral_axis[0]
            + (companion_center[1] - seat_center[1]) * lateral_axis[1]
        )
        allowed = _companion_lane_half_width(seat, companion, lateral_axis)
        companion_id = str(companion["id"])
        row = rows_by_anchor[anchor_id]
        row["companion_ids"].append(companion_id)
        if lateral_offset > allowed:
            row["misaligned_companion_ids"].append(companion_id)
            failures.append(
                f"`{companion_id}` belonging to `{anchor_id}` is outside "
                f"`{seat['id']}`'s front lane"
            )

    table_id = str(table["id"])
    related_ids = sorted(
        {
            str(obj["id"])
            for obj in [*seats, *anchors, *companions]
            if obj.get("id")
        }
    )
    if failures:
        reason = (
            "Dining place settings must be centered on the front axis of their "
            "one-to-one assigned seats. "
            + "; ".join(failures[:8])
            + ". Move each plate/bowl together with its nearby cutlery, drinkware, "
            "and napkin toward that seat's centerline; do not move it to another "
            "table edge. When available, call `align_dining_place_settings` so "
            "world-space targets are converted onto the correct segmented tabletop "
            "support surface."
        )
        label = "fail"
        confidence = 0.94
    else:
        reason = (
            f"All {len(assignment)} dining place setting(s) are centered in the "
            "front lanes of distinct nearby seats, with companions grouped to the "
            "same seats."
        )
        label = "pass"
        confidence = 0.9
    return {
        "check_id": f"fd_{table_id}_{RELATION_TYPE}",
        "metric": "functional_dependency",
        "label": label,
        "confidence": confidence,
        "primary_object": table_id,
        "related_objects": related_ids,
        "selected_related_objects": related_ids,
        "blocking_objects": [],
        "relation_type": RELATION_TYPE,
        "reason": reason,
        "diagnostics": {"assignments": diagnostics},
        "evidence": {
            "association": "minimum_seat_front_lane_cost_one_to_one",
            # 2026-07-14 修改原因：同一餐桌可能由多个连续 mesh surface 构成；
            # 修复必须沿座椅前轴选最近的真实支撑面，不能按最大桌面 footprint
            # 把端部餐位错误折叠到中央桌面条带。
            "alignment": "strict_anchor_centerline_projection_to_nearest_support_region",
        },
        "evaluation_source": "scenesmith_dining_place_setting_alignment",
        "scoring_tier": "core",
    }


def _seat_lane_assignment(
    table: dict[str, Any],
    seats: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    table_center: tuple[float, float],
) -> list[tuple[dict[str, Any], dict[str, Any]]] | None:
    seat_centers = [bbox_center_xy(seat) for seat in seats]
    anchor_centers = [bbox_center_xy(anchor) for anchor in anchors]
    if any(center is None for center in [*seat_centers, *anchor_centers]):
        return None
    count = len(seats)
    costs: list[list[float]] = []
    for seat_index, seat in enumerate(seats):
        seat_center = seat_centers[seat_index]
        assert seat_center is not None
        forward = _usable_seat_front(seat, seat_center, table_center)
        lateral_axis = (-forward[1], forward[0])
        row: list[float] = []
        for anchor_index, anchor_center in enumerate(anchor_centers):
            assert anchor_center is not None
            dx = anchor_center[0] - seat_center[0]
            dy = anchor_center[1] - seat_center[1]
            lateral = abs(dx * lateral_axis[0] + dy * lateral_axis[1])
            longitudinal = dx * forward[0] + dy * forward[1]
            target, _surface_id = _recommended_anchor_center(
                table, seat, anchors[anchor_index], seat_center, forward
            )
            if target is None:
                distance = math.hypot(dx, dy)
            else:
                distance = math.hypot(
                    anchor_center[0] - target[0], anchor_center[1] - target[1]
                )
            # 2026-07-13 修改原因：四角餐盘到多个座椅的欧氏距离相近，纯最近
            # 匹配会把餐盘分给错误桌边。优先最小化座椅前轴横向偏差，再用距离
            # 消除同边多座位歧义；位于座椅背后的候选附加尺度相关惩罚。
            behind_penalty = 8.0 * distance if longitudinal <= 0.0 else 0.0
            row.append(4.0 * lateral + distance + behind_penalty)
        costs.append(row)

    # 2026-07-13 修改原因：逐椅贪心会让相邻座位争用同一餐盘。位掩码动态规划
    # 求全局最短一对一分配，适配长桌、圆桌和非对称座椅布局。
    @functools.lru_cache(maxsize=None)
    def solve(seat_index: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
        if seat_index == count:
            return 0.0, ()
        best = (math.inf, ())
        for anchor_index in range(count):
            if used_mask & (1 << anchor_index):
                continue
            remaining_cost, remaining = solve(
                seat_index + 1, used_mask | (1 << anchor_index)
            )
            candidate = (
                costs[seat_index][anchor_index] + remaining_cost,
                (anchor_index, *remaining),
            )
            if candidate < best:
                best = candidate
        return best

    # Avoid exponential work for unusually large banquet layouts; deterministic
    # lowest-lane-cost assignment remains a safe report-only fallback above 12 seats.
    if count <= 12:
        _cost, indices = solve(0, 0)
    else:
        available = set(range(count))
        picked: list[int] = []
        for seat_index in range(count):
            anchor_index = min(
                available, key=lambda index: (costs[seat_index][index], index)
            )
            available.remove(anchor_index)
            picked.append(anchor_index)
        indices = tuple(picked)
    return [(seats[index], anchors[indices[index]]) for index in range(count)]


def _prompt_requests_place_settings(case_pack: dict[str, Any]) -> bool:
    task = str(case_pack.get("task_instruction") or "")
    required = _required_groups(task)
    return bool(required & {"plate", "bowl"}) or bool(
        re.search(r"\b(?:table|place)\s*settings?\b", task.lower())
    )


def _associated_discrete_seats(
    table: dict[str, Any], objects_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    table_center = bbox_center_xy(table)
    table_scale = _footprint_short_side(table)
    if table_center is None or table_scale is None:
        return []
    seats: list[dict[str, Any]] = []
    for seat in objects_by_id.values():
        if not _is_dining_seat(seat) or "bench" in _object_identity_text(seat):
            continue
        seat_center = bbox_center_xy(seat)
        seat_scale = _footprint_short_side(seat)
        gap = _bbox_gap_xy(table, seat)
        if seat_center is None or seat_scale is None or gap is None:
            continue
        fx, fy = front_vector(seat)
        tx, ty = table_center[0] - seat_center[0], table_center[1] - seat_center[1]
        target_distance = math.hypot(tx, ty)
        if target_distance <= 1e-9:
            continue
        front_alignment = (fx * tx + fy * ty) / target_distance
        # 2026-07-13 修改原因：餐位归属不仅看中心距离，还要求座椅朝向该桌；
        # 允许一个座椅短边的桌椅间隙，以覆盖正常拉椅空间但排除相邻桌组。
        association_gap = max(seat_scale, 0.25 * table_scale)
        if gap <= association_gap and front_alignment >= 0.5:
            seats.append(seat)
    return sorted(seats, key=lambda item: str(item.get("id") or ""))


def _is_place_anchor(obj: dict[str, Any]) -> bool:
    text = _object_text(obj)
    return _matches_item_group("plate", text) or _matches_item_group("bowl", text)


def _is_place_companion(obj: dict[str, Any]) -> bool:
    text = _object_text(obj)
    return _matches_item_group("drinkware", text) or _matches_item_group(
        "napkin", text
    ) or any(_matches_item_group(group, text) for group in CUTLERY_GROUPS)


def _usable_seat_front(
    seat: dict[str, Any],
    seat_center: tuple[float, float],
    table_center: tuple[float, float],
) -> tuple[float, float]:
    fx, fy = front_vector(seat)
    tx, ty = table_center[0] - seat_center[0], table_center[1] - seat_center[1]
    target_norm = math.hypot(tx, ty)
    if target_norm <= 1e-9:
        return fx, fy
    # If front metadata is missing or contradicts the nearby table, the dining-set
    # furniture relation owns that orientation failure. Use the geometric seat-table
    # axis here so manipuland feedback still identifies the correct table edge.
    if fx * tx + fy * ty <= 0.2 * target_norm:
        return tx / target_norm, ty / target_norm
    return fx, fy


def _anchor_centerline_tolerance(
    seat: dict[str, Any], item: dict[str, Any], lateral_axis: tuple[float, float]
) -> float:
    seat_span = _projected_span(seat, lateral_axis)
    item_span = _projected_span(item, lateral_axis)
    if seat_span is None:
        seat_span = _short_side(seat) or 0.45
    if item_span is None:
        item_span = _short_side(item) or 0.2
    # 2026-07-13 修改原因：“落在椅宽范围内”仍会产生肉眼明显的四角餐盘。
    # 餐盘/餐碗锚点必须接近座椅中心线；容差由座椅宽和锚点尺寸共同缩放，
    # 兼容不同尺寸的椅子、碗盘及长桌，而不是使用固定四人桌坐标。
    return max(0.04, min(0.2 * seat_span, 0.3 * item_span))


def _anchor_longitudinal_tolerance(
    item: dict[str, Any], forward: tuple[float, float]
) -> float:
    item_span = _projected_span(item, forward)
    if item_span is None:
        item_span = _short_side(item) or 0.2
    return max(0.04, 0.35 * item_span)


def _recommended_anchor_center(
    table: dict[str, Any],
    seat: dict[str, Any],
    anchor: dict[str, Any],
    seat_center: tuple[float, float],
    forward: tuple[float, float],
) -> tuple[tuple[float, float] | None, str | None]:
    region_entry = _nearest_tabletop_region_entry(table, seat_center, forward)
    if region_entry is None:
        return None, None
    surface_id, boundary, usable_depth = region_entry
    anchor_span = _projected_span(anchor, forward)
    if anchor_span is None:
        anchor_span = _short_side(anchor) or 0.2
    table_scale = _short_side(table) or anchor_span
    # 2026-07-13 修改原因：只投影到座椅中心线会保留餐盘靠近桌心的纵向位置，
    # 居中后容易撞中央花瓶并诱使模型再次横移。沿座椅前轴找到桌面入射边界，
    # 再按盘碗半径和桌尺度向内留边，得到可达且远离桌心装饰物的通用槽位。
    edge_inset = 0.5 * anchor_span + max(0.03, 0.05 * table_scale)
    # 2026-07-14 修改原因：HSSD 桌面常被拆成窄而连续的 plank/surface 区域。
    # 不能把常规桌边 inset 推出该单独支撑面；缩放到该 ray 穿过区域的可用深度，
    # 既保留“靠椅子一侧”的餐位，也保证目标仍可实际落在该 surface 上。
    bounded_inset = min(edge_inset, max(0.02, 0.35 * usable_depth))
    return (
        (
            boundary[0] + bounded_inset * forward[0],
            boundary[1] + bounded_inset * forward[1],
        ),
        surface_id,
    )


def _tabletop_regions(
    table: dict[str, Any],
) -> list[tuple[str | None, list[tuple[float, float]]]]:
    """Return actual tabletop support polygons, falling back to the footprint."""
    candidates: list[tuple[str | None, list[Any]]] = []
    for region in table.get("support_regions") or []:
        if isinstance(region, dict):
            polygon = region.get("polygon_world_xy")
            if isinstance(polygon, list):
                candidates.append(
                    (str(region.get("region_id") or "") or None, polygon)
                )
    # 2026-07-14 修改原因：support regions 是可放置面的权威几何；只有提取
    # 失败时才回退到家具整体 footprint，避免桌腿/桌框扩大餐盘目标区域。
    if not candidates:
        footprint = table.get("footprint_world")
        if isinstance(footprint, list):
            candidates.append((None, footprint))
    normalized = [
        (surface_id, [(float(point[0]), float(point[1])) for point in polygon])
        for surface_id, polygon in candidates
        if len(polygon) >= 3
        and all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in polygon)
    ]
    return _coalesce_adjacent_tabletop_regions(normalized)


def _coalesce_adjacent_tabletop_regions(
    regions: list[tuple[str | None, list[tuple[float, float]]]],
) -> list[tuple[str | None, list[tuple[float, float]]]]:
    """Treat connected coplanar tabletop strips as one placement region.

    HSSD support extraction can split a continuous tabletop at mesh seams.  The
    critic must retain the table's outer boundary while ignoring those internal
    seams; otherwise a seat whose front axis lands on the seam is moved to one
    side of the table.  Region records do not always carry height, so the
    geometry-only merge is intentionally limited to rectangles whose projected
    bounding boxes touch or overlap.  The input order is preserved and the first
    region ID remains the stable execution ID used by the manipuland tool.
    """
    if len(regions) < 2:
        return regions

    def bounds(polygon: list[tuple[float, float]]) -> tuple[float, float, float, float]:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        return min(xs), max(xs), min(ys), max(ys)

    def adjacent(
        first: list[tuple[float, float]], second: list[tuple[float, float]]
    ) -> bool:
        a = bounds(first)
        b = bounds(second)
        tolerance = 0.04
        overlap_x = min(a[1], b[1]) - max(a[0], b[0])
        overlap_y = min(a[3], b[3]) - max(a[2], b[2])
        gap_x = max(0.0, max(a[0], b[0]) - min(a[1], b[1]))
        gap_y = max(0.0, max(a[2], b[2]) - min(a[3], b[3]))
        return (overlap_y >= -tolerance and gap_x <= tolerance) or (
            overlap_x >= -tolerance and gap_y <= tolerance
        )

    groups: list[list[tuple[str | None, list[tuple[float, float]]]]] = []
    remaining = list(regions)
    while remaining:
        group = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            for candidate in list(remaining):
                if any(adjacent(item[1], candidate[1]) for item in group):
                    group.append(candidate)
                    remaining.remove(candidate)
                    changed = True
        groups.append(group)

    merged: list[tuple[str | None, list[tuple[float, float]]]] = []
    for group in groups:
        if len(group) == 1:
            merged.extend(group)
            continue
        all_points = [point for _region_id, polygon in group for point in polygon]
        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)
        merged.append(
            (
                group[0][0],
                [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)],
            )
        )
    return merged


def _nearest_tabletop_region_entry(
    table: dict[str, Any],
    origin: tuple[float, float],
    direction: tuple[float, float],
) -> tuple[str | None, tuple[float, float], float] | None:
    candidates: list[tuple[float, str, str | None, tuple[float, float], float]] = []
    for surface_id, polygon in _tabletop_regions(table):
        interval = _ray_polygon_interval(origin, direction, polygon)
        if interval is None:
            continue
        entry_t, exit_t = interval
        if exit_t - entry_t <= 1e-8:
            continue
        entry = (
            origin[0] + entry_t * direction[0],
            origin[1] + entry_t * direction[1],
        )
        candidates.append(
            (entry_t, str(surface_id or ""), surface_id, entry, exit_t - entry_t)
        )
    if not candidates:
        return None
    _distance, _sort_id, surface_id, entry, depth = min(candidates)
    return surface_id, entry, depth


def _ray_polygon_interval(
    origin: tuple[float, float],
    direction: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> tuple[float, float] | None:
    intersections: list[float] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        edge = (end[0] - start[0], end[1] - start[1])
        denominator = direction[0] * edge[1] - direction[1] * edge[0]
        if abs(denominator) <= 1e-9:
            continue
        offset = (start[0] - origin[0], start[1] - origin[1])
        ray_t = (offset[0] * edge[1] - offset[1] * edge[0]) / denominator
        edge_t = (offset[0] * direction[1] - offset[1] * direction[0]) / denominator
        if ray_t >= 0.0 and -1e-8 <= edge_t <= 1.0 + 1e-8:
            intersections.append(ray_t)
    if not intersections:
        return None
    unique = sorted({round(value, 10) for value in intersections})
    if len(unique) < 2:
        return None
    return unique[0], unique[-1]


def _companion_lane_half_width(
    seat: dict[str, Any], item: dict[str, Any], lateral_axis: tuple[float, float]
) -> float:
    seat_span = _projected_span(seat, lateral_axis)
    item_span = _projected_span(item, lateral_axis)
    if seat_span is None:
        seat_span = _short_side(seat) or 0.45
    if item_span is None:
        item_span = _short_side(item) or 0.1
    # 2026-07-14 修改原因：酒杯、餐具等配套物通常在盘子侧边，而不是严格
    # 落在椅子中心线上。旧的 0.5*seat + 0.1*item 容差会把正常的侧向摆放
    # 判成整套餐位失败，模型随后反复横移盘子/酒杯。保留座椅尺度约束，同时
    # 给物件自身尺寸留出比例化的侧向空间；明显跨到另一把椅子的物件仍会失败。
    return max(0.04, 0.55 * seat_span + 0.25 * item_span)


def _nearest_cluster_anchor(
    companion: dict[str, Any],
    anchors: list[dict[str, Any]],
    table_short_side: float,
) -> dict[str, Any] | None:
    center = bbox_center_xy(companion)
    if center is None:
        return None
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for anchor in anchors:
        anchor_center = bbox_center_xy(anchor)
        if anchor_center is None:
            continue
        distance = math.hypot(
            center[0] - anchor_center[0], center[1] - anchor_center[1]
        )
        ranked.append((distance, str(anchor.get("id") or ""), anchor))
    if not ranked:
        return None
    distance, _anchor_id, anchor = min(ranked)
    anchor_scale = _short_side(anchor) or 0.2
    cluster_radius = max(1.5 * anchor_scale, 0.18 * table_short_side)
    return anchor if distance <= cluster_radius else None


def _projected_span(
    obj: dict[str, Any], axis: tuple[float, float]
) -> float | None:
    size = (obj.get("bbox_world") or {}).get("size") or []
    if len(size) < 2:
        return None
    return abs(axis[0]) * float(size[0]) + abs(axis[1]) * float(size[1])


def _short_side(obj: dict[str, Any]) -> float | None:
    size = (obj.get("bbox_world") or {}).get("size") or []
    if len(size) < 2:
        return None
    positive = [float(value) for value in size[:2] if float(value) > 1e-6]
    return min(positive) if positive else None
