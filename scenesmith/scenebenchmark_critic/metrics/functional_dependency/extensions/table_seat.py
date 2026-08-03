"""Prompt-authorized chair distribution checks for rectangular tables.

The geometry rule applies to any rectangular table whenever the prompt
explicitly requests a table-edge seating topology.
"""

from __future__ import annotations

import itertools
import math
import re

from typing import Any

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
    front_vector,
    object_footprint_polygon,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.manipuland_completeness import (
    _bbox_gap_xy,
    _footprint_short_side,
    _is_dining_seat,
    _is_dining_table,
    _object_identity_text,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.seat_surface_assignment import (
    is_dining_context,
)
from scenesmith.scenebenchmark_critic.intent_contract import (
    bound_ids,
    contract_constraints,
    contract_relation_requested,
)

RELATION_TYPE = "table_seat_distribution"
_ONE_PER_EDGE_TABLE_GAP_M = 0.05


def evaluate_table_seat_distribution(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check that chairs are centered or evenly spaced along each table edge."""
    objects = [
        obj
        for obj in ((case_pack.get("scene_geometry") or {}).get("objects") or [])
        if isinstance(obj, dict) and obj.get("id")
    ]
    objects_by_id = {str(obj["id"]): obj for obj in objects}
    task_instruction = str(case_pack.get("task_instruction") or "")
    requested_one_per_edge = _requests_one_seat_per_edge(task_instruction)
    dining_context = is_dining_context(
        task_instruction=task_instruction,
        room_type=str(case_pack.get("room_type") or ""),
    )
    if not (
        contract_relation_requested(case_pack, "one_per_side")
        or requested_one_per_edge
        or dining_context
    ):
        # Outside an explicit dining context, only a prompt-authorized seating
        # topology may enable this table-edge rule.
        return []
    long_side_distribution = _requests_long_side_distribution(case_pack)
    single_long_side_distribution = _requests_single_long_side_distribution(case_pack)
    single_short_side_seat_distribution = _requests_single_short_side_seat_distribution(
        case_pack
    )
    # Legacy direct evaluator callers do not always carry a compiled contract.
    # Keep their explicit four-edge dining prompt behavior, while the new
    # long-side contract remains distinct from one-seat-per-edge positioning.
    one_per_edge = requested_one_per_edge and not long_side_distribution
    tables = [
        obj
        for obj in objects
        if (
            _is_dining_table(obj)
            or (dining_context and _is_generic_table(obj))
            or (long_side_distribution and _is_generic_table(obj))
        )
        and not _is_round_table(obj)
    ]
    seats_by_table = _positionally_associated_seats(
        tables,
        objects_by_id,
        include_unassociated=one_per_edge or long_side_distribution,
    )
    constraints = contract_constraints(
        case_pack,
        relations=("one_per_side",),
        include_auxiliary=False,
    )
    if long_side_distribution:
        bound_table_ids = {
            table_id
            for constraint in constraints
            for table_id in bound_ids(constraint.get("targets"), objects)
        }
        if bound_table_ids:
            tables = [obj for obj in tables if str(obj["id"]) in bound_table_ids]
    results: list[dict[str, Any]] = []
    for table in tables:
        result = _evaluate_table(
            table,
            seats_by_table.get(str(table["id"]), []),
            enforce_one_per_edge=one_per_edge,
            enforce_long_side_distribution=long_side_distribution,
            enforce_single_long_side_distribution=single_long_side_distribution,
            enforce_single_short_side_seat_distribution=(
                single_short_side_seat_distribution
            ),
        )
        if result is not None:
            table_id = str(table["id"])
            constraint = next(
                (
                    row
                    for row in constraints
                    if table_id in bound_ids(row.get("targets"), objects)
                ),
                None,
            )
            if constraint is not None:
                result["evidence"]["intent_constraint"] = constraint
            results.append(result)
    return results


def _evaluate_table(
    table: dict[str, Any],
    seats: list[dict[str, Any]],
    *,
    enforce_one_per_edge: bool = False,
    enforce_long_side_distribution: bool = False,
    enforce_single_long_side_distribution: bool = False,
    enforce_single_short_side_seat_distribution: bool = False,
) -> dict[str, Any] | None:
    center = bbox_center_xy(table)
    seats = [seat for seat in seats if "bench" not in _object_identity_text(seat)]
    if center is None or not seats:
        return None
    yaw = math.radians(float(table.get("yaw_deg") or 0.0))
    tangent_x = (math.cos(yaw), math.sin(yaw))
    tangent_y = (-math.sin(yaw), math.cos(yaw))
    width = _footprint_extent(table, tangent_x)
    depth = _footprint_extent(table, tangent_y)
    if width is None or depth is None or min(width, depth) <= 1e-6:
        return None
    seat_local_positions: list[tuple[dict[str, Any], float, float]] = []
    for seat in seats:
        seat_center = bbox_center_xy(seat)
        if seat_center is None:
            continue
        dx, dy = seat_center[0] - center[0], seat_center[1] - center[1]
        seat_local_positions.append(
            (
                seat,
                dx * tangent_x[0] + dy * tangent_x[1],
                dx * tangent_y[0] + dy * tangent_y[1],
            )
        )

    grouped: dict[str, list[tuple[dict[str, Any], float]]] = {
        "left": [],
        "right": [],
        "front": [],
        "back": [],
    }
    if enforce_one_per_edge and len(seat_local_positions) == 4:
        edges = ("left", "right", "front", "back")
        assignment = min(
            itertools.permutations(edges),
            key=lambda candidate: sum(
                _edge_distance(local_x, local_y, edge, width=width, depth=depth)
                for (_, local_x, local_y), edge in zip(seat_local_positions, candidate)
            ),
        )
        for (seat, local_x, local_y), edge in zip(seat_local_positions, assignment):
            grouped[edge].append(
                (seat, local_y if edge in {"left", "right"} else local_x)
            )
    else:
        for seat, local_x, local_y in seat_local_positions:
            # 2026-07-15 修改原因：座椅因碰撞或净空向外拉开后，按“到无限延长
            # 桌边直线的距离”会把短边椅误归到长边。改用有限桌边线段距离，确保
            # 桌角之外的座椅仍由其实际相邻桌边负责，且不依赖某个场景的绝对尺寸。
            edge, tangent_position = _nearest_table_edge(
                local_x, local_y, width=width, depth=depth
            )
            grouped[edge].append((seat, tangent_position))

    diagnostics: list[dict[str, Any]] = []
    failures: list[str] = []
    long_edges = {"front", "back"} if width >= depth else {"left", "right"}
    topology_failures: list[str] = []
    if enforce_long_side_distribution:
        short_edges = sorted(set(grouped) - long_edges)
        short_edge_seats = {
            edge: sorted(str(seat["id"]) for seat, _position in grouped[edge])
            for edge in short_edges
        }
        occupied_short_edges = [edge for edge in short_edges if grouped[edge]]
        if enforce_single_short_side_seat_distribution:
            if (
                len(occupied_short_edges) != 1
                or len(short_edge_seats[occupied_short_edges[0]]) != 1
            ):
                topology_failures.append(
                    "chairs must occupy exactly one short table edge with one chair; "
                    "keep the opposite short edge clear"
                )
        elif any(short_edge_seats.values()):
            seat_ids = sorted(
                seat_id for seats in short_edge_seats.values() for seat_id in seats
            )
            topology_failures.append(
                "chairs must occupy only the two long table edges; move "
                + ", ".join(f"`{seat_id}`" for seat_id in seat_ids)
                + " to a long edge"
            )
        occupied_long_edges = [edge for edge in sorted(long_edges) if grouped[edge]]
        long_edge_counts = [len(grouped[edge]) for edge in occupied_long_edges]
        if enforce_single_long_side_distribution and len(occupied_long_edges) != 1:
            topology_failures.append(
                "chairs must occupy exactly one long table edge; keep the opposite long edge clear"
            )
        elif not enforce_single_long_side_distribution and (
            len(long_edge_counts) != 2
            or not all(long_edge_counts)
            or long_edge_counts[0] != long_edge_counts[1]
        ):
            topology_failures.append(
                "chairs must be split into equal nonzero groups on the two long table edges"
            )
    failures.extend(topology_failures)
    for edge, members in grouped.items():
        if not members:
            continue
        edge_length = depth if edge in {"left", "right"} else width
        count = len(members)
        # 2026-07-21 修改原因：长边多椅不能只在扣除两端边距后均分椅子
        # 中心。那会把两把椅子推到桌边附近，使中间空隙远大于两端空隙。
        # 将整条有限桌边分成 count 个等长段，每把椅子取自己段落的中心：
        # 对两把等宽椅子，端部空隙为 g，中间空隙自然为 2g；count > 2
        # 时同一公式继续适用，不依赖固定桌长或固定座椅数量。
        slots = _equal_edge_segment_slots(edge_length, count)
        actual = sorted(members, key=lambda row: (row[1], str(row[0]["id"])))
        for segment_index, ((seat, position), slot) in enumerate(zip(actual, slots)):
            seat_center = bbox_center_xy(seat)
            if seat_center is None:
                continue
            chair_span = _seat_tangent_span(seat, edge, yaw)
            deviation = abs(position - slot)
            allowed = max(0.08, min(0.35 * chair_span, 0.08 * edge_length))
            passed = deviation <= allowed
            # 2026-07-14 修改原因：多椅同边可以平行朝向桌边而不必都斜指桌心；
            # 单椅边位才要求严格正对。2026-07-15 的有限边归类会保证拉远后的
            # 短边椅仍各自落在单椅边位，不再因错误分组漏掉 180° 翻转。
            facing_error = _seat_facing_error_deg(seat, center) if count == 1 else None
            facing_passed = facing_error is None or facing_error <= 10.0
            edge_tangent = tangent_y if edge in {"left", "right"} else tangent_x
            target_xy = (
                seat_center[0] + (slot - position) * edge_tangent[0],
                seat_center[1] + (slot - position) * edge_tangent[1],
            )
            normal_deviation: float | None = None
            target_normal_position: float | None = None
            allowed_normal_deviation: float | None = None
            if enforce_one_per_edge:
                # A "one on each side" arrangement has one unambiguous slot per
                # edge.  Its target must include the outward normal as well as the
                # along-edge coordinate; otherwise a remote chair is only slid
                # parallel to a table edge and can remain against a wall or land
                # on the tabletop.
                target_local_x, target_local_y = _one_per_edge_target_local(
                    seat,
                    edge,
                    slot,
                    width=width,
                    depth=depth,
                    tangent_x=tangent_x,
                    tangent_y=tangent_y,
                )
                target_xy = (
                    center[0]
                    + target_local_x * tangent_x[0]
                    + target_local_y * tangent_y[0],
                    center[1]
                    + target_local_x * tangent_x[1]
                    + target_local_y * tangent_y[1],
                )
                current_dx = seat_center[0] - center[0]
                current_dy = seat_center[1] - center[1]
                current_local_x = current_dx * tangent_x[0] + current_dy * tangent_x[1]
                current_local_y = current_dx * tangent_y[0] + current_dy * tangent_y[1]
                current_normal_position = (
                    current_local_x if edge in {"left", "right"} else current_local_y
                )
                target_normal_position = (
                    target_local_x if edge in {"left", "right"} else target_local_y
                )
                normal_deviation = abs(current_normal_position - target_normal_position)
                allowed_normal_deviation = max(0.08, 0.2 * chair_span)
                passed = passed and normal_deviation <= allowed_normal_deviation
            seat_front = front_vector(seat)
            diagnostics.append(
                {
                    "seat_id": str(seat["id"]),
                    "edge": edge,
                    "segment_index": segment_index,
                    "segment_count": count,
                    "segment_length_m": round(edge_length / count, 4),
                    "tangent_position_m": round(position, 4),
                    "target_position_m": round(slot, 4),
                    "deviation_m": round(deviation, 4),
                    "allowed_deviation_m": round(allowed, 4),
                    "normal_deviation_m": (
                        round(normal_deviation, 4)
                        if normal_deviation is not None
                        else None
                    ),
                    "target_normal_position_m": (
                        round(target_normal_position, 4)
                        if target_normal_position is not None
                        else None
                    ),
                    "allowed_normal_deviation_m": (
                        round(allowed_normal_deviation, 4)
                        if allowed_normal_deviation is not None
                        else None
                    ),
                    "aligned": passed,
                    "target_center_xy_m": [round(value, 6) for value in target_xy],
                    "edge_tangent_xy": [round(value, 6) for value in edge_tangent],
                    "current_front_xy": [round(value, 6) for value in seat_front],
                    "facing_target_xy_m": [round(value, 6) for value in center],
                    "facing_error_deg": (
                        round(facing_error, 2) if facing_error is not None else None
                    ),
                    "facing_allowed_error_deg": 10.0,
                    "facing_aligned": facing_passed,
                }
            )
            if not passed:
                if deviation > allowed:
                    direction = "positive" if slot > position else "negative"
                    failures.append(
                        f"`{seat['id']}` on the {edge} edge is {deviation:.2f}m from "
                        f"its equal-segment slot; move it in the {direction} edge direction"
                    )
                if (
                    normal_deviation is not None
                    and allowed_normal_deviation is not None
                    and normal_deviation > allowed_normal_deviation
                ):
                    failures.append(
                        f"`{seat['id']}` on the {edge} edge is {normal_deviation:.2f}m "
                        "away from its table-edge clearance slot; move it perpendicular "
                        "to that edge, outside the table footprint"
                    )
            if not facing_passed:
                failures.append(
                    f"`{seat['id']}` on the {edge} edge is rotated {facing_error:.1f}° "
                    "away from the table center; rotate it in place so its front normal "
                    "faces the table, preserving its table-edge slot and clearance"
                )
    if not diagnostics:
        return None
    topology_repair_slots = (
        _topology_repair_slots(
            seat_local_positions,
            center=center,
            width=width,
            depth=depth,
            tangent_x=tangent_x,
            tangent_y=tangent_y,
            long_edges=long_edges,
            single_long_side=enforce_single_long_side_distribution,
            single_short_side=enforce_single_short_side_seat_distribution,
        )
        if topology_failures
        else []
    )
    table_id = str(table["id"])
    related = sorted(str(seat["id"]) for seat in seats)
    failed = bool(failures)
    reason = (
        "Chairs on each rectangular table edge must be centered when alone "
        "and centered in equal edge segments when multiple chairs share the edge. "
        "For two chairs, this keeps the two end gaps equal and the middle gap "
        "approximately twice either end gap; apply the same equal-segment rule "
        "for any number of chairs. "
        "Use an exact table-local slot and do not use generic "
        "center snapping or shift the chair along the edge normal to resolve a "
        "door conflict; move the table or door-compatible layout instead. "
        + "; ".join(failures)
        if failed
        else "Chairs are centered when alone and centered in equal segments along their respective table edges."
    )
    return {
        "check_id": f"fd_{table_id}_{RELATION_TYPE}",
        "metric": "functional_dependency",
        "label": "fail" if failed else "pass",
        "confidence": 0.93 if failed else 0.89,
        "primary_object": table_id,
        "related_objects": related,
        "selected_related_objects": related,
        "blocking_objects": [],
        "relation_type": RELATION_TYPE,
        "reason": reason,
        "diagnostics": {
            "seat_slots": diagnostics,
            "coordinated_one_per_edge": enforce_one_per_edge,
            "long_side_distribution": enforce_long_side_distribution,
            "single_long_side_distribution": enforce_single_long_side_distribution,
            "single_short_side_seat_distribution": (
                enforce_single_short_side_seat_distribution
            ),
            "long_edges": sorted(long_edges),
            "topology_repair_slots": topology_repair_slots,
        },
        "evidence": {"distribution": "table_local_equal_edge_segments"},
        "evaluation_source": "scenesmith_table_seat_distribution",
        "scoring_tier": "core",
    }


def _footprint_extent(obj: dict[str, Any], axis: tuple[float, float]) -> float | None:
    polygon = object_footprint_polygon(obj) or []
    if polygon:
        projections = [x * axis[0] + y * axis[1] for x, y in polygon]
        return max(projections) - min(projections)
    size = (obj.get("bbox_world") or {}).get("size") or []
    if len(size) < 2:
        return None
    return abs(axis[0]) * float(size[0]) + abs(axis[1]) * float(size[1])


def _equal_edge_segment_slots(edge_length: float, count: int) -> list[float]:
    """Return local tangent centers for ``count`` equal segments of an edge."""
    if count <= 0 or edge_length <= 1e-6:
        return []
    segment_length = edge_length / count
    return [
        -edge_length / 2.0 + (index + 0.5) * segment_length for index in range(count)
    ]


def _topology_repair_slots(
    seats: list[tuple[dict[str, Any], float, float]],
    *,
    center: tuple[float, float],
    width: float,
    depth: float,
    tangent_x: tuple[float, float],
    tangent_y: tuple[float, float],
    long_edges: set[str],
    single_long_side: bool,
    single_short_side: bool,
) -> list[dict[str, Any]]:
    """Return the minimum-motion complete seating topology for deterministic repair."""
    long_edges_ordered = sorted(long_edges)
    if single_long_side:
        long_edges_ordered = long_edges_ordered[:1]
    short_edges = sorted({"left", "right", "front", "back"} - long_edges)
    long_seat_count = len(seats) - int(single_short_side)
    if (
        not seats
        or long_seat_count <= 0
        or long_seat_count % len(long_edges_ordered) != 0
    ):
        return []
    per_long_edge = long_seat_count // len(long_edges_ordered)
    desired_edges = [
        (edge, slot)
        for edge in long_edges_ordered
        for slot in _equal_edge_segment_slots(
            depth if edge in {"left", "right"} else width, per_long_edge
        )
    ]
    short_edge_options = short_edges if single_short_side else [None]
    best: tuple[float, tuple[tuple[str, float], ...]] | None = None
    for short_edge in short_edge_options:
        edges = list(desired_edges)
        if short_edge is not None:
            edges.append((short_edge, 0.0))
        for assignment in itertools.permutations(edges):
            distance = 0.0
            for (seat, local_x, local_y), (edge, slot) in zip(seats, assignment):
                target_x, target_y = _one_per_edge_target_local(
                    seat,
                    edge,
                    slot,
                    width=width,
                    depth=depth,
                    tangent_x=tangent_x,
                    tangent_y=tangent_y,
                )
                distance += (local_x - target_x) ** 2 + (local_y - target_y) ** 2
            candidate = (distance, assignment)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return []
    slots: list[dict[str, Any]] = []
    for (seat, _local_x, _local_y), (edge, slot) in zip(seats, best[1]):
        target_x, target_y = _one_per_edge_target_local(
            seat,
            edge,
            slot,
            width=width,
            depth=depth,
            tangent_x=tangent_x,
            tangent_y=tangent_y,
        )
        target_xy = (
            center[0] + target_x * tangent_x[0] + target_y * tangent_y[0],
            center[1] + target_x * tangent_x[1] + target_y * tangent_y[1],
        )
        slots.append(
            {
                "seat_id": str(seat["id"]),
                "edge": edge,
                "target_center_xy_m": [round(value, 6) for value in target_xy],
                "current_front_xy": [round(value, 6) for value in front_vector(seat)],
                "facing_target_xy_m": [round(value, 6) for value in center],
            }
        )
    return slots


def _requests_long_side_distribution(case_pack: dict[str, Any]) -> bool:
    """Return whether the compiled table-seat contract names long sides."""
    return any(
        re.search(r"\blong\s+(?:side|edge)s?\b", str(row.get("evidence_span") or ""))
        for row in contract_constraints(
            case_pack, relations=("one_per_side",), include_auxiliary=False
        )
    )


def _requests_single_long_side_distribution(case_pack: dict[str, Any]) -> bool:
    """Return whether the compiled contract requires one occupied long edge."""
    instruction = _table_seat_contract_evidence(case_pack)
    return _requests_long_side_distribution(case_pack) and bool(
        re.search(r"\b(?:one|single)\s+long\s+(?:side|edge)\b", instruction)
        and re.search(
            r"\bopposite\s+long\s+(?:side|edge)\b[^.]{0,80}\b(?:free|clear)\b",
            instruction,
        )
    )


def _requests_single_short_side_seat_distribution(case_pack: dict[str, Any]) -> bool:
    """Return whether the compiled contract requires one occupied short edge."""
    instruction = _table_seat_contract_evidence(case_pack)
    return _requests_long_side_distribution(case_pack) and bool(
        re.search(
            r"\b(?:one|single)(?:\s+[a-z]+){0,3}\s+(?:chair|seat)s?\b"
            r"[^.]{0,100}\b(?:on|along|at)\s+(?:one\s+)?short\s+(?:side|edge)\b",
            instruction,
        )
        and re.search(
            r"\bopposite\s+short\s+(?:side|edge)\b[^.]{0,80}\b(?:free|clear)\b",
            instruction,
        )
    )


def _table_seat_contract_evidence(case_pack: dict[str, Any]) -> str:
    """Join model-authored evidence for the table-seat topology contract."""
    return " ".join(
        str(row.get("evidence_span") or "").lower()
        for row in contract_constraints(
            case_pack, relations=("one_per_side",), include_auxiliary=False
        )
    )


def _seat_tangent_span(seat: dict[str, Any], edge: str, table_yaw: float) -> float:
    size = (seat.get("bbox_world") or {}).get("size") or []
    if len(size) < 2:
        return 0.45
    axis = (
        (math.cos(table_yaw), math.sin(table_yaw))
        if edge in {"front", "back"}
        else (-math.sin(table_yaw), math.cos(table_yaw))
    )
    return max(0.2, abs(axis[0]) * float(size[0]) + abs(axis[1]) * float(size[1]))


def _one_per_edge_target_local(
    seat: dict[str, Any],
    edge: str,
    tangent_slot: float,
    *,
    width: float,
    depth: float,
    tangent_x: tuple[float, float],
    tangent_y: tuple[float, float],
) -> tuple[float, float]:
    """Return the complete table-local center for a one-seat edge slot."""
    normal_axis = tangent_x if edge in {"left", "right"} else tangent_y
    seat_normal_span = _footprint_extent(seat, normal_axis)
    if seat_normal_span is None:
        seat_normal_span = 0.5
    outward_offset = seat_normal_span / 2.0 + _ONE_PER_EDGE_TABLE_GAP_M
    if edge == "left":
        return -width / 2.0 - outward_offset, tangent_slot
    if edge == "right":
        return width / 2.0 + outward_offset, tangent_slot
    if edge == "front":
        return tangent_slot, -depth / 2.0 - outward_offset
    return tangent_slot, depth / 2.0 + outward_offset


def _positionally_associated_seats(
    tables: list[dict[str, Any]],
    objects_by_id: dict[str, dict[str, Any]],
    *,
    include_unassociated: bool = False,
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
        elif include_unassociated and len(tables) == 1:
            associated[str(tables[0]["id"])].append(seat)
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


def _edge_distance(
    local_x: float,
    local_y: float,
    edge: str,
    *,
    width: float,
    depth: float,
) -> float:
    half_width = width / 2.0
    half_depth = depth / 2.0
    if edge == "left":
        return math.hypot(
            local_x + half_width,
            local_y - min(max(local_y, -half_depth), half_depth),
        )
    if edge == "right":
        return math.hypot(
            local_x - half_width,
            local_y - min(max(local_y, -half_depth), half_depth),
        )
    if edge == "front":
        return math.hypot(
            local_x - min(max(local_x, -half_width), half_width),
            local_y + half_depth,
        )
    return math.hypot(
        local_x - min(max(local_x, -half_width), half_width),
        local_y - half_depth,
    )


def _requests_one_seat_per_edge(prompt: str) -> bool:
    text = prompt.lower().replace("_", " ")
    return bool(
        re.search(
            r"\b(?:one|1)\s+"
            r"(?:(?:chair|seat|stool)\s+)?"
            r"on\s+each(?:\s+of\s+the)?\s+"
            r"(?:(?:four|4)\s+)?(?:side|edge)s?\b",
            text,
        )
    )


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


def _is_generic_table(obj: dict[str, Any]) -> bool:
    """Recognize a plain table only after the prompt establishes dining context."""
    text = _object_identity_text(obj)
    if "table" not in text:
        return False
    return not any(
        token in text
        for token in (
            "bar table",
            "coffee table",
            "console table",
            "desk",
            "end table",
            "nightstand",
            "side table",
        )
    )
