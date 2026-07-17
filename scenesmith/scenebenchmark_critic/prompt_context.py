"""Agent-aware prompt context for SceneBenchmark critic results."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from scenesmith.agent_utils.room import AgentType, RoomScene
from scenesmith.scenebenchmark_critic.reports import (
    format_prompt_context as format_full_prompt_context,
)

ISSUE_LABELS = {"fail", "degraded", "unknown"}
LABEL_RANK = {"fail": 3, "degraded": 2, "unknown": 1}
ORIENTATION_CONTRACT_SOURCE = "scenesmith_orientation_contract"
ARCHITECTURE_CATEGORIES = {"wall", "floor", "ceiling"}
COMPUTER_PERIPHERAL_CATEGORIES = {
    "keyboard",
    "mouse",
    "trackpad",
    "touchpad",
}
COMPUTER_SCREEN_CATEGORIES = {
    "display",
    "laptop",
    "monitor",
    "projection_screen",
    "screen",
    "tablet",
    "tablet_computer",
}
MEDIA_CATEGORIES = {
    "entertainment_center",
    "entertainment_center_entertainment",
    "media_console",
    "media_center",
    "projection_screen",
    "screen",
    "television",
    "tv",
    "tv_console",
    "tv_stand",
    "wall_mounted_television",
    "wall_mounted_tv",
}
FURNITURE_RELATIONS = {
    "back_against_wall",
    "bedside_group_alignment",
    # 2026-07-10 修改原因：bedside_pair 的 FD 结果包含床头柜 front 轴平行性，
    # 需要传给 furniture critic 执行位置/朝向修复。
    "bedside_pair",
    "dining_seat_distribution",
    "furniture_faces_furniture",
    "seat_faces_surface",
    "seating_to_media",
    "seating_to_work_surface",
    "side_or_back_against_wall",
    "room_center_alignment",
}
MANIPULAND_RELATIONS = {
    "computer_peripheral_faces_screen",
    "dining_place_setting_alignment",
    "display_faces_user",
    "object_on_support",
    "seating_to_media",
    "seating_to_work_surface",
}
WORKSTATION_CATEGORIES = (
    COMPUTER_PERIPHERAL_CATEGORIES
    | COMPUTER_SCREEN_CATEGORIES
    | {"computer", "notebook_computer"}
)


def format_agent_prompt_context(
    payload: dict[str, Any],
    *,
    scene: RoomScene | None = None,
    agent_type: AgentType | str,
    current_furniture_id: str | None = None,
    max_issues: int = 8,
    debug_output_dir: Path | None = None,
) -> str:
    """Format SceneBenchmark issues for the agent that can act on them."""
    filtered = filter_prompt_results_for_agent(
        payload,
        scene=scene,
        agent_type=agent_type,
        current_furniture_id=current_furniture_id,
    )
    if debug_output_dir is not None:
        _write_debug_context(debug_output_dir, payload, filtered, agent_type)
    if not filtered:
        counted = [
            result
            for result in payload.get("results") or []
            if not _is_ignored_scoring_tier(result)
        ]
        context = (
            "SceneBenchmark geometry critic: no degraded or failed checks relevant "
            f"to the current {_agent_value(agent_type)} agent in "
            f"{len(counted)} counted rule checks."
        )
    else:
        context = format_full_prompt_context({"results": filtered}, max_issues=max_issues)

    # 2026-07-11 修改原因：仅注入 failed/degraded 结果时，已经通过的稳定朝向
    # contract 对 LLM 不可见，critic 会重新解释原始 prompt，把靠墙 guest chair
    # 再次绑定到远处书桌。把当前 contract 明示为权威方向拓扑，避免局部循环。
    contract_context = _format_orientation_contract_context(payload, agent_type)
    if contract_context:
        context = f"{context}\n\n{contract_context}"
    # 2026-07-12 修改原因：仅显示失败项会让视觉 critic 看不到已通过的成组
    # manipuland 库存，并可能把已存在物品误报为全部缺失。显式提供当前支撑家具的
    # 权威 completeness 结果，避免基于单张视图推翻确定性场景数据。
    completeness_context = _format_manipuland_completeness_context(
        payload, agent_type, current_furniture_id
    )
    if completeness_context:
        context = f"{context}\n\n{completeness_context}"
    orientation_context = _format_manipuland_orientation_context(
        payload, agent_type, current_furniture_id
    )
    if orientation_context:
        context = f"{context}\n\n{orientation_context}"
    wall_media_context = _format_wall_media_window_context(
        payload, filtered, agent_type
    )
    if wall_media_context:
        context = f"{context}\n\n{wall_media_context}"
    room_center_context = _format_room_center_contract_context(payload, agent_type)
    if room_center_context:
        context = f"{context}\n\n{room_center_context}"
    bedside_context = _format_bedside_group_contract_context(payload, agent_type)
    if bedside_context:
        context = f"{context}\n\n{bedside_context}"
    return context


def _format_manipuland_completeness_context(
    payload: dict[str, Any],
    agent_type: AgentType | str,
    current_furniture_id: str | None,
) -> str:
    if _agent_value(agent_type) != AgentType.MANIPULAND.value:
        return ""
    furniture_id = str(current_furniture_id or "").strip()
    if not furniture_id:
        return ""
    rows: list[str] = []
    for result in payload.get("results") or []:
        if not isinstance(result, dict):
            continue
        if (
            result.get("metric") != "manipuland_completeness"
            or result.get("label") != "pass"
            or str(result.get("primary_object") or "") != furniture_id
        ):
            continue
        diagnostics = result.get("diagnostics") or {}
        place_count = diagnostics.get("place_count")
        required = diagnostics.get("required_groups") or []
        counts = diagnostics.get("counts") or {}
        required_text = ", ".join(str(item) for item in required)
        counts_text = ", ".join(
            f"{key}={value}" for key, value in sorted(counts.items()) if value
        )
        rows.append(
            f"- `{furniture_id}`: place_count={place_count}; "
            f"required_groups=[{required_text}]; observed_counts=[{counts_text}]"
        )
    if not rows:
        return ""
    return "\n".join(
        [
            "Authoritative deterministic manipuland completeness checks passed:",
            *rows,
            "Do not claim these required groups are absent or request wholesale "
            "removal/regeneration based only on visual ambiguity. You may still "
            "report concrete geometry, spacing, or presentation defects.",
        ]
    )


def _format_manipuland_orientation_context(
    payload: dict[str, Any],
    agent_type: AgentType | str,
    current_furniture_id: str | None,
) -> str:
    # 2026-07-17 修改原因：仅注入 failed/degraded 项会让 critic 看不到已经通过的
    # 显示器朝向，视觉误判仍可能触发反向 rotation；把当前家具的 pass/fail 方向
    # 合同一并注入，并明确 rotation 使用 parent surface-local frame。
    """Expose deterministic display orientation, including passing checks."""
    if _agent_value(agent_type) != AgentType.MANIPULAND.value:
        return ""
    furniture_id = str(current_furniture_id or "").strip()
    if not furniture_id:
        return ""

    objects = _objects_by_id(payload)
    furniture = objects.get(furniture_id)
    if furniture is None:
        return ""
    owned_surface_ids = _surface_ids(furniture)

    rows: list[str] = []
    for result in payload.get("results") or []:
        if not isinstance(result, dict):
            continue
        if result.get("relation_type") != "display_faces_user":
            continue
        if _is_ignored_scoring_tier(result):
            continue

        display_id = str(result.get("primary_object") or "")
        display = objects.get(display_id)
        diagnostics = result.get("diagnostics") or {}
        desk_id = str(diagnostics.get("desk_id") or "")
        if desk_id and desk_id != furniture_id:
            continue
        if not desk_id and _parent_surface_id(display) not in owned_surface_ids:
            continue

        related_ids = _related_ids(result)
        seat_id = str(diagnostics.get("seat_id") or "")
        if not seat_id:
            seat_id = next(
                (
                    object_id
                    for object_id in related_ids
                    if _is_seating(objects.get(object_id))
                ),
                "unknown_seat",
            )
        label = str(result.get("label") or "unknown")
        angle = diagnostics.get("angle_to_user_deg")
        angle_text = f"; angle_to_user={angle}°" if angle is not None else ""
        rows.append(
            f"- `{display_id}` -> `{seat_id}`: label={label}{angle_text}; "
            f"desk=`{furniture_id}`"
        )

    if not rows:
        return ""
    return "\n".join(
        [
            "Authoritative deterministic display-to-user orientation checks:",
            *rows,
            "These checks use the display's world front after composing its parent "
            "surface transform with its local placement angle. A `pass` result is "
            "authoritative: do not rotate that display based on visual ambiguity "
            "alone. For `fail` or `degraded`, issue rotation in the parent surface "
            "local frame and re-evaluate the same check afterward.",
        ]
    )


def _format_orientation_contract_context(
    payload: dict[str, Any], agent_type: AgentType | str
) -> str:
    if _agent_value(agent_type) != AgentType.FURNITURE.value:
        return ""
    rows: list[tuple[str, str, tuple[str, ...]]] = []
    for check in (payload.get("case_pack") or {}).get("checks") or []:
        if not isinstance(check, dict):
            continue
        if check.get("check_source") != ORIENTATION_CONTRACT_SOURCE:
            continue
        subject_id = str(check.get("subject_id") or "").strip()
        relation_type = str(check.get("relation_type") or "").strip()
        target_ids = tuple(
            str(item).strip() for item in check.get("target_ids") or [] if str(item).strip()
        )
        if subject_id and relation_type and target_ids:
            rows.append((subject_id, relation_type, target_ids))
    if not rows:
        return ""
    lines = [
        "Active stable seating orientation contracts (authoritative directional topology):"
    ]
    for subject_id, relation_type, target_ids in sorted(rows):
        targets = ", ".join(target_ids)
        lines.append(f"- `{subject_id}`: `{relation_type}` -> `{targets}`")
        if relation_type in {"seating_to_media", "seating_to_work_surface"}:
            # 2026-07-16 修改原因：客厅椅按就近原则可绑定茶几或 TV；提示 critic
            # 修复当前 contract 选中的室内焦点，避免用宽松 is_facing 结果忽略反向 yaw。
            lines.append(
                "  This chosen functional focus is authoritative for this evaluation: validate the exact "
                "seat/target pair and make the seat point into that activity area. A broad "
                "`is_facing=true` result must not override the stricter SceneBenchmark angle."
            )
        if relation_type == "back_against_wall":
            lines.append(
                "  Validate its front as normal to that wall and pointing into the room. "
                "Do not test or rotate this standalone wall chair toward a desk/table."
            )
    return "\n".join(lines)


def _format_room_center_contract_context(
    payload: dict[str, Any], agent_type: AgentType | str
) -> str:
    """Expose prompt center anchors even when their deterministic check passes."""
    if _agent_value(agent_type) != AgentType.FURNITURE.value:
        return ""
    rows = [
        result
        for result in payload.get("results") or []
        if isinstance(result, dict)
        and result.get("relation_type") == "room_center_alignment"
    ]
    if not rows:
        return ""
    lines = [
        "Authoritative prompt room-center placement contracts:",
        "These anchors must remain near the room center while local accessibility "
        "or clearance issues are repaired.",
    ]
    for result in rows:
        diagnostics = result.get("diagnostics") or {}
        room_center = diagnostics.get("room_center_xy") or []
        object_center = diagnostics.get("object_center_xy") or []
        offset = diagnostics.get("offset_m")
        allowed = diagnostics.get("allowed_offset_m")
        related = ", ".join(result.get("related_objects") or []) or "none"
        lines.append(
            f"- `{result.get('primary_object')}`: label={result.get('label')}; "
            f"target_room_center={room_center}; current_center={object_center}; "
            f"offset={offset}m (allowed={allowed}m); associated_seating={related}."
        )
    lines.append(
        "If a center anchor has accessibility problems, move the anchor and its "
        "associated seating as a coordinated group, recompute table-local seating "
        "slots, and recheck spatial_accessibility and interaction_clearance. Do not "
        "move the anchor alone, and after any checkpoint reset call "
        "get_current_scene_state() before using absolute x/y targets."
    )
    return "\n".join(lines)


def _format_bedside_group_contract_context(
    payload: dict[str, Any], agent_type: AgentType | str
) -> str:
    """Keep a passing bed group rigid while later clearance issues are repaired."""
    if _agent_value(agent_type) != AgentType.FURNITURE.value:
        return ""
    rows = [
        result
        for result in payload.get("results") or []
        if isinstance(result, dict)
        and result.get("relation_type") == "bedside_group_alignment"
        and result.get("label") == "pass"
    ]
    if not rows:
        return ""
    # 2026-07-15 修改原因：通过项默认不会进入家具 critic prompt，后续门净空
    # 修复可能再次单独移动床或床头柜；显式保留已验证的 bed-local 刚性拓扑。
    lines = ["Authoritative stable bedside-group contracts:"]
    for result in rows:
        related = ", ".join(result.get("related_objects") or []) or "none"
        diagnostics = result.get("diagnostics") or {}
        lines.append(
            f"- `{result.get('primary_object')}` with [{related}]: headboard_wall="
            f"{diagnostics.get('headboard_wall') or 'unresolved'}; all tables are in "
            "validated bed-local head-side slots."
        )
    lines.append(
        "Do not move a passed bed or nightstand independently. If a later physics, "
        "accessibility, or door-clearance issue appears, move/rotate the complete "
        "bedside group to a clear wall and reconstruct the same head-end left/right "
        "slots before rechecking."
    )
    return "\n".join(lines)


def _format_wall_media_window_context(
    payload: dict[str, Any],
    filtered: list[dict[str, Any]],
    agent_type: AgentType | str,
) -> str:
    """Add actionable same-wall window guidance for media alignment issues."""
    if _agent_value(agent_type) != AgentType.WALL_MOUNTED.value:
        return ""
    geometry = (payload.get("case_pack") or {}).get("scene_geometry") or {}
    objects = {
        str(obj.get("id")): obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    }
    windows = [
        window
        for window in ((geometry.get("scene_shell") or {}).get("windows") or [])
        if isinstance(window, dict) and window.get("id")
    ]
    if not windows:
        return ""

    rows: list[str] = []
    seen: set[tuple[str, str]] = set()
    for result in filtered:
        relation_type = str(result.get("relation_type") or "")
        if relation_type == "media_over_support_alignment":
            media = objects.get(str(result.get("primary_object") or ""))
            support_ids = _related_ids(result)
            support = objects.get(support_ids[0]) if support_ids else None
            if not _is_wall_mounted_object(media) or not _is_media(media):
                continue
            media_id = str(media.get("id") or result.get("primary_object") or "")
            support_id = str(support.get("id") or support_ids[0]) if support else ""
            # 2026-07-15 修改原因：TV 与 TV stand 可能使用不同命名体系的墙面
            # ID（如 living_room_south / south_wall）；prompt 必须使用 critic
            # 解析出的目标墙和真实窗口，而不是让模型再次选择任意侧墙。
            diagnostics = result.get("diagnostics") or {}
            target_surface_id = str(
                diagnostics.get("target_wall_surface_id") or ""
            )
            target_window_ids = [
                str(item)
                for item in diagnostics.get("target_wall_window_ids") or []
                if str(item)
            ]
            if target_window_ids:
                rows.append(
                    f"- `{media_id}` must move to the wall containing support "
                    f"`{support_id}` (`{target_surface_id}`), but that wall has "
                    f"window(s) {', '.join(target_window_ids)}. First call "
                    "`list_windows()` and repair the target opening in this order: "
                    "shrink it, move it on the same wall, then remove it only if "
                    "necessary; afterward call "
                    f"`align_wall_object_over_support(object_id=\"{media_id}\", "
                    f"support_object_id=\"{support_id}\")`. Do not move the TV "
                    "to an arbitrary side wall to avoid the window."
                )
            elif target_surface_id:
                rows.append(
                    f"- `{media_id}` must be on support `{support_id}`'s wall "
                    f"(`{target_surface_id}`), centered above it. Call "
                    f"`align_wall_object_over_support(object_id=\"{media_id}\", "
                    f"support_object_id=\"{support_id}\")`; do not leave the TV "
                    "on an arbitrary side wall."
                )
            matching_windows = [
                window for window in windows if _objects_share_wall(media, window)
            ]
            if not matching_windows and not target_window_ids:
                rows.append(
                    f"- `{media_id}` is not centered over support `{support_id}`. "
                    "Call `align_wall_object_over_support` after checking openings; "
                    "do not move the TV to an arbitrary side of the wall."
                )
            for window in matching_windows:
                key = (media_id, str(window.get("id") or ""))
                if key in seen:
                    continue
                seen.add(key)
                direction = str(window.get("wall_direction") or "the same wall")
                rows.append(
                    f"- `{media_id}` must be centered over `{support_id}` but shares "
                    f"the {direction} wall with window `{window.get('id')}`. Use the "
                    "wall designer tools in this exact order: `list_windows()`; "
                    f"`resize_window(window_id=\"{window.get('id')}\", width=0.6)`; "
                    "then `align_wall_object_over_support(object_id=\""
                    f"{media_id}\", support_object_id=\"{support_id}\")`. "
                    "If the resized opening still blocks the alignment, use "
                    "`move_window` on the same wall, and only then `remove_window`. "
                    "Never leave the TV shifted sideways merely to avoid the window."
                )
            continue
        if relation_type != "seating_to_media":
            continue
        for target_id in _related_ids(result):
            media = objects.get(target_id)
            if not _is_wall_mounted_object(media) or not _is_media(media):
                continue
            for window in windows:
                if not _objects_share_wall(media, window):
                    continue
                key = (target_id, str(window.get("id") or ""))
                if key in seen:
                    continue
                seen.add(key)
                direction = str(window.get("wall_direction") or "the same wall")
                rows.append(
                    f"- `{target_id}` has a seating-to-media issue and shares the "
                    f"{direction} wall with window `{window.get('id')}`. If that "
                    "opening prevents a centered, direct media view, repair the "
                    "window first in this order: shrink it, move it, then remove "
                    "it; afterward center/rotate the media."
                )
    if not rows:
        return ""
    return "\n".join(
        [
            "Wall-mounted media/window coordination guidance:",
            *rows,
            "Do not mark a window as blocked solely because it shares a wall; change "
            "it only when the opening prevents the required media alignment. The "
            "window edit tools rebuild the wall/SDF geometry and refresh the excluded "
            "regions; call `list_windows()` again after every edit.",
        ]
    )


def _objects_share_wall(media: dict[str, Any], window: dict[str, Any]) -> bool:
    """Return whether a wall-mounted object and shell window use the same wall."""
    window_direction = str(window.get("wall_direction") or "").strip().lower()
    if window_direction not in {"north", "south", "east", "west"}:
        return False
    placement = media.get("placement_info") or {}
    surface_id = str(placement.get("parent_surface_id") or "").lower()
    if window_direction in surface_id:
        return True

    bbox = media.get("bbox_world") or {}
    window_bbox = window.get("bbox") or {}
    omin, omax = bbox.get("min"), bbox.get("max")
    wmin, wmax = window_bbox.get("min"), window_bbox.get("max")
    if not all(isinstance(value, (list, tuple)) for value in (omin, omax, wmin, wmax)):
        return False
    if any(len(value) < 2 for value in (omin, omax, wmin, wmax)):
        return False
    wall_axis = 1 if window_direction in {"north", "south"} else 0
    wall_coord = float(
        wmax[wall_axis]
        if window_direction in {"north", "east"}
        else wmin[wall_axis]
    )
    distance_to_wall = max(
        float(omin[wall_axis]) - wall_coord,
        wall_coord - float(omax[wall_axis]),
        0.0,
    )
    window_depth = abs(float(wmax[wall_axis]) - float(wmin[wall_axis]))
    return distance_to_wall <= max(0.12, window_depth + 0.05)


def filter_prompt_results_for_agent(
    payload: dict[str, Any],
    *,
    scene: RoomScene | None = None,
    agent_type: AgentType | str,
    current_furniture_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return issues that are actionable for the current SceneSmith agent."""
    objects = _objects_by_id(payload)
    agent = _agent_value(agent_type)
    current_furniture_id = str(current_furniture_id or "").strip() or None
    scope = _scope_for_agent(objects, agent, current_furniture_id)

    selected: list[dict[str, Any]] = []
    for result in payload.get("results") or []:
        if not _is_prompt_issue(result):
            continue
        if _is_self_relation(result):
            continue
        if agent == AgentType.FURNITURE.value:
            if not _furniture_issue_is_relevant(result, objects, scope):
                continue
        elif agent == AgentType.WALL_MOUNTED.value:
            if not _wall_mounted_issue_is_relevant(result, objects, scope):
                continue
        elif agent == AgentType.MANIPULAND.value:
            if not _manipuland_issue_is_relevant(result, objects, scope):
                continue
        else:
            continue
        selected.append(result)

    return _dedupe_and_sort(selected)


def _scope_for_agent(
    objects: dict[str, dict[str, Any]],
    agent: str,
    current_furniture_id: str | None,
) -> dict[str, set[str]]:
    if agent == AgentType.FURNITURE.value:
        furniture_ids = {
            object_id
            for object_id, obj in objects.items()
            if _scene_object_type(obj) == "furniture"
        }
        return {
            "object_ids": furniture_ids,
            "support_object_ids": set(),
            "workstation_ids": set(),
        }

    if agent == AgentType.WALL_MOUNTED.value:
        wall_object_ids = {
            object_id
            for object_id, obj in objects.items()
            if _is_wall_mounted_object(obj)
        }
        return {
            "object_ids": wall_object_ids,
            "support_object_ids": set(),
            "workstation_ids": set(),
        }

    support_ids = set()
    manipuland_ids = set()
    workstation_ids = set()
    if current_furniture_id and current_furniture_id in objects:
        support_ids.add(current_furniture_id)
        owned_surfaces = _surface_ids(objects[current_furniture_id])
        for object_id, obj in objects.items():
            if _parent_surface_id(obj) in owned_surfaces:
                manipuland_ids.add(object_id)
                if _is_workstation_object(obj):
                    workstation_ids.add(object_id)
    else:
        for object_id, obj in objects.items():
            if _scene_object_type(obj) == "manipuland":
                manipuland_ids.add(object_id)
                if _is_workstation_object(obj):
                    workstation_ids.add(object_id)
            elif _scene_object_type(obj) == "furniture":
                support_ids.add(object_id)

    if workstation_ids:
        for object_id, obj in objects.items():
            if _is_seating(obj) or _is_work_surface(obj):
                support_ids.add(object_id)
    return {
        "object_ids": manipuland_ids | support_ids | workstation_ids,
        "support_object_ids": support_ids,
        "workstation_ids": workstation_ids,
    }


def _furniture_issue_is_relevant(
    result: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    scope: dict[str, set[str]],
) -> bool:
    relation_type = str(result.get("relation_type") or "")
    subject_id = str(result.get("primary_object") or "")
    related_ids = _related_ids(result)
    if result.get("metric") == "spatial_accessibility":
        return subject_id in scope["object_ids"]
    if relation_type and relation_type not in FURNITURE_RELATIONS:
        return False
    if relation_type == "seating_to_media":
        if not _is_seating(objects.get(subject_id)) or not any(
            _is_media(objects.get(target_id)) for target_id in related_ids
        ):
            return False
    if relation_type == "seating_to_work_surface":
        if not _is_seating(objects.get(subject_id)) or not any(
            _is_seating_work_surface_target(objects.get(target_id))
            for target_id in related_ids
        ):
            return False
    involved = {subject_id, *related_ids}
    if not involved & scope["object_ids"]:
        return False
    return all(
        _scene_object_type(objects.get(object_id)) != "manipuland"
        for object_id in involved
        if object_id in objects
    )


def _manipuland_issue_is_relevant(
    result: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    scope: dict[str, set[str]],
) -> bool:
    relation_type = str(result.get("relation_type") or "")
    subject_id = str(result.get("primary_object") or "")
    related_ids = _related_ids(result)
    involved = {subject_id, *related_ids}

    if result.get("metric") == "spatial_accessibility":
        return subject_id in scope["object_ids"] and _is_workstation_object(
            objects.get(subject_id)
        )
    if relation_type not in MANIPULAND_RELATIONS:
        return False
    if not involved & scope["object_ids"]:
        return False
    if relation_type in {"seating_to_media", "seating_to_work_surface"}:
        return _is_seating(objects.get(subject_id)) and bool(
            set(related_ids) & (scope["workstation_ids"] | scope["support_object_ids"])
        )
    if relation_type == "object_on_support":
        return subject_id in scope["object_ids"] and bool(
            set(related_ids) & scope["support_object_ids"]
        )
    if relation_type == "dining_place_setting_alignment":
        return subject_id in scope["support_object_ids"] and bool(
            set(related_ids) & scope["object_ids"]
        )
    if relation_type == "computer_peripheral_faces_screen":
        return _is_computer_peripheral(objects.get(subject_id)) and any(
            _is_computer_screen(objects.get(target_id)) for target_id in related_ids
        )
    if relation_type == "display_faces_user":
        return _is_computer_screen(objects.get(subject_id)) and bool(
            set(related_ids) & scope["object_ids"]
        )
    return False


def _wall_mounted_issue_is_relevant(
    result: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    scope: dict[str, set[str]],
) -> bool:
    """Select critic issues actionable by the wall-mounted agent."""
    involved = {
        str(result.get("primary_object") or ""),
        *_related_ids(result),
    }
    # Window IDs live in scene_shell rather than scene_geometry.objects. A
    # window-clearance failure is therefore relevant when one of its related
    # blockers is a current wall-mounted object.
    if str(result.get("check_id") or "").startswith("window_clearance__"):
        return bool(involved & scope["object_ids"])
    if result.get("metric") == "interaction_clearance":
        return bool(involved & scope["object_ids"])
    return bool(involved & scope["object_ids"])


def _dedupe_and_sort(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, tuple[str, ...]], dict[str, Any]] = {}
    for result in results:
        key = (
            str(result.get("metric") or ""),
            str(result.get("relation_type") or ""),
            str(result.get("primary_object") or ""),
            tuple(sorted(_related_ids(result))),
        )
        existing = by_key.get(key)
        if existing is None or _result_rank(result) > _result_rank(existing):
            by_key[key] = result
    return sorted(by_key.values(), key=_sort_key)


def _sort_key(result: dict[str, Any]) -> tuple[int, str, str]:
    return (
        -LABEL_RANK.get(str(result.get("label") or ""), 0),
        str(result.get("metric") or ""),
        str(result.get("check_id") or ""),
    )


def _result_rank(result: dict[str, Any]) -> tuple[int, float]:
    try:
        confidence = float(result.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return LABEL_RANK.get(str(result.get("label") or ""), 0), confidence


def _objects_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    geometry = ((payload.get("case_pack") or {}).get("scene_geometry") or {})
    return {
        str(obj.get("id") or ""): obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    }


def _is_prompt_issue(result: dict[str, Any]) -> bool:
    return (
        result.get("label") in ISSUE_LABELS
        and not _is_ignored_scoring_tier(result)
    )


def _is_ignored_scoring_tier(result: dict[str, Any]) -> bool:
    return str(result.get("scoring_tier") or "").strip().lower() == "ignored"


def _is_self_relation(result: dict[str, Any]) -> bool:
    subject_id = str(result.get("primary_object") or "")
    return bool(subject_id) and subject_id in _related_ids(result)


def _related_ids(result: dict[str, Any]) -> list[str]:
    return [str(item) for item in (result.get("related_objects") or []) if str(item)]


def _scene_object_type(obj: dict[str, Any] | None) -> str:
    if not obj:
        return ""
    hints = obj.get("functional_hints") or {}
    hinted_type = str(hints.get("scene_object_type") or "").strip().lower()
    declared_type = str(obj.get("object_type") or "").strip().lower()
    # 2026-07-14 修改原因：HSSD/资产标注偶尔把 wall-mounted TV 的 functional
    # hint 标成 furniture；只要任一来源明确声明壁挂物，wall agent 就必须收到
    # 相关 critic issue，不能被错误 hint 覆盖。
    if {hinted_type, declared_type} & {"wall_mounted", "wall-mounted", "mounted"}:
        return AgentType.WALL_MOUNTED.value
    return hinted_type or declared_type


def _is_wall_mounted_object(obj: dict[str, Any] | None) -> bool:
    if not obj:
        return False
    hints = obj.get("functional_hints") or {}
    object_types = {
        str(value).strip().lower()
        for value in (hints.get("scene_object_type"), obj.get("object_type"))
        if value
    }
    return bool(object_types & {"wall_mounted", "wall-mounted", "mounted"})


def _category(obj: dict[str, Any] | None) -> str:
    return str(
        (obj or {}).get("category_norm") or (obj or {}).get("category") or ""
    ).strip().lower()


def _category_text(obj: dict[str, Any] | None) -> str:
    obj = obj or {}
    return " ".join(
        str(obj.get(key) or "").strip().lower().replace("-", "_").replace(" ", "_")
        for key in ("id", "name", "category", "category_norm", "description")
    )


def _is_seating(obj: dict[str, Any] | None) -> bool:
    hints = (obj or {}).get("functional_hints") or {}
    category_group = str(hints.get("category_group") or "").strip().lower()
    return category_group == "seating" or _category(obj) in {
        "armchair",
        "bench",
        "chair",
        "dining_chair",
        "office_chair",
        "sofa",
        "stool",
    }


def _is_work_surface(obj: dict[str, Any] | None) -> bool:
    hints = (obj or {}).get("functional_hints") or {}
    return str(hints.get("category_group") or "").strip().lower() in {
        "storage_surface",
        "work_surface",
    } or _category(obj) in {"desk", "table", "counter", "island"}


def _is_seating_work_surface_target(obj: dict[str, Any] | None) -> bool:
    hints = (obj or {}).get("functional_hints") or {}
    category_group = str(hints.get("category_group") or "").strip().lower()
    category = _category(obj)
    text = _category_text(obj)
    if category_group == "work_surface":
        return True
    if category in {
        "coffee_table",
        "counter",
        "desk",
        "dining_table",
        "island",
        "table",
        "work_table",
        "writing_desk",
    }:
        return True
    return any(
        token in text
        for token in ("desk", "dining_table", "work_table", "writing_desk")
    )


def _is_workstation_object(obj: dict[str, Any] | None) -> bool:
    return _is_computer_screen(obj) or _is_computer_peripheral(obj)


def _is_computer_screen(obj: dict[str, Any] | None) -> bool:
    category = _category(obj)
    text = _category_text(obj)
    return category in COMPUTER_SCREEN_CATEGORIES or any(
        token in text for token in ("computer_monitor", "display", "screen")
    )


def _is_media(obj: dict[str, Any] | None) -> bool:
    category = _category(obj)
    text = _category_text(obj)
    return category in MEDIA_CATEGORIES or any(
        token in text
        for token in (
            "entertainment_center",
            "media",
            "projector",
            "television",
            "tv",
            "tv_stand",
        )
    )


def _is_computer_peripheral(obj: dict[str, Any] | None) -> bool:
    category = _category(obj)
    text = _category_text(obj)
    return category in COMPUTER_PERIPHERAL_CATEGORIES or any(
        token in text for token in ("keyboard", "mouse", "trackpad", "touchpad")
    )


def _surface_ids(obj: dict[str, Any]) -> set[str]:
    return {
        str(region.get("region_id") or "")
        for region in obj.get("support_regions") or []
        if isinstance(region, dict) and region.get("region_id")
    }


def _parent_surface_id(obj: dict[str, Any]) -> str:
    placement = obj.get("placement_info") or {}
    return str(placement.get("parent_surface_id") or "")


def _agent_value(agent_type: AgentType | str) -> str:
    if isinstance(agent_type, AgentType):
        return agent_type.value
    return str(agent_type or "").strip().lower()


def _write_debug_context(
    output_dir: Path,
    payload: dict[str, Any],
    filtered: list[dict[str, Any]],
    agent_type: AgentType | str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_issues = [
        result for result in payload.get("results") or [] if _is_prompt_issue(result)
    ]
    filtered_ids = {str(result.get("check_id") or "") for result in filtered}
    filtered_out = [
        {
            "check_id": result.get("check_id"),
            "reason": _debug_filter_reason(result, filtered_ids),
        }
        for result in raw_issues
        if str(result.get("check_id") or "") not in filtered_ids
    ]
    debug_payload = {
        "schema_version": "scenesmith.scenebenchmark_critic.prompt_context_debug.v1",
        "agent_type": _agent_value(agent_type),
        "raw_issue_count": len(raw_issues),
        "filtered_issue_count": len(filtered),
        "raw_issue_ids": [result.get("check_id") for result in raw_issues],
        "filtered_issue_ids": sorted(filtered_ids),
        "filtered_out": filtered_out,
    }
    (output_dir / "scenebenchmark_prompt_context_debug.json").write_text(
        json.dumps(debug_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _debug_filter_reason(result: dict[str, Any], filtered_ids: set[str]) -> str:
    check_id = str(result.get("check_id") or "")
    if check_id in filtered_ids:
        return "kept"
    if _is_self_relation(result):
        return "self_relation"
    relation_type = str(result.get("relation_type") or "")
    if relation_type:
        return "not_relevant_to_current_agent_scope_or_relation_policy"
    return "not_relevant_to_current_agent_scope"
