"""Stable orientation contracts for SceneBenchmark functional dependencies."""

from __future__ import annotations

import logging
import re

from typing import Any

from scenesmith.agent_utils.room import RoomScene
from scenesmith.scenebenchmark_critic.config import CriticConfig
from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_gap_xy,
    distance_xy,
    object_affordances,
    object_category,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.profiles import (
    object_function_profile,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.seat_surface_assignment import (
    ASSIGNMENT_SOURCE,
    SeatSurfaceAssignment,
    assign_work_seats_to_surfaces,
    room_bounds_from_case_pack,
    work_seat_candidates,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.semantics import (
    _classroom_student_role,
    _is_actionable_seating_surface_pair,
    _is_classroom_student_pair,
)

console_logger = logging.getLogger(__name__)

CONTRACT_CHECK_SOURCE = "scenesmith_orientation_contract"
CONTRACT_ATTR = "_scenebenchmark_orientation_contracts"

SEATING_RELATIONS = {"seating_to_media", "seating_to_work_surface"}
CONTRACT_RELATIONS = SEATING_RELATIONS | {
    "back_against_wall",
    "furniture_faces_furniture",
}
CONFLICTING_ORIENTATION_RELATIONS = CONTRACT_RELATIONS | {
    "furniture_faces_furniture",
    "seat_faces_surface",
    "workstation",
}
MEDIA_CATEGORIES = {
    "display",
    "display_board",
    "entertainment_center",
    "media_console",
    "monitor",
    "projection_screen",
    "screen",
    "television",
    "tv",
    "tv_stand",
}
MEDIA_TEXT_HINTS = (
    "display",
    "entertainment",
    "media",
    "monitor",
    "projector",
    "screen",
    "television",
    "tv",
)
MEDIA_REJECT_HINTS = (
    "coffee table",
    "side table",
    "end table",
    "lamp",
    "remote control",
    "remote_control",
)
MEDIA_INTENT_HINTS = MEDIA_TEXT_HINTS + ("viewing", "watch", "watching")
MEDIA_ROOM_HINTS = ("family", "living", "media", "theater", "tv")
LIVING_SEATING = {"armchair", "chair", "loveseat", "sofa"}
LIVING_NEAREST_FOCUS_SEATING = {"armchair", "chair"}
WORK_SURFACE_CATEGORIES = {
    "bar_table",
    "coffee_table",
    "counter",
    "desk",
    "dining_table",
    "island",
    "side_table",
    "table",
}
# 2026-07-12 修改原因：墙边独立座椅判定应随资产尺寸缩放，避免用 guest/visitor
# 名称和单个书房回放标定的绝对米制阈值决定功能关系。
# Keep this proportional to the seat's shortest footprint axis.  Rendered HSSD
# seats often have a small collision/annotation offset from the wall; a fixed
# absolute threshold would drop otherwise valid wall-backed seating.
# Initial LLM layouts can leave an independent seat about one seat-depth from
# its intended wall.  Treat that as the same wall-backed topology and let the
# deterministic repair snap it to the wall; the separate surface-separation
# and wall-preference checks prevent ordinary table seating from matching.
WALL_ANCHOR_GAP_RATIO = 1.0
SURFACE_SEPARATION_RATIO = 0.5
WALL_PREFERENCE_MARGIN_RATIO = 0.2


def stabilize_orientation_contracts(
    case_pack: dict[str, Any],
    scene: RoomScene,
    config: CriticConfig,
    *,
    stage: str,
) -> None:
    """Keep seating orientation targets stable across SceneBenchmark stages.

    SceneBenchmark may evaluate the same in-progress scene multiple times. Without
    a stable contract, its FD proposer can pick a fresh target from current geometry
    each time, which is noisy for seating that could reasonably face either a table
    or a media focal point. This stores a room-local contract on the live RoomScene
    object and injects matching FD checks into the current case pack.
    """
    if not _enabled(config):
        return

    geometry = case_pack.get("scene_geometry") or {}
    objects = [item for item in geometry.get("objects") or [] if isinstance(item, dict)]
    objects_by_id = {
        str(item.get("id") or ""): item for item in objects if item.get("id")
    }
    if not objects_by_id:
        return

    memory = getattr(scene, CONTRACT_ATTR, None)
    if not isinstance(memory, dict):
        memory = {}
        setattr(scene, CONTRACT_ATTR, memory)

    task_text = str(case_pack.get("task_instruction") or "")
    room_type = str(case_pack.get("room_type") or "")
    media_focus = _best_media_focus(objects, task_text=task_text, room_type=room_type)
    media_intent = _has_media_intent(task_text, room_type) and media_focus is not None
    fixed_pairs = {
        str(subject_id): str((contract.get("target_ids") or [""])[0])
        for subject_id, contract in memory.items()
        if isinstance(contract, dict)
        and contract.get("assignment_source") == ASSIGNMENT_SOURCE
        and contract.get("target_ids")
    }
    work_assignments = assign_work_seats_to_surfaces(
        objects,
        task_instruction=task_text,
        room_type=room_type,
        fixed_pairs=fixed_pairs,
        room_bounds=room_bounds_from_case_pack(case_pack),
    )
    work_assignment_by_seat = {
        assignment.seat_id: assignment for assignment in work_assignments
    }
    work_cohort_ids = {
        str(obj.get("id") or "")
        for obj in work_seat_candidates(
            objects,
            task_instruction=task_text,
            room_type=room_type,
        )
    }

    checks_added = 0
    for subject in objects:
        subject_id = str(subject.get("id") or "")
        if not subject_id or not _is_seating(subject):
            continue

        explicit_facing_target = _explicit_prompt_facing_target(
            subject,
            objects,
            task_text=task_text,
        )
        explicit_wall_target = _explicit_prompt_wall_target(
            subject,
            objects,
            task_text=task_text,
            explicit_facing_target=explicit_facing_target,
        )
        if explicit_wall_target is not None:
            # A later stage brief can refine an earlier one-line summary from
            # desk-facing to wall-normal inward-facing seating.
            explicit_facing_target = None
        existing = memory.get(subject_id)
        if _contract_is_usable(
            existing,
            objects_by_id,
            subject,
            objects,
            media_intent,
            media_focus,
            work_assignment_by_seat.get(subject_id),
            subject_id in work_cohort_ids,
            explicit_facing_target,
            explicit_wall_target,
        ):
            contract = dict(existing)
            contract["stage_last_seen"] = stage
            assignment = work_assignment_by_seat.get(subject_id)
            if (
                assignment is not None
                and contract.get("assignment_source") == ASSIGNMENT_SOURCE
            ):
                # Keep the stable surface identity, but recompute its local slot
                # after a coordinated repair moves or rotates the desk.
                evidence = assignment.evidence()
                contract["target_slot"] = evidence["target_slot"]
                contract["assignment_cost"] = evidence["assignment_cost"]
                contract["annotation_sources"] = evidence["annotation_sources"]
        else:
            contract = _plan_contract(
                subject,
                objects,
                media_focus=media_focus,
                media_intent=media_intent,
                work_assignment=work_assignment_by_seat.get(subject_id),
                work_cohort_member=subject_id in work_cohort_ids,
                explicit_facing_target=explicit_facing_target,
                explicit_wall_target=explicit_wall_target,
                stage=stage,
            )

        if not contract:
            memory.pop(subject_id, None)
            continue

        memory[subject_id] = contract
        _replace_contract_check(case_pack, subject, contract)
        checks_added += 1

    if checks_added:
        console_logger.info(
            "SceneBenchmark orientation contracts active for %d seating object(s) "
            "at stage %s",
            checks_added,
            stage,
        )


def orientation_contract_subjects(case_pack: dict[str, Any]) -> set[str]:
    """Return subjects whose seating FD target is fixed by a contract check."""
    subjects: set[str] = set()
    for check in case_pack.get("checks") or []:
        if not isinstance(check, dict):
            continue
        if check.get("check_source") != CONTRACT_CHECK_SOURCE:
            continue
        if str(check.get("relation_type") or "") not in CONTRACT_RELATIONS:
            continue
        subject_id = str(check.get("subject_id") or "")
        if subject_id:
            subjects.add(subject_id)
    return subjects


def _enabled(config: CriticConfig) -> bool:
    value = config.extra.get("stable_orientation_contracts", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _contract_is_usable(
    contract: Any,
    objects_by_id: dict[str, dict[str, Any]],
    subject: dict[str, Any],
    objects: list[dict[str, Any]],
    media_intent: bool,
    media_focus: dict[str, Any] | None,
    work_assignment: SeatSurfaceAssignment | None,
    work_cohort_member: bool,
    explicit_facing_target: dict[str, Any] | None,
    explicit_wall_target: dict[str, Any] | None,
) -> bool:
    if not isinstance(contract, dict):
        return False
    target_ids = [str(item) for item in contract.get("target_ids") or [] if str(item)]
    if not target_ids or any(
        target_id not in objects_by_id for target_id in target_ids
    ):
        return False
    relation_type = str(contract.get("relation_type") or "")
    if relation_type not in CONTRACT_RELATIONS:
        return False

    if explicit_facing_target is not None:
        return (
            bool(contract.get("prompt_explicit_facing"))
            and relation_type == "furniture_faces_furniture"
            and target_ids == [str(explicit_facing_target.get("id") or "")]
        )
    if contract.get("prompt_explicit_facing"):
        return False

    if explicit_wall_target is not None:
        return (
            bool(contract.get("prompt_explicit_wall"))
            and relation_type == "back_against_wall"
            and target_ids == [str(explicit_wall_target.get("id") or "")]
        )
    if contract.get("prompt_explicit_wall"):
        return False

    if work_assignment is not None:
        return relation_type == "seating_to_work_surface" and target_ids == [
            work_assignment.surface_id
        ]
    if work_cohort_member:
        return False

    classroom_target = _classroom_student_partner(subject, objects)
    if classroom_target is not None:
        return relation_type == "seating_to_work_surface" and target_ids == [
            str(classroom_target.get("id") or "")
        ]

    # 2026-07-14 修改原因：dining_chair 被门净空或桌椅碰撞推到墙边后，旧逻辑
    # 会把它重新识别为 back_against_wall，覆盖“餐椅属于餐桌”的功能依赖，导致
    # 椅子不再保持餐桌座位线。餐椅存在餐桌时，餐桌 contract 优先于墙 contract。
    if relation_type == "back_against_wall" and _nearest_dining_table(subject, objects):
        return False

    # 2026-07-11 修改原因：书房访客椅可能先被临时绑定到书桌，随后才按
    # prompt 移到侧墙。椅子已经贴墙且远离桌面时必须废弃旧 contract，
    # 否则稳定目标会持续强迫空闲椅朝向书桌，破坏背靠墙且相互平行的布局。
    if relation_type == "seating_to_work_surface":
        target = objects_by_id[target_ids[0]]
        if _nearest_dining_table(
            subject, objects
        ) is None and _is_wall_anchored_standalone_seating(subject, target, objects):
            return False
    elif relation_type == "back_against_wall":
        wall = _standalone_wall_target(subject, objects)
        if wall is None or str(wall.get("id") or "") != target_ids[0]:
            return False

    # 2026-07-16 修改原因：客厅独立椅允许按就近原则朝茶几或媒体焦点；旧的
    # media contract 会永久锁定 TV，即使茶几明显更近，也会把正常内收姿态
    # 强行转走。每次评估都重新确认最近功能焦点，只有拓扑未变化才复用 contract。
    nearest_focus = _nearest_living_seat_focus(
        subject,
        objects,
        media_focus=media_focus if media_intent else None,
    )
    uses_nearest_living_focus = (
        relation_type in SEATING_RELATIONS and nearest_focus is not None
    )
    if uses_nearest_living_focus and nearest_focus is not None:
        preferred_target, preferred_relation = nearest_focus
        if relation_type != preferred_relation or target_ids[0] != str(
            preferred_target.get("id") or ""
        ):
            return False

    # A newly available semantic focal point is a legitimate topology change for
    # ordinary seating, but not for a guest chair whose explicit topology is the wall.
    if (
        media_intent
        and media_focus is not None
        and relation_type not in {"seating_to_media", "back_against_wall"}
        and not uses_nearest_living_focus
    ):
        return False
    return True


def _plan_contract(
    subject: dict[str, Any],
    objects: list[dict[str, Any]],
    *,
    media_focus: dict[str, Any] | None,
    media_intent: bool,
    work_assignment: SeatSurfaceAssignment | None,
    work_cohort_member: bool,
    explicit_facing_target: dict[str, Any] | None,
    explicit_wall_target: dict[str, Any] | None,
    stage: str,
) -> dict[str, Any] | None:
    if explicit_facing_target is not None:
        return _contract(
            subject,
            explicit_facing_target,
            relation_type="furniture_faces_furniture",
            stage=stage,
            reason=(
                "task instruction explicitly requires this seating role to face "
                "the referenced work surface"
            ),
            dependency={
                "subject_face": "front",
                "target_face": "any",
                "max_angle_deg": 60.0,
                # Prompt-facing topology can intentionally span a meeting zone.
                "max_distance_m": 6.5,
            },
            prompt_explicit_facing=True,
        )

    if explicit_wall_target is not None:
        return _contract(
            subject,
            explicit_wall_target,
            relation_type="back_against_wall",
            stage=stage,
            reason=(
                "task instruction explicitly places this seating role against a "
                "wall, with its front normal to the wall and facing into the room"
            ),
            dependency={
                "subject_face": "back",
                "max_angle_deg": 10.0,
                "max_distance_m": 0.25,
            },
            prompt_explicit_wall=True,
        )

    if work_assignment is not None:
        target = next(
            (
                obj
                for obj in objects
                if str(obj.get("id") or "") == work_assignment.surface_id
            ),
            None,
        )
        if target is not None:
            return _contract(
                subject,
                target,
                relation_type="seating_to_work_surface",
                stage=stage,
                reason=(
                    "functional annotations and surface-local geometry assign this "
                    "work seat to a distinct work-surface slot"
                ),
                assignment=work_assignment,
            )

    classroom_desk = _classroom_student_partner(subject, objects)
    if classroom_desk is not None:
        return _contract(
            subject,
            classroom_desk,
            relation_type="seating_to_work_surface",
            stage=stage,
            reason=(
                "indexed classroom student chair is paired with its matching "
                "student desk; pairing takes priority over incidental wall proximity"
            ),
        )

    if work_cohort_member:
        return None

    # 2026-07-14 修改原因：餐桌座位关系是显式功能拓扑，优先于几何上更近的
    # 墙面；否则门净空把 dining_chair 推近墙后会发生 wall/table contract 抖动。
    dining_table = _nearest_dining_table(subject, objects)
    if dining_table is not None:
        return _contract(
            subject,
            dining_table,
            relation_type="seating_to_work_surface",
            stage=stage,
            reason=(
                "dining chair belongs to the nearest dining table; table seating "
                "topology takes priority over incidental wall proximity"
            ),
        )
    wall = _standalone_wall_target(subject, objects)
    if wall is not None:
        return _contract(
            subject,
            wall,
            relation_type="back_against_wall",
            stage=stage,
            reason=(
                "wall-anchored standalone seating keeps its back at the "
                "wall and its front normal to the wall"
            ),
            dependency={
                "subject_face": "back",
                "max_angle_deg": 10.0,
                "max_distance_m": 0.25,
            },
        )

    nearest_focus = _nearest_living_seat_focus(
        subject,
        objects,
        media_focus=media_focus if media_intent else None,
    )
    if nearest_focus is not None:
        target, relation_type = nearest_focus
        # 2026-07-16 修改原因：客厅扶手椅可朝最近的茶几而不必统一朝 TV；
        # 锁定最近的有效室内焦点后，FD 只需阻止椅子反向朝向房间外侧。
        return _contract(
            subject,
            target,
            relation_type=relation_type,
            stage=stage,
            reason=(
                "living-room chair uses the nearest valid coffee-table or media "
                "focus so its front points into the local activity area"
            ),
        )

    # 2026-07-11 修改原因：study wall 阶段新增 tv_0 后，fresh evaluate_scenes
    # 曾把两把远端 guest chairs 从 wall contract 改绑到 TV。standalone wall
    # guest seating 的显式拓扑必须优先于后来出现的 media focus。
    if media_intent and media_focus is not None and _should_face_media(subject):
        return _contract(
            subject,
            media_focus,
            relation_type="seating_to_media",
            stage=stage,
            reason=(
                "room/task has a media focal point; seating keeps that facing "
                "target across stages"
            ),
        )

    surface = _nearest_work_surface(subject, objects)
    if surface is None:
        return None
    return _contract(
        subject,
        surface,
        relation_type="seating_to_work_surface",
        stage=stage,
        reason=(
            "no media focal point is active; seating uses the nearest functional "
            "surface"
        ),
    )


def _classroom_student_partner(
    subject: dict[str, Any], objects: list[dict[str, Any]]
) -> dict[str, Any] | None:
    role = _classroom_student_role(subject)
    if role is None or role[0] != "chair":
        return None
    candidates = [obj for obj in objects if _is_classroom_student_pair(subject, obj)]
    candidates.sort(key=lambda obj: str(obj.get("id") or ""))
    return candidates[0] if candidates else None


def _explicit_prompt_facing_target(
    subject: dict[str, Any],
    objects: list[dict[str, Any]],
    *,
    task_text: str,
) -> dict[str, Any] | None:
    """Resolve an explicit prompt role-facing-surface relation for one seat."""
    if not task_text.strip():
        return None
    subject_aliases = _prompt_subject_aliases(subject)
    if not subject_aliases:
        return None
    other_seating_aliases = tuple(
        sorted(
            {
                alias
                for obj in objects
                if obj.get("id") != subject.get("id") and _is_seating(obj)
                for alias in _prompt_subject_aliases(obj)
            },
            key=lambda value: (-len(value.split()), value),
        )
    )

    normalized_task = _normalize_prompt_text(task_text)
    matches: list[tuple[int, dict[str, Any]]] = []
    for target in objects:
        if target.get("id") == subject.get("id") or not _is_work_surface(target):
            continue
        target_aliases = _prompt_target_aliases(target)
        specificity = _explicit_facing_pair_specificity(
            normalized_task,
            subject_aliases,
            target_aliases,
            other_seating_aliases=other_seating_aliases,
        )
        if specificity is not None:
            matches.append((specificity, target))
    if not matches:
        return None
    matches.sort(
        key=lambda item: (
            -item[0],
            *_surface_rank(subject, item[1]),
        )
    )
    return matches[0][1]


def _explicit_prompt_wall_target(
    subject: dict[str, Any],
    objects: list[dict[str, Any]],
    *,
    task_text: str,
    explicit_facing_target: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve an explicitly wall-backed seating role from the task text.

    Prompt topology takes priority over a proximity-based work-seat assignment.
    A direct instruction to face a work surface remains stronger unless a later
    stage constraint explicitly changes the same seating role to inward-facing,
    wall-normal placement.
    """
    if not task_text.strip():
        return None
    subject_aliases = _prompt_subject_aliases(subject)
    if not subject_aliases:
        return None
    other_seating_aliases = tuple(
        sorted(
            {
                alias
                for obj in objects
                if obj.get("id") != subject.get("id") and _is_seating(obj)
                for alias in _prompt_subject_aliases(obj)
            },
            key=lambda value: (-len(value.split()), value),
        )
    )
    normalized_task = _normalize_prompt_text(task_text)
    matches: list[tuple[int, int, str]] = []
    for subject_alias in subject_aliases:
        pattern = re.compile(
            rf"\b{re.escape(subject_alias)}\b"
            # A comma commonly starts the next furniture noun phrase, e.g.
            # "four dining chairs ..., a sideboard against the wall". Do not
            # transfer that later object's wall relation back to the seats.
            rf"(?P<between>[^,.!?;\n]{{0,160}}?)\b"
            rf"(?:(?:are|is)\s+)?"
            rf"(?:(?:placed|positioned|arranged|set|located|kept)\s+)?"
            rf"(?:against|along|by|beside|near|at)\s+"
            rf"(?:the\s+|an?\s+)?"
            rf"(?P<wall_descriptor>"
            rf"(?:(?:left|right|side|adjacent|back|rear|front|north|south|east|west)\s+)?"
            rf"wall)\b"
        )
        for match in pattern.finditer(normalized_task):
            between = match.group("between")
            if any(
                re.search(rf"\b{re.escape(alias)}\b", between)
                for alias in other_seating_aliases
            ):
                continue
            descriptor = str(match.group("wall_descriptor") or "wall").strip()
            specificity = len(subject_alias.split()) + (
                1 if descriptor != "wall" else 0
            )
            matches.append((specificity, match.start(), descriptor))
    if not matches:
        return None
    if explicit_facing_target is not None:
        latest_facing = _latest_explicit_facing_pair_position(
            normalized_task,
            subject_aliases,
            _prompt_target_aliases(explicit_facing_target),
            other_seating_aliases=other_seating_aliases,
        )
        latest_inward = _latest_inward_facing_position(
            normalized_task,
            subject_aliases,
            other_seating_aliases=other_seating_aliases,
        )
        if latest_inward is None or (
            latest_facing is not None and latest_inward <= latest_facing
        ):
            return None
    matches.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return _prompt_wall_target(subject, objects, descriptor=matches[0][2])


def _prompt_wall_target(
    subject: dict[str, Any],
    objects: list[dict[str, Any]],
    *,
    descriptor: str,
) -> dict[str, Any] | None:
    walls = [obj for obj in objects if object_category(obj) == "wall"]
    if not walls:
        return None
    direction = descriptor.split()[0] if descriptor != "wall" else ""
    if direction in {"north", "south", "east", "west"}:
        named_walls = [
            wall for wall in walls if re.search(rf"\b{direction}\b", _object_text(wall))
        ]
        if named_walls:
            walls = named_walls
    elif direction in {"side", "adjacent", "left", "right"}:
        side_walls = _side_walls_for_work_surface(subject, objects, walls)
        if side_walls:
            walls = side_walls
    # "Back" and "rear" are relative to the intended seating composition,
    # not a compass direction. Keep every wall eligible: deriving a target
    # from a coffee table or TV stand selects the wall opposite a correctly
    # wall-backed sofa. The geometry rank below binds the adjacent wall.
    walls.sort(key=lambda wall: _prompt_wall_rank(subject, wall))
    return walls[0] if walls else None


def _side_walls_for_work_surface(
    subject: dict[str, Any],
    objects: list[dict[str, Any]],
    walls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    backing_wall = _primary_work_surface_backing_wall(subject, objects, walls)
    backing_normal_axis = _wall_normal_axis(backing_wall)
    if backing_normal_axis is None:
        return walls
    side_walls = [
        wall
        for wall in walls
        if _wall_normal_axis(wall) not in {None, backing_normal_axis}
    ]
    return side_walls or walls


def _primary_work_surface_backing_wall(
    subject: dict[str, Any],
    objects: list[dict[str, Any]],
    walls: list[dict[str, Any]],
) -> dict[str, Any] | None:
    surfaces = [obj for obj in objects if _is_work_surface(obj)]
    if not surfaces or not walls:
        return None
    candidates: list[tuple[int, float, tuple[float, float, str], dict[str, Any]]] = []
    for surface in surfaces:
        category = object_category(surface)
        desk_rank = 0 if category == "desk" or "desk" in _object_text(surface) else 1
        for wall in walls:
            wall_gap = _wall_gap(surface, wall)
            if wall_gap is None:
                continue
            candidates.append(
                (desk_rank, wall_gap, _surface_rank(subject, surface), wall)
            )
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (item[0], item[1], item[2], str(item[3].get("id") or ""))
    )
    return candidates[0][3]


def _wall_normal_axis(wall: dict[str, Any] | None) -> int | None:
    if not isinstance(wall, dict):
        return None
    size = (wall.get("bbox_world") or {}).get("size") or []
    if len(size) < 2:
        return None
    return 0 if float(size[0]) <= float(size[1]) else 1


def _prompt_wall_rank(
    subject: dict[str, Any], wall: dict[str, Any]
) -> tuple[float, float, str]:
    gap = _wall_gap(subject, wall)
    distance = distance_xy(subject, wall)
    return (
        gap if gap is not None else 999.0,
        distance if distance is not None else 999.0,
        str(wall.get("id") or ""),
    )


def _explicit_facing_pair_specificity(
    task_text: str,
    subject_aliases: tuple[str, ...],
    target_aliases: tuple[str, ...],
    *,
    other_seating_aliases: tuple[str, ...],
) -> int | None:
    best: int | None = None
    for subject_alias in subject_aliases:
        for target_alias in target_aliases:
            pattern = re.compile(
                rf"\b{re.escape(subject_alias)}\b"
                rf"(?P<between>[^.!?;\n]{{0,160}}?)\b"
                rf"(?P<facing>(?:are\s+|is\s+)?"
                rf"(?:oriented\s+(?:to\s+)?)?(?:face|faces|facing))\s+"
                rf"(?!away(?:\s+from)?\b)"
                rf"(?:directly\s+)?(?:(?:toward|towards)\s+)?"
                rf"(?:the\s+|an?\s+)?\b{re.escape(target_alias)}\b"
            )
            for match in pattern.finditer(task_text):
                if _prompt_facing_match_is_negated(task_text, match):
                    continue
                between = match.group("between")
                if any(
                    re.search(rf"\b{re.escape(alias)}\b", between)
                    for alias in other_seating_aliases
                ):
                    continue
                specificity = len(subject_alias.split()) + len(target_alias.split())
                best = specificity if best is None else max(best, specificity)
    return best


def _prompt_facing_match_is_negated(task_text: str, match: re.Match[str]) -> bool:
    clause_start = max(
        task_text.rfind(mark, 0, match.start("facing"))
        for mark in (".", "!", "?", ";", "\n")
    )
    prefix = task_text[clause_start + 1 : match.start("facing")]
    return bool(
        re.search(
            r"\b(?:avoid|never|not|do not|must not|should not|without)\b",
            prefix,
        )
    )


def _latest_explicit_facing_pair_position(
    task_text: str,
    subject_aliases: tuple[str, ...],
    target_aliases: tuple[str, ...],
    *,
    other_seating_aliases: tuple[str, ...],
) -> int | None:
    latest: int | None = None
    for subject_alias in subject_aliases:
        for target_alias in target_aliases:
            pattern = re.compile(
                rf"\b{re.escape(subject_alias)}\b"
                rf"(?P<between>[^.!?;\n]{{0,160}}?)\b"
                rf"(?P<facing>(?:are\s+|is\s+)?"
                rf"(?:oriented\s+(?:to\s+)?)?(?:face|faces|facing))\s+"
                rf"(?!away(?:\s+from)?\b)"
                rf"(?:directly\s+)?(?:(?:toward|towards)\s+)?"
                rf"(?:the\s+|an?\s+)?\b{re.escape(target_alias)}\b"
            )
            for match in pattern.finditer(task_text):
                if _prompt_facing_match_is_negated(task_text, match):
                    continue
                between = match.group("between")
                if any(
                    re.search(rf"\b{re.escape(alias)}\b", between)
                    for alias in other_seating_aliases
                ):
                    continue
                latest = match.start() if latest is None else max(latest, match.start())
    return latest


def _latest_inward_facing_position(
    task_text: str,
    subject_aliases: tuple[str, ...],
    *,
    other_seating_aliases: tuple[str, ...],
) -> int | None:
    latest: int | None = None
    for subject_alias in subject_aliases:
        pattern = re.compile(
            rf"\b{re.escape(subject_alias)}\b"
            rf"(?P<between>[^.!?;\n]{{0,220}}?)\b"
            rf"(?:fronts?\s+)?"
            rf"(?:are\s+|is\s+)?"
            rf"(?:oriented\s+(?:with\s+(?:their|its)\s+fronts?\s+)?)?"
            rf"(?:face|faces|facing|point|pointing)\s+"
            rf"(?:inward|inwards|into\s+(?:the\s+)?room|toward\s+(?:the\s+)?room\s+center)\b"
        )
        for match in pattern.finditer(task_text):
            between = match.group("between")
            if any(
                re.search(rf"\b{re.escape(alias)}\b", between)
                for alias in other_seating_aliases
            ):
                continue
            latest = match.start() if latest is None else max(latest, match.start())
    return latest


def _prompt_subject_aliases(obj: dict[str, Any]) -> tuple[str, ...]:
    identity_aliases = {
        alias
        for key in ("id", "name")
        if (alias := _normalize_prompt_object_identity(obj.get(key)))
    }
    aliases = set(identity_aliases)
    if not aliases:
        category = _normalize_prompt_object_identity(object_category(obj))
        if category:
            aliases.add(category)

    identity_text = " ".join(aliases)
    if re.search(r"\b(?:guest|visitor)\s+(?:arm\s*)?chair\b", identity_text):
        aliases.update(
            {"guest chair", "visitor chair", "guest armchair", "visitor armchair"}
        )
    return _with_prompt_plurals(aliases)


def _prompt_target_aliases(obj: dict[str, Any]) -> tuple[str, ...]:
    aliases = {
        alias
        for value in (obj.get("id"), obj.get("name"), object_category(obj))
        if (alias := _normalize_prompt_object_identity(value))
    }
    return _with_prompt_plurals(aliases)


def _with_prompt_plurals(aliases: set[str]) -> tuple[str, ...]:
    expanded = set(aliases)
    for alias in aliases:
        if alias.endswith("chair") or alias.endswith("desk") or alias.endswith("table"):
            expanded.add(f"{alias}s")
    return tuple(sorted(expanded, key=lambda value: (-len(value.split()), value)))


def _normalize_prompt_object_identity(value: Any) -> str:
    text = _normalize_prompt_text(str(value or ""))
    text = re.sub(r"\s+[0-9]+$", "", text).strip()
    return text if text and "\n" not in text else ""


def _normalize_prompt_text(value: str) -> str:
    text = value.lower().replace("_", " ").replace("-", " ")
    # Preserve commas so relation parsers can distinguish adjacent furniture
    # noun phrases instead of treating the whole sentence as one subject.
    text = re.sub(r"[^a-z0-9,.!?;\n]+", " ", text)
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def _nearest_dining_table(
    subject: dict[str, Any], objects: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return the closest dining table for a dining chair, if one exists."""
    if object_category(subject) != "dining_chair":
        return None
    candidates = [
        obj
        for obj in objects
        if obj.get("id") != subject.get("id")
        and (
            object_category(obj) in {"dining_table", "dining_table_set"}
            or "dining_table" in _object_text(obj)
        )
    ]
    candidates.sort(
        key=lambda obj: (
            (
                distance_xy(subject, obj)
                if distance_xy(subject, obj) is not None
                else 999.0
            ),
            str(obj.get("id") or ""),
        )
    )
    return candidates[0] if candidates else None


def _contract(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    relation_type: str,
    stage: str,
    reason: str,
    assignment: SeatSurfaceAssignment | None = None,
    dependency: dict[str, Any] | None = None,
    prompt_explicit_facing: bool = False,
    prompt_explicit_wall: bool = False,
) -> dict[str, Any]:
    contract = {
        "schema_version": "scenesmith.scenebenchmark_critic.orientation_contract.v1",
        "subject_id": str(subject.get("id") or ""),
        "target_ids": [str(target.get("id") or "")],
        "relation_type": relation_type,
        "stage_created": stage,
        "stage_last_seen": stage,
        "target_category": object_category(target),
        "policy": (
            "lock_primary_target_until_target_removed_or_semantic_focal_point_added"
        ),
        "reason": reason,
    }
    if assignment is not None:
        contract.update(assignment.evidence())
    if dependency:
        contract["dependency"] = dict(dependency)
    if prompt_explicit_facing:
        contract["prompt_explicit_facing"] = True
    if prompt_explicit_wall:
        contract["prompt_explicit_wall"] = True
    return contract


def _replace_contract_check(
    case_pack: dict[str, Any],
    subject: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    subject_id = str(contract.get("subject_id") or subject.get("id") or "")
    target_ids = [str(item) for item in contract.get("target_ids") or [] if str(item)]
    relation_type = str(contract.get("relation_type") or "")
    if not subject_id or not target_ids or relation_type not in CONTRACT_RELATIONS:
        return

    checks = [
        check
        for check in case_pack.get("checks") or []
        if not (
            isinstance(check, dict)
            and (
                (
                    check.get("check_source") == CONTRACT_CHECK_SOURCE
                    and str(check.get("subject_id") or "") == subject_id
                )
                or _check_conflicts_with_orientation_contract(
                    check, subject_id, relation_type
                )
            )
        )
    ]
    check_id = f"fd_contract_{subject_id}_{'_'.join(target_ids)}_{relation_type}"
    checks.append(
        {
            "check_id": check_id,
            "metric": "functional_dependency",
            "subject_id": subject_id,
            "target_ids": target_ids,
            "relation_type": relation_type,
            "expected_use": _expected_use(relation_type),
            "priority_weight": 0.9,
            "question": (
                f"Does the stable orientation contract `{relation_type}` hold "
                f"for `{subject_id}`?"
            ),
            "evidence": {
                "source": CONTRACT_CHECK_SOURCE,
                "policy": contract.get("policy"),
                "reason": contract.get("reason"),
                "stage_created": contract.get("stage_created"),
                "stage_last_seen": contract.get("stage_last_seen"),
                "assignment_source": contract.get("assignment_source"),
                "topology_required": contract.get("topology_required"),
                "target_slot": contract.get("target_slot"),
                "annotation_sources": contract.get("annotation_sources"),
                "dependency": contract.get("dependency"),
                "prompt_explicit_facing": contract.get("prompt_explicit_facing"),
                "prompt_explicit_wall": contract.get("prompt_explicit_wall"),
            },
            "evidence_refs": ["scene_geometry"],
            "check_source": CONTRACT_CHECK_SOURCE,
            "scoring_tier": "core",
        }
    )
    case_pack["checks"] = checks


def _check_conflicts_with_orientation_contract(
    check: dict[str, Any],
    contract_subject_id: str,
    contract_relation_type: str,
) -> bool:
    relation_type = str(check.get("relation_type") or "")
    if relation_type not in CONFLICTING_ORIENTATION_RELATIONS:
        return False
    check_subject_id = str(check.get("subject_id") or "")
    target_ids = {str(item) for item in check.get("target_ids") or [] if str(item)}
    involved = (
        check_subject_id == contract_subject_id or contract_subject_id in target_ids
    )
    if not involved:
        return False
    if relation_type == "workstation":
        return contract_relation_type == "back_against_wall"
    # 2026-07-11 修改原因：稳定 wall contract 不能与资产标注/模板残留的
    # guest-chair<->desk 朝向 FD 并存；否则远端椅仍会被计作 desk companion，
    # 后续 critic/SA 又可能把它拉回书桌。稳定 contract 应替换冲突拓扑。
    return check.get("check_source") != CONTRACT_CHECK_SOURCE


def _expected_use(relation_type: str) -> str:
    if relation_type == "seating_to_media":
        return "sit and view the room's chosen media focal point"
    if relation_type == "back_against_wall":
        return "remain wall-backed with the seating front normal to the wall"
    if relation_type == "furniture_faces_furniture":
        return "face the work surface explicitly referenced by the task instruction"
    return "sit at and use the chosen table or work surface"


def _best_media_focus(
    objects: list[dict[str, Any]], *, task_text: str, room_type: str
) -> dict[str, Any] | None:
    candidates = [obj for obj in objects if _is_media_target(obj)]
    if not candidates:
        return None
    has_media_intent = _has_media_intent(task_text, room_type)
    if not has_media_intent:
        return None
    candidates.sort(key=lambda obj: (_media_rank(obj), str(obj.get("id") or "")))
    return candidates[0]


def _has_media_intent(task_text: str, room_type: str) -> bool:
    text = f"{task_text} {room_type}".lower()
    return any(hint in text for hint in MEDIA_INTENT_HINTS + MEDIA_ROOM_HINTS)


def _is_media_target(obj: dict[str, Any]) -> bool:
    category = object_category(obj)
    text = _object_text(obj)
    if any(hint in text for hint in MEDIA_REJECT_HINTS):
        return False
    return category in MEDIA_CATEGORIES or any(
        hint in text for hint in MEDIA_TEXT_HINTS
    )


def _media_rank(obj: dict[str, Any]) -> tuple[int, float]:
    category = object_category(obj)
    text = _object_text(obj)
    if category in {"television", "tv", "monitor", "screen", "projection_screen"}:
        kind_rank = 0
    elif any(hint in text for hint in ("television", "tv", "monitor", "screen")):
        kind_rank = 1
    elif category in {"entertainment_center", "media_console", "tv_stand"}:
        kind_rank = 2
    else:
        kind_rank = 3
    center = (obj.get("bbox_world") or {}).get("center") or [0.0, 0.0]
    try:
        centrality = abs(float(center[0])) + abs(float(center[1])) * 0.05
    except Exception:
        centrality = 999.0
    return kind_rank, centrality


def _is_seating(obj: dict[str, Any]) -> bool:
    # 2026-07-08 修改原因：asset affordances 会把部分非座椅误标成 sittable，
    # orientation contract 必须使用归一化后的功能画像，避免给桌、灯、小物生成 seating 关系。
    hints = obj.get("functional_hints") or {}
    scene_object_type = (
        str(obj.get("object_type") or hints.get("scene_object_type") or "")
        .strip()
        .lower()
    )
    # 2026-07-11 修改原因：final living 回放中 throw pillow 因错误 sittable
    # affordance 被当成 seating，并生成 pillow -> TV remote 的稳定 contract。
    # 朝向 contract 仅适用于家具，不能作用于 manipuland/cushion/decor。
    if scene_object_type != "furniture":
        return False
    return object_function_profile(obj).is_seating and (
        "sittable" in object_affordances(obj) or object_category(obj) in LIVING_SEATING
    )


def _should_face_media(subject: dict[str, Any]) -> bool:
    category = object_category(subject)
    return _is_seating(subject) and category in LIVING_SEATING


def _nearest_living_seat_focus(
    subject: dict[str, Any],
    objects: list[dict[str, Any]],
    *,
    media_focus: dict[str, Any] | None,
) -> tuple[dict[str, Any], str] | None:
    """Choose the nearest coffee-table/media focus for an independent chair."""
    if object_category(subject) not in LIVING_NEAREST_FOCUS_SEATING:
        return None
    surfaces = [
        obj
        for obj in objects
        if obj.get("id") != subject.get("id")
        and object_category(obj) == "coffee_table"
        and _is_actionable_seating_surface_pair(subject, obj)
    ]
    candidates = [(obj, "seating_to_work_surface") for obj in surfaces]
    if media_focus is not None:
        candidates.append((media_focus, "seating_to_media"))
    if not candidates:
        return None
    # 2026-07-16 修改原因：目标选择只使用距离/间隙，不使用当前 yaw；否则一把
    # 已朝外的椅子会因角度惩罚换目标，critic 无法稳定修回最近活动区。
    candidates.sort(
        key=lambda item: (
            (
                bbox_gap_xy(subject, item[0])
                if bbox_gap_xy(subject, item[0]) is not None
                else 999.0
            ),
            (
                distance_xy(subject, item[0])
                if distance_xy(subject, item[0]) is not None
                else 999.0
            ),
            str(item[0].get("id") or ""),
        )
    )
    return candidates[0]


def _nearest_work_surface(
    subject: dict[str, Any], objects: list[dict[str, Any]]
) -> dict[str, Any] | None:
    candidates = [
        obj
        for obj in objects
        if obj.get("id") != subject.get("id")
        and _is_work_surface(obj)
        and _is_actionable_seating_surface_pair(subject, obj)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda obj: _surface_rank(subject, obj))
    if _is_wall_anchored_standalone_seating(subject, candidates[0], objects):
        return None
    return candidates[0]


def _is_work_surface(obj: dict[str, Any]) -> bool:
    category = object_category(obj)
    if category in {"bookcase", "bookshelf", "shelf", "wall_shelf"}:
        return False
    if category in WORK_SURFACE_CATEGORIES:
        return True
    return object_function_profile(obj).is_work_surface and not _is_media_target(obj)


def _surface_rank(
    subject: dict[str, Any], target: dict[str, Any]
) -> tuple[float, float, str]:
    gap = bbox_gap_xy(subject, target)
    dist = distance_xy(subject, target)
    return (
        gap if gap is not None else 999.0,
        dist if dist is not None else 999.0,
        str(target.get("id") or ""),
    )


def _is_wall_anchored_standalone_seating(
    subject: dict[str, Any],
    target: dict[str, Any],
    objects: list[dict[str, Any]],
) -> bool:
    wall_gap = _nearest_wall_gap(subject, objects)
    surface_gap = bbox_gap_xy(subject, target)
    if wall_gap is None or surface_gap is None:
        return False
    category = object_category(subject)
    if category not in {
        "armchair",
        "bar_stool",
        "bench",
        "chair",
        "dining_chair",
        "office_chair",
        "stool",
    }:
        return False
    footprint_scale = _seat_footprint_scale(subject)
    if footprint_scale is None:
        return False
    # 2026-07-12 修改原因：以座椅短边为尺度，同时要求墙面明显比工作面更近。
    # 这样可覆盖不同房间尺度、chair/stool 同义类别和旋转布局，也不会把靠墙但
    # 实际紧邻桌面的工作椅误判为空闲墙椅。
    return (
        wall_gap <= footprint_scale * WALL_ANCHOR_GAP_RATIO
        and surface_gap >= footprint_scale * SURFACE_SEPARATION_RATIO
        and wall_gap + footprint_scale * WALL_PREFERENCE_MARGIN_RATIO < surface_gap
    )


def _standalone_wall_target(
    subject: dict[str, Any], objects: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if not _is_wall_anchor_candidate(subject):
        return None
    surfaces = [
        obj
        for obj in objects
        if obj.get("id") != subject.get("id") and _is_work_surface(obj)
    ]
    surfaces.sort(key=lambda obj: _surface_rank(subject, obj))
    if surfaces and not _is_wall_anchored_standalone_seating(
        subject, surfaces[0], objects
    ):
        return None
    walls = [obj for obj in objects if object_category(obj) == "wall"]
    walls.sort(
        key=lambda obj: (
            _wall_gap(subject, obj) if _wall_gap(subject, obj) is not None else 999.0,
            str(obj.get("id") or ""),
        )
    )
    if not walls:
        return None
    wall_gap = _wall_gap(subject, walls[0])
    footprint_scale = _seat_footprint_scale(subject)
    if wall_gap is None or footprint_scale is None:
        return None
    return walls[0] if wall_gap <= footprint_scale * WALL_ANCHOR_GAP_RATIO else None


def _is_wall_anchor_candidate(subject: dict[str, Any]) -> bool:
    return object_category(subject) in {
        "armchair",
        "bar_stool",
        "bench",
        "chair",
        "dining_chair",
        "office_chair",
        "stool",
    }


def _seat_footprint_scale(subject: dict[str, Any]) -> float | None:
    size = (subject.get("bbox_world") or {}).get("size") or []
    if len(size) < 2:
        return None
    footprint = [float(value) for value in size[:2] if float(value) > 1e-6]
    return min(footprint) if footprint else None


def _nearest_wall_gap(
    subject: dict[str, Any], objects: list[dict[str, Any]]
) -> float | None:
    subject_bbox = subject.get("bbox_world") or {}
    subject_min = subject_bbox.get("min") or []
    subject_max = subject_bbox.get("max") or []
    if len(subject_min) < 2 or len(subject_max) < 2:
        return None
    best: float | None = None
    for obj in objects:
        if object_category(obj) != "wall":
            continue
        wall_bbox = obj.get("bbox_world") or {}
        wall_min = wall_bbox.get("min") or []
        wall_max = wall_bbox.get("max") or []
        if len(wall_min) < 2 or len(wall_max) < 2:
            continue
        dx = max(
            float(wall_min[0] - subject_max[0]),
            float(subject_min[0] - wall_max[0]),
            0.0,
        )
        dy = max(
            float(wall_min[1] - subject_max[1]),
            float(subject_min[1] - wall_max[1]),
            0.0,
        )
        gap = (dx * dx + dy * dy) ** 0.5
        if best is None or gap < best:
            best = gap
    return best


def _wall_gap(subject: dict[str, Any], wall: dict[str, Any]) -> float | None:
    subject_bbox = subject.get("bbox_world") or {}
    wall_bbox = wall.get("bbox_world") or {}
    subject_min = subject_bbox.get("min") or []
    subject_max = subject_bbox.get("max") or []
    wall_min = wall_bbox.get("min") or []
    wall_max = wall_bbox.get("max") or []
    if min(len(subject_min), len(subject_max), len(wall_min), len(wall_max)) < 2:
        return None
    dx = max(
        float(wall_min[0] - subject_max[0]),
        float(subject_min[0] - wall_max[0]),
        0.0,
    )
    dy = max(
        float(wall_min[1] - subject_max[1]),
        float(subject_min[1] - wall_max[1]),
        0.0,
    )
    return (dx * dx + dy * dy) ** 0.5


def _object_text(obj: dict[str, Any]) -> str:
    parts = [
        obj.get("id"),
        obj.get("name"),
        obj.get("description"),
        obj.get("category"),
        obj.get("category_norm"),
    ]
    metadata = obj.get("metadata")
    if isinstance(metadata, dict):
        parts.extend(
            [
                metadata.get("category"),
                metadata.get("asset_category"),
                metadata.get("semantic_label"),
            ]
        )
    return " ".join(str(part).lower() for part in parts if part)
