"""Prompt-driven room-center placement checks.

This rule only activates when the room prompt explicitly places a major object
in the room center. It ignores centerlines relative to another object or wall,
such as ``desk centered against the back wall`` and
``coffee table centered between the sofa and TV``.
"""

from __future__ import annotations

import re
from typing import Any

from scenesmith.scenebenchmark_critic.core.geometry import (
    bbox_center_xy,
)

RELATION_TYPE = "room_center_alignment"
_CENTER_RE = re.compile(
    r"\b(?:center|centre|middle|central|centrally|centered|centred)\b"
)
_RELATIVE_CENTER_RE = re.compile(
    r"\b(?:between|against|beside|near|opposite)\b|"
    r"\b(?:on|along)\s+(?:the\s+)?(?:\w+\s+){0,2}wall\b|"
    r"\b(?:center|centre|middle)\s+of\s+(?:the\s+)?"
    r"(?:table|desk|bed|sofa|couch|island|workbench)\b"
)
_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:[,;]|\band\b)\s*", re.IGNORECASE)
_ROOM_CONTEXT_RE = re.compile(r"\b(?:room|space|area)\b")
_ANCHOR_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dining_table", ("dining table", "dining_table")),
    ("table", ("table",)),
    ("desk", ("desk",)),
    ("bed", ("bed",)),
    ("sofa", ("sofa", "couch")),
    ("workbench", ("workbench", "work bench")),
    ("island", ("island",)),
)
_SEATING_TOKENS = ("chair", "seat", "stool", "bench", "sofa", "couch")


def evaluate_room_center_alignment(case_pack: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate objects explicitly requested at the room center."""
    geometry = case_pack.get("scene_geometry") or {}
    objects = [
        obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    ]
    rooms = [
        room
        for room in geometry.get("rooms") or []
        if isinstance(room, dict) and room.get("id")
    ]
    prompt = str(case_pack.get("task_instruction") or "")
    if not objects or not rooms or not prompt:
        return []

    room = rooms[0]
    room_bbox = room.get("bbox") or {}
    room_min = room_bbox.get("min") or []
    room_max = room_bbox.get("max") or []
    if len(room_min) < 2 or len(room_max) < 2:
        return []
    room_center = (
        (float(room_min[0]) + float(room_max[0])) / 2.0,
        (float(room_min[1]) + float(room_max[1])) / 2.0,
    )
    room_span = (
        abs(float(room_max[0]) - float(room_min[0])),
        abs(float(room_max[1]) - float(room_min[1])),
    )
    positive_spans = [item for item in room_span if item > 1e-6]
    if not positive_spans:
        return []
    scale = min(positive_spans)

    objects_by_id = {str(obj["id"]): obj for obj in objects}
    used_anchor_ids: set[str] = set()
    results: list[dict[str, Any]] = []
    for alias, sentences in _prompt_center_sentences(prompt):
        candidates = [
            obj
            for obj in objects
            if str(obj.get("id")) not in used_anchor_ids
            and _matches_alias(obj, alias)
        ]
        if not candidates:
            continue
        # If several instances share the category, use the object nearest to
        # the room center as the prompt's singular anchor.
        anchor = min(
            candidates,
            key=lambda obj: _distance_to_center(obj, room_center),
        )
        anchor_id = str(anchor["id"])
        used_anchor_ids.add(anchor_id)
        center = bbox_center_xy(anchor)
        if center is None:
            continue
        dx = float(center[0]) - room_center[0]
        dy = float(center[1]) - room_center[1]
        offset = (dx * dx + dy * dy) ** 0.5
        allowed = max(0.15, 0.08 * scale)
        if offset <= allowed:
            label = "pass"
        elif offset <= 2.0 * allowed:
            label = "degraded"
        else:
            label = "fail"
        related = _associated_seating(anchor, objects_by_id, room_center)
        phrase = "; ".join(sentences)
        advice = _repair_advice(anchor_id, related, room_center, label)
        if label == "pass":
            reason = (
                f"`{anchor_id}` is centered in the room within {offset:.2f}m; "
                f"allowed offset is {allowed:.2f}m. Prompt evidence: {phrase}."
            )
        else:
            reason = (
                f"Prompt requests `{anchor_id}` at the room center, but its center "
                f"is {offset:.2f}m from the room center ({dx:+.2f}m x, {dy:+.2f}m y); "
                f"allowed offset is {allowed:.2f}m. Prompt evidence: {phrase}."
            )
        results.append(
            {
                "check_id": f"fd_{anchor_id}_{RELATION_TYPE}",
                "metric": "functional_dependency",
                "label": label,
                "confidence": 0.94 if label != "pass" else 0.9,
                "primary_object": anchor_id,
                "related_objects": related,
                "selected_related_objects": related,
                "blocking_objects": [],
                "relation_type": RELATION_TYPE,
                "reason": reason,
                "repair_advice": advice,
                "diagnostics": {
                    "prompt_anchor": alias,
                    "prompt_evidence": phrase,
                    "room_id": str(room.get("id")),
                    "room_center_xy": [room_center[0], room_center[1]],
                    "object_center_xy": [float(center[0]), float(center[1])],
                    "offset_xy": [dx, dy],
                    "offset_m": offset,
                    "allowed_offset_m": allowed,
                    "room_span_xy": [room_span[0], room_span[1]],
                },
                "evidence": {
                    "constraint": "prompt_explicit_room_center",
                    "coordinate_frame": "room_world_xy",
                },
                "evaluation_source": "scenesmith_room_center_alignment",
                "scoring_tier": "core",
            }
        )
    return results


def _prompt_center_sentences(prompt: str) -> list[tuple[str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    for sentence in re.split(r"(?<=[.!?])\s+", prompt):
        for clause in _CLAUSE_SPLIT_RE.split(sentence):
            clause = clause.strip()
            lower = clause.lower()
            if not clause or not _CENTER_RE.search(lower):
                continue
            if _RELATIVE_CENTER_RE.search(lower):
                continue
            for alias, aliases in _ANCHOR_ALIASES:
                if _has_room_center_anchor(lower, aliases):
                    grouped.setdefault(alias, []).append(clause)
    return list(grouped.items())


def _has_room_center_anchor(clause: str, aliases: tuple[str, ...]) -> bool:
    """Require the center wording to describe the candidate furniture.

    2026-07-15 修改原因：此前按整句同时命中 ``table`` 和 ``middle``，会把
    ``centerpiece vase ... middle of the table`` 误判成餐桌位于房间中心。
    只有家具在中心短语前、明确使用 ``central`` 修饰家具，或句子明确说
    ``center of the room`` 时才建立 room-center contract。
    """
    alias_matches = [
        match
        for alias in aliases
        for match in re.finditer(rf"\b{re.escape(alias)}\b", clause)
    ]
    center_matches = list(_CENTER_RE.finditer(clause))
    if not alias_matches or not center_matches:
        return False

    for center in center_matches:
        for alias in alias_matches:
            if alias.end() <= center.start():
                before_center = clause[: center.start()]
                # ``the table's center`` is a local object relation, not a
                # placement request for the table in the room.
                if re.search(
                    r"\b(?:table|desk|bed|sofa|couch|island|workbench)\s*"
                    r"(?:['’]s|of\s+the)\s*$",
                    before_center,
                ):
                    continue
                return True

            if alias.start() < center.start():
                continue

            # Support natural wording such as ``the center of the room
            # contains a dining table`` where the anchor follows the center
            # phrase.
            if _ROOM_CONTEXT_RE.search(clause[center.end() : alias.start()]):
                return True

            # ``central dining table`` and ``centrally positioned dining table``
            # put the center adjective before the anchor. Limit the accepted
            # gap so ``centerpiece ... table`` cannot become a contract.
            if center.group(0).lower() not in {"central", "centrally"}:
                continue
            between = clause[center.end() : alias.start()]
            if len(re.findall(r"\b[\w_]+\b", between)) <= 3:
                return True
    return False


def _matches_alias(obj: dict[str, Any], alias: str) -> bool:
    text = " ".join(
        str(obj.get(key) or "").lower()
        for key in ("category", "category_norm", "name", "description")
    )
    if alias == "dining_table":
        return "dining" in text and "table" in text
    return alias in text or (alias == "sofa" and "couch" in text)


def _distance_to_center(obj: dict[str, Any], center: tuple[float, float]) -> float:
    point = bbox_center_xy(obj)
    if point is None:
        return float("inf")
    return (
        (float(point[0]) - center[0]) ** 2 + (float(point[1]) - center[1]) ** 2
    ) ** 0.5


def _associated_seating(
    anchor: dict[str, Any],
    objects_by_id: dict[str, dict[str, Any]],
    room_center: tuple[float, float],
) -> list[str]:
    anchor_text = " ".join(
        str(anchor.get(key) or "").lower()
        for key in ("category", "category_norm", "name", "description")
    )
    if "table" not in anchor_text:
        return []
    anchor_center = bbox_center_xy(anchor) or room_center
    result: list[tuple[float, str]] = []
    for obj_id, obj in objects_by_id.items():
        text = " ".join(
            str(obj.get(key) or "").lower()
            for key in ("category", "category_norm", "name", "description")
        )
        if not any(token in text for token in _SEATING_TOKENS):
            continue
        point = bbox_center_xy(obj)
        if point is None:
            continue
        distance = (
            (float(point[0]) - anchor_center[0]) ** 2
            + (float(point[1]) - anchor_center[1]) ** 2
        ) ** 0.5
        if distance <= 2.5:
            result.append((distance, obj_id))
    return [obj_id for _, obj_id in sorted(result)]


def _repair_advice(
    anchor_id: str,
    related: list[str],
    room_center: tuple[float, float],
    label: str,
) -> str:
    if label == "pass":
        return ""
    if related:
        return (
            f"Keep `{anchor_id}` near the room center at approximately "
            f"({room_center[0]:.2f}, {room_center[1]:.2f}). Repair the anchor and "
            f"associated seating ({', '.join(related)}) as one coordinated group: "
            "apply any "
            "shared translation first, then recompute table-local seating slots "
            "and recheck spatial_accessibility, interaction_clearance, and "
            "functional_dependency. Do not move the anchor alone to solve a local "
            "chair or sideboard conflict."
        )
    return (
        f"Keep `{anchor_id}` near the room center at approximately "
        f"({room_center[0]:.2f}, {room_center[1]:.2f}); resolve local conflicts by "
        "moving secondary furniture while preserving this anchor."
    )
