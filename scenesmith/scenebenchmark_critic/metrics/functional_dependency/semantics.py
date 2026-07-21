from __future__ import annotations

import re

from typing import Any

from scenesmith.scenebenchmark_critic.core.geometry import (
    angle_to_target_deg,
    bbox_gap_xy,
    is_small_object,
    object_affordances,
    object_category,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.constants import *
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.profiles import (
    object_function_profile,
)


_CLASSROOM_STUDENT_OBJECT_RE = re.compile(
    r"\b(?:student|classroom[_\s-]*student)[_\s-]*(chair|desk)[_\s-]*(\d+)\b",
    re.IGNORECASE,
)


def _classroom_student_role(obj: dict[str, Any]) -> tuple[str, int] | None:
    """Return the indexed student-chair/desk role encoded by an asset identity."""
    for key in ("id", "name", "category", "category_norm", "asset_id"):
        value = str(obj.get(key) or "")
        match = _CLASSROOM_STUDENT_OBJECT_RE.search(value)
        if match is None:
            continue
        return match.group(1).lower(), int(match.group(2))
    return None


def _is_classroom_student_pair(
    subject: dict[str, Any], target: dict[str, Any]
) -> bool:
    """Whether a student chair and desk share the same explicit instance index."""
    subject_role = _classroom_student_role(subject)
    target_role = _classroom_student_role(target)
    return bool(
        subject_role
        and target_role
        and subject_role == ("chair", target_role[1])
        and target_role[0] == "desk"
    )


def _is_work_surface_target(target: dict[str, Any]) -> bool:
    profile = object_function_profile(target)
    category = object_category(target)
    if category in {"bookcase", "bookshelf", "shelf", "wall_shelf"}:
        return False
    if (
        profile.source == "explicit"
        and profile.is_small_placeable
        and _scene_object_type(target) == "manipuland"
        and not profile.is_work_surface
    ):
        return False
    if (
        profile.source == "explicit"
        and profile.is_work_surface
        and not profile.is_seating
        and not profile.is_media_target
    ):
        return True
    category_group = _category_group(target)
    if is_small_object(target):
        return False
    if category in SEATING or category in MEDIA:
        return False
    if category_group in WORK_SURFACE_REJECT_GROUPS:
        return False
    if _is_any_lamp_object(target):
        return False
    if _raw_text_has_any(target, WORK_SURFACE_TARGET_REJECT_HINTS):
        return False
    if _category_token_has_any(target, SMALL_OBJECT_TEXT_HINTS):
        return False
    if _category_token_has_any(target, SOFT_SUPPORT_TARGET_REJECT_HINTS):
        return False
    if category in WORK_SURFACES | NIGHTSTANDS | {
        "end_table",
        "console",
        "credenza",
        "sideboard",
        "buffet",
        "counter",
        "island",
    }:
        return True
    if (
        category_group in WORK_SURFACE_CATEGORY_GROUPS
        and _has_support_storage_semantics(target)
    ):
        return True
    return _category_surface_family_match(target)


def _is_media_target(target: dict[str, Any]) -> bool:
    if _raw_text_has_any(target, MEDIA_TARGET_REJECT_HINTS):
        return False
    if _text_has_any(target, ("remote", "controller", "device")):
        return False
    profile = object_function_profile(target)
    if profile.source == "explicit" and profile.is_media_target:
        return True
    category = object_category(target)
    if category in MEDIA:
        return True
    if is_small_object(target):
        return False
    return _text_has_any(target, MEDIA_TEXT_HINTS + ("tv_stand",))


def _is_computer_peripheral_subject(subject: dict[str, Any]) -> bool:
    category = object_category(subject)
    return category in {
        "keyboard",
        "mouse",
        "trackpad",
        "touchpad",
    } or _text_has_any(subject, ("keyboard", "mouse", "trackpad", "touchpad"))


def _is_computer_screen_target(target: dict[str, Any]) -> bool:
    category = object_category(target)
    return category in {
        "display",
        "laptop",
        "monitor",
        "notebook_computer",
        "projection_screen",
        "screen",
        "tablet",
        "tablet_computer",
    } or _text_has_any(
        target,
        (
            "computer_monitor",
            "display",
            "laptop",
            "monitor",
            "notebook_computer",
            "screen",
            "tablet_computer",
        ),
    )


def _is_nightstand_target(target: dict[str, Any]) -> bool:
    profile = object_function_profile(target)
    if profile.source == "explicit" and profile.is_bedside_surface:
        return True
    category = object_category(target)
    if is_small_object(target):
        return False
    return category in NIGHTSTANDS or _text_has_any(
        target, ("nightstand", "side_table")
    )


def _is_supported_small_subject(subject: dict[str, Any]) -> bool:
    profile = object_function_profile(subject)
    if (
        profile.source == "explicit"
        and profile.is_small_placeable
        and _is_surface_placed_small_placeable(subject)
        and not (
            profile.is_seating
            or profile.is_work_surface
            or profile.is_media_target
            or profile.is_sleeping_surface
            or _is_any_lamp_object(subject)
        )
    ):
        # 2026-07-08 修改原因：pen cup/tray/jewelry box 等 surface object
        # 可能被误标为 furniture 或可收纳，但自身仍应能作为被支撑小物。
        return _support_subject_size_is_reasonable(subject)
    if (
        profile.source == "explicit"
        and profile.is_small_placeable
        and not (
            profile.can_support_top
            or profile.has_internal_shelf
            or profile.is_seating
            or profile.is_work_surface
            or _is_any_lamp_object(subject)
        )
    ):
        return True
    if _scene_object_type(subject) in {"wall_mounted", "ceiling_mounted"}:
        return False
    if _token_text_has_any(subject, SUPPORT_SUBJECT_REJECT_HINTS):
        return False
    category = object_category(subject)
    category_group = _category_group(subject)
    if category not in SUPPORTED_SMALL and (
        category in SUPPORTS or category_group in SUPPORT_CATEGORY_GROUPS
    ):
        return False
    if is_small_object(subject):
        return category in SUPPORTED_SMALL or _token_text_has_any(
            subject, SUPPORT_SUBJECT_TEXT_HINTS
        )
    if category in SUPPORTED_SMALL or _token_text_has_any(
        subject, SUPPORT_SUBJECT_TEXT_HINTS
    ):
        return _support_subject_size_is_reasonable(subject)
    return False


def _is_surface_placed_small_placeable(subject: dict[str, Any]) -> bool:
    scene_type = _scene_object_type(subject)
    if scene_type in {"wall_mounted", "ceiling_mounted"}:
        return False
    if scene_type == "manipuland":
        return True
    hints = subject.get("functional_hints") or {}
    placement_class = str(hints.get("placement_class") or "").strip().lower()
    if placement_class == "surface_object":
        return True
    placement = subject.get("placement_info") or {}
    return isinstance(placement, dict) and bool(placement.get("parent_surface_id"))


def _is_upright_reading_material(subject: dict[str, Any]) -> bool:
    category = object_category(subject)
    if category not in UPRIGHT_THIN_READING_MATERIALS and not _token_text_has_any(
        subject,
        UPRIGHT_THIN_READING_MATERIALS,
    ):
        return False
    bbox = subject.get("bbox_world") or {}
    size = bbox.get("size") or []
    if len(size) < 3:
        return False
    width = max(float(size[0]), 0.0)
    depth = max(float(size[1]), 0.0)
    height = max(float(size[2]), 0.0)
    return height >= 0.16 and height >= max(width, depth) * 0.85


def _support_subject_size_is_reasonable(subject: dict[str, Any]) -> bool:
    bbox = subject.get("bbox_world") or {}
    size = bbox.get("size") or []
    if len(size) < 3:
        return False
    width = max(float(size[0]), 0.0)
    depth = max(float(size[1]), 0.0)
    height = max(float(size[2]), 0.0)
    return width <= 0.7 and depth <= 0.7 and width * depth <= 0.35 and height <= 1.1


def _is_lamp_subject(subject: dict[str, Any]) -> bool:
    if not _is_any_lamp_object(subject):
        return False
    if _is_floor_lamp_subject(subject) or _is_mounted_lamp_subject(subject):
        return False
    return object_category(subject) in LAMPS or _text_has_any(
        subject, ("desk_lamp", "table_lamp", "bedside_lamp")
    )


def _is_any_lamp_object(obj: dict[str, Any]) -> bool:
    category = object_category(obj)
    if category in LAMPS | {"floor_lamp"}:
        return True
    return _token_text_has_any(
        obj,
        (
            "lamp",
            "desk_lamp",
            "table_lamp",
            "floor_lamp",
            "lightfixture",
            "recessed_light",
            "track_light",
            "wall_sconce",
            "wall_light",
            "pendant",
            "chandelier",
        ),
    )


def _is_floor_lamp_subject(subject: dict[str, Any]) -> bool:
    if not _is_any_lamp_object(subject):
        return False
    if _token_text_has_any(subject, FLOOR_LAMP_TEXT_HINTS):
        return True
    bbox = subject.get("bbox_world") or {}
    size = bbox.get("size") or []
    bmin = bbox.get("min") or []
    if len(size) < 3 or len(bmin) < 3:
        return False
    max_dim = max(abs(float(size[0])), abs(float(size[1])), abs(float(size[2])))
    return float(bmin[2]) <= 0.08 and max_dim >= 0.95


def _is_mounted_lamp_subject(subject: dict[str, Any]) -> bool:
    if not _is_any_lamp_object(subject):
        return False
    if _scene_object_type(subject) in {"wall_mounted", "ceiling_mounted"}:
        return True
    if _token_text_has_any(subject, MOUNTED_LAMP_TEXT_HINTS):
        return True
    bbox = subject.get("bbox_world") or {}
    bmin = bbox.get("min") or []
    if len(bmin) < 3:
        return False
    if float(bmin[2]) < 0.8:
        return False
    return _token_text_has_any(
        subject, ("mount", "mounting", "mounting_plate", "housing", "lightfixture")
    )


def _is_seating_subject(subject: dict[str, Any]) -> bool:
    # 2026-07-11 修改原因：living final 的 throw pillow 带错误 explicit
    # sittable/media 标注，模板 proposer 因而生成 pillow -> TV remote FD。
    # manipuland/wall/ceiling objects 不能成为 seating relation 的 subject。
    if _scene_object_type(subject) in {
        "manipuland",
        "wall_mounted",
        "ceiling_mounted",
    }:
        return False
    profile = object_function_profile(subject)
    if profile.source == "explicit" and profile.is_seating:
        return not _raw_text_has_any(subject, SEATING_SUBJECT_REJECT_HINTS)
    return object_category(subject) in SEATING and not _raw_text_has_any(
        subject, SEATING_SUBJECT_REJECT_HINTS
    )


def _is_directional_facing_subject(subject: dict[str, Any]) -> bool:
    # 2026-07-08 修改原因：资产标注会给对称桌面、装饰物、落地灯生成 front-facing
    # 依赖；只让有明确使用正面的对象参与 furniture_faces_furniture。
    category = object_category(subject)
    profile = object_function_profile(subject)
    if category in {"wall", "floor", "ceiling"}:
        return False
    if _scene_object_type(subject) == "ceiling_mounted":
        return False
    if _is_seating_subject(subject) or _is_media_target(subject):
        return True
    if (
        profile.source == "explicit"
        and profile.is_small_placeable
        and _scene_object_type(subject) == "manipuland"
    ):
        return False
    if category in {
        "bookshelf",
        "cabinet",
        "console",
        "credenza",
        "desk",
        "dresser",
        "media_console",
        "nightstand",
        "sideboard",
        "storage_furniture",
        "tv_stand",
        "wardrobe",
    }:
        return True
    group = _category_group(subject)
    if group in {"seating", "media"}:
        return True
    if group not in {"storage", "storage_surface", "work_surface"}:
        return False
    if not (profile.has_internal_shelf or profile.can_support_top):
        return False
    return _front_access_terms(subject)


def _is_facing_relation_target(target: dict[str, Any]) -> bool:
    category = object_category(target)
    if category in {"wall", "floor", "ceiling"}:
        return False
    if _is_seating_subject(target) or _is_media_target(target):
        return True
    if _is_work_surface_target(target) or _is_nightstand_target(target):
        return True
    if category in BEDS:
        return True
    if category in {
        "bookshelf",
        "cabinet",
        "console",
        "credenza",
        "dresser",
        "media_console",
        "sideboard",
        "storage_furniture",
        "tv_stand",
        "wardrobe",
    }:
        return True
    if _scene_object_type(target) in {"wall_mounted", "ceiling_mounted"}:
        return False
    if is_small_object(target):
        return False
    group = _category_group(target)
    return group in {"storage", "storage_surface", "work_surface"}


def _front_access_terms(obj: dict[str, Any]) -> bool:
    hints = obj.get("functional_hints") or {}
    access_type = hints.get("access_type") or {}
    primary_access = (
        str(access_type.get("primary") or "").strip().lower()
        if isinstance(access_type, dict)
        else ""
    )
    if primary_access in {"front", "front_open", "front-open"}:
        return True
    surface_map = hints.get("interaction_surface_map") or {}
    if not isinstance(surface_map, dict):
        return False
    front_terms = " ".join(
        str(item or "").strip().lower() for item in surface_map.get("front") or []
    )
    return any(
        term in front_terms
        for term in ("drawer", "door", "storage", "shelf", "screen", "viewing", "work")
    )


def _text_has_any(obj: dict[str, Any], hints: tuple[str, ...]) -> bool:
    text = _object_text(obj)
    return any(hint in text for hint in hints)


def _token_text_has_any(obj: dict[str, Any], hints: tuple[str, ...]) -> bool:
    tokens = _object_tokens(obj)
    if not tokens:
        return False
    for hint in hints:
        parts = [
            part
            for part in re.split(r"[^a-z0-9]+", str(hint or "").strip().lower())
            if part
        ]
        if parts and all(part in tokens for part in parts):
            return True
    return False


def _category_token_has_any(obj: dict[str, Any], hints: tuple[str, ...]) -> bool:
    tokens: set[str] = set()
    for key in ("category", "category_norm"):
        tokens.update(
            token
            for token in re.split(
                r"[^a-z0-9]+", str(obj.get(key) or "").strip().lower()
            )
            if token
        )
    if not tokens:
        return False
    for hint in hints:
        parts = [
            part
            for part in re.split(r"[^a-z0-9]+", str(hint or "").strip().lower())
            if part
        ]
        if parts and all(part in tokens for part in parts):
            return True
    return False


def _object_text(obj: dict[str, Any]) -> str:
    hints = obj.get("functional_hints") or {}
    parts: list[str] = [
        str(obj.get("category") or "").strip().lower(),
        str(obj.get("category_norm") or "").strip().lower(),
        str(hints.get("category_group") or "").strip().lower(),
        str(hints.get("placement_class") or "").strip().lower(),
        str(hints.get("scene_object_type") or "").strip().lower(),
        str(hints.get("benchmark_relevance") or "").strip().lower(),
        str(hints.get("classification_source") or "").strip().lower(),
    ]
    for keyword in hints.get("category_keywords") or []:
        parts.append(str(keyword or "").strip().lower())
    access_type = hints.get("access_type") or {}
    if isinstance(access_type, dict):
        for value in access_type.values():
            parts.append(str(value or "").strip().lower())
    surface_map = hints.get("interaction_surface_map") or {}
    if isinstance(surface_map, dict):
        for values in surface_map.values():
            for value in values or []:
                parts.append(str(value or "").strip().lower())
    return " ".join(part for part in parts if part)


def _object_tokens(obj: dict[str, Any]) -> set[str]:
    tokens = {token for token in re.split(r"[^a-z0-9]+", _object_text(obj)) if token}
    singular_aliases = {
        "bookends": "bookend",
        "books": "book",
        "bottles": "bottle",
        "bowls": "bowl",
        "cups": "cup",
        "glasses": "glass",
        "magazines": "magazine",
        "mugs": "mug",
        "newspapers": "newspaper",
        "novels": "novel",
        "phones": "phone",
        "plants": "plant",
        "plates": "plate",
        "remotes": "remote",
        "trays": "tray",
        "tumblers": "tumbler",
        "vases": "vase",
    }
    tokens.update(alias for token, alias in singular_aliases.items() if token in tokens)
    return tokens


def _scene_object_type(obj: dict[str, Any]) -> str:
    hints = obj.get("functional_hints") or {}
    text = (
        str(hints.get("scene_object_type") or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    return (
        text
        if text
        in {"wall_mounted", "manipuland", "ceiling_mounted", "furniture", "unknown"}
        else "unknown"
    )


def _is_core_work_surface_target(target: dict[str, Any]) -> bool:
    category = object_category(target)
    if category in {"bookcase", "bookshelf", "shelf", "wall_shelf"}:
        return False
    category_group = _category_group(target)
    if _raw_text_has_any(target, WORK_SURFACE_TARGET_REJECT_HINTS):
        return False
    if category in SEATING or category in MEDIA:
        return False
    if category_group in WORK_SURFACE_REJECT_GROUPS:
        return False
    if _is_lamp_subject(target):
        return False
    if category in WORK_SURFACES:
        return True
    if (
        category_group in WORK_SURFACE_CATEGORY_GROUPS
        and _has_support_storage_semantics(target)
    ):
        return True
    return _category_surface_family_match(target)


def _is_actionable_seating_surface_pair(
    subject: dict[str, Any], target: dict[str, Any]
) -> bool:
    if not _is_seating_subject(subject) or not _is_work_surface_target(target):
        return False
    gap = bbox_gap_xy(subject, target)
    if gap is None:
        return False

    # Classroom assets encode the intended chair-desk pairing in their indexed
    # identities. Keep this relation actionable even after a bad layout has
    # separated the pair beyond the generic proximity threshold, so the critic
    # reports a concrete failure instead of treating the chair as independent.
    if _is_classroom_student_pair(subject, target):
        return True

    subject_category = object_category(subject)
    target_category = object_category(target)
    subject_profile = object_function_profile(subject)
    target_profile = object_function_profile(target)
    if (
        subject_profile.source == "explicit"
        and target_profile.source == "explicit"
        and subject_profile.is_seating
        and target_profile.is_work_surface
    ):
        return gap <= DIRECT_SEATING_SURFACE_MAX_GAP_M
    if target_category in DIRECT_SEATING_WORK_SURFACES:
        return gap <= DIRECT_SEATING_SURFACE_MAX_GAP_M
    if target_category == "coffee_table":
        max_gap = (
            LIVING_ROOM_COFFEE_TABLE_MAX_GAP_M
            if subject_category in LIVING_ROOM_SEATING
            else COFFEE_TABLE_SEATING_MAX_GAP_M
        )
        return gap <= max_gap
    if _is_side_surface_target(target):
        return gap <= 0.95

    angle = angle_to_target_deg(subject, target)
    if subject_category == "stool" and target_category in {
        "counter",
        "island",
        "bar_table",
    }:
        return gap <= STOOL_COUNTER_MAX_GAP_M
    if (
        target_category in OPTIONAL_SEATING_WORK_SURFACES
        or _category_surface_family_match(target)
    ):
        return (
            gap <= OPTIONAL_SEATING_SURFACE_MAX_GAP_M
            and angle is not None
            and angle <= OPTIONAL_SEATING_SURFACE_MAX_ANGLE_DEG
        )
    return False


def _is_core_media_target(target: dict[str, Any]) -> bool:
    category = object_category(target)
    if _raw_text_has_any(target, MEDIA_TARGET_REJECT_HINTS):
        return False
    if category in MEDIA:
        return True
    return _category_text_has_any(target, ("television", "tv", "monitor", "screen"))


def _is_side_surface_target(target: dict[str, Any]) -> bool:
    return object_category(target) in {
        "side_table",
        "nightstand",
    } or _category_text_has_any(target, SIDE_SURFACE_HINTS)


def _category_text_has_any(obj: dict[str, Any], hints: tuple[str, ...]) -> bool:
    text = " ".join(
        str(obj.get(key) or "").strip().lower() for key in ("category", "category_norm")
    )
    return any(hint in text for hint in hints)


def _category_group(obj: dict[str, Any]) -> str:
    return (
        str(((obj.get("functional_hints") or {}).get("category_group") or ""))
        .strip()
        .lower()
    )


def _normalized_category_phrases(obj: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    for key in ("category", "category_norm"):
        value = str(obj.get(key) or "").strip().lower()
        if not value:
            continue
        normalized = re.sub(r"[_-]+", " ", value)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if normalized:
            phrases.append(normalized)
    return phrases


def _category_surface_family_match(obj: dict[str, Any]) -> bool:
    for phrase in _normalized_category_phrases(obj):
        if phrase in {
            "desk",
            "table",
            "dining table",
            "coffee table",
            "bar table",
            "side table",
            "nightstand",
            "end table",
        }:
            return True
        if any(
            phrase == family or phrase.startswith(f"{family} ")
            for family in WORK_SURFACE_PREFIX_FAMILIES
        ):
            return True
    return False


def _raw_text_has_any(obj: dict[str, Any], hints: tuple[str, ...]) -> bool:
    text = " ".join(
        str(obj.get(key) or "").strip().lower()
        for key in ("id", "category", "asset_id", "asset_annotation_path")
    )
    return any(hint in text for hint in hints)


def _has_support_storage_semantics(target: dict[str, Any]) -> bool:
    affordances = object_affordances(target)
    category_group = (
        str(((target.get("functional_hints") or {}).get("category_group") or ""))
        .strip()
        .lower()
    )
    if affordances & SUPPORT_TOP_SURFACE_AFFORDANCES:
        return True
    if (
        category_group in SUPPORT_CATEGORY_GROUPS
        and affordances & SUPPORT_INTERNAL_AFFORDANCES
    ):
        return True
    modes = _support_modes(target)
    return (
        "top_surface" in modes or "open_shelf" in modes or "internal_storage" in modes
    )


def _support_modes(target: dict[str, Any]) -> set[str]:
    hints = target.get("functional_hints") or {}
    affordances = object_affordances(target)
    category = object_category(target)
    modes: set[str] = set()
    surface_map = hints.get("interaction_surface_map") or {}
    access_type = hints.get("access_type") or {}
    category_group = str(hints.get("category_group") or "").strip().lower()
    primary_access = (
        str((access_type.get("primary") if isinstance(access_type, dict) else "") or "")
        .strip()
        .lower()
    )

    top_terms = [
        str(value or "").strip().lower() for value in (surface_map.get("top") or [])
    ]
    side_terms: list[str] = []
    if isinstance(surface_map, dict):
        for face_name, values in surface_map.items():
            if str(face_name) == "top":
                continue
            side_terms.extend(
                str(value or "").strip().lower() for value in (values or [])
            )

    if (
        category in SUPPORTS
        or affordances & SUPPORT_TOP_SURFACE_AFFORDANCES
        or any(any(term in value for term in TOP_SURFACE_TERMS) for value in top_terms)
    ):
        modes.add("top_surface")
    if (
        category in {"shelf", "bookshelf", "wall_shelf"}
        or primary_access in {"front_open", "front-open"}
        or any(
            any(term in value for term in SHELF_SURFACE_TERMS) for value in side_terms
        )
        or affordances & {"storage_shelf", "storage"}
    ):
        modes.add("open_shelf")
    if (
        category_group in SUPPORT_CATEGORY_GROUPS
        and (
            affordances & SUPPORT_INTERNAL_AFFORDANCES
            or category
            in {"cabinet", "drawer", "dresser", "console", "sideboard", "credenza"}
        )
    ) or any(
        any(term in value for term in STORAGE_SURFACE_TERMS) for value in side_terms
    ):
        modes.add("internal_storage")
    return modes
