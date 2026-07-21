"""Rule checks for manipuland set completeness."""

from __future__ import annotations

import re

from collections import Counter
from typing import Any


DINING_ITEM_PATTERNS = {
    "plate": r"\bplate\b",
    "bowl": r"\bbowl\b",
    "drinkware": (
        r"\b(?:water|wine|drinking) glass\b|\bglassware\b|\bgoblet\b|"
        r"\bcup\b|\bmug\b|\btumbler\b"
    ),
    "fork": r"\bfork\b",
    "knife": r"\bknife\b|\bknives\b",
    "spoon": r"\b(?:tea|table)?spoon\b",
    "chopsticks": r"\bchopsticks?\b",
    "utensil": r"\b(?:cutlery|flatware|silverware|utensil)\b",
    "napkin": r"\bnapkin\b",
}
DINING_ITEM_GROUPS = tuple(DINING_ITEM_PATTERNS)
CUTLERY_GROUPS = ("fork", "knife", "spoon", "chopsticks", "utensil")
PROMPT_GROUP_PATTERNS = {
    "plate": r"\bplates?\b",
    "bowl": r"\bbowls?\b",
    "drinkware": (
        r"\bglasses\b|\bglassware\b|\b(?:water|wine|drinking) glass\b|"
        r"\bcups?\b|\bmugs?\b|\bgoblets?\b|\btumblers?\b"
    ),
    "fork": r"\bforks?\b",
    "knife": r"\bknife\b|\bknives\b",
    "spoon": r"\bspoons?\b",
    "chopsticks": r"\bchopsticks?\b",
    "napkin": r"\bnapkins?\b",
}
GENERIC_CUTLERY_PATTERN = r"\bcutlery\b|\bflatware\b|\bsilverware\b|\butensils?\b"
COUNT_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def evaluate_manipuland_completeness(case_pack: dict[str, Any]) -> list[dict[str, Any]]:
    """Return extra rule results for missing manipulands in established sets."""
    geometry = case_pack.get("scene_geometry") or {}
    objects = [
        obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    ]
    if not objects:
        return []

    surface_owner = _surface_owner_map(objects)
    objects_by_id = {str(obj["id"]): obj for obj in objects}
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
        result = _evaluate_dining_table_setting(
            table=table,
            surface_items=surface_items,
            objects_by_id=objects_by_id,
            task_instruction=str(case_pack.get("task_instruction") or ""),
        )
        if result is not None:
            results.append(result)
    return results


def _evaluate_dining_table_setting(
    *,
    table: dict[str, Any],
    surface_items: list[dict[str, Any]],
    objects_by_id: dict[str, dict[str, Any]],
    task_instruction: str,
) -> dict[str, Any] | None:
    counts = Counter()
    item_ids_by_group: dict[str, list[str]] = {key: [] for key in DINING_ITEM_GROUPS}
    for item in surface_items:
        text = _object_text(item)
        for group in DINING_ITEM_GROUPS:
            if _matches_item_group(group, text):
                counts[group] += 1
                item_ids_by_group[group].append(str(item["id"]))
                break

    required_groups = _required_groups(task_instruction)
    if not required_groups:
        return None
    requested_place_count = _requested_place_count(task_instruction)
    # 2026-07-12 修改原因：prompt 明确给出席位数时应作为权威数量，不能被
    # 相邻家具的语义噪声或 bbox 邻近误计放大；未明确数量时再由餐具锚点和座位推断。
    place_count = requested_place_count or max(
        counts["plate"],
        counts["bowl"],
        _nearby_dining_seat_count(table, objects_by_id),
    )
    # Only enforce this when geometry or an explicit setting count establishes plurality.
    if place_count < 2:
        return None

    missing: dict[str, int] = {}
    for group in sorted(required_groups):
        available = (
            sum(counts[item_group] for item_group in CUTLERY_GROUPS)
            if group == "cutlery"
            else counts[group]
        )
        deficit = place_count - available
        if deficit > 0:
            missing[group] = deficit

    related = sorted(
        {
            item_id
            for group in DINING_ITEM_GROUPS
            for item_id in item_ids_by_group[group]
        }
    )
    table_id = str(table["id"])
    if missing:
        missing_text = ", ".join(
            f"{group} x{count}" for group, count in sorted(missing.items())
        )
        counts_text = ", ".join(
            f"{group}={counts[group]}" for group in DINING_ITEM_GROUPS
        )
        # 2026-07-09 修改原因：critic 通过后物理后处理可能删除餐具/餐巾；
        # 最终规则报告必须显式暴露成组 tabletop manipuland 缺失。
        return {
            "check_id": f"manipuland_completeness__{table_id}__dining_place_setting",
            "metric": "functional_dependency",
            "label": "fail",
            "confidence": 0.9,
            "primary_object": table_id,
            "related_objects": related,
            "selected_related_objects": related,
            "blocking_objects": [],
            "reason": (
                f"Dining table has {place_count} place setting(s) implied by "
                f"the task/anchors/seats, but required tabletop items are missing: "
                f"{missing_text}. Counts: {counts_text}."
            ),
            "diagnostics": {
                "place_count": place_count,
                "counts": {group: counts[group] for group in DINING_ITEM_GROUPS},
                "required_groups": sorted(required_groups),
                "missing": missing,
            },
            "evidence": {
                "surface_item_ids": sorted(str(item["id"]) for item in surface_items)
            },
            "scoring_tier": "core",
        }

    return {
        "check_id": f"manipuland_completeness__{table_id}__dining_place_setting",
        "metric": "functional_dependency",
        "label": "pass",
        "confidence": 0.85,
        "primary_object": table_id,
        "related_objects": related,
        "selected_related_objects": related,
        "blocking_objects": [],
        "reason": (
            f"Dining table place setting is complete for {place_count} "
            "implied place setting(s)."
        ),
        "diagnostics": {
            "place_count": place_count,
            "counts": {group: counts[group] for group in DINING_ITEM_GROUPS},
            "required_groups": sorted(required_groups),
            "missing": {},
        },
        "evidence": {
            "surface_item_ids": sorted(str(item["id"]) for item in surface_items)
        },
        "scoring_tier": "core",
    }


def _surface_owner_map(objects: list[dict[str, Any]]) -> dict[str, str]:
    owners: dict[str, str] = {}
    for obj in objects:
        obj_id = str(obj.get("id") or "")
        for surface in obj.get("support_surfaces") or []:
            surface_id = str(surface.get("surface_id") or surface.get("id") or "")
            if surface_id:
                owners[surface_id] = obj_id
        for region in obj.get("support_regions") or []:
            region_id = str(region.get("region_id") or region.get("surface_id") or "")
            if region_id:
                owners[region_id] = obj_id
    return owners


def _nearby_dining_seat_count(
    table: dict[str, Any], objects_by_id: dict[str, dict[str, Any]]
) -> int:
    count = 0
    for obj in objects_by_id.values():
        if not _is_dining_seat(obj):
            continue
        gap = _bbox_gap_xy(table, obj)
        table_scale = _footprint_short_side(table)
        seat_scale = _footprint_short_side(obj)
        if gap is None or table_scale is None or seat_scale is None:
            continue
        # 2026-07-12 修改原因：餐椅归属按桌椅 bbox 间隙及双方尺寸缩放，
        # 避免 1.8m 中心半径在长桌、小桌或相邻桌组中误计座位。
        association_gap = max(0.6 * seat_scale, 0.15 * table_scale)
        if gap <= association_gap:
            count += 1
    return count


def _bbox_gap_xy(first: dict[str, Any], second: dict[str, Any]) -> float | None:
    first_bounds = _bbox_bounds_xy(first)
    second_bounds = _bbox_bounds_xy(second)
    if first_bounds is None or second_bounds is None:
        return None
    ax0, ay0, ax1, ay1 = first_bounds
    bx0, by0, bx1, by1 = second_bounds
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def _bbox_bounds_xy(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = obj.get("bbox_world") or {}
    minimum = bbox.get("min") or []
    maximum = bbox.get("max") or []
    if len(minimum) >= 2 and len(maximum) >= 2:
        return (
            float(minimum[0]),
            float(minimum[1]),
            float(maximum[0]),
            float(maximum[1]),
        )
    center = bbox.get("center") or []
    size = bbox.get("size") or []
    if len(center) < 2 or len(size) < 2:
        return None
    return (
        float(center[0]) - float(size[0]) / 2.0,
        float(center[1]) - float(size[1]) / 2.0,
        float(center[0]) + float(size[0]) / 2.0,
        float(center[1]) + float(size[1]) / 2.0,
    )


def _footprint_short_side(obj: dict[str, Any]) -> float | None:
    size = (obj.get("bbox_world") or {}).get("size") or []
    if len(size) < 2:
        return None
    positive = [float(value) for value in size[:2] if float(value) > 1e-6]
    return min(positive) if positive else None


def _required_groups(task_instruction: str) -> set[str]:
    text = task_instruction.lower().replace("_", " ")
    required = {
        group
        for group, pattern in PROMPT_GROUP_PATTERNS.items()
        if re.search(pattern, text)
    }
    if re.search(GENERIC_CUTLERY_PATTERN, text):
        # 2026-07-12 修改原因：generic cutlery 只表示每席需要可用餐具，不能
        # 强制西式 fork+knife+spoon 全套；明确点名的器具仍按各自类别检查。
        required.add("cutlery")
    return required


def _requested_place_count(task_instruction: str) -> int:
    text = task_instruction.lower().replace("_", " ")
    number = r"\d+|" + "|".join(COUNT_WORDS)
    patterns = (
        rf"\b(?:table|place)?\s*settings?\s+for\s+({number})\b",
        rf"\b({number})\s+(?:table|place)?\s*settings?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        token = match.group(1)
        value = int(token) if token.isdigit() else COUNT_WORDS[token]
        return value if value >= 2 else 0
    return 0


def _matches_item_group(group: str, text: str) -> bool:
    if group == "drinkware" and re.search(r"\bvase\b", text):
        # 2026-07-12 修改原因：材质词 glass 不能让玻璃花瓶冒充饮用玻璃杯。
        return False
    if group == "plate" and re.search(r"\bdecorative\b|\bwall plate\b", text):
        return False
    return re.search(DINING_ITEM_PATTERNS[group], text) is not None


def _placement_surface_id(obj: dict[str, Any]) -> str:
    placement = obj.get("placement_info") or {}
    return str(placement.get("parent_surface_id") or "")


def _scene_object_type(obj: dict[str, Any]) -> str:
    hints = obj.get("functional_hints") or {}
    return str(hints.get("scene_object_type") or obj.get("object_type") or "").lower()


def _is_dining_table(obj: dict[str, Any]) -> bool:
    text = _object_identity_text(obj)
    return (
        _scene_object_type(obj) == "furniture" and "dining" in text and "table" in text
    )


def _is_dining_seat(obj: dict[str, Any]) -> bool:
    text = _object_identity_text(obj)
    return _scene_object_type(obj) == "furniture" and any(
        token in text for token in ("chair", "seat", "stool", "bench")
    )


def _object_identity_text(obj: dict[str, Any]) -> str:
    """Return stable category identity without descriptive relation noise."""
    hints = obj.get("functional_hints") or {}
    # 2026-07-12 修改原因：VLM description/functional_categories 常提到相邻
    # dining table/chairs，若用于角色识别会把 sideboard 当餐桌或额外座位。
    parts = [
        obj.get("id"),
        obj.get("name"),
        obj.get("category"),
        obj.get("category_norm"),
        hints.get("category_group"),
    ]
    return " ".join(
        str(part).lower().replace("_", " ") for part in parts if part
    )


def _object_text(obj: dict[str, Any]) -> str:
    hints = obj.get("functional_hints") or {}
    parts = [
        obj.get("id"),
        obj.get("name"),
        obj.get("description"),
        obj.get("category"),
        obj.get("category_norm"),
        hints.get("category_group"),
        " ".join(str(item) for item in hints.get("functional_categories") or []),
    ]
    return " ".join(str(part).lower().replace("_", " ") for part in parts if part)
