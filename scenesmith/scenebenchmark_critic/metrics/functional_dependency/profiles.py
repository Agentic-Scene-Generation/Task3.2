from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scenesmith.scenebenchmark_critic.core.geometry import (
    is_small_object,
    object_affordances,
    object_category,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.constants import (
    BEDS,
    MEDIA,
    NIGHTSTANDS,
    SEATING,
    SUPPORT_CATEGORY_GROUPS,
    SUPPORT_INTERNAL_AFFORDANCES,
    SUPPORT_TOP_SURFACE_AFFORDANCES,
    WORK_SURFACE_CATEGORY_GROUPS,
    WORK_SURFACES,
)


@dataclass(frozen=True)
class ObjectFunctionProfile:
    can_support_top: bool = False
    has_internal_shelf: bool = False
    is_small_placeable: bool = False
    is_seating: bool = False
    is_work_surface: bool = False
    is_media_target: bool = False
    is_bedside_surface: bool = False
    is_sleeping_surface: bool = False
    source: str = "inferred"


def object_function_profile(obj: dict[str, Any]) -> ObjectFunctionProfile:
    category = object_category(obj)
    hints = obj.get("functional_hints") or {}
    group = str(hints.get("category_group") or "").strip().lower()
    affordances = object_affordances(obj)
    support_regions = obj.get("support_regions") or []
    region_kinds = {
        str(region.get("support_kind") or "").strip().lower()
        for region in support_regions
        if isinstance(region, dict)
    }
    access_type = hints.get("access_type") or {}
    primary_access = (
        str(access_type.get("primary") or "").strip().lower()
        if isinstance(access_type, dict)
        else ""
    )

    can_support_top = bool(
        affordances & SUPPORT_TOP_SURFACE_AFFORDANCES
        or "top_surface" in region_kinds
        or group in WORK_SURFACE_CATEGORY_GROUPS
    )
    has_internal_shelf = bool(
        affordances & SUPPORT_INTERNAL_AFFORDANCES
        or any(
            any(term in kind for term in ("shelf", "drawer", "cabinet", "storage"))
            for kind in region_kinds
        )
        or primary_access
        in {
            "front_open",
            "front-open",
            "open_shelf",
            "openable_storage",
            "internal_storage",
        }
        or group in SUPPORT_CATEGORY_GROUPS
    )
    is_work_surface = category in WORK_SURFACES or group in WORK_SURFACE_CATEGORY_GROUPS
    is_bedside_surface = category in NIGHTSTANDS
    is_sleeping_surface = category in BEDS or group == "sleeping"
    is_seating = category in SEATING or group == "seating"
    is_media_target = category in MEDIA or group == "media"

    inferred = ObjectFunctionProfile(
        can_support_top=can_support_top or is_work_surface or is_bedside_surface,
        has_internal_shelf=has_internal_shelf,
        is_small_placeable=is_small_object(obj) or group in {"small_object", "decor"},
        is_seating=is_seating,
        is_work_surface=is_work_surface,
        is_media_target=is_media_target,
        is_bedside_surface=is_bedside_surface,
        is_sleeping_surface=is_sleeping_surface,
    )
    explicit = obj.get("object_function_profile")
    if not isinstance(explicit, dict):
        return inferred
    return ObjectFunctionProfile(
        can_support_top=bool(explicit.get("can_support_top", inferred.can_support_top)),
        has_internal_shelf=bool(
            explicit.get("has_internal_shelf", inferred.has_internal_shelf)
        ),
        is_small_placeable=bool(
            explicit.get("is_small_placeable", inferred.is_small_placeable)
        ),
        is_seating=bool(explicit.get("is_seating", inferred.is_seating)),
        is_work_surface=bool(explicit.get("is_work_surface", inferred.is_work_surface)),
        is_media_target=bool(explicit.get("is_media_target", inferred.is_media_target)),
        is_bedside_surface=bool(
            explicit.get("is_bedside_surface", inferred.is_bedside_surface)
        ),
        is_sleeping_surface=bool(
            explicit.get("is_sleeping_surface", inferred.is_sleeping_surface)
        ),
        source="explicit",
    )
