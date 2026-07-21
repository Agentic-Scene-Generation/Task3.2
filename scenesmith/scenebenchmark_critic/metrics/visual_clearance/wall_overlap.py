"""Same-wall visual mesh overlap checks."""

from __future__ import annotations

from typing import Any

from scenesmith.scenebenchmark_critic.metrics.visual_clearance.classification import (
    is_wall_mounted_visual_subject,
)
from scenesmith.scenebenchmark_critic.metrics.visual_clearance.geometry import (
    rect_area,
    rect_intersection,
    wall_direction,
    wall_projection_rect,
)

PASS_OVERLAP_RATIO = 0.01
FAIL_OVERLAP_RATIO = 0.05
RELATION_TYPE = "wall_mounted_overlap"


def evaluate_wall_mounted_overlap(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    """Report each pair of overlapping decorative objects once."""
    geometry = case_pack.get("scene_geometry") or {}
    objects = [
        obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    ]
    subjects = [
        obj
        for obj in objects
        if is_wall_mounted_visual_subject(obj) and wall_direction(obj)
    ]
    results: list[dict[str, Any]] = []
    for index, first in enumerate(subjects):
        first_surface = str(
            (first.get("placement_info") or {}).get("parent_surface_id") or ""
        )
        first_direction = wall_direction(first)
        first_rect = wall_projection_rect(first, direction=first_direction)
        first_area = rect_area(first_rect)
        if first_rect is None or first_area <= 1e-9:
            continue
        for second in subjects[index + 1 :]:
            second_surface = str(
                (second.get("placement_info") or {}).get("parent_surface_id") or ""
            )
            if not first_surface or first_surface != second_surface:
                continue
            second_rect = wall_projection_rect(second, direction=first_direction)
            second_area = rect_area(second_rect)
            intersection = rect_intersection(first_rect, second_rect)
            overlap_area = rect_area(intersection)
            if second_rect is None or second_area <= 1e-9 or overlap_area <= 1e-9:
                continue
            ratio = overlap_area / min(first_area, second_area)
            primary, related = sorted(
                (first, second),
                key=lambda obj: (
                    rect_area(
                        wall_projection_rect(obj, direction=first_direction)
                    ),
                    str(obj["id"]),
                ),
            )
            primary_id = str(primary["id"])
            related_id = str(related["id"])
            if ratio <= PASS_OVERLAP_RATIO:
                label = "pass"
            elif ratio < FAIL_OVERLAP_RATIO:
                label = "degraded"
            else:
                label = "fail"
            results.append(
                {
                    "check_id": f"wall_overlap__{primary_id}__{related_id}",
                    "metric": "visual_clearance",
                    "label": label,
                    "confidence": 0.99,
                    "primary_object": primary_id,
                    "related_objects": [related_id],
                    "selected_related_objects": [related_id],
                    "blocking_objects": [related_id],
                    "relation_type": RELATION_TYPE,
                    "reason": (
                        f"Wall-mounted objects {primary_id!r} and {related_id!r} "
                        f"overlap by {overlap_area:.4f} m2 ({ratio * 100:.2f}% "
                        "of the smaller projected object)."
                    ),
                    "repair_advice": (
                        f"Move {primary_id!r} laterally or vertically on the same wall "
                        "inside wall bounds, clear of openings, furniture, and other "
                        "wall-mounted objects."
                    ),
                    "diagnostics": {
                        "parent_surface_id": first_surface,
                        "wall_direction": first_direction,
                        "intersection_area_m2": round(overlap_area, 6),
                        "smaller_projection_area_m2": round(
                            min(first_area, second_area), 6
                        ),
                        "overlap_ratio": round(ratio, 8),
                        "object_ids": [str(first["id"]), str(second["id"])],
                    },
                    "evidence": {
                        "constraint": "same_wall_projected_visual_mesh_overlap",
                        "drake_collision_proxy_used": False,
                    },
                    "evaluation_source": "scenesmith_wall_mounted_overlap",
                    "scoring_tier": "core",
                }
            )
    return results
