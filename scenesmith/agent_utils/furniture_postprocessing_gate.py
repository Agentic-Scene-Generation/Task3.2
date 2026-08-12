"""Deterministic guards for furniture after projection/simulation and rendering."""

from __future__ import annotations

import math

from pathlib import Path
from typing import Any

import numpy as np

from PIL import Image, ImageStat

from scenesmith.agent_utils.room import ObjectType
from scenesmith.floor_plan_agents.tools.polygon_geometry import (
    room_geometry_covers_object,
)


def snapshot_furniture_transforms(scene: Any) -> dict[str, Any]:
    """Capture the last Designer-approved pose for every furniture object."""
    return {
        str(obj.object_id): obj.transform
        for obj in scene.objects.values()
        if getattr(obj, "object_type", None) == ObjectType.FURNITURE
    }


def repair_invalid_postprocessed_furniture(
    scene: Any,
    original_transforms: dict[str, Any],
    *,
    max_xy_displacement_m: float = 1.5,
    max_z_displacement_m: float = 0.5,
    floor_penetration_tolerance_m: float = 0.10,
    max_floor_gap_m: float = 0.25,
) -> list[dict[str, Any]]:
    """Rollback unsafe simulation poses; delete only when rollback is also invalid."""
    audit: list[dict[str, Any]] = []
    for obj in list(scene.objects.values()):
        if getattr(obj, "object_type", None) != ObjectType.FURNITURE:
            continue
        object_id = str(obj.object_id)
        original = original_transforms.get(object_id)
        reasons = _postprocess_reasons(
            scene,
            obj,
            original,
            max_xy_displacement_m=max_xy_displacement_m,
            max_z_displacement_m=max_z_displacement_m,
            floor_penetration_tolerance_m=floor_penetration_tolerance_m,
            max_floor_gap_m=max_floor_gap_m,
            check_displacement=True,
        )
        if not reasons:
            continue
        record: dict[str, Any] = {"object_id": object_id, "reasons": reasons}
        if original is not None:
            obj.transform = original
            rollback_reasons = _postprocess_reasons(
                scene,
                obj,
                None,
                max_xy_displacement_m=max_xy_displacement_m,
                max_z_displacement_m=max_z_displacement_m,
                floor_penetration_tolerance_m=floor_penetration_tolerance_m,
                max_floor_gap_m=max_floor_gap_m,
                check_displacement=False,
            )
            if not rollback_reasons:
                record["action"] = "rolled_back_to_pre_simulation_transform"
                audit.append(record)
                continue
            record["rollback_reasons"] = rollback_reasons
        scene.remove_object(obj.object_id)
        record["action"] = "removed_after_invalid_rollback"
        audit.append(record)
    return audit


def validate_render_directory(render_dir: Path) -> list[Path]:
    """Reject missing, black/flat, or effectively empty render outputs."""
    render_dir = Path(render_dir)
    image_paths = sorted(render_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"Render validity gate found no PNG files in {render_dir}")
    valid: list[Path] = []
    failures: list[str] = []
    for path in image_paths:
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                if image.width < 32 or image.height < 32:
                    failures.append(f"{path.name}: canvas too small")
                    continue
                array = np.asarray(image, dtype=np.uint8)
                nonblack_ratio = float(np.mean(np.max(array, axis=2) > 8))
                extrema = ImageStat.Stat(image).extrema
                dynamic_range = max(high - low for low, high in extrema)
                if nonblack_ratio < 0.01:
                    failures.append(f"{path.name}: almost entirely black")
                    continue
                if dynamic_range < 3:
                    failures.append(f"{path.name}: flat/solid-color image")
                    continue
                valid.append(path)
        except Exception as exc:
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
    if not valid:
        raise RuntimeError("Render validity gate failed: " + "; ".join(failures))
    return valid


def _postprocess_reasons(
    scene: Any,
    obj: Any,
    original_transform: Any | None,
    *,
    max_xy_displacement_m: float,
    max_z_displacement_m: float,
    floor_penetration_tolerance_m: float,
    max_floor_gap_m: float,
    check_displacement: bool,
) -> list[str]:
    reasons: list[str] = []
    if not room_geometry_covers_object(scene.room_geometry, obj):
        reasons.append("POLYGON_CONTAINMENT_CONFLICT")
    bounds = obj.compute_world_bounds()
    if bounds is None:
        reasons.append("MISSING_WORLD_BOUNDS")
        return reasons
    lower, upper = bounds
    bottom_z = float(lower[2])
    center_z = float((lower[2] + upper[2]) / 2.0)
    if not math.isfinite(bottom_z) or not math.isfinite(center_z):
        reasons.append("NONFINITE_HEIGHT")
    elif bottom_z < -floor_penetration_tolerance_m or center_z < -max_z_displacement_m:
        reasons.append("BELOW_FLOOR")
    elif bottom_z > max_floor_gap_m:
        reasons.append("NO_FLOOR_SUPPORT")
    if check_displacement and original_transform is not None:
        before = np.asarray(original_transform.translation(), dtype=float)
        after = np.asarray(obj.transform.translation(), dtype=float)
        if float(np.linalg.norm(after[:2] - before[:2])) > max_xy_displacement_m:
            reasons.append("EXTREME_XY_DISPLACEMENT")
        if abs(float(after[2] - before[2])) > max_z_displacement_m:
            reasons.append("EXTREME_Z_DISPLACEMENT")
    return reasons
