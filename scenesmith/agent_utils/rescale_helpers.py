"""Shared rescale operation logic for all placement agents.

This module provides the common implementation for rescaling objects across
all placement agents (furniture, manipuland, wall, ceiling).
"""

from __future__ import annotations

import logging

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from scenesmith.agent_utils.rescale_result import RescaleErrorType, RescaleResult
from scenesmith.agent_utils.response_datatypes import BoundingBox3D
from scenesmith.agent_utils.room import RoomScene, SceneObject
from scenesmith.agent_utils.sdf_generator import rescale_sdf

if TYPE_CHECKING:
    from scenesmith.agent_utils.asset_registry import AssetRegistry

console_logger = logging.getLogger(__name__)


@dataclass
class _ObjectScaleSnapshot:
    obj: SceneObject
    bbox_min: np.ndarray | None
    bbox_max: np.ndarray | None
    support_surfaces: list
    scale_factor: float


def _snapshot_object_scale(obj: SceneObject) -> _ObjectScaleSnapshot:
    return _ObjectScaleSnapshot(
        obj=obj,
        bbox_min=None if obj.bbox_min is None else obj.bbox_min.copy(),
        bbox_max=None if obj.bbox_max is None else obj.bbox_max.copy(),
        # apply_scale replaces this list; its elements are not mutated.
        support_surfaces=obj.support_surfaces,
        scale_factor=float(obj.scale_factor),
    )


def _restore_object_scale(snapshot: _ObjectScaleSnapshot) -> None:
    snapshot.obj.bbox_min = snapshot.bbox_min
    snapshot.obj.bbox_max = snapshot.bbox_max
    snapshot.obj.support_surfaces = snapshot.support_surfaces
    snapshot.obj.scale_factor = snapshot.scale_factor


def rescale_object_common(
    scene: RoomScene,
    object_id: str,
    scale_factor: float,
    object_type_name: str,
    asset_registry: AssetRegistry | None = None,
    post_validate: Callable[[list[SceneObject]], tuple[bool, str]] | None = None,
) -> RescaleResult:
    """Shared rescale logic for all placement agents.

    IMPORTANT: This rescales the ASSET (SDF file), affecting ALL instances that
    share the same sdf_path. This is intentional - if one instance is the wrong
    size, they all are.

    Args:
        scene: The RoomScene containing the object.
        object_id: ID of the object to rescale.
        scale_factor: Scale multiplier (e.g., 1.5 = 50% larger).
        object_type_name: Human-readable object type for messages (e.g., "furniture").
        asset_registry: Optional registry to update after rescaling. If provided,
            all registry entries with matching sdf_path will be updated.
        post_validate: Optional domain validator run against every affected scene
            instance after the tentative scale. A rejection rolls back the SDF,
            bounding boxes, runtime scale, and support surfaces atomically.

    Returns:
        RescaleResult with operation outcome.
    """
    # Validate scale factor.
    if scale_factor <= 0:
        return RescaleResult(
            success=False,
            message=f"Scale factor must be positive, got {scale_factor}",
            object_id=object_id,
            error_type=RescaleErrorType.INVALID_SCALE_FACTOR,
        )

    if scale_factor == 1.0:
        return RescaleResult(
            success=False,
            message="Scale factor of 1.0 has no effect",
            object_id=object_id,
            error_type=RescaleErrorType.NO_OP,
        )

    # Find the object.
    obj = scene.get_object(object_id)
    if obj is None:
        return RescaleResult(
            success=False,
            message=f"{object_type_name.capitalize()} '{object_id}' not found in scene",
            object_id=object_id,
            error_type=RescaleErrorType.OBJECT_NOT_FOUND,
        )

    # Check immutability.
    if obj.immutable:
        return RescaleResult(
            success=False,
            message=f"{object_type_name.capitalize()} '{object_id}' is immutable",
            object_id=object_id,
            error_type=RescaleErrorType.IMMUTABLE_OBJECT,
        )

    # Check SDF path exists.
    if obj.sdf_path is None or not obj.sdf_path.exists():
        return RescaleResult(
            success=False,
            message=f"{object_type_name.capitalize()} '{object_id}' has no valid SDF file",
            object_id=object_id,
            error_type=RescaleErrorType.NO_SDF,
        )

    sdf_path = obj.sdf_path

    # Capture previous dimensions.
    previous_dims = None
    if obj.bbox_min is not None and obj.bbox_max is not None:
        size = obj.bbox_max - obj.bbox_min
        previous_dims = BoundingBox3D(
            width=float(size[0]), depth=float(size[1]), height=float(size[2])
        )

    # Find ALL objects that share the same SDF path.
    # Note: scene.objects is a dict, so iterate over values not keys.
    affected_objects = [o for o in scene.objects.values() if o.sdf_path == sdf_path]
    affected_object_ids = [str(o.object_id) for o in affected_objects]
    registry_objects = (
        [
            asset
            for asset in asset_registry.list_all()
            if asset.sdf_path == sdf_path
        ]
        if asset_registry is not None
        else []
    )
    all_objects: list[SceneObject] = []
    seen_object_refs: set[int] = set()
    for affected in [*affected_objects, *registry_objects]:
        if id(affected) in seen_object_refs:
            continue
        seen_object_refs.add(id(affected))
        all_objects.append(affected)
    snapshots = [_snapshot_object_scale(affected) for affected in all_objects]
    try:
        original_sdf = sdf_path.read_bytes()
    except OSError as exc:
        return RescaleResult(
            success=False,
            message=f"Could not begin rescale transaction: {exc}",
            object_id=object_id,
            error_type=RescaleErrorType.RESCALE_FAILED,
        )

    console_logger.info(
        f"Rescaling {object_type_name} '{object_id}' by {scale_factor:.3f}x "
        f"(affects {len(affected_objects)} object(s))"
    )

    try:
        # Rescale the SDF file in-place.
        rescale_sdf(sdf_path=sdf_path, scale_factor=scale_factor)

        # Update all affected objects' bounding boxes and invalidate surfaces.
        for affected_obj in all_objects:
            affected_obj.apply_scale(scale_factor)

        invalid_dimensions = [
            str(affected.object_id)
            for affected in affected_objects
            if affected.bbox_min is not None
            and affected.bbox_max is not None
            and (
                not np.all(np.isfinite(affected.bbox_min))
                or not np.all(np.isfinite(affected.bbox_max))
                or np.any(affected.bbox_max <= affected.bbox_min)
            )
        ]
        if invalid_dimensions:
            raise ValueError(
                "Rescale produced invalid bounds for: "
                + ", ".join(invalid_dimensions)
            )

        if post_validate is not None:
            is_valid, validation_message = post_validate(affected_objects)
            if not is_valid:
                raise ValueError(validation_message or "Post-rescale validation failed")

        # Persist registry only after the complete mutation has passed.
        if asset_registry is not None and asset_registry.auto_save_path:
            asset_registry.save_to_file(asset_registry.auto_save_path)

    except Exception as e:
        rollback_errors: list[str] = []
        try:
            sdf_path.write_bytes(original_sdf)
        except OSError as rollback_error:
            rollback_errors.append(f"SDF restore failed: {rollback_error}")
        for snapshot in snapshots:
            try:
                _restore_object_scale(snapshot)
            except Exception as rollback_error:
                rollback_errors.append(
                    f"object {snapshot.obj.object_id} restore failed: {rollback_error}"
                )
        if asset_registry is not None and asset_registry.auto_save_path:
            try:
                asset_registry.save_to_file(asset_registry.auto_save_path)
            except Exception as rollback_error:
                rollback_errors.append(
                    f"registry restore failed: {rollback_error}"
                )
        console_logger.error(f"Failed to rescale SDF: {e}")
        rollback_note = (
            " Rollback diagnostics: " + "; ".join(rollback_errors)
            if rollback_errors
            else ""
        )
        return RescaleResult(
            success=False,
            message=f"Failed to rescale SDF: {e}.{rollback_note}",
            object_id=object_id,
            error_type=RescaleErrorType.RESCALE_FAILED,
        )

    # Capture new dimensions from the triggering object.
    new_dims = None
    if obj.bbox_min is not None and obj.bbox_max is not None:
        size = obj.bbox_max - obj.bbox_min
        new_dims = BoundingBox3D(
            width=float(size[0]), depth=float(size[1]), height=float(size[2])
        )

    # Get the new cumulative scale.
    new_asset_scale = obj.scale_factor

    # Build success message.
    if len(affected_objects) > 1:
        message = (
            f"Rescaled {object_type_name} '{object_id}' by {scale_factor:.2f}x. "
            f"Updated {len(affected_objects)} object(s) sharing this asset."
        )
    else:
        message = f"Rescaled {object_type_name} '{object_id}' by {scale_factor:.2f}x"

    if new_dims:
        message += (
            f". New size: {new_dims.width:.2f}m x {new_dims.depth:.2f}m "
            f"x {new_dims.height:.2f}m"
        )

    return RescaleResult(
        success=True,
        message=message,
        asset_id=str(sdf_path),
        object_id=object_id,
        affected_object_ids=affected_object_ids,
        scale_factor=scale_factor,
        new_asset_scale=new_asset_scale,
        previous_dimensions=previous_dims,
        new_dimensions=new_dims,
    )
