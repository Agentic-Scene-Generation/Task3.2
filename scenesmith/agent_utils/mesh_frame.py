"""Coordinate helpers for glTF meshes and SceneSmith geometry."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def scene_dimensions_to_gltf_y_up(dimensions: Sequence[float]) -> list[float]:
    """Map SceneSmith ``[width, depth, height]`` to glTF Y-up extents."""
    if len(dimensions) != 3:
        raise ValueError(f"Expected three dimensions, got {dimensions}")
    width, depth, height = (float(value) for value in dimensions)
    return [width, height, depth]


def gltf_y_up_bounds_to_scene_z_up(
    bounds: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a glTF Y-up AABB to SceneSmith's Z-up object frame."""
    array = np.asarray(bounds, dtype=float)
    if array.shape != (2, 3):
        raise ValueError(f"Expected bounds with shape (2, 3), got {array.shape}")
    source_min, source_max = array
    scene_min = np.array([source_min[0], -source_max[2], source_min[1]])
    scene_max = np.array([source_max[0], -source_min[2], source_max[1]])
    return scene_min, scene_max


def validate_uniform_dimension_fit(
    actual_dimensions: Sequence[float],
    requested_dimensions: Sequence[float],
    *,
    min_ratio: float = 0.5,
    max_ratio: float = 1.75,
) -> None:
    """Reject a retrieved mesh whose proportions cannot fit uniformly."""
    actual = np.asarray(actual_dimensions, dtype=float)
    requested = np.asarray(requested_dimensions, dtype=float)
    if actual.shape != (3,) or requested.shape != (3,):
        raise ValueError("Actual and requested dimensions must each have 3 values")
    if np.any(actual <= 0) or np.any(requested <= 0):
        raise ValueError(
            f"Dimensions must be positive, got actual={actual}, requested={requested}"
        )
    ratios = actual / requested
    if np.any(ratios < min_ratio) or np.any(ratios > max_ratio):
        raise ValueError(
            "Uniformly scaled asset does not fit requested proportions: "
            f"actual={actual.tolist()}, requested={requested.tolist()}, "
            f"ratios={ratios.round(3).tolist()}, allowed=[{min_ratio}, {max_ratio}]"
        )
