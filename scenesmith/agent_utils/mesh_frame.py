"""Coordinate helpers for glTF meshes and SceneSmith geometry."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations

import numpy as np

ASSET_FRAME_CONTRACT_VERSION = "3.0"
CANONICAL_FRONT_AXIS = "+Y"
CANONICAL_UP_AXIS = "+Z"
CANONICAL_DIMENSION_ORDER = ("width", "depth", "height")


def scene_dimensions_to_gltf_y_up(dimensions: Sequence[float]) -> list[float]:
    """Map SceneSmith ``[width, depth, height]`` to glTF Y-up extents."""
    if len(dimensions) != 3:
        raise ValueError(f"Expected three dimensions, got {dimensions}")
    width, depth, height = (float(value) for value in dimensions)
    return [width, height, depth]


def gltf_y_up_dimensions_to_scene_z_up(
    dimensions: Sequence[float],
) -> list[float]:
    """Map glTF Y-up extents to SceneSmith ``[width, depth, height]``."""
    if len(dimensions) != 3:
        raise ValueError(f"Expected three dimensions, got {dimensions}")
    width, height, depth = (float(value) for value in dimensions)
    return [width, depth, height]


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
    fit_axes: Sequence[int] = (0, 1, 2),
    epsilon: float = 1e-6,
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
    normalized_fit_axes = tuple(dict.fromkeys(int(axis) for axis in fit_axes))
    if not normalized_fit_axes or any(
        axis not in {0, 1, 2} for axis in normalized_fit_axes
    ):
        raise ValueError(f"fit_axes must contain axes 0, 1, or 2, got {fit_axes}")
    ratios = (actual / requested)[list(normalized_fit_axes)]
    if np.any(ratios < min_ratio - epsilon) or np.any(ratios > max_ratio + epsilon):
        raise ValueError(
            "Uniformly scaled asset does not fit requested proportions: "
            f"actual={actual.tolist()}, requested={requested.tolist()}, "
            f"ratios={ratios.round(3).tolist()}, allowed=[{min_ratio}, {max_ratio}]"
        )


def uniform_scale_shape_error(
    actual_dimensions: Sequence[float], requested_dimensions: Sequence[float]
) -> float:
    """Return scale-invariant log-ratio error under uniform scaling."""
    actual = np.asarray(actual_dimensions, dtype=float)
    requested = np.asarray(requested_dimensions, dtype=float)
    if actual.shape != (3,) or requested.shape != (3,):
        raise ValueError("Actual and requested dimensions must each have 3 values")
    if np.any(actual <= 0) or np.any(requested <= 0):
        return float("inf")
    scale = float(np.median(requested / actual))
    return float(np.sum(np.abs(np.log((actual * scale) / requested))))


def axis_agnostic_uniform_scale_shape_error(
    actual_dimensions: Sequence[float],
    requested_dimensions: Sequence[float],
) -> float:
    """Rank unresolved source-frame dimensions across all axis orders."""
    actual = np.asarray(actual_dimensions, dtype=float)
    requested = np.asarray(requested_dimensions, dtype=float)
    if actual.shape != (3,) or requested.shape != (3,):
        raise ValueError("Actual and requested dimensions must each have 3 values")
    return min(
        uniform_scale_shape_error(permutation, requested)
        for permutation in permutations(actual.tolist())
    )


def axis_agnostic_uniform_fit_exists(
    actual_dimensions: Sequence[float],
    requested_dimensions: Sequence[float],
    *,
    min_ratio: float,
    max_ratio: float,
) -> bool:
    """Return whether any unresolved axis order can meet a uniform-fit gate."""
    actual = np.asarray(actual_dimensions, dtype=float)
    requested = np.asarray(requested_dimensions, dtype=float)
    if actual.shape != (3,) or requested.shape != (3,):
        return False
    if np.any(actual <= 0) or np.any(requested <= 0):
        return False
    for permutation in permutations(actual.tolist()):
        ordered = np.asarray(permutation, dtype=float)
        predicted = ordered * float(np.median(requested / ordered))
        try:
            validate_uniform_dimension_fit(
                predicted,
                requested,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
            )
            return True
        except ValueError:
            continue
    return False


def choose_uniform_scale_for_contract(
    source_dimensions: Sequence[float],
    requested_dimensions: Sequence[float],
    *,
    min_ratio: float,
    max_ratio: float,
    minimum_dimensions: Sequence[float] | None = None,
    maximum_dimensions: Sequence[float] | None = None,
    enforce_requested_ratio: bool = True,
) -> tuple[float, np.ndarray]:
    """Choose one feasible scale under residual-ratio and family-size bounds."""
    source = np.asarray(source_dimensions, dtype=float)
    requested = np.asarray(requested_dimensions, dtype=float)
    if source.shape != (3,) or requested.shape != (3,):
        raise ValueError("Source and requested dimensions must each have 3 values")
    if np.any(source <= 0) or np.any(requested <= 0):
        raise ValueError(
            f"Dimensions must be positive, got source={source}, requested={requested}"
        )
    if min_ratio <= 0 or max_ratio < min_ratio:
        raise ValueError(f"Invalid dimension ratio interval [{min_ratio}, {max_ratio}]")

    normalized = requested.copy()
    minimum = None
    maximum = None
    if minimum_dimensions is not None:
        minimum = np.asarray(minimum_dimensions, dtype=float)
        if minimum.shape != (3,):
            raise ValueError("Minimum dimensions must contain 3 values")
        normalized = np.maximum(normalized, minimum)
    if maximum_dimensions is not None:
        maximum = np.asarray(maximum_dimensions, dtype=float)
        if maximum.shape != (3,):
            raise ValueError("Maximum dimensions must contain 3 values")
        normalized = np.minimum(normalized, maximum)
    if minimum is not None and maximum is not None and np.any(minimum > maximum):
        raise ValueError(
            f"Invalid family size bounds: minimum={minimum}, maximum={maximum}"
        )

    lower = 0.0
    upper = float("inf")
    if enforce_requested_ratio:
        lower = float(np.max(min_ratio * normalized / source))
        upper = float(np.min(max_ratio * normalized / source))
    if minimum is not None:
        lower = max(lower, float(np.max(minimum / source)))
    if maximum is not None:
        upper = min(upper, float(np.min(maximum / source)))
    if lower > upper + 1e-9:
        raise ValueError(
            "Asset proportions cannot satisfy the uniform dimension contract: "
            f"source={source.round(4).tolist()}, "
            f"normalized_target={normalized.round(4).tolist()}, "
            f"feasible_scale=[{lower:.5f}, {upper:.5f}]"
        )

    preferred = float(np.median(normalized / source))
    scale = float(np.clip(preferred, lower, upper))
    return scale, normalized


def hssd_dimension_shape_error(
    mesh_extents: Sequence[float],
    requested_scene_dimensions: Sequence[float],
) -> float:
    """Score HSSD extents without assuming one corpus-wide source frame."""
    return axis_agnostic_uniform_scale_shape_error(
        mesh_extents,
        requested_scene_dimensions,
    )
