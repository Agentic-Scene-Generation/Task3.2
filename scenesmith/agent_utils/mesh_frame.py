"""Coordinate helpers for standard glTF meshes and SceneSmith geometry.

Blender exports canonical assets using glTF's Y-up convention, while SceneSmith
placement, dimensions, and bounding boxes use a Z-up convention.  Keeping this
conversion explicit prevents depth from being mistaken for height.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations

import numpy as np

ASSET_FRAME_CONTRACT_VERSION = "3.0"
CANONICAL_FRONT_AXIS = "+Y"
CANONICAL_UP_AXIS = "+Z"
CANONICAL_DIMENSION_ORDER = ("width", "depth", "height")


def scene_dimensions_to_gltf_y_up(
    dimensions: Sequence[float],
) -> list[float]:
    """Map SceneSmith ``[width, depth, height]`` to glTF axis extents."""
    if len(dimensions) != 3:
        raise ValueError(f"Expected three dimensions, got {dimensions}")
    width, depth, height = (float(value) for value in dimensions)
    return [width, height, depth]


def gltf_y_up_bounds_to_scene_z_up(
    bounds: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a glTF Y-up AABB to SceneSmith's Z-up object frame.

    Blender's inverse glTF axis mapping is ``(x, y, z) -> (x, -z, y)``.
    The sign change swaps the source Z minimum and maximum.
    """
    array = np.asarray(bounds, dtype=float)
    if array.shape != (2, 3):
        raise ValueError(f"Expected bounds with shape (2, 3), got {array.shape}")
    source_min, source_max = array
    scene_min = np.array([source_min[0], -source_max[2], source_min[1]], dtype=float)
    scene_max = np.array([source_max[0], -source_min[2], source_max[1]], dtype=float)
    return scene_min, scene_max


def validate_uniform_dimension_fit(
    actual_dimensions: Sequence[float],
    requested_dimensions: Sequence[float],
    *,
    min_ratio: float = 0.5,
    max_ratio: float = 1.75,
    epsilon: float = 1e-6,
) -> None:
    """Reject a semantically unsuitable mesh after uniform scaling.

    Uniform scaling preserves asset proportions.  A large residual mismatch on
    any axis therefore indicates that the retrieved object is the wrong shape
    (for example, a bench retrieved for a rug), not that more scaling is needed.
    """
    actual = np.asarray(actual_dimensions, dtype=float)
    requested = np.asarray(requested_dimensions, dtype=float)
    if actual.shape != (3,) or requested.shape != (3,):
        raise ValueError("Actual and requested dimensions must each have 3 values")
    if np.any(actual <= 0) or np.any(requested <= 0):
        raise ValueError(
            f"Dimensions must be positive, got actual={actual}, requested={requested}"
        )

    ratios = actual / requested
    # Mesh export/import and float32 bounds can move an exact boundary by a few
    # ULPs (for example 0.5 -> 0.49999997).  Treat that as numerical noise,
    # while preserving the semantic proportion gate itself.
    if np.any(ratios < min_ratio - epsilon) or np.any(ratios > max_ratio + epsilon):
        raise ValueError(
            "Uniformly scaled asset does not fit requested proportions: "
            f"actual={actual.tolist()}, requested={requested.tolist()}, "
            f"ratios={ratios.round(3).tolist()}, allowed=[{min_ratio}, {max_ratio}]"
        )


def uniform_scale_shape_error(
    actual_dimensions: Sequence[float], requested_dimensions: Sequence[float]
) -> float:
    """Return scale-invariant log-ratio error under the production scaler."""
    actual = np.asarray(actual_dimensions, dtype=float)
    requested = np.asarray(requested_dimensions, dtype=float)
    if actual.shape != (3,) or requested.shape != (3,):
        raise ValueError("Actual and requested dimensions must each have 3 values")
    if np.any(actual <= 0) or np.any(requested <= 0):
        return float("inf")
    uniform_scale = float(np.median(requested / actual))
    scaled = actual * uniform_scale
    return float(np.sum(np.abs(np.log(scaled / requested))))


def axis_agnostic_uniform_scale_shape_error(
    actual_dimensions: Sequence[float],
    requested_dimensions: Sequence[float],
) -> float:
    """Return the best cheap shape score before an asset frame is resolved.

    HSSD bounding boxes may be reported in the source frame, evaluated glTF
    frame, or an already aligned frame.  Their order is therefore not a semantic
    ``[width, depth, height]`` contract.  This score is intentionally
    axis-agnostic and is only suitable for candidate ranking.  Admission must
    happen after the asset's up/front axes have been canonicalized.
    """
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
    """Choose one feasible uniform scale under the shared dimension contract.

    The requested dimensions are first normalized to configured real-world
    family bounds.  A single scale factor is then chosen as close as possible to
    the designer request while satisfying both residual-ratio and family-size
    constraints.  If no such factor exists, the candidate's proportions are
    incompatible and the caller must try another asset.
    """
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
    """Score HSSD extents across its mixed evaluated coordinate frames.

    ``trimesh`` may expose an HSSD glTF's evaluated scene transform as Z-up,
    while aligned/exported candidates can remain in raw glTF Y-up.  Dataset
    metadata is sparse, so a single unconditional depth/height swap is not a
    valid corpus-wide contract.  Rank against both interpretations and let
    semantic admission plus post-canonical dimension validation decide.
    """
    return axis_agnostic_uniform_scale_shape_error(
        mesh_extents,
        requested_scene_dimensions,
    )
