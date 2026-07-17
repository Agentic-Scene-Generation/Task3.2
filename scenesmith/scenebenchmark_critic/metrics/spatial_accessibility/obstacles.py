from __future__ import annotations

import math

from typing import Any

import numpy as np

from scenesmith.scenebenchmark_critic.core.geometry import (
    GeometryStore,
    bbox_min_max_xy,
    is_walkway_obstacle,
    object_footprint_polygon,
)
from scenesmith.scenebenchmark_critic.metrics.spatial_accessibility.grid import (
    _dilate_mask,
    _polygon_mask,
)


def _obstacle_mask(
    store: GeometryStore,
    subject_id: str,
    xs: np.ndarray,
    ys: np.ndarray,
    params: dict[str, float],
    profile: dict[str, Any],
    *,
    ignored_object_ids: set[str] | None = None,
) -> np.ndarray:
    mask = np.zeros(xs.shape, dtype=bool)
    inflation = float(profile["clearance_width_m"]) * 0.5
    dilation_cells = int(math.ceil(inflation / params["grid_resolution_m"]))
    ignored = ignored_object_ids or set()
    for obj_id, obj in store.objects.items():
        if obj_id == subject_id or obj_id in ignored:
            continue
        if not is_walkway_obstacle(
            obj, height_threshold_m=params["height_threshold_m"]
        ):
            continue
        footprint = object_footprint_polygon(obj)
        if not footprint:
            bounds = bbox_min_max_xy(obj)
            if bounds is None:
                continue
            x0, y0, x1, y1 = bounds
            obstacle = (
                (xs >= x0 - inflation)
                & (xs <= x1 + inflation)
                & (ys >= y0 - inflation)
                & (ys <= y1 + inflation)
            )
            mask |= obstacle
            continue
        obstacle = _polygon_mask(xs, ys, footprint)
        mask |= _dilate_mask(obstacle, dilation_cells)
    return mask
