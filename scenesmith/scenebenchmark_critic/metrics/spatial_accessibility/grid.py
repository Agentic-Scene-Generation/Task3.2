from __future__ import annotations

from collections import deque

import numpy as np

from scenesmith.scenebenchmark_critic.core.geometry import (
    GeometryStore,
    point_in_polygon_xy,
    polygon_bounds_xy,
)


def _build_grid(
    polygon: list[tuple[float, float]],
    resolution: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if len(polygon) < 3:
        return None
    xs_raw = [point[0] for point in polygon]
    ys_raw = [point[1] for point in polygon]
    margin = max(resolution, 0.05)
    x_values = np.arange(
        min(xs_raw) - margin, max(xs_raw) + margin + resolution, resolution
    )
    y_values = np.arange(
        min(ys_raw) - margin, max(ys_raw) + margin + resolution, resolution
    )
    if len(x_values) < 2 or len(y_values) < 2:
        return None
    xs, ys = np.meshgrid(x_values, y_values, indexing="xy")
    floor_mask = np.vectorize(
        lambda x, y: _point_in_polygon(float(x), float(y), polygon)
    )(xs, ys)
    return xs, ys, floor_mask.astype(bool)


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, point in enumerate(polygon):
        xi, yi = point
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (
            (yj - yi) or 1e-9
        ) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def _polygon_mask(
    xs: np.ndarray, ys: np.ndarray, polygon: list[tuple[float, float]]
) -> np.ndarray:
    x0, y0, x1, y1 = polygon_bounds_xy(polygon)
    candidates = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    mask = np.zeros(xs.shape, dtype=bool)
    rows, cols = np.where(candidates)
    for row, col in zip(rows.tolist(), cols.tolist()):
        mask[row, col] = point_in_polygon_xy(
            float(xs[row, col]), float(ys[row, col]), polygon
        )
    return mask


def _dilate_mask(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return mask
    height, width = mask.shape
    dilated = np.zeros_like(mask, dtype=bool)
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dr * dr + dc * dc > radius_cells * radius_cells:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(height, height - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(width, width - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(height, height + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(width, width + dc)
            dilated[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return dilated


def _entry_component(
    store: GeometryStore,
    walkable: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    floor_polygon: list[tuple[float, float]],
) -> np.ndarray | None:
    seeds = _opening_seed_points(store)
    if not seeds:
        return None
    best_seed: tuple[int, int] | None = None
    best_dist = float("inf")
    for sx, sy in seeds:
        if not point_in_polygon_xy(sx, sy, floor_polygon):
            continue
        dist = (xs - sx) ** 2 + (ys - sy) ** 2
        rows, cols = np.where(walkable)
        if len(rows) == 0:
            continue
        idx = int(np.argmin(dist[rows, cols]))
        row, col = int(rows[idx]), int(cols[idx])
        if float(dist[row, col]) < best_dist:
            best_seed = (row, col)
            best_dist = float(dist[row, col])
    if best_seed is None:
        return None
    return _component_from_seed(walkable, best_seed)


def _opening_seed_points(store: GeometryStore) -> list[tuple[float, float]]:
    seeds: list[tuple[float, float]] = []
    shell = store.raw.get("scene_shell") or {}
    for opening in list(shell.get("doors") or []) + list(shell.get("windows") or []):
        if not isinstance(opening, dict):
            continue
        for key in ("center", "position"):
            value = opening.get(key)
            if isinstance(value, list) and len(value) >= 2:
                seeds.append((float(value[0]), float(value[1])))
                break
        else:
            bbox = opening.get("bbox") or {}
            bmin = bbox.get("min") or []
            bmax = bbox.get("max") or []
            if len(bmin) >= 2 and len(bmax) >= 2:
                seeds.append(
                    (
                        (float(bmin[0]) + float(bmax[0])) / 2.0,
                        (float(bmin[1]) + float(bmax[1])) / 2.0,
                    )
                )
    return seeds


def _largest_component(walkable: np.ndarray) -> np.ndarray | None:
    height, width = walkable.shape
    visited = np.zeros_like(walkable, dtype=bool)
    best_cells: list[tuple[int, int]] = []
    for row in range(height):
        for col in range(width):
            if visited[row, col] or not walkable[row, col]:
                continue
            cells: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if nr < 0 or nc < 0 or nr >= height or nc >= width:
                        continue
                    if visited[nr, nc] or not walkable[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))
            if len(cells) > len(best_cells):
                best_cells = cells
    if not best_cells:
        return None
    component = np.zeros_like(walkable, dtype=bool)
    for row, col in best_cells:
        component[row, col] = True
    return component


def _component_from_seed(
    walkable: np.ndarray, seed: tuple[int, int]
) -> np.ndarray | None:
    height, width = walkable.shape
    row, col = seed
    if row < 0 or col < 0 or row >= height or col >= width or not walkable[row, col]:
        return None
    component = np.zeros_like(walkable, dtype=bool)
    queue: deque[tuple[int, int]] = deque([(row, col)])
    component[row, col] = True
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nr < 0 or nc < 0 or nr >= height or nc >= width:
                continue
            if component[nr, nc] or not walkable[nr, nc]:
                continue
            component[nr, nc] = True
            queue.append((nr, nc))
    return component
