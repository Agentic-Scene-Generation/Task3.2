"""Pure-Python render view construction helpers."""

import math


def generate_angled_drawer_view(
    surface: dict,
    joint_name: str,
    drawer_direction: list[float] | None = None,
    view_index: int = 0,
) -> dict:
    """Build a serializable camera direction looking into an open drawer."""
    surface_id = surface.get("surface_id", f"surface_{view_index}")

    if drawer_direction is not None:
        dx, dy, _ = drawer_direction
        horizontal_magnitude = math.sqrt(dx * dx + dy * dy)
        if horizontal_magnitude > 0.01:
            horizontal_scale = 0.7
            direction = [
                dx / horizontal_magnitude * horizontal_scale,
                dy / horizontal_magnitude * horizontal_scale,
                0.7,
            ]
        else:
            direction = [0.0, 0.7, 0.7]
    else:
        direction = [0.0, 0.7, 0.7]

    norm = math.sqrt(sum(component * component for component in direction))
    normalized_direction = [component / norm for component in direction]
    return {
        "name": f"drawer_{joint_name}_{surface_id}",
        "direction": normalized_direction,
        "is_side": False,
        "surface_data": surface,
        "is_drawer_view": True,
    }
