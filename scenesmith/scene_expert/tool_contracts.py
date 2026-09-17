"""Side-effect-free checks at model-authored tool request boundaries."""

from __future__ import annotations

import math


def asset_generation_argument_error(
    object_descriptions: list[str],
    short_names: list[str],
    desired_dimensions: list[list[float]],
) -> str | None:
    """Explain an invalid batch without inventing dimensions or dropping items.

    Call before asset-budget accounting, size policies and retrieval. Parallel
    lists must stay aligned: downstream zip operations cannot validate them.
    """
    if not object_descriptions:
        return "object_descriptions must contain at least one object."
    if not (len(object_descriptions) == len(short_names) == len(desired_dimensions)):
        return (
            "object_descriptions, short_names and desired_dimensions must have "
            f"the same nonzero length; received {len(object_descriptions)}, "
            f"{len(short_names)} and {len(desired_dimensions)} respectively. "
            "Supply one [width, depth, height] in meters for every object."
        )
    for field, values in (
        ("object_descriptions", object_descriptions),
        ("short_names", short_names),
    ):
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value.strip():
                return f"{field}[{index}] must be a nonempty string."
    for index, dimensions in enumerate(desired_dimensions):
        if not isinstance(dimensions, (list, tuple)) or len(dimensions) != 3:
            return (
                f"desired_dimensions[{index}] must contain exactly three values: "
                "[width, depth, height] in meters."
            )
        for value in dimensions:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                return (
                    f"desired_dimensions[{index}] must contain only finite, "
                    "strictly positive numbers in meters."
                )
    return None
