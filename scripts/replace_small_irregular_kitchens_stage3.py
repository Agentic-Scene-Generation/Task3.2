#!/usr/bin/env python3
"""Scale the three small irregular kitchens into usable 15+ m² polygons."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REPLACEMENT_SLOTS = ["000086", "000087", "000088"]
TARGET_AREAS = [16.20, 18.00, 19.50]
FURNISHING_VARIANTS = [
    "Use a practical galley arrangement with one refrigerator, one stove, one sink, one dishwasher, five kitchen cabinets, and one microwave. Leave a clear working route.",
    "Arrange one refrigerator, one stove, one sink, one dishwasher, six kitchen cabinets, and one microwave, with a compact pantry cabinet while preserving clear circulation.",
    "Furnish one refrigerator, one stove, one sink, four kitchen cabinets, one dishwasher, and one microwave. Add a narrow dining table with two chairs only where it leaves a clear route through the kitchen.",
]


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def polygon_area(vertices: list[list[float]]) -> float:
    return abs(sum(vertices[i][0] * vertices[(i + 1) % len(vertices)][1] - vertices[(i + 1) % len(vertices)][0] * vertices[i][1] for i in range(len(vertices))) / 2.0)


def polygon_centroid(vertices: list[list[float]]) -> tuple[float, float]:
    signed_twice_area = sum(vertices[i][0] * vertices[(i + 1) % len(vertices)][1] - vertices[(i + 1) % len(vertices)][0] * vertices[i][1] for i in range(len(vertices)))
    if abs(signed_twice_area) < 1e-9:
        return (sum(v[0] for v in vertices) / len(vertices), sum(v[1] for v in vertices) / len(vertices))
    cx = sum((vertices[i][0] + vertices[(i + 1) % len(vertices)][0]) * (vertices[i][0] * vertices[(i + 1) % len(vertices)][1] - vertices[(i + 1) % len(vertices)][0] * vertices[i][1]) for i in range(len(vertices))) / (3.0 * signed_twice_area)
    cy = sum((vertices[i][1] + vertices[(i + 1) % len(vertices)][1]) * (vertices[i][0] * vertices[(i + 1) % len(vertices)][1] - vertices[(i + 1) % len(vertices)][0] * vertices[i][1]) for i in range(len(vertices))) / (3.0 * signed_twice_area)
    return cx, cy


def scale_polygon(vertices: list[list[float]], target_area: float) -> list[list[float]]:
    current_area = polygon_area(vertices)
    factor = math.sqrt(target_area / current_area)
    cx, cy = polygon_centroid(vertices)
    return [[round(cx + factor * (x - cx), 6), round(cy + factor * (y - cy), 6)] for x, y in vertices]


def build_prompt(vertices: str, shape_type: str, area: float, furnishing: str) -> str:
    return (
        f"Create exactly one irregular kitchen using these ordered floor-boundary vertices in meters: {vertices}. "
        f"This {shape_type} polygon has an area of approximately {area:.2f} m². "
        "Use this exact simple polygon rather than its bounding rectangle; the vertices define the complete floor boundary, "
        f"and every complete furniture footprint must remain inside it. {furnishing}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    by_id = {row["PromptID"]: row for row in rows}
    if len(rows) != 100 or sum(row["GeometryType"] == "irregular" for row in rows) != 30:
        raise ValueError("input must contain 100 rows with 30 irregular cases")

    for slot, target_area, furnishing in zip(REPLACEMENT_SLOTS, TARGET_AREAS, FURNISHING_VARIANTS):
        row = by_id.get(slot)
        if row is None or row["GeometryType"] != "irregular" or row["RoomType"] != "kitchen":
            raise ValueError(f"{slot} is not an irregular kitchen")
        old_source = row["SourcePromptID"]
        old_vertices = json.loads(row["Vertices"])
        new_vertices = scale_polygon(old_vertices, target_area)
        computed_area = polygon_area(new_vertices)
        vertices_text = compact_json(new_vertices)
        row.update({
            "SourcePromptID": f"scaled_from_{old_source}",
            "Vertices": vertices_text,
            "Area": f"{computed_area:.2f}",
            "Prompt": build_prompt(vertices_text, row["ShapeType"], computed_area, furnishing),
            "DimensionDescriptionRemoved": "false",
        })

    if any(row["RoomType"] in {"game room", "bathroom"} for row in rows if row["GeometryType"] == "irregular"):
        raise AssertionError("irregular game room or bathroom remains")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    print("scaled irregular kitchens: " + ", ".join(REPLACEMENT_SLOTS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
