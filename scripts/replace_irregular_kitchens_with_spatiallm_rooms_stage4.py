#!/usr/bin/env python3
"""Replace three irregular kitchens with exact ~15 m² SpatialLM rooms."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPLACEMENTS = {
    "000086": (
        "scene_001618_01_0",
        "children's bedroom",
        "Place one bunk bed, one compact desk with one chair, one wardrobe, one bookcase, one nightstand, one rug, and one basket while keeping a clear route.",
    ),
    "000087": (
        "scene_001335_02_0",
        "living room",
        "Arrange one sofa, one armchair, one coffee table, one television stand, one floor lamp, one rug, and one basket as a compact conversation area with clear circulation.",
    ),
    "000088": (
        "scene_001616_00_0",
        "master bedroom",
        "Place one double bed with two nightstands, one wardrobe, one chest of drawers, one armchair, one floor lamp, and one rug while preserving a clear route.",
    ),
}


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def polygon_area(vertices: list[list[float]]) -> float:
    return abs(sum(vertices[i][0] * vertices[(i + 1) % len(vertices)][1] - vertices[(i + 1) % len(vertices)][0] * vertices[i][1] for i in range(len(vertices))) / 2.0)


def build_prompt(room_type: str, shape_type: str, vertices: str, area: float, furnishing: str) -> str:
    return (
        f"Create exactly one irregular {room_type} using these ordered floor-boundary vertices in meters: {vertices}. "
        f"This {shape_type} polygon has an area of approximately {area:.2f} m². "
        "Use this exact simple polygon rather than its bounding rectangle; the vertices define the complete floor boundary, "
        f"and every complete furniture footprint must remain inside it. {furnishing}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    with args.source.open(newline="", encoding="utf-8") as handle:
        source_by_id = {row["SourceID"]: row for row in csv.DictReader(handle)}

    if len(rows) != 100 or sum(row["GeometryType"] == "irregular" for row in rows) != 30:
        raise ValueError("input must contain 100 rows with 30 irregular cases")

    by_prompt_id = {row["PromptID"]: row for row in rows}
    for prompt_id, (source_id, room_type, furnishing) in REPLACEMENTS.items():
        row = by_prompt_id.get(prompt_id)
        source = source_by_id.get(source_id)
        if row is None or row["GeometryType"] != "irregular" or row["RoomType"] != "kitchen":
            raise ValueError(f"{prompt_id} is not an irregular kitchen")
        if source is None or source["RoomType"] != room_type:
            raise ValueError(f"{source_id} is not the expected SpatialLM room type {room_type}")
        vertices = compact_json(json.loads(source["Vertices"]))
        area = polygon_area(json.loads(vertices))
        if not 14 <= area <= 16:
            raise ValueError(f"{source_id} area {area:.2f} is not approximately 15 m²")
        row.update({
            "SourcePromptID": source_id,
            "RoomType": room_type,
            "SceneEvalRoomType": room_type,
            "Width": "",
            "Length": "",
            "Vertices": vertices,
            "ShapeType": source["ShapeType"],
            "Area": f"{area:.2f}",
            "DimensionDescriptionRemoved": "false",
            "Prompt": build_prompt(room_type, source["ShapeType"], vertices, area, furnishing),
        })

    if any(row["RoomType"] == "kitchen" for row in rows if row["PromptID"] in REPLACEMENTS):
        raise AssertionError("replacement kitchen remains")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    print("replaced irregular kitchens with exact SpatialLM rooms: " + ", ".join(REPLACEMENTS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
