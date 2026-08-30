#!/usr/bin/env python3
"""Replace irregular game-room/bathroom rows with SpatialLM bedroom polygons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPLACEMENT_SLOTS = ["000071", "000072", "000081", "000082", "000083", "000084"]
BEDROOM_SOURCE_IDS = [
    "scene_001452_04_0",  # L-shaped, 10.19 m²
    "scene_001393_02_0",  # octagon, 11.96 m²
    "scene_001405_02_0",  # complex, 14.33 m²
    "scene_000289_01_0",  # octagon, 15.16 m²
    "scene_001456_00_0",  # complex, 17.29 m²
    "scene_001663_00_0",  # L-shaped, 18.95 m²
]

FURNISHING_VARIANTS = [
    "Place one bed, one nightstand, one wardrobe, and a compact desk with one chair. Add one rug and keep a clear route.",
    "Arrange one bed with two nightstands, one wardrobe, and one slim bookcase. Add one rug while preserving clear circulation.",
    "Furnish the bedroom with one double bed, two nightstands, one wardrobe, one chest of drawers, and a compact desk with one chair. Keep a clear route.",
    "Create a calm sleeping area with one double bed and two nightstands. Add one wardrobe, one armchair, one floor lamp, and one rug without blocking circulation.",
    "Use one bed, two nightstands, one wardrobe, one chest of drawers, one desk, one chair, and one bookcase, with a soft rug and clear circulation.",
    "Place one bed, two nightstands, one wardrobe, one chest of drawers, one bench, and one floor mirror. Add one rug and retain a clear route through the room.",
]


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_prompt(vertices: str, shape_type: str, area: float, furnishing: str) -> str:
    return (
        f"Create exactly one irregular bedroom using these ordered floor-boundary vertices in meters: {vertices}. "
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
        source_rows = list(csv.DictReader(handle))
    source_by_id = {row["SourceID"]: row for row in source_rows}

    if len(rows) != 100 or sum(row["GeometryType"] == "irregular" for row in rows) != 30:
        raise ValueError("input must contain the 100-row stage-1 CSV with 30 irregular rows")
    if any(slot not in {row["PromptID"] for row in rows} for slot in REPLACEMENT_SLOTS):
        raise ValueError("replacement prompt IDs are missing from input")

    replacements = {}
    for slot, source_id, furnishing in zip(REPLACEMENT_SLOTS, BEDROOM_SOURCE_IDS, FURNISHING_VARIANTS):
        source = source_by_id.get(source_id)
        if source is None or source["RoomType"] != "bedroom":
            raise ValueError(f"source {source_id} is not an ordinary bedroom")
        vertices = compact_json(json.loads(source["Vertices"]))
        area = float(source["Area"])
        replacements[slot] = {
            "SourcePromptID": source_id,
            "GeometryType": "irregular",
            "RoomType": "bedroom",
            "SceneEvalRoomType": "bedroom",
            "Width": "",
            "Length": "",
            "Vertices": vertices,
            "ShapeType": source["ShapeType"],
            "Area": f"{area:.2f}",
            "DimensionDescriptionRemoved": "false",
            "Prompt": build_prompt(vertices, source["ShapeType"], area, furnishing),
        }

    for row in rows:
        replacement = replacements.get(row["PromptID"])
        if replacement:
            for key, value in replacement.items():
                row[key] = value

    remaining_bad = [row["PromptID"] for row in rows if row["GeometryType"] == "irregular" and row["RoomType"] in {"game room", "bathroom"}]
    if remaining_bad:
        raise AssertionError(f"game/bathroom rows remain: {remaining_bad}")
    if sum(row["RoomType"] == "bedroom" and row["GeometryType"] == "irregular" for row in rows) < 10:
        raise AssertionError("expected at least ten irregular ordinary bedrooms after replacement")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"replaced {len(replacements)} irregular game/bathroom rows with ordinary bedrooms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
