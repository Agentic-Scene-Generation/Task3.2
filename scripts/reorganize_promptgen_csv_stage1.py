#!/usr/bin/env python3
"""Reorganize PromptGen v3.4 CSV into numeric/non-numeric rectangular groups.

Stage 1 intentionally does not change room geometry.  It only assigns the
35 rectangular rows with the most convenient text split to the numeric group
and normalizes geometry wording in the prose.  Runtime geometry remains in
Width/Length/Area and is injected by prepare_promptgen_scene_cases.py.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


# These patterns remove only geometric measurements, not furniture counts.
GEOMETRY_PATTERNS = (
    re.compile(r"\s+measuring\s+\d+(?:\.\d+)?\s*m\s*wide\s+by\s+\d+(?:\.\d+)?\s*m\s*long(?:,?\s*with\s+\d+(?:\.\d+)?\s*m²\s+of\s+area)?", re.I),
    re.compile(r"\s+measuring\s+\d+(?:\.\d+)?\s*m\s+by\s+\d+(?:\.\d+)?\s*m(?:\s*\([^)]*m²[^)]*\))?", re.I),
    re.compile(r"\s+in\s+a\s+rectangular\s+\d+(?:\.\d+)?\s*m\s+by\s+\d+(?:\.\d+)?\s*m\s+(?:layout|footprint|room|space),?", re.I),
    re.compile(r"\s+a\s+rectangular\s+\d+(?:\.\d+)?\s*m\s+by\s+\d+(?:\.\d+)?\s*m\s+(?:room|space|footprint),?", re.I),
    re.compile(r"\s+rectangular\s+\d+(?:\.\d+)?\s*m\s+by\s+\d+(?:\.\d+)?\s*m", re.I),
    re.compile(r"\s+\d+(?:\.\d+)?\s*m\s+wide\s+by\s+\d+(?:\.\d+)?\s*m\s+long", re.I),
    re.compile(r"\s+\d+(?:\.\d+)?\s*m\s+by\s+\d+(?:\.\d+)?\s*m(?:\s*\([^)]*m²[^)]*\))?", re.I),
    re.compile(r",?\s+(?:with\s+)?(?:an\s+)?area\s+of\s+\d+(?:\.\d+)?\s*m²(?:\s+of\s+area)?", re.I),
    re.compile(r",?\s+(?:totaling|covering|occupying)\s+\d+(?:\.\d+)?\s*m²(?:\s+of\s+area)?", re.I),
    re.compile(r",?\s+\d+(?:\.\d+)?\s*m²\s+of\s+(?:floor\s+)?(?:area|space)", re.I),
    re.compile(r",?\s+with\s+\d+(?:\.\d+)?\s*m²\s+of\s+(?:floor\s+)?(?:area|space)", re.I),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*m²\s*\)", re.I),
    re.compile(r",?\s+\d+(?:\.\d+)?\s*m²\b", re.I),
)


def strip_geometry(text: str) -> str:
    revised = text
    for pattern in GEOMETRY_PATTERNS:
        revised = pattern.sub("", revised)
    revised = re.sub(r"\s+,", ",", revised)
    revised = re.sub(r",\s*with,", ",", revised, flags=re.I)
    revised = re.sub(r",\s*with\s+arranged\b", " arranged", revised, flags=re.I)
    revised = re.sub(r",\s+(is|was|are|were)\b", r" \1", revised, flags=re.I)
    revised = re.sub(r",\s*,", ",", revised)
    revised = re.sub(r"\s{2,}", " ", revised)
    revised = re.sub(r"\s+([.!?])", r"\1", revised)
    return revised.strip(" ,")


def has_numeric_geometry(text: str) -> bool:
    return bool(re.search(r"\d+(?:\.\d+)?\s*m(?:²|\s+(?:wide|long|by))", text, re.I))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])

    rect = [row for row in rows if row["GeometryType"] == "rectangular"]
    irregular = [row for row in rows if row["GeometryType"] == "irregular"]
    if len(rect) != 70 or len(irregular) != 30:
        raise ValueError(f"expected 70 rectangular + 30 irregular rows, got {len(rect)} + {len(irregular)}")

    # The 31 out-of-range rows are placed in the non-numeric group first.
    # Add the four smallest in-range rows to fill the fixed 35-row group.
    out_of_range = [row for row in rect if not 10 <= float(row["Area"]) <= 20]
    in_range = [row for row in rect if 10 <= float(row["Area"]) <= 20]
    # Fill the fixed 35-row non-numeric group from overrepresented room
    # types, while retaining at least one numeric example for every type.
    by_room_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in in_range:
        by_room_type[row["RoomType"]].append(row)
    for candidates in by_room_type.values():
        candidates.sort(key=lambda row: (float(row["Area"]), row["PromptID"]))
    fill_rows: list[dict[str, str]] = []
    room_type_order = sorted(by_room_type, key=lambda room_type: (-len(by_room_type[room_type]), room_type))
    # Round-robin across the largest groups so the numeric group retains
    # kitchen/living/bedroom/children coverage instead of losing one type.
    for room_type in room_type_order:
        if len(fill_rows) >= 4:
            break
        if len(by_room_type[room_type]) > 1:
            fill_rows.append(by_room_type[room_type].pop(0))
    if len(fill_rows) != 4:
        raise AssertionError("cannot fill non-numeric group without removing a room type")
    non_numeric_ids = {row["PromptID"] for row in out_of_range + fill_rows}
    if len(non_numeric_ids) != 35:
        raise AssertionError(f"non-numeric group should have 35 rows, got {len(non_numeric_ids)}")

    numeric_ids = {row["PromptID"] for row in rect} - non_numeric_ids
    numeric_areas = [float(row["Area"]) for row in rect if row["PromptID"] in numeric_ids]
    if len(numeric_ids) != 35 or not all(10 <= area <= 20 for area in numeric_areas):
        raise AssertionError("numeric rectangular group is not exactly 35 rows in [10, 20] m²")

    for row in rows:
        if row["GeometryType"] == "rectangular":
            is_non_numeric = row["PromptID"] in non_numeric_ids
            prose = strip_geometry(row["Prompt"])
            if not is_non_numeric:
                width = float(row["Width"])
                length = float(row["Length"])
                area = float(row["Area"])
                prefix = f"The rectangular room measures {width:.2f} m by {length:.2f} m, with an area of {area:.2f} m². "
                prose = prefix + prose
            if is_non_numeric and has_numeric_geometry(prose):
                raise AssertionError(f"geometry text remains in non-numeric row {row['PromptID']}: {prose}")
            if not is_non_numeric and not has_numeric_geometry(prose):
                raise AssertionError(f"geometry text missing in numeric row {row['PromptID']}: {prose}")
            row["Prompt"] = prose
            row["DimensionDescriptionRemoved"] = "true" if is_non_numeric else "false"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"rectangular: numeric={len(numeric_ids)}, non_numeric={len(non_numeric_ids)}")
    print(f"numeric area range: {min(numeric_areas):.2f}-{max(numeric_areas):.2f} m²")
    print("non_numeric_ids=" + ",".join(sorted(non_numeric_ids)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
