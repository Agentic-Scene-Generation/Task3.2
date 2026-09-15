#!/usr/bin/env python3
"""Build the stage-5 PromptGen CSV with area-controlled furniture inventories."""

from __future__ import annotations

import csv
from pathlib import Path


SOURCE = Path(
    "/mnt/afs/visitor33/Datasets/promptGen/outputs/"
    "v3_4_cola_integrated_100_larger/"
    "prompts_70_rect_30_irregular_reorganized_stage4_spatiallm_15m2.csv"
)
OUTPUT = SOURCE.with_name(
    "prompts_70_rect_30_irregular_reorganized_stage5_area_controlled_furniture.csv"
)
SMOKE_OUTPUT = SOURCE.with_name(
    "prompts_stage5_smoke_rect_irregular_bedroom_living_room.csv"
)
SMOKE_PROMPT_IDS = ("000009", "000049", "000071", "000087")


# Only the furniture-enumeration suffix is replaced. The prefix retains the
# original geometry, room description, dimensions, shape, and stylistic text.
# Rugs remain named when they were present in the source prompt.
REWRITES: dict[str, tuple[str, str]] = {
    "000001": (
        "with one shower stall",
        "with one shower stall, one sink, and one slim cabinet for storage. "
        "The room should feel as fully furnished as its size allows.",
    ),
    "000002": (
        "It includes one shower stall",
        "It includes one shower stall, one sink, and one slim cabinet for organized "
        "storage. The room should feel as fully furnished as its size allows.",
    ),
    "000009": (
        "A double bed anchors",
        "Place one double bed, one nightstand, one wardrobe, and one soft rug. "
        "The room should feel as fully furnished as its size allows.",
    ),
    "000010": (
        "Furnish it with",
        "Furnish it with one double bed, one nightstand, one wardrobe, and one rug. "
        "The room should feel as fully furnished as its size allows.",
    ),
    "000011": (
        "with one double bed",
        "furnished with one double bed, one nightstand, one wardrobe, and one rug. "
        "The room should feel as fully furnished as its size allows.",
    ),
    "000012": (
        "The room centers on",
        "Place one bed, one nightstand, one wardrobe, and one rug. The room should "
        "feel as fully furnished as its size allows.",
    ),
    "000019": (
        "The room includes",
        "The room includes one bunk bed, one compact desk with one chair, one "
        "wardrobe, and one soft rug. The room should feel as fully furnished as its "
        "size allows.",
    ),
    "000020": (
        "A very full yet workable",
        "A practical rectangular children's bedroom is arranged with one bunk bed, "
        "one compact desk with one chair, one wardrobe, and one soft rug. The room "
        "should feel as fully furnished as its size allows.",
    ),
    "000021": (
        "arranged with a space-saving bunk bed",
        "arranged with one space-saving bunk bed, one compact desk with one chair, "
        "one wardrobe, and one soft rug. The room should feel as fully furnished as "
        "its size allows.",
    ),
    "000022": (
        "One bunk bed makes",
        "Include one bunk bed, one small desk with one chair, one wardrobe, and one "
        "rug. The room should feel as fully furnished as its size allows.",
    ),
    "000023": (
        "with a television receiver",
        "with one television receiver on one TV stand, one video game console, one "
        "compact sofa, and one rug. The room should feel as fully furnished as its "
        "size allows.",
    ),
    "000024": (
        "This balanced game room centers on",
        "This balanced game room centers on one television set on one compact TV "
        "stand, one video game console, one beanbag chair, and one small rug. The "
        "room should feel as fully furnished as its size allows.",
    ),
    "000025": (
        "uses a television receiver",
        "uses one television receiver on one TV stand, one video game console, one "
        "beanbag chair, and one small rug. The room should feel as fully furnished "
        "as its size allows.",
    ),
    "000026": (
        "with a television receiver",
        "with one television receiver on one TV stand, one video game console, one "
        "beanbag chair, and one rug. The room should feel as fully furnished as its "
        "size allows.",
    ),
    "000027": (
        "This rectangular game room centers on",
        "This rectangular game room centers on one television receiver on one TV "
        "stand, one video game console, one compact sofa, and one rug. The room "
        "should feel as fully furnished as its size allows.",
    ),
    "000028": (
        "A television receiver sits",
        "Include one television receiver on one TV stand, one video game console, "
        "one compact sofa, and one rug. The room should feel as fully furnished as "
        "its size allows.",
    ),
    "000029": (
        "with a television receiver",
        "with one television receiver on one slim TV stand, one video game console, "
        "one beanbag chair, and one rug. The room should feel as fully furnished as "
        "its size allows.",
    ),
    "000030": (
        "The setup centers on",
        "The setup centers on one television receiver, one video game console, one "
        "low TV stand, one beanbag chair, and one rug. The room should feel as fully "
        "furnished as its size allows.",
    ),
    "000031": (
        "one television receiver sits",
        "one television receiver sits on one slim TV stand with one video game "
        "console, one beanbag chair, and one rug. The room should feel as fully "
        "furnished as its size allows.",
    ),
    "000032": (
        "One television receiver sits",
        "Include one television receiver on one slim TV stand, one video game "
        "console, one compact sofa, and one rug. The room should feel as fully "
        "furnished as its size allows.",
    ),
    "000033": (
        "The room is furnished",
        "The room is furnished with one refrigerator, one stove, one sink, and three "
        "kitchen cabinets. The room should feel as fully furnished as its size "
        "allows.",
    ),
    "000034": (
        "The room includes",
        "The room includes one refrigerator, one stove, one sink, and three kitchen "
        "cabinets. The room should feel as fully furnished as its size allows.",
    ),
    "000035": (
        "It includes",
        "It includes one refrigerator, one stove, one sink, and three kitchen "
        "cabinets. The room should feel as fully furnished as its size allows.",
    ),
    "000036": (
        "with one refrigerator",
        "with one refrigerator, one stove, one sink, and three kitchen cabinets. The "
        "room should feel as fully furnished as its size allows.",
    ),
    "000037": (
        "One refrigerator",
        "Include one refrigerator, one stove, one sink, and three kitchen cabinets. "
        "The room should feel as fully furnished as its size allows.",
    ),
    "000038": (
        "one refrigerator",
        "one refrigerator, one stove, one sink, and three kitchen cabinets. The room "
        "should feel as fully furnished as its size allows.",
    ),
    "000039": (
        "with one refrigerator",
        "with one refrigerator, one stove, one sink, and three kitchen cabinets. The "
        "room should feel as fully furnished as its size allows.",
    ),
    "000040": (
        "One refrigerator anchors",
        "Include one refrigerator, one stove, one sink, and three kitchen cabinets. "
        "The room should feel as fully furnished as its size allows.",
    ),
    "000041": (
        "The compact layout includes",
        "The compact layout includes one refrigerator, one stove, one sink, and "
        "three kitchen cabinets. The room should feel as fully furnished as its size "
        "allows.",
    ),
    "000042": (
        "One refrigerator",
        "Include one refrigerator, one stove, one sink, and three kitchen cabinets. "
        "The room should feel as fully furnished as its size allows.",
    ),
    "000049": (
        "One slim sofa",
        "Include one slim sofa, one compact coffee table, and one rug. The room "
        "should feel as fully furnished as its size allows.",
    ),
    "000050": (
        "a compact sofa anchors",
        "one compact sofa, one narrow coffee table, and one warm rug form the main "
        "seating area. The room should feel as fully furnished as its size allows.",
    ),
    "000051": (
        "One sofa faces",
        "Include one sofa, one small coffee table, and one rug. The room should feel "
        "as fully furnished as its size allows.",
    ),
    "000052": (
        "with one slim sofa",
        "with one slim sofa, one narrow coffee table, and one small rug. The room "
        "should feel as fully furnished as its size allows.",
    ),
    "000062": (
        "In a rectangular master bedroom",
        "In the rectangular master bedroom, place one double bed, one nightstand, "
        "one wardrobe, and one rug. The room should feel as fully furnished as its "
        "size allows.",
    ),
    "000063": (
        "with a double bed",
        "furnished with one double bed, one nightstand, one wardrobe, and one soft "
        "rug. The room should feel as fully furnished as its size allows.",
    ),
    "000064": (
        "It contains",
        "It contains one bed, one nightstand, one wardrobe, and one rug. The room "
        "should feel as fully furnished as its size allows.",
    ),
    "000065": (
        "The room centers on",
        "Place one standard bed, one nightstand, one wardrobe, and one rug. The room "
        "should feel as fully furnished as its size allows.",
    ),
    "000066": (
        "The room is centered on",
        "Place one double bed, one nightstand, one wardrobe, and one soft rug. The "
        "room should feel as fully furnished as its size allows.",
    ),
    "000067": (
        "with a comfortable bed",
        "furnished with one comfortable bed, one nightstand, one wardrobe, and one "
        "rug. The room should feel as fully furnished as its size allows.",
    ),
    "000068": (
        "It features",
        "It features one double bed, one nightstand, one wardrobe, and one rug. The "
        "room should feel as fully furnished as its size allows.",
    ),
    "000069": (
        "furnished for sleeping and reading",
        "furnished with one double bed, one nightstand, one wardrobe, and one rug. "
        "The room should feel as fully furnished as its size allows.",
    ),
    "000070": (
        "with a double bed",
        "with one double bed, one nightstand, one wardrobe, and one warm rug. The "
        "room should feel as fully furnished as its size allows.",
    ),
    "000071": (
        "Place one bed",
        "Place one bed, one nightstand, one wardrobe, and one rug. The room should "
        "feel as fully furnished as its size allows.",
    ),
    "000072": (
        "Arrange one bed",
        "Arrange one bed, one nightstand, one wardrobe, and one rug. The room should "
        "feel as fully furnished as its size allows.",
    ),
    "000077": (
        "Furnish it with",
        "Furnish it with one space-saving bunk bed, one compact desk with one chair, "
        "one wardrobe, and one soft rug. The room should feel as fully furnished as "
        "its size allows.",
    ),
    "000078": (
        "Furnish it with",
        "Furnish it with one bunk bed, one compact desk with one chair, one wardrobe, "
        "and one rug. The room should feel as fully furnished as its size allows.",
    ),
    "000079": (
        "Furnish it with",
        "Furnish it with one bunk bed, one compact desk with one chair, one wardrobe, "
        "and one rug. The room should feel as fully furnished as its size allows.",
    ),
    "000080": (
        "Furnish it with",
        "Furnish it with one child-sized bed, one compact desk with one chair, one "
        "wardrobe, and one rug. The room should feel as fully furnished as its size "
        "allows.",
    ),
    "000081": (
        "Furnish the bedroom",
        "Furnish the bedroom with one double bed, one nightstand, and one wardrobe. "
        "The room should feel as fully furnished as its size allows.",
    ),
    "000087": (
        "Arrange one sofa",
        "Arrange one sofa, one coffee table, and one rug. The room should feel as "
        "fully furnished as its size allows.",
    ),
    "000088": (
        "Place one double bed",
        "Place one double bed, one nightstand, one wardrobe, and one rug. The room "
        "should feel as fully furnished as its size allows.",
    ),
}


def should_reduce(row: dict[str, str]) -> bool:
    area = float(row["Area"])
    geometry = row["GeometryType"]
    return (geometry == "rectangular" and area < 12.0) or (
        geometry == "irregular" and area < 15.0
    )


def main() -> None:
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    expected_ids = {row["PromptID"] for row in rows if should_reduce(row)}
    if expected_ids != set(REWRITES):
        missing = sorted(expected_ids - set(REWRITES))
        extra = sorted(set(REWRITES) - expected_ids)
        raise RuntimeError(f"rewrite coverage mismatch: missing={missing}, extra={extra}")

    changed_ids: list[str] = []
    for row in rows:
        prompt_id = row["PromptID"]
        if prompt_id not in REWRITES:
            continue
        marker, replacement = REWRITES[prompt_id]
        prompt = row["Prompt"]
        marker_count = prompt.count(marker)
        if marker_count != 1:
            raise RuntimeError(
                f"{prompt_id}: expected marker {marker!r} once, found {marker_count}"
            )
        prefix, _ = prompt.split(marker, 1)
        row["Prompt"] = prefix + replacement
        changed_ids.append(prompt_id)

    if len(rows) != 100 or len(changed_ids) != 52:
        raise RuntimeError(
            f"unexpected output counts: rows={len(rows)}, changed={len(changed_ids)}"
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rows_by_id = {row["PromptID"]: row for row in rows}
    smoke_rows = [rows_by_id[prompt_id] for prompt_id in SMOKE_PROMPT_IDS]
    with SMOKE_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(smoke_rows)

    print(f"wrote {OUTPUT}")
    print(f"rows={len(rows)} changed={len(changed_ids)} unchanged={len(rows)-len(changed_ids)}")
    print(f"wrote {SMOKE_OUTPUT}")
    print(f"smoke_rows={len(smoke_rows)} ids={','.join(SMOKE_PROMPT_IDS)}")


if __name__ == "__main__":
    main()
