#!/usr/bin/env python3
"""Complete asset-local semantic direction annotations for the HSSD library."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any


AXES = {
    "front": ([0.0, 0.0, 1.0], "+Z asset-local", 3, "back.png"),
    "up": ([0.0, 1.0, 0.0], "+Y asset-local", None, "top.png"),
    "down": ([0.0, -1.0, 0.0], "-Y asset-local", None, "bottom (not in six-view bundle)"),
}

# These are functional faces, not merely the gravity up axis of every upright asset.
DOWN_CATEGORIES = {
    "ceiling fan", "ceiling lamp", "ceiling lamp.n.01,track lighting",
    "chandelier", "pendant lamp", "range hood", "showerhead", "streetlight",
    "track lighting", "table lamp", "floor lamp", "lamp", "lantern",
    "pathway light",
}

UP_CATEGORIES = {
    "air hockey table", "ashcan", "ashtray", "basket", "bath mat",
    "bathroom scale", "birdbath", "bowl", "bucket", "caddy", "cafeteria tray",
    "cake stand", "casserole", "chopping board", "coaster", "coffee table",
    "coffee table.n.01,aquarium", "colander", "cookie sheet", "countertop",
    "crate", "cup", "dining table", "electric frying pan", "end table",
    "frying pan", "game table.n.01,chessboard", "glass", "hot tub",
    "kitchen island", "mat", "mixing bowl", "mousepad", "pan", "pet bowl",
    "picnic rug", "picnic table", "place mat", "planter", "plate", "pond",
    "pool table", "punch bowl", "rug", "salver", "saucepan", "shower pan",
    "sink", "soap dish", "stove", "stove.n.01,oven", "sugar bowl",
    "swimming pool", "table", "table runner", "table-tennis table", "tray",
    "tureen", "wok", "workbench", "potted plant", "plant", "flower",
    "flower in vase", "vase", "hamper", "storage box", "jar", "bottle",
    "canister", "carafe", "jug", "kettle", "pitcher", "teapot", "thermos",
    "tumbler", "wineglass", "wine bottle", "cocktail shaker", "cruet",
    "wine bucket", "umbrella stand", "pen holder", "spice holder",
    "soap dispenser", "watering can", "pestle.n.03,mortar", "roaster",
    "tissue box", "toilet brush.n.01,toilet brush holder", "candle",
    "candlestick", "candlestick.n.01,candle", "candle.n.01,candlestick",
    "candelabrum", "candle.n.01,candlestick.n.01,wall mirror", "firepit",
    "punching bag", "umbrella", "christmas tree", "clothes tree",
    "coatrack", "hall tree", "plant stand", "trampoline", "drum",
    "cake", "butter dish", "soda can", "purse", "bag", "shopping bag",
    "overnighter", "shoebox", "gift box", "toiletry.n.01,bottle",
    "toiletry.n.01,perfume.n.02,bottle",
}

# Categories whose function/display/access side provides a stable horizontal face.
HORIZONTAL_EXACT = {
    "air conditioner", "alarm clock", "aquarium", "audio system", "bar",
    "barbecue", "base cabinet", "bench grinder", "bicycle", "birdcage",
    "birdhouse", "blackboard", "blender", "board game", "board game.n.01,chess",
    "book", "bookcase", "bread-bin", "bulletin board", "cabinet", "camera",
    "camcorder", "car", "cellular telephone", "chain saw", "chest",
    "computer screen", "cooker.n.01,steamer", "credenza", "darts", "desk",
    "desk calendar", "dish rack", "dishwasher", "dollhouse", "door", "doorbell",
    "double door", "drawer", "drawer unit", "drawer unit.n.01,file",
    "dressing table", "dryer", "dvd player", "easel", "electric fan",
    "elevator", "espresso maker", "exercise bike", "faucet", "file", "fireplace",
    "garage door", "gate", "grandfather clock", "guitar", "gym equipment",
    "hand glass", "handcart", "headboard", "heating system", "ironing board",
    "jewelry box", "kitchen appliance", "kitchen appliance.n.01,slicer",
    "kitchen scale", "kitchen timer", "knocker", "ladder", "laptop",
    "lawn mower", "lectern", "loudspeaker", "mailbox", "makeup mirror",
    "mantel clock", "mantel clock.n.01,radio receiver", "media player",
    "medicine chest", "microphone", "microwave", "mirror", "mixer", "monitor",
    "motorcycle", "music stand", "oven", "paper organizer", "piano",
    "pinball machine", "postbox", "power saw", "printer", "projector",
    "radio receiver", "radio receiver.n.01,alarm clock", "record player",
    "refrigerator", "safe", "sewing machine", "shoe rack",
    "shoe rack.n.01,cabinet", "shower faucet", "shower stall", "sink stand",
    "socket", "space heater", "switch", "switch.n.01,dimmer", "telephone",
    "telescope", "television receiver", "timer", "toaster", "toaster oven",
    "toilet", "toilet flush plate", "treadmill", "urinal", "vacuum",
    "video game console", "wall mounted screen", "wall organizer", "wall panel",
    "wall sign", "wall socket", "wall sticker", "wall unit", "washer",
    "washer.n.03,dryer", "water heater", "whiteboard", "window",
    "chest of drawers", "chest of drawers.n.01,end table", "nightstand",
    "buffet", "picture frame", "tv stand", "bathtub",
    "bathtub.n.01,shower stall", "chaise longue", "l-shaped couch",
    "storage bench", "storage box", "trunk", "blanket chest", "toy box",
    "sculpture", "plush toy", "shoe", "serving cart", "clothing rack",
    "towel rack", "towel rail", "towel ring", "towel rack.n.01,radiator",
    "coffee maker", "crib", "railing", "stairway.n.01,stairs", "fence",
    "gazebo", "playhouse", "playhouse.n.01,play area", "pet house",
    "cabin", "shed", "greenhouse", "room divider", "trellis", "balcony",
    "roof", "hammock", "swing", "swing.n.02,play area", "slide",
    "playpen", "play area", "drying rack", "clothes dryer.n.01,drying rack",
    "rack", "magazine rack", "bicycle rack", "luggage rack", "wine rack",
    "ladder bookcase", "hook", "grab bar", "handle", "broom", "spade",
    "reamer", "strainer", "bottle opener", "barrow", "scooter",
    "water scooter", "surfboard", "skateboard", "foosball table",
    "tablet computer", "videodisk", "notebook", "binder", "magazine",
    "chess", "subwoofer", "earphone", "spectacles", "fan", "radiator",
    "shower caddy", "bidet", "washbasin", "pedestal sink", "sauna",
    "toilet brush", "toilet paper holder", "paper towel.n.01,paper towel holder",
    "console table", "ottoman", "mattress", "cradle", "tent", "step ladder",
    "conference table", "desk organizer", "spicemill", "coffeepot",
    "pepper mill", "spice rack", "cage", "smoke detector", "mantel",
    "tapestry", "valve", "hourglass", "bookend", "potholder",
    "throw pillow", "cushion", "bolster", "seat cushion",
}

HORIZONTAL_TERMS = (
    "armchair", "armoire", "bed", "bench", "cabinet", "chair", "clock",
    "curtain", "door", "drawer", "dryer", "fridge", "headboard", "mirror", "poster",
    "screen", "shelf", "shelving", "sofa", "stool", "television", "toilet",
    "wall art", "wall calendar", "wall decor", "wall hook", "wall lamp",
    "wardrobe", "washer", "window blind", "window curtain", "window shade",
)

# Asset categories for which no stable face/direction can honestly be inferred
# from category semantics and the canonical upright pose alone.
NO_STABLE_CATEGORIES = {
    "armrest", "ball", "beam", "blanket", "bolster", "bouquet", "bridge",
    "christmas stocking", "coaster", "cushion", "football", "globe", "hedge",
    "hobby", "magnet", "mobile", "napkin", "net", "paperweight", "plaything",
    "quilt", "rock", "soap", "soccer ball",
    "spoon", "string lights", "throw", "towel", "bath towel",
    "bathrobe", "weight", "wind chime", "wreath",
}

SUPPLEMENTAL_UP_TERMS = (
    "bottle", "canister", "carafe", "jar", "jug", "kettle", "pitcher",
    "teapot", "thermos", "tumbler", "vase", "wineglass",
)


def load_lookup(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_lookup(path: Path, lookup: dict[str, dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8") as handle:
        json.dump(lookup, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    temp.replace(path)


def write_records(path: Path, lookup: dict[str, dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for hssd_id in sorted(lookup):
            handle.write(json.dumps(lookup[hssd_id], ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def has_term(category: str, terms: tuple[str, ...]) -> bool:
    return any(term in category for term in terms)


def category_policy(category: str, prior: dict[str, Any]) -> dict[str, Any]:
    existing_semantic = (
        bool(prior.get("canonical_orientation_is_semantic_front"))
        and prior.get("canonical_orientation_source")
        != "full_library_category_semantic_direction_audit"
    )
    directions: list[str] = []
    primary: str | None = None
    source = "explicit_category_semantics"

    if existing_semantic:
        primary = "front"
        directions.append("front")
        source = "preserve_existing_asset_verified_front"
    elif category in NO_STABLE_CATEGORIES:
        source = "explicit_no_stable_category_direction"
    elif category in DOWN_CATEGORIES:
        primary = "down"
        directions.append("down")
    elif category in UP_CATEGORIES:
        primary = "up"
        directions.append("up")
    elif category in HORIZONTAL_EXACT or has_term(category, HORIZONTAL_TERMS):
        primary = "front"
        directions.append("front")

    if category in UP_CATEGORIES or has_term(category, SUPPLEMENTAL_UP_TERMS):
        if "up" not in directions:
            directions.append("up")
    if category in DOWN_CATEGORIES and "down" not in directions:
        directions.append("down")

    return {
        "primary_direction_kind": primary,
        "semantic_direction_kinds": directions,
        "has_semantic_direction": primary is not None,
        "policy_source": source,
    }


def direction_entry(kind: str, confidence: float, primary: bool) -> dict[str, Any]:
    axis, label, view_index, view_name = AXES[kind]
    face = {
        "front": "functional, display, access, or user-facing horizontal face",
        "up": "upward-facing opening, receiving, support, or usable surface",
        "down": "downward-facing output, airflow, water, or light-emission face",
    }[kind]
    return {
        "kind": kind,
        "axis": axis,
        "axis_frame": "asset_local_hssd_y_up",
        "direction": label,
        "semantic_face": face,
        "confidence": confidence,
        "is_primary": primary,
        "is_strict_positive_front": kind == "front",
        "render_evidence_view": view_name,
        "front_view_image_index": view_index,
    }


def patch_record(record: dict[str, Any], policy: dict[str, Any]) -> str:
    front = record.setdefault("canonical_front", {})
    kinds = policy["semantic_direction_kinds"]
    primary = policy["primary_direction_kind"]
    vertical_kinds = [kind for kind in kinds if kind in {"up", "down"}]
    record["functional_directions"] = [
        direction_entry(kind, 0.82 if kind == primary else 0.74, kind == primary)
        for kind in vertical_kinds
    ]
    for direction in record["functional_directions"]:
        direction["is_strict_positive_front"] = False
        direction["direction_role"] = "non_front_functional_direction"

    if not primary:
        front["asset_local_front_axis"] = [0.0, 0.0, 1.0]
        front["canonical_front_direction"] = "+Z asset-local horizontal fallback"
        front["canonical_orientation_axis"] = [0.0, 0.0, 1.0]
        front["canonical_orientation_axis_frame"] = "asset_local"
        front["canonical_orientation_confidence"] = 0.2
        front["canonical_orientation_source"] = "fallback_asset_library_horizontal_axis"
        front["semantic_directions"] = []
        front["semantic_direction_kind"] = None
        front["canonical_orientation_is_semantic_front"] = False
        front["is_strict_front"] = False
        front["is_strict_positive_front"] = False
        front["validation_status"] = "explicit_no_stable_category_direction"
        front["semantic_direction_audit_status"] = "explicit_no_stable_category_direction"
        return "no_stable_direction"

    preserved = policy["policy_source"] == "preserve_existing_asset_verified_front"
    confidence = float(front.get("canonical_orientation_confidence") or 0.0) if preserved else 0.82
    confidence = max(confidence, 0.82 if primary != "front" else 0.78)
    horizontal_entries = [
        direction_entry("front", confidence, True)
    ] if primary == "front" else []
    front["semantic_directions"] = horizontal_entries
    front["semantic_direction_kind"] = "front" if primary == "front" else None
    front["semantic_direction_schema_version"] = "hssd_semantic_direction@1.0"
    front["semantic_direction_audit_status"] = (
        "preserved_asset_verified" if preserved else "category_semantics_verified_axis"
    )

    if preserved:
        # Keep the already audited per-asset horizontal axis and its confidence.
        horizontal_entries[0]["axis"] = list(front["canonical_orientation_axis"])
        horizontal_entries[0]["direction"] = front.get("canonical_front_direction")
        horizontal_entries[0]["front_view_image_index"] = front.get("front_view_image_index")
        horizontal_entries[0]["render_evidence_view"] = front.get("front_view_image_name")
        horizontal_entries[0]["is_strict_positive_front"] = bool(front.get("is_strict_positive_front"))
        return "preserved"

    if primary in {"up", "down"}:
        front.update(
            {
                "asset_local_front_axis": [0.0, 0.0, 1.0],
                "canonical_front_direction": "+Z asset-local horizontal fallback",
                "canonical_front_face": None,
                "canonical_orientation_axis": [0.0, 0.0, 1.0],
                "canonical_orientation_axis_frame": "asset_local",
                "canonical_orientation_confidence": 0.2,
                "canonical_orientation_is_semantic_front": False,
                "canonical_orientation_source": "fallback_asset_library_horizontal_axis",
                "confidence": 0.2,
                "front_strictness_label": "default_horizontal_direction_not_strict",
                "front_view_image_index": None,
                "front_view_image_name": None,
                "is_strict_front": False,
                "is_strict_positive_front": False,
                "method": "horizontal_fallback_plus_separate_functional_direction",
                "notes": (
                    f"No stable horizontal semantic front; {primary} is stored "
                    "separately in record.functional_directions."
                ),
                "status": "default_horizontal_direction",
                "validation_status": "functional_direction_not_canonical_front",
                "world_front_axis": None,
            }
        )
        front["semantic_direction_audit_status"] = (
            "vertical_functional_direction_separate_from_horizontal_front"
        )
        front["evidence"] = [
            item
            for item in (front.get("evidence") or [])
            if not str(item).startswith("full-library-semantic-direction-v1:")
        ]
        return f"promoted_functional_{primary}"

    axis, label, view_index, view_name = AXES[primary]
    strict = primary == "front"
    front.update(
        {
            "asset_local_front_axis": axis,
            "canonical_front_direction": label,
            "canonical_front_face": horizontal_entries[0]["semantic_face"],
            "canonical_orientation_axis": axis,
            "canonical_orientation_axis_frame": "asset_local",
            "canonical_orientation_confidence": confidence,
            "canonical_orientation_is_semantic_front": True,
            "canonical_orientation_source": "full_library_category_semantic_direction_audit",
            "confidence": confidence,
            "front_strictness_label": (
                "strict_positive_front_category_verified"
                if strict
                else f"semantic_{primary}_face_not_strict_positive_front"
            ),
            "front_view_image_index": view_index,
            "front_view_image_name": view_name,
            "is_strict_front": strict,
            "is_strict_positive_front": strict,
            "method": "category_semantics_with_hssd_axis_convention",
            "notes": horizontal_entries[0]["semantic_face"],
            "status": "verified_semantic_direction",
            "validation_status": "category_semantics_verified_axis",
            "world_front_axis": None,
        }
    )
    evidence = list(front.get("evidence") or [])
    note = (
        "full-library-semantic-direction-v1: HSSD is asset-local Y-up; "
        "horizontal +Z is shown by back.png, +Y by top.png, and -Y is the "
        "functional underside (no bottom image in the shared six-view bundle)."
    )
    if note not in evidence:
        evidence.append(note)
    front["evidence"] = evidence
    return f"promoted_{primary}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    lookup_path = args.data_root / "hssd_annotation_lookup.json.gz"
    lookup = load_lookup(lookup_path)
    categories: dict[str, dict[str, Any]] = {}
    outcomes: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for hssd_id in sorted(lookup):
        record = lookup[hssd_id]
        category = str(record.get("category") or "").strip().lower()
        policy = category_policy(category, record.get("canonical_front") or {})
        categories.setdefault(category, policy)
        outcome = patch_record(record, policy)
        outcomes[outcome] += 1
        front = record["canonical_front"]
        rows.append(
            {
                "hssd_id": hssd_id,
                "category": category,
                "primary_direction_kind": front.get("semantic_direction_kind"),
                "annotation_kind": policy["primary_direction_kind"],
                "axis": json.dumps(front.get("canonical_orientation_axis")),
                "is_semantic": front.get("canonical_orientation_is_semantic_front"),
                "is_strict_positive_front": front.get("is_strict_positive_front"),
                "confidence": front.get("canonical_orientation_confidence"),
                "validation_status": front.get("validation_status"),
                "outcome": outcome,
            }
        )

    write_lookup(lookup_path, lookup)
    records_path = args.data_root / "records.jsonl"
    if records_path.exists():
        write_records(records_path, lookup)

    audit_path = args.audit_out or args.data_root / "SEMANTIC_DIRECTION_AUDIT.json"
    csv_path = args.csv_out or args.data_root / "SEMANTIC_DIRECTION_ASSETS.csv"
    audit = {
        "schema_version": "hssd_semantic_direction_audit@1.0",
        "asset_count": len(lookup),
        "category_count": len(categories),
        "outcomes": dict(sorted(outcomes.items())),
        "semantic_front_asset_count": sum(bool(r["is_semantic"]) for r in rows),
        "strict_positive_front_count": sum(bool(r["is_strict_positive_front"]) for r in rows),
        "horizontal_semantic_front_count": sum(
            bool(r["is_semantic"]) for r in rows
        ),
        "functional_direction_counts": dict(
            sorted(
                Counter(
                    direction["kind"]
                    for record in lookup.values()
                    for direction in record.get("functional_directions", [])
                ).items()
            )
        ),
        "annotation_policy_counts": dict(
            sorted(Counter(r["annotation_kind"] or "none" for r in rows).items())
        ),
        "coordinate_mapping": {
            "asset_up_axis": "+Y",
            "horizontal_views": {
                "back.png": "+Z", "front.png": "-Z", "right.png": "+X", "left.png": "-X"
            },
            "vertical_views": {"top.png": "+Y", "bottom": "-Y (not rendered)"},
        },
        "categories": categories,
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: audit[k] for k in ("asset_count", "category_count", "semantic_front_asset_count", "strict_positive_front_count", "functional_direction_counts", "annotation_policy_counts", "outcomes")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
