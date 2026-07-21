#!/usr/bin/env python3
"""Cross-check HSSD physics annotations, ID alignment, and mass sanity."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_lookup(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"lookup must be an object: {path}")
    return value


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("physics_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rounded_expected_mass(reference: float, scale: float) -> float:
    return round(max(0.05, reference * scale), 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    standalone = load_lookup(args.standalone)
    scene = load_lookup(args.scene)
    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))
    generator = load_generator(args.generator)
    records = list(standalone.values())
    medians = generator.category_median_volumes(records)

    standalone_ids = set(standalone)
    scene_ids = set(scene)
    render_ids = {item.name for item in args.render_root.iterdir() if item.is_dir()}
    embedded_id_mismatches = sorted(
        key for key, record in standalone.items() if record.get("hssd_id") != key
    )
    scene_payload_mismatches = sorted(
        key
        for key in standalone_ids & scene_ids
        if any(
            standalone[key].get(field) != scene[key].get(field)
            for field in ("asset_physics", "asset_quality")
        )
    )

    formula_mismatches: list[dict[str, Any]] = []
    invalid_ranges: list[str] = []
    invalid_dimensions: list[str] = []
    extreme_bbox_density: list[dict[str, Any]] = []
    audit_statuses: Counter[str] = Counter()
    audit_flags: Counter[str] = Counter()
    category_rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "materials": Counter(), "masses": [], "flags": Counter()}
    )

    for hid, record in standalone.items():
        category = str(record.get("category") or record.get("category_key") or "")
        physics = record.get("asset_physics") or {}
        basis = physics.get("estimation_basis") or {}
        audit = physics.get("audit") or {}
        status = str(audit.get("status") or "missing")
        audit_statuses[status] += 1
        for flag in audit.get("flags") or []:
            audit_flags[str(flag)] += 1

        expected_material, _, expected_material_rule = generator.classify_material(category)
        expected_reference, expected_mass_rule = generator.reference_mass(category)
        dims, _ = generator.dimensions(record)
        if category.lower() == "motorcycle" and dims and max(dims) < 0.6:
            expected_reference = 1.0
            expected_mass_rule = "category_size_override"
        median = medians.get(category)
        raw_scale = 1.0
        scale = 1.0
        if dims and median and median > 0 and expected_mass_rule != "category_size_override":
            cfg = profiles["mass_estimation"]
            raw_scale = (math.prod(dims) / median) ** float(
                cfg["dimension_scale_exponent"]
            )
            scale = min(
                float(cfg["dimension_scale_max"]),
                max(float(cfg["dimension_scale_min"]), raw_scale),
            )
        expected_mass = rounded_expected_mass(expected_reference, scale)
        expected_friction = profiles["materials"][expected_material][
            "friction_coefficient"
        ]
        checks = {
            "material": physics.get("material") == expected_material,
            "material_rule": basis.get("material_rule") == expected_material_rule,
            "mass_rule": basis.get("mass_rule") == expected_mass_rule,
            "reference_mass": basis.get("reference_mass_kg") == expected_reference,
            "mass": physics.get("mass_kg") == expected_mass,
            "friction": physics.get("friction_coefficient") == expected_friction,
        }
        if not all(checks.values()):
            formula_mismatches.append(
                {"hssd_id": hid, "category": category, "failed": sorted(k for k, ok in checks.items() if not ok)}
            )

        mass = physics.get("mass_kg")
        mass_range = physics.get("mass_range_kg")
        if not (
            isinstance(mass, (int, float))
            and math.isfinite(mass)
            and mass > 0
            and isinstance(mass_range, list)
            and len(mass_range) == 2
            and 0 < mass_range[0] <= mass <= mass_range[1]
        ):
            invalid_ranges.append(hid)
        if dims:
            volume = math.prod(dims)
            if not math.isfinite(volume) or volume <= 0:
                invalid_dimensions.append(hid)
            elif isinstance(mass, (int, float)):
                bbox_density = mass / volume
                # Bounding-box density is only a screening signal for hollow objects.
                if bbox_density < 0.05 or bbox_density > 5000.0:
                    extreme_bbox_density.append(
                        {
                            "hssd_id": hid,
                            "category": category,
                            "mass_kg": mass,
                            "bbox_density_kg_m3": round(bbox_density, 3),
                        }
                    )

        row = category_rows[category]
        row["count"] += 1
        row["materials"][str(physics.get("material"))] += 1
        if isinstance(mass, (int, float)):
            row["masses"].append(float(mass))
        for flag in audit.get("flags") or []:
            row["flags"][str(flag)] += 1

    render_file_counts = {
        hid: sum(1 for item in (args.render_root / hid).iterdir() if item.is_file())
        for hid in standalone_ids & render_ids
    }
    empty_render_dirs = sorted(hid for hid, count in render_file_counts.items() if count == 0)
    render_hash_groups: dict[str, list[str]] = defaultdict(list)
    render_probe_files: dict[str, str] = {}
    for hid in sorted(standalone_ids & render_ids):
        directory = args.render_root / hid
        candidates = [directory / "iso.png", directory / "front.png"]
        candidates.extend(sorted(item for item in directory.iterdir() if item.is_file()))
        probe = next((item for item in candidates if item.is_file()), None)
        if probe is None:
            continue
        digest = hashlib.sha256(probe.read_bytes()).hexdigest()
        render_hash_groups[digest].append(hid)
        render_probe_files[hid] = probe.name
    duplicate_render_groups = [
        ids for ids in render_hash_groups.values() if len(ids) > 1
    ]
    cross_category_render_groups = []
    compatible_alias_render_groups = []
    compatible_category_sets = {frozenset(("sofa", "l-shaped couch"))}
    for ids in duplicate_render_groups:
        categories = sorted({str(standalone[hid].get("category")) for hid in ids})
        if len(categories) > 1:
            group = {
                "hssd_ids": ids,
                "categories": categories,
                "probe_files": {hid: render_probe_files[hid] for hid in ids},
            }
            if frozenset(categories) in compatible_category_sets:
                group["review_outcome"] = "compatible_category_alias_duplicate"
                compatible_alias_render_groups.append(group)
            else:
                cross_category_render_groups.append(group)

    csv_rows: list[dict[str, Any]] = []
    for category, row in sorted(category_rows.items()):
        masses = row["masses"]
        reference, mass_rule = generator.reference_mass(category)
        material, confidence, material_rule = generator.classify_material(category)
        csv_rows.append(
            {
                "category": category,
                "asset_count": row["count"],
                "material": material,
                "material_confidence": confidence,
                "material_rule": material_rule,
                "reference_mass_kg": reference,
                "mass_rule": mass_rule,
                "minimum_mass_kg": min(masses),
                "median_mass_kg": round(statistics.median(masses), 3),
                "maximum_mass_kg": max(masses),
                "audit_flags": ";".join(f"{key}:{value}" for key, value in sorted(row["flags"].items())),
            }
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    result = {
        "schema_version": "hssd_asset_physics_quality_audit@1.0",
        "method_version": generator.METHOD_VERSION,
        "scope": {
            "asset_count": len(standalone),
            "category_count": len(category_rows),
            "direct_measurement": False,
            "semantic_render_content_verified": False,
            "note": "ID/file alignment and deterministic physics rules are checked exhaustively. Render-directory naming cannot prove that image pixels depict the intended asset; representative renders require visual review.",
        },
        "id_alignment": {
            "standalone_only": sorted(standalone_ids - scene_ids),
            "scene_only": sorted(scene_ids - standalone_ids),
            "render_missing": sorted(standalone_ids - render_ids),
            "render_extra": sorted(render_ids - standalone_ids),
            "embedded_id_mismatches": embedded_id_mismatches,
            "scene_payload_mismatches": scene_payload_mismatches,
            "empty_render_directories": empty_render_dirs,
        },
        "annotation_consistency": {
            "formula_mismatch_count": len(formula_mismatches),
            "formula_mismatches": formula_mismatches,
            "invalid_mass_range_count": len(invalid_ranges),
            "invalid_mass_range_ids": invalid_ranges,
            "invalid_dimension_count": len(invalid_dimensions),
            "invalid_dimension_ids": invalid_dimensions,
        },
        "review_distribution": {
            "statuses": dict(sorted(audit_statuses.items())),
            "flags": dict(sorted(audit_flags.items())),
            "extreme_bbox_density_count": len(extreme_bbox_density),
            "extreme_bbox_density_examples": extreme_bbox_density[:200],
        },
        "render_content_screening": {
            "representative_images_hashed": len(render_probe_files),
            "exact_duplicate_group_count": len(duplicate_render_groups),
            "compatible_alias_duplicate_group_count": len(compatible_alias_render_groups),
            "compatible_alias_duplicate_groups": compatible_alias_render_groups,
            "unresolved_cross_category_duplicate_group_count": len(cross_category_render_groups),
            "cross_category_exact_duplicate_groups": cross_category_render_groups,
            "note": "Exact representative-image duplicates across different categories are suspicious and require visual review; they are not automatically treated as annotation mismatches.",
        },
        "category_audit_csv": args.output_csv.name,
    }
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 1 if any(
        (
            result["id_alignment"]["standalone_only"],
            result["id_alignment"]["scene_only"],
            result["id_alignment"]["render_missing"],
            result["id_alignment"]["embedded_id_mismatches"],
            formula_mismatches,
            invalid_ranges,
            invalid_dimensions,
        )
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
