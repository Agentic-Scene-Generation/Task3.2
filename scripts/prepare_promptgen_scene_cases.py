#!/usr/bin/env python3
"""
Prepare PromptGen v3.4 scene cases for OpenRouter GPT-5.6 Luna Pro synthesis.

Reads the 100-case CSV with mixed rectangular and irregular geometries,
validates all polygon topologies, and generates derived manifests for the
parallel runner without modifying the source CSV.

Output structure:
    <output_dir>/
        rectangular.csv          # 70 room-mode cases, ID,Description format
        irregular.csv            # 30 polygon-mode cases, ID,Description format
        cases.jsonl              # Full mapping: PromptID -> geometry + run config
        validation_report.json   # Geometry checks, edge stats, conversion audit
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_vertices(vertices_str: str) -> List[Tuple[float, float]]:
    """Parse JSON-encoded vertices string to list of (x, y) tuples."""
    raw = json.loads(vertices_str)
    return [(float(v[0]), float(v[1])) for v in raw]


def calculate_edge_lengths(vertices: List[Tuple[float, float]]) -> List[float]:
    """Calculate all edge lengths for a closed polygon."""
    edges = []
    n = len(vertices)
    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        length = math.sqrt((v2[0] - v1[0])**2 + (v2[1] - v1[1])**2)
        edges.append(length)
    return edges


def calculate_polygon_area(vertices: List[Tuple[float, float]]) -> float:
    """Shoelace formula for signed polygon area (absolute value)."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def build_runner_safe_description(description: str) -> str:
    """Collapse formatting that the shell runner reserves for record framing."""
    if "\x00" in description or "|" in description:
        raise ValueError("Description contains a reserved runner separator")
    return " ".join(description.split())


def check_self_intersection(vertices: List[Tuple[float, float]]) -> bool:
    """
    Simple O(n²) self-intersection check.
    Returns True if any non-adjacent edge pair intersects.
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def segments_intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    n = len(vertices)
    for i in range(n):
        seg1_a = vertices[i]
        seg1_b = vertices[(i + 1) % n]
        for j in range(i + 2, n):
            # Skip adjacent edges and the wrap-around edge
            if j == (i + 1) % n or (i == 0 and j == n - 1):
                continue
            seg2_a = vertices[j]
            seg2_b = vertices[(j + 1) % n]
            if segments_intersect(seg1_a, seg1_b, seg2_a, seg2_b):
                return True
    return False


def validate_irregular_geometry(
    prompt_id: str,
    vertices_str: str,
    shape_type: str,
    declared_area: float,
    area_tolerance: float = 0.01,
) -> Dict[str, Any]:
    """
    Validate polygon geometry: parse vertices, check topology, measure edges.

    Returns dict with:
        - valid: bool
        - vertices: List[Tuple[float, float]] if parsable
        - edge_lengths: List[float]
        - computed_area: float
        - area_error_ratio: float
        - min_edge: float
        - rejection_reasons: List[str]
    """
    result = {
        "valid": True,
        "vertices": None,
        "edge_lengths": [],
        "computed_area": 0.0,
        "area_error_ratio": 0.0,
        "min_edge": 0.0,
        "rejection_reasons": [],
    }

    try:
        vertices = parse_vertices(vertices_str)
    except Exception as e:
        result["valid"] = False
        result["rejection_reasons"].append(f"Vertices parse failed: {e}")
        return result

    result["vertices"] = vertices
    n = len(vertices)

    if n < 3:
        result["valid"] = False
        result["rejection_reasons"].append(f"Insufficient vertices: {n}")
        return result

    # Check self-intersection
    if check_self_intersection(vertices):
        result["valid"] = False
        result["rejection_reasons"].append("Self-intersecting polygon")
        return result

    # Compute edges
    edges = calculate_edge_lengths(vertices)
    result["edge_lengths"] = edges
    result["min_edge"] = min(edges) if edges else 0.0

    # Compute area and compare
    computed = calculate_polygon_area(vertices)
    result["computed_area"] = computed
    if declared_area > 0:
        error_ratio = abs(computed - declared_area) / declared_area
        result["area_error_ratio"] = error_ratio
        if error_ratio > area_tolerance:
            result["valid"] = False
            result["rejection_reasons"].append(
                f"Area mismatch: declared={declared_area:.2f}, computed={computed:.2f}, "
                f"error={error_ratio*100:.1f}%"
            )

    return result


def build_rectangular_description(row: Dict[str, str]) -> str:
    """
    Build Description for rectangular room cases.
    Prepend an unambiguous geometry constraint to cover dimension-removed samples.
    """
    width = float(row["Width"])
    length = float(row["Length"])
    original_prompt = row["Prompt"].strip()

    prefix = (
        f"Use a rectangular room with the exact interior dimensions "
        f"Width={width:.2f}m and Length={length:.2f}m. "
        f"Do not infer or replace these dimensions from the prose description below.\n\n"
    )
    return prefix + original_prompt


def build_irregular_description(row: Dict[str, str]) -> str:
    """
    Build Description for irregular polygon cases.
    Prepend machine-readable geometry specification with strict constraints.
    """
    vertices_str = row["Vertices"]
    shape_type = row["ShapeType"]
    area = float(row["Area"])
    original_prompt = row["Prompt"].strip()

    # Parse for human-readable coordinate summary
    vertices = parse_vertices(vertices_str)
    vertices_readable = ", ".join(f"({x:.2f},{y:.2f})" for x, y in vertices)

    prefix = (
        f"GEOMETRY SPECIFICATION (floor_plan_agent.mode=polygon):\n"
        f"  Shape type: {shape_type}\n"
        f"  Vertices (ordered, defining exact boundary): {vertices_readable}\n"
        f"  Expected area: {area:.2f} m²\n"
        f"STRICT CONSTRAINTS:\n"
        f"  - Use the provided vertices in exact order; do not reorder, scale, or repair.\n"
        f"  - Do not replace this polygon with a bounding rectangle or regular shape.\n"
        f"  - Vertices must be used as-is for floor plan generation.\n\n"
        f"SCENE DESCRIPTION:\n"
    )
    return prefix + original_prompt


def prepare_scene_cases(
    input_csv_path: Path,
    output_dir: Path,
    min_edge_candidate_m: float = 0.30,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Main preparation function: read CSV, validate, generate derived manifests.

    Returns summary dict with counts, validation stats, and output paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rectangular_cases = []
    irregular_cases = []
    all_case_records = []
    validation_results = []

    stats = {
        "total": 0,
        "rectangular": 0,
        "irregular": 0,
        "irregular_by_shape": {},
        "dimension_removed": 0,
        "validation_passed": 0,
        "validation_failed": 0,
        "min_edge_overall": float("inf"),
        "min_edge_case_id": None,
    }

    with open(input_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total"] += 1
            prompt_id = row["PromptID"]
            geometry_type = row["GeometryType"]

            if row.get("DimensionDescriptionRemoved", "").lower() == "true":
                stats["dimension_removed"] += 1

            case_record = {
                "prompt_id": prompt_id,
                "source_prompt_id": row["SourcePromptID"],
                "geometry_type": geometry_type,
                "room_type": row["RoomType"],
                "scene_eval_room_type": row["SceneEvalRoomType"],
            }

            if geometry_type == "rectangular":
                stats["rectangular"] += 1
                width = float(row["Width"])
                length = float(row["Length"])
                area = float(row["Area"])

                description = build_rectangular_description(row)
                rectangular_cases.append({
                    "ID": prompt_id,
                    "Description": build_runner_safe_description(description),
                })

                case_record.update({
                    "mode": "room",
                    "width": width,
                    "length": length,
                    "area": area,
                    "description": description,
                })

            elif geometry_type == "irregular":
                stats["irregular"] += 1
                shape_type = row["ShapeType"]
                stats["irregular_by_shape"][shape_type] = \
                    stats["irregular_by_shape"].get(shape_type, 0) + 1

                declared_area = float(row["Area"])
                vertices_str = row["Vertices"]

                # Validate geometry
                val_result = validate_irregular_geometry(
                    prompt_id, vertices_str, shape_type, declared_area
                )
                validation_results.append({
                    "prompt_id": prompt_id,
                    "shape_type": shape_type,
                    **val_result,
                })

                if val_result["valid"]:
                    stats["validation_passed"] += 1
                else:
                    stats["validation_failed"] += 1

                min_edge = val_result["min_edge"]
                if min_edge < stats["min_edge_overall"]:
                    stats["min_edge_overall"] = min_edge
                    stats["min_edge_case_id"] = prompt_id

                description = build_irregular_description(row)
                irregular_cases.append({
                    "ID": prompt_id,
                    "Description": build_runner_safe_description(description),
                })

                case_record.update({
                    "mode": "polygon",
                    "shape_type": shape_type,
                    "vertices": val_result["vertices"],
                    "area": declared_area,
                    "computed_area": val_result["computed_area"],
                    "edge_lengths": val_result["edge_lengths"],
                    "min_edge": min_edge,
                    "validation": val_result,
                    "description": description,
                })

            all_case_records.append(case_record)

    # Write derived CSVs
    rect_csv_path = output_dir / "rectangular.csv"
    irreg_csv_path = output_dir / "irregular.csv"

    if not dry_run:
        with open(rect_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "Description"])
            writer.writeheader()
            writer.writerows(rectangular_cases)

        with open(irreg_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "Description"])
            writer.writeheader()
            writer.writerows(irregular_cases)

        # Write JSONL manifest
        cases_jsonl_path = output_dir / "cases.jsonl"
        with open(cases_jsonl_path, "w", encoding="utf-8") as f:
            for record in all_case_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Write validation report
        report = {
            "input_csv": str(input_csv_path),
            "output_dir": str(output_dir),
            "conversion_version": "v1",
            "min_edge_candidate_m": min_edge_candidate_m,
            "stats": stats,
            "validation_details": validation_results,
        }
        report_path = output_dir / "validation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    summary = {
        "output_paths": {
            "rectangular_csv": str(rect_csv_path),
            "irregular_csv": str(irreg_csv_path),
            "cases_jsonl": str(output_dir / "cases.jsonl"),
            "validation_report": str(output_dir / "validation_report.json"),
        },
        "stats": stats,
        "validation_failures": [
            v for v in validation_results if not v["valid"]
        ],
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PromptGen v3.4 scene cases for OpenRouter synthesis"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/mnt/afs/visitor33/Datasets/promptGen/outputs/"
                     "v3_4_cola_integrated_100/prompts_70_rect_30_irregular.csv"),
        help="Input CSV with 100 cases (default: v3.4 integrated CSV)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for derived manifests",
    )
    parser.add_argument(
        "--min-edge-candidate-m",
        type=float,
        default=0.30,
        help="Candidate minimum edge length threshold (m) for reporting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only; do not write output files",
    )

    args = parser.parse_args()

    if not args.input_csv.exists():
        print(f"ERROR: Input CSV not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Preparing scene cases from: {args.input_csv}")
    print(f"Output directory: {args.output_dir}")
    if args.dry_run:
        print("DRY RUN: validation only, no files will be written")
    print()

    summary = prepare_scene_cases(
        args.input_csv,
        args.output_dir,
        min_edge_candidate_m=args.min_edge_candidate_m,
        dry_run=args.dry_run,
    )

    stats = summary["stats"]
    print("=" * 60)
    print("PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Total cases:           {stats['total']}")
    print(f"  Rectangular:         {stats['rectangular']}")
    print(f"  Irregular:           {stats['irregular']}")
    print()
    print("Irregular by shape:")
    for shape, count in sorted(stats["irregular_by_shape"].items()):
        print(f"  {shape:12s}  {count}")
    print()
    print(f"Dimension-removed:     {stats['dimension_removed']}")
    print(f"Validation passed:     {stats['validation_passed']}")
    print(f"Validation failed:     {stats['validation_failed']}")
    print()
    print(f"Min edge overall:      {stats['min_edge_overall']:.4f} m")
    print(f"  (Case: {stats['min_edge_case_id']})")
    print()

    if summary["validation_failures"]:
        print("!" * 60)
        print("VALIDATION FAILURES")
        print("!" * 60)
        for failure in summary["validation_failures"]:
            print(f"PromptID {failure['prompt_id']}:")
            for reason in failure["rejection_reasons"]:
                print(f"  - {reason}")
        print()
        sys.exit(1)

    if not args.dry_run:
        print("Output files written:")
        for name, path in summary["output_paths"].items():
            print(f"  {name:20s} -> {path}")

    print()
    print("✓ Preparation complete. All geometries valid.")


if __name__ == "__main__":
    main()
