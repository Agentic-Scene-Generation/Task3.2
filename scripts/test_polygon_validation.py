#!/usr/bin/env python3
"""
Test polygon geometry validation against PromptGen v3.4 irregular cases.

This script validates that the 30 irregular polygon cases can pass through
the polygon_geometry validator with the candidate min_edge_length_m threshold.
"""
import json
import sys
from pathlib import Path

# Add scenesmith to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scenesmith.floor_plan_agents.tools.polygon_geometry import (
    PolygonValidationConfig,
    PolygonValidationError,
    canonicalize_polygon,
    polygon_edges,
)


def test_polygon_validation(
    manifest_path: Path,
    min_edge_length_m: float = 0.30,
    min_dimension_m: float = 1.40,
    verbose: bool = False,
):
    """Test all irregular cases from the manifest."""

    with open(manifest_path, "r") as f:
        cases = [json.loads(line) for line in f]

    irregular_cases = [c for c in cases if c["mode"] == "polygon"]

    print(f"Testing {len(irregular_cases)} irregular polygon cases")
    print(
        "Validation config: "
        f"min_edge_length_m={min_edge_length_m}, "
        f"min_dimension_m={min_dimension_m}"
    )
    print("=" * 70)

    config = PolygonValidationConfig(
        min_edge_length_m=min_edge_length_m,
        min_area_m2=4.0,
        min_interior_angle_deg=20.0,
        min_dimension_m=min_dimension_m,
    )

    results = {
        "total": len(irregular_cases),
        "passed": 0,
        "failed": 0,
        "failures": [],
    }

    for case in irregular_cases:
        prompt_id = case["prompt_id"]
        shape_type = case["shape_type"]
        vertices = case["vertices"]
        declared_area = case["area"]
        min_edge = case["min_edge"]

        try:
            # Canonicalize (this runs full validation)
            canonical = canonicalize_polygon(vertices, config)
            edges = polygon_edges(canonical)

            # Verify
            computed_min_edge = min(e.length for e in edges)

            results["passed"] += 1

            if verbose:
                print(f"✓ {prompt_id} ({shape_type})")
                print(f"    Min edge: {min_edge:.4f} m (computed: {computed_min_edge:.4f} m)")
                print(f"    Vertices: {len(vertices)}, Area: {declared_area:.2f} m²")

        except PolygonValidationError as e:
            results["failed"] += 1
            failure = {
                "prompt_id": prompt_id,
                "shape_type": shape_type,
                "min_edge": min_edge,
                "declared_area": declared_area,
                "error": str(e),
            }
            results["failures"].append(failure)

            print(f"✗ {prompt_id} ({shape_type})")
            print(f"    Min edge: {min_edge:.4f} m")
            print(f"    ERROR: {e}")
            print()

    print("=" * 70)
    print(f"RESULTS: {results['passed']}/{results['total']} passed")

    if results["failures"]:
        print()
        print("FAILED CASES:")
        for failure in results["failures"]:
            print(f"  {failure['prompt_id']}: {failure['error']}")
        return False
    else:
        print("✓ All polygon cases passed validation")
        return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test polygon validation against PromptGen v3.4 cases"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/mnt/afs/visitor33/Task3.2/test_data/promptgen_v3_4/cases.jsonl"),
        help="Path to cases.jsonl manifest",
    )
    parser.add_argument(
        "--min-edge-length-m",
        type=float,
        default=0.30,
        help="Minimum edge length threshold (m)",
    )
    parser.add_argument(
        "--min-dimension-m",
        type=float,
        default=1.40,
        help="Minimum polygon AABB dimension threshold (m)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output for all cases",
    )

    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    success = test_polygon_validation(
        args.manifest,
        min_edge_length_m=args.min_edge_length_m,
        min_dimension_m=args.min_dimension_m,
        verbose=args.verbose,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
