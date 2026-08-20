#!/usr/bin/env python3
"""Export critic-grounded SceneExpert trajectories as a validated DPO package."""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenesmith.scene_expert.slow_memory.dpo import export_dpo_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact-context chosen/rejected pairs from SceneSmith audit "
            "evidence. Ambiguous rows are retained only as diagnostics."
        )
    )
    parser.add_argument(
        "--trajectory-source",
        type=Path,
        action="append",
        required=True,
        help=(
            "A trajectories.jsonl file or run/output directory. Repeat this "
            "option to combine independent runs."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-quality-margin", type=float, default=0.05)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write diagnostics and exit successfully when no train pair exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = export_dpo_dataset(
        trajectory_sources=args.trajectory_source,
        output_dir=args.output_dir,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_quality_margin=args.min_quality_margin,
    )
    print(json.dumps(manifest["stats"], indent=2, ensure_ascii=False))
    validation = manifest["validation"]
    if validation["valid"]:
        print(f"Validated DPO package: {args.output_dir / 'manifest.json'}")
        return 0
    if args.allow_empty and validation["errors"] == ["training split is empty"]:
        print(
            "No exact-context pair is available yet; diagnostics were written "
            f"to {args.output_dir / 'rejected_pair_diagnostics.jsonl'}."
        )
        return 0
    print("DPO package is not trainable:", file=sys.stderr)
    for error in validation["errors"]:
        print(f"- {error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
