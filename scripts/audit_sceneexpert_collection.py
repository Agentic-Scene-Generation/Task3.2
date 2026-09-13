#!/usr/bin/env python3
"""Audit a fresh or archived Qwen Slow Memory collection without rerunning scenes."""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenesmith.scene_expert.slow_memory.collection import audit_collection


def main() -> int:
    """Always write diagnostic artifacts; fail only the requested collection gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-model", default="unsloth/Qwen3.8-27B-GGUF")
    parser.add_argument("--min-pairs", type=int, default=0)
    args = parser.parse_args()
    if not args.run_root.is_dir():
        parser.error(f"run root is not a directory: {args.run_root}")
    if args.min_pairs < 0:
        parser.error("--min-pairs must be nonnegative")
    output_dir = args.output_dir or args.run_root / "collection"
    result = audit_collection(
        args.run_root,
        output_dir,
        expected_model=args.expected_model,
        min_pairs=args.min_pairs,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "generation_complete",
                    "observer_collection_ready",
                    "dpo_export_ready",
                    "training_preflight_status",
                    "trajectory_count",
                    "designer_count",
                    "contexts_with_multiple_candidates",
                    "min_pairs",
                    "gate_passed",
                    "errors",
                )
            },
            indent=2,
        )
    )
    print(f"DPO pairs: {result['dpo_stats']['eligible_pair_count']}")
    print(f"Collection audit: {output_dir / 'collection_audit.json'}")
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
