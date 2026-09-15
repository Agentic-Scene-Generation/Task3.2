#!/usr/bin/env python3
"""Import externally generated teacher candidates into Slow Memory v2."""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenesmith.scene_expert.slow_memory.importer import import_teacher_trajectories


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-rejected-rows",
        action="store_true",
        help="Return success while retaining invalid rows in diagnostics.",
    )
    args = parser.parse_args()
    summary = import_teacher_trajectories(
        source_paths=args.source,
        output_path=args.output,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["rejected_count"] and not args.allow_rejected_rows:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
