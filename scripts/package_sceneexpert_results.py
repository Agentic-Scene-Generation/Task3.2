#!/usr/bin/env python3
"""Package bounded SceneExpert review data, or verify an existing package."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenesmith.scene_expert.review_package import MIB, package_results, verify_package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--collection-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--max-file-mib", type=int, default=32)
    parser.add_argument("--max-total-mib", type=int, default=512)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    # This standalone helper also runs outside the project's managed environment.
    if sys.version_info < (3, 11):  # noqa: UP036
        parser.error("Python 3.11+ is required")
    if not args.verify and (not args.run_id or not args.output):
        parser.error("packaging requires --run-id and --output")
    try:
        result = (
            verify_package(args.verify)
            if args.verify
            else package_results(
                project_root=args.project_root,
                run_id=args.run_id,
                output=args.output,
                collection_root=args.collection_root,
                log_root=args.log_root,
                max_file_bytes=args.max_file_mib * MIB,
                max_total_bytes=args.max_total_mib * MIB,
            )
        )
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as exc:
        print(f"Package error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
