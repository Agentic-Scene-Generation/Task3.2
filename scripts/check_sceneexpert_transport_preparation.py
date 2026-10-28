#!/usr/bin/env python3
"""Check future transport bindings before allocation; never import the packer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scenesmith.scene_expert.transport_preflight import check_transport_preparation


def main() -> int:
    """Return a binding proof or an actionable error without running shell code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--entrypoint", type=Path, required=True)
    args = parser.parse_args()
    try:
        proof = check_transport_preparation(args.project_root, args.plan, args.entrypoint)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(proof, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
