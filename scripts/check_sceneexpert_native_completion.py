"""Check future native pair execution artifacts directly in the project."""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenesmith.scene_expert.native_completion import check_native_pair_completion


def main() -> None:
    """Print execution evidence and return nonzero for an incomplete pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-report", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--arm-order", nargs="+", default=["base", "adapter"])
    parser.add_argument("--resolution", type=int, default=1536)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--engine", choices=["eevee", "cycles"], default="eevee")
    args = parser.parse_args()
    report = json.loads(args.pair_report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        parser.error("pair report must be a JSON object")
    result = check_native_pair_completion(
        report, project_root=args.project_root,
        expected_arms=tuple(args.arm_order), resolution=args.resolution,
        samples=args.samples, engine=args.engine,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(result["aggregate_exit_code"])


if __name__ == "__main__":
    main()
