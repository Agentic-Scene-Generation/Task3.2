"""Create an immutable SceneEval collection/evaluation plan without a GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scenesmith.scene_expert.slow_memory.campaign import prepare_campaign, save_campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", type=Path, default=Path("scripts/assets/annotations.csv")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=128)
    parser.add_argument("--validation", type=int, default=24)
    parser.add_argument("--test", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    plan = prepare_campaign(
        args.annotations,
        seed=args.seed,
        train=args.train,
        validation=args.validation,
        test=args.test,
    )
    save_campaign(args.output, plan)
    print(json.dumps(plan["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
