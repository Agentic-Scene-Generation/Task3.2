#!/usr/bin/env python3
"""Apply the paired SceneEval promotion gate to a trained adapter."""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenesmith.scene_expert.slow_memory.evaluation import evaluate_scene_level_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configurations/slow_memory/qwen_dpo_qlora.yaml"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    thresholds = (config.get("quality_gate") or {}).get("scene_level") or {}
    report = evaluate_scene_level_paths(
        baseline=args.baseline,
        candidate=args.candidate,
        thresholds=thresholds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    if not report["promotable"]:
        for failure in report["failures"]:
            print(f"- {failure}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
