#!/usr/bin/env python3
"""Run an isolated initial candidate, or audit/export a completed ACP pair pilot."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenesmith.scene_expert.slow_memory.paired import audit_pairs, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker-group", type=Path)
    mode.add_argument("--audit-root", type=Path)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--rescore-source", type=Path)
    parser.add_argument("--rescore-output", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--expected-groups", type=int, default=2)
    parser.add_argument("--min-pairs", type=int, default=1)
    args = parser.parse_args()
    if args.rescore_source:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
        )
        if not args.rescore_output:
            parser.error("--rescore-source requires a new --rescore-output")
        from scenesmith.scene_expert.slow_memory.paired_rescore import rescore_pairs

        result = rescore_pairs(args.rescore_source, args.rescore_output)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "gate_passed",
                        "execution_integrity_passed",
                        "preference_gate_passed",
                        "candidate_count",
                        "eligible_pair_count",
                    )
                },
                indent=2,
            )
        )
        return 0 if result["gate_passed"] else 2
    if args.min_pairs < 0 or not 1 <= args.expected_groups <= 4:
        parser.error("expected groups must be 1..4 and min pairs must be nonnegative")
    if args.preflight:
        from scenesmith.scene_expert.slow_memory.paired_runtime import (
            snapshot_codec_preflight,
        )

        try:
            result = snapshot_codec_preflight()
        except (ImportError, TypeError, ValueError) as exc:
            result = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        if args.preflight_report:
            write_json(args.preflight_report, result)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "passed" else 2
    if args.worker_group:
        from scenesmith.scene_expert.slow_memory.paired_runtime import (
            run_shadow,
            shadow_environment,
        )

        worker_env = shadow_environment(args.worker_group)
        os.environ.clear()
        os.environ.update(worker_env)

        # SIGTERM should unwind the worker and close its renderer/physics servers.
        def terminate(signum: int, frame: object) -> None:
            raise RuntimeError("initial candidate worker terminated")

        signal.signal(signal.SIGTERM, terminate)
        try:
            asyncio.run(run_shadow(args.worker_group.resolve()))
        except Exception as exc:
            write_json(
                args.worker_group / "B/failure.json",
                {"error": str(exc), "type": type(exc).__name__},
            )
            traceback.print_exc()
            return 2
        return 0
    result = audit_pairs(
        args.audit_root, min_pairs=args.min_pairs, expected_groups=args.expected_groups
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "gate_passed",
                    "execution_integrity_passed",
                    "preference_gate_passed",
                    "candidate_count",
                    "eligible_pair_count",
                    "errors",
                )
            },
            indent=2,
        )
    )
    print(f"Pair audit: {args.audit_root / 'pair_audit.json'}")
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
