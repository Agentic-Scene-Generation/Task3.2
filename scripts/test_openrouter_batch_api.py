#!/usr/bin/env python3
"""Explicitly gated one-request OpenRouter Beta Batch smoke test."""

import argparse
from pathlib import Path

from test_openrouter_api import load_api_key, poll_batch_api, test_batch_api


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--key-file",
        type=Path,
        default=Path("/mnt/afs/visitor33/apikeys/openrouter.json"),
    )
    parser.add_argument(
        "--confirm-paid-api-test",
        action="store_true",
        help="Required before submitting the one-request batch",
    )
    parser.add_argument(
        "--batch-id",
        help="Resume polling an existing batch without submitting a new request",
    )
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0)
    parser.add_argument("--poll-timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    if not args.batch_id and not args.confirm_paid_api_test:
        raise SystemExit(
            "Refusing to submit: pass --confirm-paid-api-test explicitly"
        )
    api_key = load_api_key(args.key_file)
    if args.batch_id:
        result = poll_batch_api(
            api_key,
            args.batch_id,
            poll_interval_seconds=args.poll_interval_seconds,
            poll_timeout_seconds=args.poll_timeout_seconds,
        )
    else:
        result = test_batch_api(
            api_key,
            poll_interval_seconds=args.poll_interval_seconds,
            poll_timeout_seconds=args.poll_timeout_seconds,
        )
    if not result.get("success"):
        raise SystemExit(f"Batch smoke failed: {result}")


if __name__ == "__main__":
    main()
