#!/usr/bin/env python3
"""Opt-in strict image matching for a future SHA-pinned exact-byte packer."""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scenesmith.scene_expert.transport_media import load_pinned_transport


def main() -> int:
    """Preserve the native packer's return code and report literal-span handling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--packer", type=Path, required=True)
    parser.add_argument("--packer-sha256", required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--include-postpack", action="store_true")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--server-originals", action="store_true")
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if not args.verify and (not args.plan or not args.prefix):
        parser.error("packaging requires --plan and --prefix")
    root = args.project_root.resolve()
    for name in ["plan", "prefix", "verify", "cache"]:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, root / value)
    engine, matcher = load_pinned_transport(root, args.packer, args.packer_sha256)
    if args.verify:
        print(
            json.dumps(
                engine.verify(root, args.verify, args.server_originals, args.cache),
                ensure_ascii=False,
            )
        )
        return 0
    exit_code = engine.package(root, args.plan, args.prefix, args.include_postpack)
    print(
        json.dumps(
            {
                "transport_filter": "canonical_inline_images_v1",
                "skipped_image_lookalikes_retained_literal": matcher.skipped,
                "native_package_exit_code": exit_code,
                "native_quality_or_review_completion_claimed": False,
            },
            ensure_ascii=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
