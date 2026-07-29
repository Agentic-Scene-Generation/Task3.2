#!/usr/bin/env python3
"""Merge reviewed wardrobe canonical-front fields into the SceneSmith lookup."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def load_gzip(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"lookup must be an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()

    source = load_gzip(args.source)
    target = load_gzip(args.target)
    spec = json.loads(args.annotations.read_text(encoding="utf-8"))
    hssd_ids = sorted(spec["assets"])
    for hssd_id in hssd_ids:
        if hssd_id not in source or hssd_id not in target:
            raise ValueError(f"wardrobe ID missing from lookup: {hssd_id}")
        if target[hssd_id].get("category") != "wardrobe":
            raise ValueError(f"target ID is not a wardrobe: {hssd_id}")
        source_front = source[hssd_id].get("canonical_front") or {}
        if source_front.get("validation_status") != "render_multiview_verified":
            raise ValueError(f"source wardrobe is not reviewed: {hssd_id}")
        target[hssd_id]["canonical_front"] = source_front

    with gzip.open(args.target, "wt", encoding="utf-8") as handle:
        json.dump(target, handle, ensure_ascii=False, sort_keys=True)
    print(f"enriched {len(hssd_ids)} reviewed wardrobe canonical fronts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
