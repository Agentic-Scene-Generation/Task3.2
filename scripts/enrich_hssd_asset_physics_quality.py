#!/usr/bin/env python3
"""Inject standalone HSSD physics/quality fields without replacing critic data."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


FIELDS = ("asset_physics", "asset_quality")


def load(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"lookup must be an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    source = load(args.source)
    target = load(args.target)
    missing = sorted(set(target) - set(source))
    if missing:
        raise ValueError(f"source lacks {len(missing)} target ids; first={missing[0]}")
    for hssd_id, record in target.items():
        annotated = source[hssd_id]
        for field in FIELDS:
            if field not in annotated:
                raise ValueError(f"source {hssd_id} lacks {field}")
            record[field] = annotated[field]
    with gzip.open(args.target, "wt", encoding="utf-8") as handle:
        json.dump(target, handle, ensure_ascii=False, sort_keys=True)
    print(f"enriched {len(target)} records with {', '.join(FIELDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
