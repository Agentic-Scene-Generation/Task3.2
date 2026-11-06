"""Explicit mount translation for the immutable all-assets release."""

from __future__ import annotations

import json
import os

from pathlib import Path


def resolve_all_assets_path(path: str | Path) -> Path:
    """Translate a published path using an explicitly configured mount map.

    ``HSSD_ALL_ASSETS_PATH_PREFIX_MAP`` is a JSON object mapping absolute source
    directories to absolute deployment directories. No map leaves paths intact;
    a map never searches for assets or changes the published index/manifest.
    """
    original = Path(path)
    raw = os.environ.get("HSSD_ALL_ASSETS_PATH_PREFIX_MAP")
    if not raw:
        return original
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid HSSD_ALL_ASSETS_PATH_PREFIX_MAP JSON") from exc
    if not isinstance(mapping, dict):
        raise ValueError("HSSD_ALL_ASSETS_PATH_PREFIX_MAP must be an object")
    prefixes: list[tuple[Path, Path]] = []
    for source, target in mapping.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("All-assets mount prefixes must be strings")
        source_path, target_path = Path(source), Path(target)
        if (
            not source_path.is_absolute()
            or not target_path.is_absolute()
            or ".." in source_path.parts
            or ".." in target_path.parts
        ):
            raise ValueError("All-assets mount prefixes must be absolute directories")
        prefixes.append((source_path, target_path))
    for source_path, target_path in sorted(
        prefixes, key=lambda entry: len(entry[0].parts), reverse=True
    ):
        if original.is_relative_to(source_path):
            suffix = original.relative_to(source_path)
            if ".." in suffix.parts:
                raise ValueError("All-assets paths must not escape their mount prefix")
            return target_path / suffix
    return original
