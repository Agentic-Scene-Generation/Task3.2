"""Helpers for tests that depend on external glTF buffer sidecars."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Iterable


def missing_gltf_buffers(paths: Iterable[Path]) -> list[Path]:
    """Return missing files referenced by the supplied glTF documents."""
    missing: list[Path] = []
    for path in paths:
        if not path.exists():
            missing.append(path)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            missing.append(path)
            continue
        for buffer in payload.get("buffers") or []:
            uri = str(buffer.get("uri") or "")
            if not uri or uri.startswith("data:"):
                continue
            buffer_path = path.parent / uri
            if not buffer_path.exists():
                missing.append(buffer_path)
    return missing
