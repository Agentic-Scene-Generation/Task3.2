"""Read-only, portable adapters for evaluation evidence (never execute payloads)."""

from __future__ import annotations

import json
import math

from pathlib import Path
from typing import Any


def read_object(path: Path, warnings: list[str]) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(value, dict):
            return value
    except (OSError, ValueError, UnicodeError) as exc:
        warnings.append(f"unreadable_evidence:{path}:{type(exc).__name__}")
        return {}
    warnings.append(f"non_object_evidence:{path}")
    return {}


def read_rows(path: Path, warnings: list[str]) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    try:
        with path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("not an object")
                    rows.append(value)
                except ValueError:
                    warnings.append(f"invalid_evidence_row:{path}:{number}")
    except (OSError, UnicodeError) as exc:
        warnings.append(f"unreadable_evidence:{path}:{type(exc).__name__}")
    return rows


def local_reference(root: Path, value: str) -> Path | None:
    """Resolve portable scene-relative payload refs, rejecting traversal/symlinks."""
    relative = Path(str(value).replace("\\", "/"))
    if not value or relative.is_absolute() or ":" in str(relative):
        return None
    path = (root / relative).resolve()
    return path if path.is_relative_to(root.resolve()) else None


def finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError):
        return None
