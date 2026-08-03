"""Versioned admission quarantine for confirmed-invalid HSSD source assets.

The semantic VLM remains the general asset classifier. This module handles the
smaller, different problem of deterministic dataset defects that have already
been confirmed from source meshes and production renders. Quarantine is checked
before semantic caches so a stale positive decision can never re-admit a known
bad mesh.
"""

from __future__ import annotations

import json
import logging
import os

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

console_logger = logging.getLogger(__name__)

DEFAULT_HSSD_ASSET_QUARANTINE_PATH = (
    Path(__file__).resolve().parent / "data" / "hssd_asset_quarantine.json"
)
HSSD_ASSET_QUARANTINE_ENV = "SCENEEXPERT_HSSD_ASSET_QUARANTINE_PATH"


@dataclass(frozen=True)
class HssdAssetQuarantineEntry:
    """One evidence-backed HSSD admission denial."""

    mesh_id: str
    reason: str
    families: tuple[str, ...] = ()
    evidence: str = ""


def _read_quarantine_file(path: Path) -> dict[str, HssdAssetQuarantineEntry]:
    if not path.is_file():
        raise FileNotFoundError(f"HSSD asset quarantine file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: dict[str, HssdAssetQuarantineEntry] = {}
    for raw in list(payload.get("assets", []) or []):
        if str(raw.get("status", "blocked")).casefold() != "blocked":
            continue
        mesh_id = str(raw.get("mesh_id", "") or "").strip().casefold()
        reason = str(raw.get("reason", "") or "").strip()
        if not mesh_id or not reason:
            console_logger.warning(
                "Ignoring malformed HSSD quarantine entry in %s: %s",
                path,
                raw,
            )
            continue
        entries[mesh_id] = HssdAssetQuarantineEntry(
            mesh_id=mesh_id,
            reason=reason,
            families=tuple(
                str(value).strip().casefold()
                for value in list(raw.get("families", []) or [])
                if str(value).strip()
            ),
            evidence=str(raw.get("evidence", "") or "").strip(),
        )
    return entries


@lru_cache(maxsize=8)
def _load_hssd_asset_quarantine_cached(
    overlay_path: str,
) -> dict[str, HssdAssetQuarantineEntry]:
    entries = _read_quarantine_file(DEFAULT_HSSD_ASSET_QUARANTINE_PATH)
    if overlay_path:
        # An operator-supplied file extends or overrides the checked-in evidence
        # list. It never disables the baseline confirmed-invalid assets.
        entries.update(_read_quarantine_file(Path(overlay_path).expanduser()))
    return entries


def load_hssd_asset_quarantine(
    overlay_path: str | Path | None = None,
) -> dict[str, HssdAssetQuarantineEntry]:
    """Load the baseline quarantine plus an optional operator overlay."""

    resolved_overlay = (
        str(
            Path(
                overlay_path or os.environ.get(HSSD_ASSET_QUARANTINE_ENV, "")
            ).expanduser()
        )
        if (overlay_path or os.environ.get(HSSD_ASSET_QUARANTINE_ENV, ""))
        else ""
    )
    return dict(_load_hssd_asset_quarantine_cached(resolved_overlay))


def hssd_asset_quarantine_reason(mesh_id: str) -> str | None:
    """Return the confirmed rejection reason for ``mesh_id``, if any."""

    entry = load_hssd_asset_quarantine().get(str(mesh_id or "").strip().casefold())
    return entry.reason if entry is not None else None


def quarantined_hssd_asset_ids(family: str | None = None) -> frozenset[str]:
    """Return exact denied mesh IDs, optionally scoped to a semantic family."""

    entries = load_hssd_asset_quarantine()
    normalized_family = str(family or "").strip().casefold()
    if not normalized_family:
        return frozenset(entries)
    return frozenset(
        mesh_id
        for mesh_id, entry in entries.items()
        if not entry.families or normalized_family in entry.families
    )


def value_references_quarantined_hssd_asset(value: Any) -> bool:
    """Detect exact quarantined mesh IDs in a serialized memory value."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    try:
        serialized = json.dumps(value, ensure_ascii=False, default=str).casefold()
    except (TypeError, ValueError):
        serialized = str(value).casefold()
    return any(mesh_id in serialized for mesh_id in quarantined_hssd_asset_ids())
