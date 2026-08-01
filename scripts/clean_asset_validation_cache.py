"""Audit or archive stale SceneExpert HSSD semantic-cache entries.

The command is dry-run by default.  ``--apply`` moves invalid entries into a
timestamped backup directory and writes an audit report; it never deletes the
only copy.  Contract-v8 entries are reusable only when bound to a passing
structural check and the source-geometry fingerprint used by that check.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenesmith.agent_utils.asset_quarantine import hssd_asset_quarantine_reason
from scenesmith.agent_utils.asset_runtime import ASSET_SEMANTIC_CONTRACT_VERSION
from scenesmith.agent_utils.asset_structure import ASSET_STRUCTURE_CONTRACT_VERSION


def _stale_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    candidate_id = str(payload.get("candidate_id", "") or "")
    quarantine_reason = hssd_asset_quarantine_reason(candidate_id)
    if quarantine_reason:
        reasons.append(f"quarantined_asset: {quarantine_reason}")
    if str(payload.get("schema_version", "")) != ASSET_SEMANTIC_CONTRACT_VERSION:
        reasons.append("semantic_contract_mismatch")

    structure = payload.get("structural_check", {})
    if not isinstance(structure, dict):
        structure = {}
    if str(structure.get("contract_version", "")) != ASSET_STRUCTURE_CONTRACT_VERSION:
        reasons.append("structural_contract_mismatch")
    if str(structure.get("status", "")) != "pass":
        reasons.append("structural_admission_not_passed")
    if not str(structure.get("geometry_fingerprint", "") or ""):
        reasons.append("geometry_fingerprint_missing")
    return reasons


def clean_asset_validation_cache(
    *,
    cache_dir: str | Path,
    apply: bool = False,
) -> dict[str, Any]:
    """Inspect or archive entries incompatible with current admission contracts."""

    cache_root = Path(cache_dir).resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Asset validation cache not found: {cache_root}")

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stale: list[dict[str, Any]] = []
    valid_count = 0
    for cache_path in sorted(cache_root.glob("*.json")):
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("cache payload is not an object")
            reasons = _stale_reasons(payload)
        except (OSError, ValueError, TypeError) as exc:
            payload = {}
            reasons = [f"unreadable_cache: {type(exc).__name__}: {exc}"]
        if not reasons:
            valid_count += 1
            continue
        stale.append(
            {
                "path": str(cache_path),
                "candidate_id": str(payload.get("candidate_id", "") or ""),
                "reasons": reasons,
                "backup_path": "",
            }
        )

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "APPLIED" if apply else "DRY_RUN",
        "created_at": timestamp,
        "cache_dir": str(cache_root),
        "active_semantic_contract": ASSET_SEMANTIC_CONTRACT_VERSION,
        "active_structural_contract": ASSET_STRUCTURE_CONTRACT_VERSION,
        "valid_count": valid_count,
        "stale_count": len(stale),
        "entries": stale,
        "report_path": "",
    }
    if not apply or not stale:
        return report

    backup_root = cache_root / "backups" / timestamp
    migration_root = cache_root / "migrations"
    backup_root.mkdir(parents=True, exist_ok=True)
    migration_root.mkdir(parents=True, exist_ok=True)
    for entry in stale:
        source = Path(entry["path"])
        destination = backup_root / source.name
        shutil.move(str(source), destination)
        entry["backup_path"] = str(destination)

    report_path = migration_root / f"asset_validation_cache_cleanup.{timestamp}.json"
    report["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="SceneExpert HSSD semantic-cache directory.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Archive stale entries. Without this flag the command is read-only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = clean_asset_validation_cache(
        cache_dir=args.cache_dir,
        apply=bool(args.apply),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
