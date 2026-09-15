"""Export a deterministic reader-contract audit without modifying a memory bank.

Usage: python -m scenesmith.scene_expert.memory.audit_contract
       --memory-dir PATH --output PATH_OUTSIDE_BANK
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging

from pathlib import Path

from pydantic import ValidationError

from scenesmith.scene_expert.memory.contracts import (
    PAYLOAD_VERSION,
    selection_from_record,
)
from scenesmith.scene_expert.memory.schemas import FailureCase, Skill, SuccessCase

console_logger = logging.getLogger(__name__)


def audit_memory_contract(memory_dir: Path) -> dict:
    """Read raw JSONL bytes; preserve legacy files, IDs and unverified claims.

    This is a content/identity view, not proof of task applicability or benefit.
    No FastMemoryStore is opened, so even old banks without manifests stay
    byte-for-byte unchanged. Malformed rows are visible, never silently skipped.
    """
    root = memory_dir.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    files: dict[str, str] = {}
    rows: list[dict] = []
    for filename, model in (
        ("success_cases.jsonl", SuccessCase),
        ("failure_cases.jsonl", FailureCase),
        ("skills.jsonl", Skill),
    ):
        path = root / filename
        if not path.exists():
            continue
        data = path.read_bytes()
        files[filename] = hashlib.sha256(data).hexdigest()
        seen: set[str] = set()
        for number, line in enumerate(data.decode("utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = model.model_validate(json.loads(line))
                payload = selection_from_record(record, rank=number, memory_dir=root)
                reasons = list(payload.evidence_warnings)
                if record.status != "active":
                    reasons.append("record_not_active")
                if payload.memory_id in seen:
                    reasons.append("duplicate_identity")
                seen.add(payload.memory_id)
                rows.append(
                    {
                        "file": filename,
                        "line": number,
                        "status": record.status,
                        "reader_eligible": record.status == "active"
                        and "duplicate_identity" not in reasons
                        and bool(payload.injected_text),
                        "requires_task_compatibility_check": True,
                        "warnings": reasons,
                        "payload": payload.model_dump(mode="json"),
                        "spatial_status": [
                            {
                                "constraint_id": relation.evidence_ref,
                                "stored_geometry_verified": relation.geometry_verified,
                                "verified_geometry_eligible": relation.has_verified_geometry,
                                "verification_status": relation.verification_status,
                            }
                            for relation in record.spatial_relations
                        ],
                    }
                )
            except (ValidationError, ValueError, TypeError) as exc:
                rows.append(
                    {
                        "file": filename,
                        "line": number,
                        "reader_eligible": False,
                        "warnings": ["malformed_record"],
                        "error": str(exc),
                    }
                )
    identities: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if "payload" in row:
            identities.setdefault(
                (row["file"], row["payload"]["memory_id"]), []
            ).append(row)
    for duplicates in identities.values():
        if len(duplicates) > 1:
            for row in duplicates:
                row["reader_eligible"] = False
                if "duplicate_identity" not in row["warnings"]:
                    row["warnings"].append("duplicate_identity")
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {
        "contract_version": PAYLOAD_VERSION,
        "bank_files_hash": identity,
        "source_files": files,
        "record_count": len(rows),
        "records": rows,
    }


def main() -> None:
    """Create a new audit file outside the source bank; never overwrite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.memory_dir.resolve(strict=True)
    output = args.output.resolve()
    if output == root or root in output.parents:
        parser.error("--output must be outside the memory bank")
    report = audit_memory_contract(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    console_logger.info("Exported %d records to %s", report["record_count"], output)


if __name__ == "__main__":
    main()
