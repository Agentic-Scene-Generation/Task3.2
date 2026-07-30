"""Remove failure memories created by the legacy cross-stage attribution bug.

The command is dry-run by default.  ``--apply`` acquires the memory-bank lock,
creates a timestamped backup, atomically rewrites ``failure_cases.jsonl``, and
writes a migration report.  Rebuild hybrid indexes after a successful apply.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


KNOWN_P1_POLLUTED_FAILURE_IDS = (
    "trace_000000_wall_mounted_collision",
    "failure_bedroom_ceiling_mounted_e1aa559564a1",
    "trace_000000_collision_pattern",
    "trace_000002_physics_collision",
    "trace_000000_cascading_failure",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _memory_lock(memory_dir: Path) -> Iterator[None]:
    lock_path = memory_dir / ".memory.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except ImportError:
            fcntl = None
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def clean_failure_bank(
    *,
    memory_dir: str | Path,
    failure_ids: set[str],
    apply: bool = False,
) -> dict[str, object]:
    """Inspect or remove exact failure IDs without heuristic deletion."""

    memory_root = Path(memory_dir).resolve()
    failure_path = memory_root / "failure_cases.jsonl"
    if not failure_path.is_file():
        raise FileNotFoundError(f"Failure memory bank not found: {failure_path}")

    with _memory_lock(memory_root):
        source_sha256 = _sha256(failure_path)
        source_lines = failure_path.read_text(encoding="utf-8").splitlines()
        retained_lines: list[str] = []
        removed_records: list[dict] = []
        malformed_line_numbers: list[int] = []
        observed_ids: set[str] = set()

        for line_number, line in enumerate(source_lines, start=1):
            if not line.strip():
                retained_lines.append(line)
                continue
            try:
                payload = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                malformed_line_numbers.append(line_number)
                retained_lines.append(line)
                continue
            failure_id = str(payload.get("failure_id", "") or "")
            if failure_id in failure_ids:
                observed_ids.add(failure_id)
                removed_records.append(payload)
            else:
                retained_lines.append(line)

        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        report: dict[str, object] = {
            "schema_version": "1.0",
            "status": "APPLIED" if apply else "DRY_RUN",
            "created_at": timestamp,
            "memory_dir": str(memory_root),
            "failure_bank": str(failure_path),
            "source_sha256": source_sha256,
            "requested_failure_ids": sorted(failure_ids),
            "removed_failure_ids": sorted(observed_ids),
            "missing_failure_ids": sorted(failure_ids - observed_ids),
            "removed_count": len(removed_records),
            "before_count": sum(1 for line in source_lines if line.strip()),
            "after_count": sum(1 for line in retained_lines if line.strip()),
            "malformed_line_numbers_preserved": malformed_line_numbers,
            "backup_path": "",
            "report_path": "",
            "index_rebuild_required": bool(apply and removed_records),
        }
        if not apply or not removed_records:
            return report

        backup_dir = memory_root / "backups"
        migration_dir = memory_root / "migrations"
        backup_dir.mkdir(parents=True, exist_ok=True)
        migration_dir.mkdir(parents=True, exist_ok=True)
        backup_path = (
            backup_dir / f"failure_cases.before_stage_cleanup.{timestamp}.jsonl"
        )
        shutil.copy2(failure_path, backup_path)

        temporary_path = failure_path.with_suffix(
            failure_path.suffix + f".{os.getpid()}.tmp"
        )
        temporary_path.write_text(
            "\n".join(retained_lines) + ("\n" if retained_lines else ""),
            encoding="utf-8",
        )
        temporary_path.replace(failure_path)

        report["backup_path"] = str(backup_path)
        report["result_sha256"] = _sha256(failure_path)
        report_path = migration_dir / f"stage_attribution_cleanup.{timestamp}.json"
        report["report_path"] = str(report_path)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-dir",
        required=True,
        help="Memory bank root containing failure_cases.jsonl.",
    )
    parser.add_argument(
        "--failure-id",
        action="append",
        default=[],
        help=(
            "Exact failure ID to remove. Repeat to provide multiple IDs. "
            "When omitted, the five reviewed P1-contaminated IDs are used."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the cleanup. Without this flag the command is read-only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failure_ids = set(args.failure_id or KNOWN_P1_POLLUTED_FAILURE_IDS)
    report = clean_failure_bank(
        memory_dir=args.memory_dir,
        failure_ids=failure_ids,
        apply=bool(args.apply),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["index_rebuild_required"]:
        print(
            "\nIndex rebuild required. Run:\n"
            f"python scripts/build_memory_index.py --memory-dir {args.memory_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
