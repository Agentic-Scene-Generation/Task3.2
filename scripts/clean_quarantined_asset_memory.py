"""Remove memory records that reference confirmed-invalid HSSD assets.

The command is dry-run by default. ``--apply`` locks the memory bank, creates
timestamped backups, atomically rewrites affected JSONL files, and writes an
audit report. Rebuild hybrid indexes after an applied cleanup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenesmith.agent_utils.asset_quarantine import quarantined_hssd_asset_ids

MEMORY_BANKS = (
    "success_cases.jsonl",
    "failure_cases.jsonl",
    "skills.jsonl",
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


def clean_quarantined_asset_memory(
    *,
    memory_dir: str | Path,
    asset_ids: set[str] | None = None,
    apply: bool = False,
) -> dict[str, object]:
    """Inspect or remove exact quarantined asset references from all banks."""

    memory_root = Path(memory_dir).resolve()
    if not memory_root.is_dir():
        raise FileNotFoundError(f"Memory directory not found: {memory_root}")
    blocked_ids = {
        str(value).strip().casefold()
        for value in (asset_ids or set(quarantined_hssd_asset_ids()))
        if str(value).strip()
    }
    if not blocked_ids:
        raise ValueError("At least one quarantined HSSD asset ID is required")

    with _memory_lock(memory_root):
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        bank_results: dict[str, dict[str, object]] = {}
        rewrites: dict[Path, list[str]] = {}
        total_removed = 0

        for bank_name in MEMORY_BANKS:
            bank_path = memory_root / bank_name
            if not bank_path.exists():
                continue
            source_lines = bank_path.read_text(encoding="utf-8").splitlines()
            retained_lines: list[str] = []
            removed_ids: list[str] = []
            malformed_line_numbers: list[int] = []

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
                serialized = json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=str,
                ).casefold()
                matched_ids = sorted(
                    asset_id for asset_id in blocked_ids if asset_id in serialized
                )
                if matched_ids:
                    removed_ids.extend(matched_ids)
                else:
                    retained_lines.append(line)

            removed_count = sum(1 for line in source_lines if line.strip()) - sum(
                1 for line in retained_lines if line.strip()
            )
            total_removed += removed_count
            if removed_count:
                rewrites[bank_path] = retained_lines
            bank_results[bank_name] = {
                "source_sha256": _sha256(bank_path),
                "before_count": sum(1 for line in source_lines if line.strip()),
                "after_count": sum(1 for line in retained_lines if line.strip()),
                "removed_count": removed_count,
                "matched_asset_ids": sorted(set(removed_ids)),
                "malformed_line_numbers_preserved": malformed_line_numbers,
                "backup_path": "",
            }

        report: dict[str, object] = {
            "schema_version": "1.0",
            "status": "APPLIED" if apply else "DRY_RUN",
            "created_at": timestamp,
            "memory_dir": str(memory_root),
            "quarantined_asset_ids": sorted(blocked_ids),
            "removed_count": total_removed,
            "banks": bank_results,
            "report_path": "",
            "index_rebuild_required": bool(apply and total_removed),
        }
        if not apply or not rewrites:
            return report

        backup_dir = memory_root / "backups"
        migration_dir = memory_root / "migrations"
        backup_dir.mkdir(parents=True, exist_ok=True)
        migration_dir.mkdir(parents=True, exist_ok=True)
        for bank_path, retained_lines in rewrites.items():
            backup_path = (
                backup_dir
                / f"{bank_path.stem}.before_asset_quarantine.{timestamp}.jsonl"
            )
            shutil.copy2(bank_path, backup_path)
            temporary_path = bank_path.with_suffix(
                bank_path.suffix + f".{os.getpid()}.tmp"
            )
            temporary_path.write_text(
                "\n".join(retained_lines) + ("\n" if retained_lines else ""),
                encoding="utf-8",
            )
            temporary_path.replace(bank_path)
            bank_results[bank_path.name]["backup_path"] = str(backup_path)
            bank_results[bank_path.name]["result_sha256"] = _sha256(bank_path)

        report_path = migration_dir / f"quarantined_asset_cleanup.{timestamp}.json"
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
        help="Memory bank root containing JSONL memory files.",
    )
    parser.add_argument(
        "--asset-id",
        action="append",
        default=[],
        help=(
            "Exact HSSD mesh ID to remove. Repeat for multiple IDs. When omitted, "
            "the checked-in quarantine is used."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply cleanup. Without this flag the command is read-only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = clean_quarantined_asset_memory(
        memory_dir=args.memory_dir,
        asset_ids=set(args.asset_id) if args.asset_id else None,
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
