#!/usr/bin/env python3
"""Summarize persisted reasoning artifacts without printing their contents."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


def inspect_database(path: Path) -> dict | None:
    """Return artifact metadata for one SQLite DB, or None if no table exists."""
    try:
        with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as conn:
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='agent_reasoning_artifacts'"
            ).fetchone()
            if table_exists is None:
                return None
            rows = conn.execute(
                "SELECT provider, model, source_type, "
                "LENGTH(COALESCE(summary, '')), LENGTH(COALESCE(raw_json, '')) "
                "FROM agent_reasoning_artifacts"
            ).fetchall()
    except sqlite3.Error as exc:
        return {"path": str(path), "error": str(exc)}

    providers = Counter(str(row[0] or "unknown") for row in rows)
    models = Counter(str(row[1] or "unknown") for row in rows)
    source_types = Counter(str(row[2] or "unknown") for row in rows)
    return {
        "path": str(path),
        "records": len(rows),
        "nonempty_summary_records": sum(int(row[3] or 0) > 0 for row in rows),
        "nonempty_raw_records": sum(int(row[4] or 0) > 0 for row in rows),
        "summary_chars": sum(int(row[3] or 0) for row in rows),
        "providers": dict(sorted(providers.items())),
        "models": dict(sorted(models.items())),
        "source_types": dict(sorted(source_types.items())),
    }


def build_report(root: Path) -> dict:
    databases = sorted(root.rglob("*.db"))
    inspected: list[dict] = []
    errors: list[dict] = []
    for database in databases:
        result = inspect_database(database)
        if result is None:
            continue
        if "error" in result:
            errors.append(result)
        else:
            inspected.append(result)

    return {
        "root": str(root.resolve()),
        "sqlite_databases_scanned": len(databases),
        "databases_with_reasoning_table": len(inspected),
        "artifact_records": sum(item["records"] for item in inspected),
        "nonempty_summary_records": sum(
            item["nonempty_summary_records"] for item in inspected
        ),
        "summary_chars": sum(item["summary_chars"] for item in inspected),
        "databases": inspected,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include per-database metadata in stdout",
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Exit nonzero if no non-empty reasoning summary was found",
    )
    args = parser.parse_args()
    if not args.output_root.is_dir():
        raise SystemExit(f"Output directory does not exist: {args.output_root}")

    report = build_report(args.output_root)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(serialized, encoding="utf-8")
    console_report = report if args.details else {
        key: value
        for key, value in report.items()
        if key not in {"databases", "errors"}
    }
    console_report["database_errors"] = len(report["errors"])
    print(json.dumps(console_report, ensure_ascii=False, indent=2))
    if args.require_artifacts and report["nonempty_summary_records"] == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
