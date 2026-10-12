"""Select complete floor-plan checkpoints without changing original case indices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator


def _declared_paths(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"geometry_path", "sdf_path", "image_path"} and item:
                if not isinstance(item, str):
                    raise ValueError(f"invalid declared {key}")
                yield item
            else:
                yield from _declared_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _declared_paths(item)


def checkpoint_evidence(scene: Path, row: dict[str, str]) -> dict[str, Any]:
    """Check completion/identity and readable local geometry; never create a base."""
    status_path, layout_path = scene / "scene_status.json", scene / "house_layout.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") not in {"completed", "completed_with_quality_issues"}:
        raise ValueError(f"scene status is {status.get('status')}")
    if (
        status.get("scene_id") != int(row["scene_index"])
        or status.get("prompt") != row["prompt"]
    ):
        raise ValueError("checkpoint identity does not match assigned case")
    marker = "_SUCCESS" if status["status"] == "completed" else "_DEGRADED"
    if not (scene / marker).is_file():
        raise ValueError(f"completion marker missing: {marker}")
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    rooms, geometries = layout.get("rooms"), layout.get("room_geometries")
    if not isinstance(rooms, list) or not rooms or not isinstance(geometries, dict):
        raise ValueError("checkpoint has no materialized rooms")
    room_ids = [room.get("id") for room in rooms if isinstance(room, dict)]
    if len(room_ids) != len(rooms) or len(set(room_ids)) != len(room_ids):
        raise ValueError("invalid room identities")
    if any(
        not isinstance(key, str) or not (geometries.get(key) or {}).get("sdf_path")
        for key in room_ids
    ):
        raise ValueError("room geometry SDF missing from layout")
    paths = {status_path, layout_path, scene / marker}
    for reference in _declared_paths(layout):
        path = (scene / reference).resolve()
        if (
            not path.is_relative_to(scene.resolve())
            or not path.is_file()
            or not path.stat().st_size
        ):
            raise ValueError(f"missing or external checkpoint geometry: {reference}")
        if path.suffix == ".sdf":
            ET.parse(path)
        paths.add(path)
    return {
        "status": status["status"],
        "scene_path": str(scene),
        "artifact_sha256": {
            str(path.relative_to(scene)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)
        },
    }


def partition_shared_base(output_root: Path, shared_base_root: Path) -> dict[str, Any]:
    """Audit every assigned case and retain its original batch/scene mapping."""
    output_root, shared_base_root = output_root.resolve(), shared_base_root.resolve()
    with (output_root / "assigned_cases.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("assignment manifest is empty")
    cases, indices, batches = set(), set(), set()
    ready, excluded = [], []
    for row in rows:
        case, index, batch = row["case_id"], int(row["scene_index"]), row["batch_id"]
        if (
            not re.fullmatch(r"[a-zA-Z0-9_-]+", case)
            or not re.fullmatch(r"batch_[0-9]+", batch)
            or index < 0
        ):
            raise ValueError("invalid assignment identity")
        if case in cases or index in indices or batch in batches:
            raise ValueError(
                "partial shared bases require unique cases and one scene per batch"
            )
        cases.add(case)
        indices.add(index)
        batches.add(batch)
        scene = shared_base_root / batch / "hydra" / f"scene_{index:03d}"
        record = {"case_id": case, "scene_index": index, "batch_id": batch}
        try:
            record["evidence"] = checkpoint_evidence(scene, row)
            ready.append(record)
        except (OSError, ValueError, TypeError, AttributeError, ET.ParseError) as exc:
            record["reason"] = str(exc)
            excluded.append(record)
    return {
        "schema_version": "sceneexpert.collection_fault_isolation.v1",
        "scope": "checkpoint_readiness_only",
        "original_assignment_sha256": hashlib.sha256(
            (output_root / "assigned_cases.csv").read_bytes()
        ).hexdigest(),
        "original_case_count": len(rows),
        "complete": not excluded,
        "eligible": ready,
        "excluded": excluded,
        "failed_cases_become_preferences": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shared-base-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = partition_shared_base(args.output_root, args.shared_base_root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {
            "schema_version": "sceneexpert.collection_fault_isolation.v1",
            "error": str(exc),
            "eligible": [],
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["eligible"]:
        sys.stderr.write(
            "No validated shared-base checkpoint; see " + str(args.report) + "\n"
        )
        return 2
    sys.stdout.write(
        ",".join(row["case_id"] for row in report["eligible"])
        + "|"
        + str(len(report["excluded"]))
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
