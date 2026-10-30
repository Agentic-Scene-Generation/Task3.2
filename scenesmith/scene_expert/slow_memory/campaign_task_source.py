"""Resolve a frozen campaign's explicit task source before any worker starts."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CampaignTaskSource:
    """Validated input paths and controller settings, without model side effects."""

    annotations: Path
    environment: dict[str, str]
    identity: dict[str, Any]


def _path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("campaign task input must be an existing project file")
    return path


def _bytes(root: Path, ref: dict[str, Any]) -> tuple[Path, bytes]:
    path = _path(root, ref["path"])
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != ref["sha256"]:
        raise ValueError("campaign task input content changed after freezing")
    return path, content


def _rows(content: bytes) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    if not rows or len({row["ID"] for row in rows}) != len(rows):
        raise ValueError("campaign task input must have unique nonempty rows")
    return rows


def _task_id(prompt: str) -> str:
    content = json.dumps(" ".join(prompt.split()), ensure_ascii=False, sort_keys=True)
    return "task_" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def resolve_campaign_task_source(plan: dict[str, Any], root: Path) -> CampaignTaskSource:
    """Preserve legacy inputs or validate an independently namespaced CSV pool."""
    source = plan.get("task_source")
    if source is None:
        annotations = root / "scripts/assets/annotations.csv"
        digest = hashlib.sha256(annotations.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if digest != plan["annotations_sha256"]:
            raise ValueError("SceneEval annotations changed after freezing")
        return CampaignTaskSource(
            annotations,
            {"CASE_SET": "sceneeval500", "SCENEEVAL_SIZE": "500", "SCENEEVAL_ANNOTATIONS": str(annotations)},
            {"kind": "legacy_sceneeval500", "annotations_LF_sha256": digest},
        )
    if source.get("schema") != "sceneexpert.namespaced_prompt_csv_source.v1":
        raise ValueError("unsupported campaign task source schema")
    if plan.get("split_frozen") is not True:
        raise ValueError("new-pool campaign splits must be explicitly frozen before execution")
    policy_path, policy_bytes = _bytes(root, source["policy"])
    policy = json.loads(policy_bytes)
    namespace = source["namespace"]
    if namespace != policy["dataset"]["namespace"] or namespace in {"sceneeval100", "sceneeval500"}:
        raise ValueError("new task namespace does not match the direct human pool policy")
    annotations, original_bytes = _bytes(root, source["original_csv"])
    if source["original_csv"]["sha256"] != policy["dataset"]["sha256"]:
        raise ValueError("original task CSV is not the human-approved input")
    registry, registry_bytes = _bytes(root, source["registry_csv"])
    original = {row["ID"]: row for row in _rows(original_bytes)}
    registry_rows = _rows(registry_bytes)
    accepted = {}
    allowed_ids = set(policy["initial_execution_scope"]["IDs"])
    for row in registry_rows:
        original_row = original.get(row["ID"])
        if original_row is None or int(row["ID"]) not in allowed_ids:
            raise ValueError("registry contains a prefix, whole-house or unapproved task")
        if original_row["is_house"] != "FALSE" or row.get("SceneScope") != "single_room":
            raise ValueError("initial task registry must preserve the approved single-room scope")
        if {key: row.get(key) for key in original_row} != original_row:
            raise ValueError("registry changed an original prompt or annotation field")
        raw_hash = hashlib.sha256(json.dumps(original_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if row.get("DatasetNamespace") != namespace or row.get("SourceRowSHA256") != raw_hash:
            raise ValueError("registry task namespace or raw-row fingerprint does not match")
        accepted[int(row["ID"])] = (original_row, raw_hash)
    seen = set()
    assignments = plan["split_assignments"]
    for task in plan["tasks"]:
        # sceneeval_id remains a compatibility field, explicitly scoped by this
        # source identity. It is not an original SceneEval benchmark mapping.
        number = task["dataset_task_id"]
        if number != task["sceneeval_id"] or number in seen or number not in accepted:
            raise ValueError("campaign task ID mapping or uniqueness is invalid")
        seen.add(number)
        original_row, raw_hash = accepted[number]
        if task["prompt"] != original_row["Description"] or task["raw_row_sha256"] != raw_hash:
            raise ValueError("campaign task prompt or raw annotation binding changed")
        if task["dataset_namespace"] != namespace or task["task_id"] != _task_id(task["prompt"]):
            raise ValueError("campaign task identity changed")
        if task["split"] not in {"train", "validation", "test"} or assignments.get(task["task_id"]) != task["split"]:
            raise ValueError("campaign split assignment is absent or inconsistent")
    if not seen:
        raise ValueError("campaign has no tasks")
    return CampaignTaskSource(
        annotations,
        {"CASE_SET": "promptcsv", "PROMPT_CSV": str(registry)},
        {"kind": "namespaced_promptcsv", "namespace": namespace,
         "original_csv_sha256": source["original_csv"]["sha256"],
         "registry_csv_sha256": source["registry_csv"]["sha256"],
         "policy_sha256": source["policy"]["sha256"],
         "policy_path": policy_path.relative_to(root.resolve()).as_posix()},
    )
