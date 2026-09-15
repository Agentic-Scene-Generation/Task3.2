"""Convert external teacher trajectories into the canonical Slow Memory schema."""

from __future__ import annotations

import hashlib
import json
import time

from pathlib import Path
from typing import Any, Iterable

from scenesmith.scene_expert.slow_memory.schemas import (
    PreferenceEvidence,
    TrajectoryOutcome,
    TrajectoryRecord,
)
from scenesmith.scene_expert.slow_memory.trajectory import redact_sensitive_text


def _hash(value: Any, length: int = 64) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact(item) for key, item in value.items()}
    return value


def _image_refs(paths: Iterable[Any], *, source_dir: Path) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for value in paths:
        path = Path(str(value))
        if not path.is_absolute():
            path = source_dir / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"teacher image does not exist: {path}")
        references.append(
            {
                "path": str(path),
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return references


def import_teacher_trajectories(
    *,
    source_paths: Iterable[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Import explicit candidate outcomes without inventing preference labels."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: dict[str, TrajectoryRecord] = {}
    diagnostics: list[dict[str, Any]] = []
    for source_path in source_paths:
        source_path = Path(source_path)
        for line_number, line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("teacher row must be a JSON object")
                messages = _redact(raw.get("messages") or [])
                completion = _redact(raw.get("completion_messages") or [])
                tools = _redact(raw.get("tools") or [])
                spatial_context = _redact(raw.get("spatial_context") or {})
                images = _image_refs(
                    raw.get("image_paths") or [],
                    source_dir=source_path.parent,
                )
                if not messages or not completion:
                    raise ValueError(
                        "messages and completion_messages are required and lossless"
                    )
                role = str(raw.get("agent_role") or "")
                event = str(raw.get("event") or "")
                task_type = str(raw.get("task_type") or "")
                if not role or not event or not task_type:
                    raise ValueError("agent_role, event, and task_type are required")
                outcome = TrajectoryOutcome.model_validate(raw.get("outcome") or {})
                evidence_raw = raw.get("evidence") or {}
                verdict = str(evidence_raw.get("verdict") or "unlabeled")
                authoritative = bool(evidence_raw.get("authoritative", False))
                if (
                    role == "critic"
                    and verdict in {"accepted", "rejected"}
                    and not outcome.causal_link_verified
                ):
                    raise ValueError(
                        "critic preference labels require downstream causal_link_verified"
                    )
                source_ref = f"{source_path.resolve()}:{line_number}"
                report_ref = str(evidence_raw.get("report_ref") or source_ref)
                evidence = PreferenceEvidence(
                    evidence_id=str(
                        evidence_raw.get("evidence_id")
                        or "evidence_" + _hash([source_ref, evidence_raw], 24)
                    ),
                    kind=str(evidence_raw.get("kind") or "none"),
                    verdict=verdict,
                    source=str(evidence_raw.get("source") or "external_teacher"),
                    authoritative=authoritative,
                    quality_score=evidence_raw.get("quality_score"),
                    report_ref=report_ref if authoritative else "",
                    details=_redact(evidence_raw.get("details") or {}),
                )
                context_payload = {
                    "role": role,
                    "event": event,
                    "task_type": task_type,
                    "messages": messages,
                    "tools": tools,
                    "image_hashes": [image["sha256"] for image in images],
                    "spatial_context": spatial_context,
                }
                task_id = str(raw.get("task_id") or "task_" + _hash(messages, 16))
                trajectory_id = str(
                    raw.get("trajectory_id")
                    or "trajectory_"
                    + _hash([source_ref, context_payload, completion], 24)
                )
                record = TrajectoryRecord(
                    trajectory_id=trajectory_id,
                    created_at=str(
                        raw.get("created_at")
                        or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    ),
                    run_id=str(raw.get("run_id") or "external_teacher"),
                    scene_id=str(raw.get("scene_id") or "external_scene"),
                    task_id=task_id,
                    scenario_family_id=str(raw.get("scenario_family_id") or task_id),
                    experiment_signature=str(raw.get("experiment_signature") or ""),
                    config_hash=str(raw.get("config_hash") or ""),
                    model_id=str(raw.get("model_id") or ""),
                    stage=str(raw.get("stage") or ""),
                    agent_role=role,
                    event=event,
                    task_type=task_type,
                    context_hash=_hash(context_payload),
                    prompt=json.dumps(messages, ensure_ascii=False, sort_keys=True),
                    response=json.dumps(completion, ensure_ascii=False, sort_keys=True),
                    response_hash=_hash(completion),
                    evidence=evidence,
                    source_refs=[source_ref, *map(str, raw.get("source_refs") or [])],
                    messages=messages,
                    completion_messages=completion,
                    tools=tools,
                    image_refs=images,
                    spatial_context=spatial_context,
                    action_trace=_redact(raw.get("action_trace") or []),
                    outcome=outcome,
                    provenance={
                        "import_source": source_ref,
                        "external_sample_id": str(raw.get("sample_id") or ""),
                    },
                    metadata=_redact(raw.get("metadata") or {}),
                )
                existing = records.get(record.trajectory_id)
                if existing is not None and existing != record:
                    raise ValueError("trajectory_id collision with different content")
                records[record.trajectory_id] = record
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                diagnostics.append(
                    {
                        "source": str(source_path),
                        "line": line_number,
                        "error": str(exc),
                    }
                )
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sorted(records.values(), key=lambda item: item.trajectory_id):
            handle.write(record.model_dump_json() + "\n")
    diagnostics_path = output_path.with_suffix(".diagnostics.jsonl")
    with diagnostics_path.open("w", encoding="utf-8", newline="\n") as handle:
        for diagnostic in diagnostics:
            handle.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")
    return {
        "schema_version": "sceneexpert.teacher_import.v1",
        "record_count": len(records),
        "rejected_count": len(diagnostics),
        "output_path": str(output_path),
        "diagnostics_path": str(diagnostics_path),
    }
