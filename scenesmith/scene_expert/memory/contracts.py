"""Atomic, versioned reader payloads derived from persisted memory records.

No bank mutation or model calls. The same record owns its text, layout, source
identity and evidence warnings throughout recall and admission.
"""

from __future__ import annotations

import hashlib
import json

from pathlib import Path

from scenesmith.scene_expert.memory.schemas import FailureCase, Skill, SuccessCase
from scenesmith.scene_expert.schemas import RetrievedMemorySelection

PAYLOAD_VERSION = "memory-contract.v1"
MemoryRecord = SuccessCase | FailureCase | Skill


def record_content_hash(record: MemoryRecord) -> str:
    """Fingerprint the persisted content, excluding incidental usage counters."""
    content = record.model_dump(
        mode="json",
        exclude={
            "embedding_text",
            "last_used_at",
            "usage_count",
            "positive_utility_count",
            "negative_utility_count",
            "utility_observations",
        },
    )
    return hashlib.sha256(
        json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def selection_from_record(
    record: MemoryRecord,
    *,
    rank: int,
    memory_dir: Path,
    bank_id: str = "",
    bank_revision: int = 0,
    score: float | None = None,
    score_components: dict[str, float] | None = None,
) -> RetrievedMemorySelection:
    """Render one identity-bound payload with conservative legacy evidence."""
    if isinstance(record, SuccessCase):
        kind, memory_id, filename = "success", record.case_id, "success_cases.jsonl"
        text, layout = record.to_positive_guidance(), record.to_placement_text()
        if (
            not any(
                value.strip()
                for value in [*record.positive_guidance, *record.successful_pattern]
            )
            and not layout
        ):
            text = ""
    elif isinstance(record, FailureCase):
        kind, memory_id, filename = "failure", record.failure_id, "failure_cases.jsonl"
        text, layout = record.to_negative_constraint(), ""
        if not (record.negative_constraint.strip() or record.bad_pattern.strip()):
            text = ""
    else:
        kind, memory_id, filename = "skill", record.skill_name, "skills.jsonl"
        text, layout = record.to_procedure_text(), ""
        if not any(step.strip() for step in record.procedure):
            text = ""
    warnings = []
    if not text:
        warnings.append("empty_payload")
    if any(not relation.has_verified_geometry for relation in record.spatial_relations):
        warnings.append("spatial_evidence_not_verified")
    if isinstance(record, SuccessCase) and record.placement_reference:
        warnings.append("legacy_world_layout_omitted")
    return RetrievedMemorySelection(
        memory_id=memory_id,
        memory_type=kind,
        rank=rank,
        score=score,
        score_components=score_components or {},
        source_path=str((memory_dir / filename).resolve()),
        source_task_ids=sorted(
            {record.source_task_id, record.provenance.task_id, *record.source_task_ids}
            - {""}
        ),
        source_run_ids=sorted(
            {record.source_run_id, record.provenance.run_id, *record.source_run_ids}
            - {""}
        ),
        bank_id=bank_id,
        bank_revision=bank_revision,
        injected_text=text,
        placement_text=layout,
        content_hash=record_content_hash(record),
        payload_version=PAYLOAD_VERSION,
        evidence_warnings=warnings,
        spatial_relations=[
            item.model_dump(mode="json") for item in record.spatial_relations
        ],
        applicability=(
            record.applicability.model_dump(mode="json")
            if isinstance(record, Skill)
            else {}
        ),
    )
