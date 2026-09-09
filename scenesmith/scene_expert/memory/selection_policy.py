"""Conservative cross-bank admission policy for Fast Memory prompt injection."""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Any

from scenesmith.scene_expert.memory.adaptation import source_conflicts
from scenesmith.scene_expert.memory.contracts import selection_from_record
from scenesmith.scene_expert.memory.placement import experience_priority, inventory_only
from scenesmith.scene_expert.memory.schemas import FailureCase, Skill, SuccessCase
from scenesmith.scene_expert.memory.scoring import (
    object_overlap,
    record_required_objects,
    task_required_objects,
)
from scenesmith.scene_expert.memory.state import observed_roles
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.schemas import (
    MemoryPack,
    MemorySelectionDecision,
    RetrievedMemorySelection,
    SceneTaskSpec,
    SkillSelectionDecision,
    StageRelationContext,
)


@dataclass(frozen=True)
class MemoryInjectionPolicy:
    """One shared record/character budget across all memory banks."""

    max_total_records: int = 3
    max_success_cases: int = 1
    max_failure_cases: int = 1
    max_skills: int = 1
    max_total_chars: int = 8000
    require_verified_failures: bool = True
    require_failure_grounding: bool = True
    object_overlap_threshold: float = 0.15


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _relation_types(context: StageRelationContext | None) -> set[str]:
    if context is None:
        return set()
    values: set[str] = set()
    for row in context.hard_constraints:
        for key in ("relation_type", "predicate", "relation", "type"):
            value = _normalized_text(str(row.get(key) or ""))
            if value:
                values.add(value)
    return values


class BudgetedMemoryRetriever:
    """Wrap either retriever with one deterministic, inspectable admission gate."""

    # The wrapper writes the final post-policy counts through HookRunner.  A
    # hybrid delegate may still retain its detailed recall/rerank timing file.
    writes_detailed_timing = False

    def __init__(
        self,
        delegate: Any,
        *,
        store: FastMemoryStore,
        policy: MemoryInjectionPolicy,
    ) -> None:
        self._delegate = delegate
        self._store = store
        self.policy = policy

    def retrieve(
        self,
        task_spec: SceneTaskSpec,
        stage: str,
        relation_context: StageRelationContext | None = None,
        scene_state: dict | None = None,
    ) -> MemoryPack:
        candidate_pack = self._delegate.retrieve(
            task_spec,
            stage,
            relation_context=relation_context,
            **({"scene_state": scene_state} if scene_state is not None else {}),
        ).deduplicated()
        if scene_state is not None:
            candidate_pack = candidate_pack.model_copy(
                update={"current_scene_state": scene_state}
            )
        return self._apply(candidate_pack, task_spec, stage, relation_context)

    def _apply(
        self,
        pack: MemoryPack,
        task_spec: SceneTaskSpec,
        stage: str,
        relation_context: StageRelationContext | None,
    ) -> MemoryPack:
        records = self._record_index()
        selections_by_id = {
            (row.memory_type, row.memory_id): row for row in pack.selections
        }
        candidate_ids = {
            "success": list(pack.success_case_ids),
            "failure": list(pack.failure_case_ids),
            "skill": list(pack.skill_names),
        }
        for kind, ids in candidate_ids.items():
            ids.sort(
                key=lambda key: (
                    -experience_priority(records[(kind, key)])
                    if records.get((kind, key)) is not None
                    else 0
                )
            )
        limits = {
            "success": max(0, self.policy.max_success_cases),
            "failure": max(0, self.policy.max_failure_cases),
            "skill": max(0, self.policy.max_skills),
        }
        selected: dict[str, list[str]] = {key: [] for key in candidate_ids}
        decisions: list[MemorySelectionDecision] = []
        selected_text_keys: set[str] = set()
        consumed_chars = 0
        consumed_records = 0
        admitted_rows: list[RetrievedMemorySelection] = []

        # Positive patterns and verified procedures establish a plan before the
        # single negative guard is admitted. This prevents a failure-heavy bank
        # from dominating the designer context.
        for memory_type in ("success", "skill", "failure"):
            for memory_id in candidate_ids[memory_type]:
                row = selections_by_id.get((memory_type, memory_id))
                record = records.get((memory_type, memory_id))
                reasons = self._rejection_reasons(
                    memory_type=memory_type,
                    record=record,
                    task_spec=task_spec,
                    stage=stage,
                    relation_context=relation_context,
                    available_objects=observed_roles(pack.current_scene_state),
                )
                if (memory_type, memory_id) in records and record is None:
                    reasons = ["ambiguous_record_identity"]
                if not memory_id.strip():
                    reasons.append("empty_record_identity")
                canonical = None
                if record is not None:
                    canonical = selection_from_record(
                        record,
                        rank=row.rank if row is not None else 1,
                        memory_dir=self._store.memory_dir,
                        bank_id=self._store.bank_id,
                        bank_revision=self._store.revision,
                        score=row.score if row is not None else None,
                        score_components=(
                            row.score_components if row is not None else {}
                        ),
                    )
                    reasons.extend(source_conflicts(canonical, relation_context))
                    if (
                        row is not None
                        and row.content_hash
                        and row.content_hash != canonical.content_hash
                    ):
                        reasons.append("source_content_changed")
                if len(selected[memory_type]) >= limits[memory_type]:
                    reasons.append("type_budget_pruned")
                if consumed_records >= max(0, self.policy.max_total_records):
                    reasons.append("record_budget_pruned")
                text = canonical.injected_text if canonical is not None else ""
                layout = canonical.placement_text if canonical is not None else ""
                if not text.strip():
                    reasons.append("empty_payload")
                # Punctuation can encode geometry (for example -1 versus +1).
                # Lexical token normalization is not semantic deduplication.
                text_key = " ".join((text + "\n" + layout).split()).casefold()
                if text_key and text_key in selected_text_keys:
                    reasons.append("duplicate_content")
                # Account for the actual, selected record's complete content.
                # Never charge or retain a different candidate's shared layout.
                extra_chars = len(text) + len(layout) + (2 if layout else 0)
                if consumed_chars + extra_chars > max(0, self.policy.max_total_chars):
                    reasons.append("prompt_budget_pruned")
                if reasons:
                    decisions.append(
                        self._decision(memory_id, memory_type, row, "rejected", reasons)
                    )
                    continue
                selected[memory_type].append(memory_id)
                assert canonical is not None
                admitted_rows.append(
                    canonical.model_copy(update={"rank": len(selected[memory_type])})
                )
                consumed_records += 1
                consumed_chars += extra_chars
                if text_key:
                    selected_text_keys.add(text_key)
                decisions.append(
                    self._decision(memory_id, memory_type, row, "selected", [])
                )

        success_ids = selected["success"]
        failure_ids = selected["failure"]
        skill_names = selected["skill"]
        # Legacy maps cannot express cross-bank ID collisions. Canonical rows
        # always carry typed provenance; omit ambiguous keys from legacy maps.
        id_counts = {
            row.memory_id: sum(
                other.memory_id == row.memory_id for other in admitted_rows
            )
            for row in admitted_rows
        }
        skill_decisions = self._updated_skill_decisions(
            pack.skill_filter_decisions,
            selected=set(skill_names),
        )
        return pack.model_copy(
            update={
                "success_hints": [
                    row.injected_text
                    for row in admitted_rows
                    if row.memory_type == "success"
                ],
                "failure_hints": [
                    row.injected_text
                    for row in admitted_rows
                    if row.memory_type == "failure"
                ],
                "skill_texts": [
                    row.injected_text
                    for row in admitted_rows
                    if row.memory_type == "skill"
                ],
                "placement_reference": "",
                "success_case_ids": success_ids,
                "failure_case_ids": failure_ids,
                "skill_names": skill_names,
                "retrieved_source_task_ids": {
                    row.memory_id: row.source_task_ids
                    for row in admitted_rows
                    if id_counts[row.memory_id] == 1
                },
                "retrieved_source_run_ids": {
                    row.memory_id: row.source_run_ids
                    for row in admitted_rows
                    if id_counts[row.memory_id] == 1
                },
                "selections": admitted_rows,
                "selection_decisions": decisions,
                "selection_policy": {
                    "schema_version": "sceneexpert.memory_injection_policy.v2",
                    "max_total_records": self.policy.max_total_records,
                    "max_success_cases": self.policy.max_success_cases,
                    "max_failure_cases": self.policy.max_failure_cases,
                    "max_skills": self.policy.max_skills,
                    "max_total_chars": self.policy.max_total_chars,
                    "require_verified_failures": self.policy.require_verified_failures,
                    "require_failure_grounding": self.policy.require_failure_grounding,
                    "selected_records": consumed_records,
                    "selected_chars": consumed_chars,
                },
                "skill_filter_decisions": skill_decisions,
            }
        ).deduplicated()

    def _rejection_reasons(
        self,
        *,
        memory_type: str,
        record: SuccessCase | FailureCase | Skill | None,
        task_spec: SceneTaskSpec,
        stage: str,
        relation_context: StageRelationContext | None,
        available_objects: list[str] | None = None,
    ) -> list[str]:
        if record is None:
            return ["missing_record"]
        reasons: list[str] = []
        if record.stage != stage:
            reasons.append("stage_mismatch")
        if record.placement_experience is not None and not experience_priority(record):
            reasons.append("invalid_placement_experience")
        if isinstance(record, SuccessCase) and inventory_only(
            record.positive_guidance or record.successful_pattern
        ):
            reasons.append("redundant_inventory_restatement")
        if isinstance(record, Skill) and inventory_only(record.procedure):
            reasons.append("redundant_inventory_restatement")
        if memory_type != "failure" or not isinstance(record, FailureCase):
            return reasons
        if self.policy.require_verified_failures and not (
            record.repair_verified or record.is_deterministic
        ):
            reasons.append("unverified_failure")
        if not self.policy.require_failure_grounding:
            return reasons
        if (
            record.is_deterministic
            and record.scope in {"global", "stage"}
            and not (record_required_objects(record) or record.spatial_relations)
        ):
            return reasons
        task_objects = task_required_objects(task_spec, stage) + (
            available_objects or []
        )
        object_match = bool(record_required_objects(record)) and (
            object_overlap(record_required_objects(record), task_objects)
            >= self.policy.object_overlap_threshold
        )
        active_relations = _relation_types(relation_context)
        record_relations = {
            _normalized_text(relation.relation_type)
            for relation in record.spatial_relations
            if _normalized_text(relation.relation_type)
        }
        relation_match = bool(record_relations & active_relations)
        if not object_match and not relation_match:
            reasons.append(
                "relation_mismatch" if record_relations else "object_mismatch"
            )
        return reasons

    def _record_index(
        self,
    ) -> dict[tuple[str, str], SuccessCase | FailureCase | Skill | None]:
        output: dict[tuple[str, str], SuccessCase | FailureCase | Skill | None] = {}
        for kind, field, records in (
            ("success", "case_id", self._store.active_success_cases),
            ("failure", "failure_id", self._store.active_failure_cases),
            ("skill", "skill_name", self._store.active_skills),
        ):
            for record in records:
                key = (kind, getattr(record, field))
                output[key] = None if key in output else record
        return output

    @staticmethod
    def _decision(
        memory_id: str,
        memory_type: str,
        row: RetrievedMemorySelection | None,
        decision: str,
        reasons: list[str],
    ) -> MemorySelectionDecision:
        return MemorySelectionDecision(
            memory_id=memory_id,
            memory_type=memory_type,
            decision=decision,
            reasons=list(dict.fromkeys(reasons)),
            retrieval_rank=row.rank if row is not None else 0,
            retrieval_score=row.score if row is not None else None,
        )

    @staticmethod
    def _updated_skill_decisions(
        decisions: list[SkillSelectionDecision],
        *,
        selected: set[str],
    ) -> list[SkillSelectionDecision]:
        output: list[SkillSelectionDecision] = []
        for decision in decisions:
            if decision.decision == "rejected":
                output.append(decision)
                continue
            output.append(
                decision.model_copy(
                    update={
                        "decision": (
                            "selected"
                            if decision.skill_name in selected
                            else "not_selected"
                        )
                    }
                )
            )
        return output
