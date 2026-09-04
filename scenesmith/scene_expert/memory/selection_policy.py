"""Conservative cross-bank admission policy for Fast Memory prompt injection."""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Any

from scenesmith.scene_expert.memory.schemas import FailureCase, Skill, SuccessCase
from scenesmith.scene_expert.memory.scoring import (
    object_overlap,
    record_required_objects,
    task_required_objects,
)
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
    ) -> MemoryPack:
        candidate_pack = self._delegate.retrieve(
            task_spec,
            stage,
            relation_context=relation_context,
        ).deduplicated()
        return self._apply(candidate_pack, task_spec, stage, relation_context)

    def _apply(
        self,
        pack: MemoryPack,
        task_spec: SceneTaskSpec,
        stage: str,
        relation_context: StageRelationContext | None,
    ) -> MemoryPack:
        records = self._record_index()
        selections_by_id = {row.memory_id: row for row in pack.selections}
        candidate_ids = {
            "success": list(pack.success_case_ids),
            "failure": list(pack.failure_case_ids),
            "skill": list(pack.skill_names),
        }
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

        # Positive patterns and verified procedures establish a plan before the
        # single negative guard is admitted. This prevents a failure-heavy bank
        # from dominating the designer context.
        for memory_type in ("success", "skill", "failure"):
            for memory_id in candidate_ids[memory_type]:
                row = selections_by_id.get(memory_id)
                record = records.get((memory_type, memory_id))
                reasons = self._rejection_reasons(
                    memory_type=memory_type,
                    record=record,
                    task_spec=task_spec,
                    stage=stage,
                    relation_context=relation_context,
                )
                if len(selected[memory_type]) >= limits[memory_type]:
                    reasons.append("type_budget_pruned")
                if consumed_records >= max(0, self.policy.max_total_records):
                    reasons.append("record_budget_pruned")
                text = str(row.injected_text if row is not None else "")
                text_key = _normalized_text(text)
                if text_key and text_key in selected_text_keys:
                    reasons.append("duplicate_content")
                extra_chars = len(text)
                if memory_type == "success" and not selected["success"]:
                    extra_chars += len(pack.placement_reference)
                if consumed_chars + extra_chars > max(0, self.policy.max_total_chars):
                    reasons.append("prompt_budget_pruned")
                if reasons:
                    decisions.append(
                        self._decision(memory_id, memory_type, row, "rejected", reasons)
                    )
                    continue
                selected[memory_type].append(memory_id)
                consumed_records += 1
                consumed_chars += extra_chars
                if text_key:
                    selected_text_keys.add(text_key)
                decisions.append(
                    self._decision(memory_id, memory_type, row, "selected", [])
                )

        selected_ids = {
            memory_id for values in selected.values() for memory_id in values
        }
        selected_rows = [
            row for row in pack.selections if row.memory_id in selected_ids
        ]
        ranks: dict[str, int] = {"success": 0, "failure": 0, "skill": 0}
        reranked_rows: list[RetrievedMemorySelection] = []
        for row in selected_rows:
            ranks[row.memory_type] += 1
            reranked_rows.append(
                row.model_copy(update={"rank": ranks[row.memory_type]})
            )

        success_ids = selected["success"]
        failure_ids = selected["failure"]
        skill_names = selected["skill"]
        source_ids = selected_ids
        skill_decisions = self._updated_skill_decisions(
            pack.skill_filter_decisions,
            selected=set(skill_names),
        )
        return pack.model_copy(
            update={
                "success_hints": self._texts_for_ids(
                    pack.success_hints, pack.success_case_ids, success_ids
                ),
                "failure_hints": self._texts_for_ids(
                    pack.failure_hints, pack.failure_case_ids, failure_ids
                ),
                "skill_texts": self._texts_for_ids(
                    pack.skill_texts, pack.skill_names, skill_names
                ),
                "placement_reference": (
                    pack.placement_reference if success_ids else ""
                ),
                "success_case_ids": success_ids,
                "failure_case_ids": failure_ids,
                "skill_names": skill_names,
                "retrieved_source_task_ids": {
                    key: value
                    for key, value in pack.retrieved_source_task_ids.items()
                    if key in source_ids
                },
                "retrieved_source_run_ids": {
                    key: value
                    for key, value in pack.retrieved_source_run_ids.items()
                    if key in source_ids
                },
                "selections": reranked_rows,
                "selection_decisions": decisions,
                "selection_policy": {
                    "schema_version": "sceneexpert.memory_injection_policy.v1",
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
    ) -> list[str]:
        if record is None:
            return ["missing_record"]
        reasons: list[str] = []
        if record.stage != stage:
            reasons.append("stage_mismatch")
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
        task_objects = task_required_objects(task_spec, stage)
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
    ) -> dict[tuple[str, str], SuccessCase | FailureCase | Skill]:
        output: dict[tuple[str, str], SuccessCase | FailureCase | Skill] = {}
        for record in self._store.active_success_cases:
            output[("success", record.case_id)] = record
        for record in self._store.active_failure_cases:
            output[("failure", record.failure_id)] = record
        for record in self._store.active_skills:
            output[("skill", record.skill_name)] = record
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
    def _texts_for_ids(
        texts: list[str],
        original_ids: list[str],
        selected_ids: list[str],
    ) -> list[str]:
        by_id = dict(zip(original_ids, texts, strict=False))
        return [by_id[memory_id] for memory_id in selected_ids if memory_id in by_id]

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
