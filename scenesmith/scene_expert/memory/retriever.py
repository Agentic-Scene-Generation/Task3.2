"""BM25-style keyword retrieval for the SceneExpert fast memory system.

MVP uses simple keyword overlap scoring — no heavy embedding model required.
Retrieval matches on: room_type, stage, and keyword overlap with task_signature.
"""

from __future__ import annotations

import re

from scenesmith.scene_expert.memory.schemas import FailureCase, Skill, SuccessCase
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.schemas import (
    MemoryPack,
    RetrievedMemorySelection,
    SceneTaskSpec,
)

_ALIASES = {
    "卧室": ["bedroom"],
    "客厅": ["living_room", "living", "room"],
    "厨房": ["kitchen"],
    "餐厅": ["dining_room", "dining"],
    "办公室": ["office"],
    "书房": ["study", "office"],
    "床头柜": ["nightstand", "bedside_table"],
    "床": ["bed"],
    "衣柜": ["wardrobe", "closet"],
    "柜子": ["cabinet"],
    "沙发": ["sofa", "couch"],
    "茶几": ["coffee_table"],
    "桌子": ["table", "desk"],
    "书桌": ["desk"],
    "椅子": ["chair"],
    "窗": ["window"],
    "窗户": ["window"],
    "门": ["door"],
    "地毯": ["rug", "carpet"],
    "灯": ["lamp", "light"],
    "吊灯": ["ceiling_light"],
    "画": ["painting", "wall_art"],
    "架子": ["shelf"],
    "书架": ["bookshelf"],
    "night stand": ["nightstand", "bedside_table"],
    "night stands": ["nightstand", "bedside_table"],
    "bedside table": ["nightstand", "bedside_table"],
    "closet": ["wardrobe"],
    "couch": ["sofa"],
}


def _tokenize(text: str) -> list[str]:
    """Tokenize English and Chinese text with light synonym expansion."""
    text = text.lower()
    tokens: list[str] = []

    for phrase, aliases in _ALIASES.items():
        if phrase in text:
            tokens.extend(aliases)

    for token in re.split(r"[^a-z0-9_]+", text):
        if len(token) > 2:
            tokens.append(token)
            for phrase, aliases in _ALIASES.items():
                if phrase.isascii() and phrase.replace(" ", "_") == token:
                    tokens.extend(aliases)

    for segment in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(segment) > 1:
            tokens.append(segment)
        if len(segment) >= 2:
            tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
        if len(segment) >= 3:
            tokens.extend(segment[i : i + 3] for i in range(len(segment) - 2))

    return tokens


def _keyword_score(query_tokens: set[str], candidate_tokens: list[str]) -> float:
    """Jaccard-like overlap score between query and candidate token sets."""
    candidate_set = set(candidate_tokens)
    if not candidate_set:
        return 0.0
    intersection = query_tokens & candidate_set
    return len(intersection) / (len(query_tokens | candidate_set) + 1e-9)


def _room_type_matches(record_room: str, task_room: str) -> bool:
    """Require room-scoped memories to match instead of merely adding a bonus."""
    generic = {"", "*", "all", "any", "generic"}
    record_norm = str(record_room or "").lower().replace("_", " ").strip()
    task_norm = str(task_room or "").lower().replace("_", " ").strip()
    return record_norm in generic or task_norm in generic or record_norm == task_norm


def _build_query_tokens(task_spec: SceneTaskSpec, stage: str) -> set[str]:
    """Build a flat token set from the task spec for retrieval matching."""
    texts = (
        [task_spec.room_type, task_spec.style, stage]
        + task_spec.required_large_objects
        + task_spec.required_wall_objects
        + task_spec.required_ceiling_objects
        + task_spec.required_small_objects
        + task_spec.functional_zones
    )
    tokens: set[str] = set()
    for t in texts:
        tokens.update(_tokenize(t))
    return tokens


def _stage_required_object_tokens(
    task_spec: SceneTaskSpec,
    stage: str,
) -> set[str]:
    by_stage = {
        "floor_plan": task_spec.required_large_objects,
        "furniture": task_spec.required_large_objects,
        "wall_mounted": task_spec.required_wall_objects,
        "ceiling_mounted": task_spec.required_ceiling_objects,
        "manipuland": task_spec.required_small_objects,
    }
    tokens: set[str] = set()
    for value in by_stage.get(stage, []):
        tokens.update(_tokenize(value))
    return tokens


class MemoryRetriever:
    """Retrieves relevant memory entries for a given task spec and stage."""

    def __init__(
        self,
        store: FastMemoryStore,
        max_success: int = 3,
        max_failure: int = 3,
        max_skills: int = 2,
        exclude_source_task_id: str = "",
    ) -> None:
        self._store = store
        self._max_success = max_success
        self._max_failure = max_failure
        self._max_skills = max_skills
        self._exclude_source_task_id = str(exclude_source_task_id or "")

    def _same_task(self, record: SuccessCase | FailureCase | Skill) -> bool:
        if not self._exclude_source_task_id:
            return False
        task_ids = list(record.source_task_ids)
        if record.source_task_id:
            task_ids.append(record.source_task_id)
        known_task_ids = {value for value in task_ids if value}
        return known_task_ids == {self._exclude_source_task_id}

    def retrieve(self, task_spec: SceneTaskSpec, stage: str) -> MemoryPack:
        """Retrieve and format memory for injection into a StageBrief."""
        self._store.refresh_if_changed()
        query_tokens = _build_query_tokens(task_spec, stage)

        success_hints, placement_reference, success_ids = self._retrieve_success(
            task_spec, stage, query_tokens
        )
        failure_hints, failure_ids = self._retrieve_failure(
            task_spec, stage, query_tokens
        )
        skill_texts, skill_names = self._retrieve_skills(task_spec, stage, query_tokens)
        source_task_ids, source_run_ids = self._selected_provenance(
            [*success_ids, *failure_ids, *skill_names]
        )
        selections = self._build_selections(
            success_ids=success_ids,
            failure_ids=failure_ids,
            skill_names=skill_names,
            source_task_ids=source_task_ids,
            source_run_ids=source_run_ids,
        )

        return MemoryPack(
            success_hints=success_hints,
            failure_hints=failure_hints,
            skill_texts=skill_texts,
            placement_reference=placement_reference,
            success_case_ids=success_ids,
            failure_case_ids=failure_ids,
            skill_names=skill_names,
            retrieved_source_task_ids=source_task_ids,
            retrieved_source_run_ids=source_run_ids,
            memory_bank_id=self._store.bank_id,
            memory_bank_revision=self._store.revision,
            selections=selections,
        ).deduplicated()

    def _build_selections(
        self,
        *,
        success_ids: list[str],
        failure_ids: list[str],
        skill_names: list[str],
        source_task_ids: dict[str, list[str]],
        source_run_ids: dict[str, list[str]],
    ) -> list[RetrievedMemorySelection]:
        """Create stable source locators even for the lightweight retriever."""
        rows: list[RetrievedMemorySelection] = []
        specs = (
            ("success", success_ids, "success_cases.jsonl"),
            ("failure", failure_ids, "failure_cases.jsonl"),
            ("skill", skill_names, "skills.jsonl"),
        )
        for memory_type, record_ids, filename in specs:
            for rank, memory_id in enumerate(record_ids, start=1):
                rows.append(
                    RetrievedMemorySelection(
                        memory_id=memory_id,
                        memory_type=memory_type,
                        rank=rank,
                        source_path=str((self._store.memory_dir / filename).resolve()),
                        source_task_ids=source_task_ids.get(memory_id, []),
                        source_run_ids=source_run_ids.get(memory_id, []),
                        bank_id=self._store.bank_id,
                        bank_revision=self._store.revision,
                    )
                )
        return rows

    def _selected_provenance(
        self, selected_ids: list[str]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Return persisted task/run provenance for the selected records."""
        wanted = set(selected_ids)
        task_ids: dict[str, list[str]] = {}
        run_ids: dict[str, list[str]] = {}
        records: list[SuccessCase | FailureCase | Skill] = [
            *self._store.active_success_cases,
            *self._store.active_failure_cases,
            *self._store.active_skills,
        ]
        for record in records:
            record_id = str(
                getattr(record, "case_id", "")
                or getattr(record, "failure_id", "")
                or getattr(record, "skill_name", "")
            )
            if record_id not in wanted:
                continue
            task_ids[record_id] = sorted(
                {
                    *[value for value in record.source_task_ids if value],
                    *([record.source_task_id] if record.source_task_id else []),
                }
            )
            run_ids[record_id] = sorted(
                {
                    *[value for value in record.source_run_ids if value],
                    *([record.source_run_id] if record.source_run_id else []),
                }
            )
        return task_ids, run_ids

    def _retrieve_success(
        self, task_spec: SceneTaskSpec, stage: str, query_tokens: set[str]
    ) -> tuple[list[str], str, list[str]]:
        """Return hints, placement reference, and source case IDs.

        hint_strings: compressed one-liners for GlobalPlanner context.
        placement_reference_text: full placement block from the top case,
            to be injected directly into the designer prompt.
        """
        scored: list[tuple[float, SuccessCase]] = []
        required_tokens = _stage_required_object_tokens(task_spec, stage)
        for case in self._store.active_success_cases:
            if self._same_task(case):
                continue
            if case.stage != stage or not _room_type_matches(
                case.room_type, task_spec.room_type
            ):
                continue
            case_object_tokens = set(
                _tokenize(" ".join(case.required_objects or case.task_signature))
            )
            if case_object_tokens and not (
                required_tokens and case_object_tokens & required_tokens
            ):
                continue
            candidate_tokens = _tokenize(
                " ".join([case.room_type, case.style] + case.task_signature)
            )
            score = _keyword_score(query_tokens, candidate_tokens) * 1.5
            scored.append((score, case))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [(s, c) for s, c in scored[: self._max_success] if s > 0]

        hints = [case.to_hint_text() for _, case in top]

        # Placement reference: take the top case that actually has placement data
        placement_reference = ""
        for _, case in top:
            ref = case.to_placement_text()
            if ref:
                placement_reference = ref
                break

        return hints, placement_reference, [case.case_id for _, case in top]

    def _retrieve_failure(
        self, task_spec: SceneTaskSpec, stage: str, query_tokens: set[str]
    ) -> tuple[list[str], list[str]]:
        scored: list[tuple[float, FailureCase]] = []
        task_object_tokens = _stage_required_object_tokens(task_spec, stage)
        for case in self._store.active_failure_cases:
            if self._same_task(case):
                continue
            if case.stage != stage or not _room_type_matches(
                case.room_type, task_spec.room_type
            ):
                continue
            case_object_tokens = set(_tokenize(case.object))
            if case_object_tokens and not (case_object_tokens & task_object_tokens):
                continue
            candidate_tokens = _tokenize(
                " ".join(
                    [case.room_type, case.object, case.failure_type, case.bad_pattern]
                )
            )
            score = _keyword_score(query_tokens, candidate_tokens) * 1.5
            scored.append((score, case))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [
            (score, case) for score, case in scored[: self._max_failure] if score > 0
        ]
        return (
            [case.to_hint_text() for _, case in top],
            [case.failure_id for _, case in top],
        )

    def _retrieve_skills(
        self, task_spec: SceneTaskSpec, stage: str, query_tokens: set[str]
    ) -> tuple[list[str], list[str]]:
        scored: list[tuple[float, Skill]] = []
        for skill in self._store.active_skills:
            if self._same_task(skill):
                continue
            if skill.stage != stage:
                continue
            if skill.applicability.excluded_room_types and any(
                _room_type_matches(room, task_spec.room_type)
                for room in skill.applicability.excluded_room_types
            ):
                continue
            skill_rooms = [
                *skill.room_types,
                *skill.applicability.room_types,
            ]
            if getattr(skill, "room_type", ""):
                skill_rooms.append(skill.room_type)
            if skill_rooms and not any(
                _room_type_matches(room, task_spec.room_type) for room in skill_rooms
            ):
                continue
            room_bonus = 1.5 if skill_rooms else 1.0
            candidate_tokens = _tokenize(
                " ".join(
                    [skill.skill_name, skill.stage]
                    + skill.room_types
                    + skill.applicability.room_types
                    + skill.applicability.required_object_roles
                    + skill.preconditions
                    + skill.procedure
                )
            )
            score = _keyword_score(query_tokens, candidate_tokens) * room_bonus
            scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [
            (score, skill) for score, skill in scored[: self._max_skills] if score > 0
        ]
        return (
            [skill.to_procedure_text() for _, skill in top],
            [skill.skill_name for _, skill in top],
        )
