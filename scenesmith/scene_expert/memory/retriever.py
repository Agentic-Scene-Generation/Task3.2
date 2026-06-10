"""BM25-style keyword retrieval for the SceneExpert fast memory system.

MVP uses simple keyword overlap scoring — no heavy embedding model required.
Retrieval matches on: room_type, stage, and keyword overlap with task_signature.
"""

from __future__ import annotations

import re
from collections import Counter

from scenesmith.scene_expert.memory.schemas import FailureCase, Skill, SuccessCase
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.schemas import MemoryPack, SceneTaskSpec


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, filter short tokens."""
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 2]


def _keyword_score(query_tokens: set[str], candidate_tokens: list[str]) -> float:
    """Jaccard-like overlap score between query and candidate token sets."""
    candidate_set = set(candidate_tokens)
    if not candidate_set:
        return 0.0
    intersection = query_tokens & candidate_set
    return len(intersection) / (len(query_tokens | candidate_set) + 1e-9)


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


class MemoryRetriever:
    """Retrieves relevant memory entries for a given task spec and stage."""

    def __init__(
        self,
        store: FastMemoryStore,
        max_success: int = 3,
        max_failure: int = 3,
        max_skills: int = 2,
    ) -> None:
        self._store = store
        self._max_success = max_success
        self._max_failure = max_failure
        self._max_skills = max_skills

    def retrieve(self, task_spec: SceneTaskSpec, stage: str) -> MemoryPack:
        """Retrieve and format memory for injection into a StageBrief."""
        query_tokens = _build_query_tokens(task_spec, stage)

        success_hints, placement_reference = self._retrieve_success(
            task_spec, stage, query_tokens
        )
        failure_hints = self._retrieve_failure(task_spec, stage, query_tokens)
        skill_texts = self._retrieve_skills(task_spec, stage, query_tokens)

        return MemoryPack(
            success_hints=success_hints,
            failure_hints=failure_hints,
            skill_texts=skill_texts,
            placement_reference=placement_reference,
        )

    def _retrieve_success(
        self, task_spec: SceneTaskSpec, stage: str, query_tokens: set[str]
    ) -> tuple[list[str], str]:
        """Return (hint_strings, placement_reference_text).

        hint_strings: compressed one-liners for GlobalPlanner context.
        placement_reference_text: full placement block from the top case,
            to be injected directly into the designer prompt.
        """
        scored: list[tuple[float, SuccessCase]] = []
        for case in self._store.success_cases:
            if case.stage != stage:
                continue
            room_bonus = 1.5 if case.room_type == task_spec.room_type else 1.0
            candidate_tokens = _tokenize(
                " ".join([case.room_type, case.style] + case.task_signature)
            )
            score = _keyword_score(query_tokens, candidate_tokens) * room_bonus
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

        return hints, placement_reference

    def _retrieve_failure(
        self, task_spec: SceneTaskSpec, stage: str, query_tokens: set[str]
    ) -> list[str]:
        scored: list[tuple[float, FailureCase]] = []
        for case in self._store.failure_cases:
            if case.stage != stage:
                continue
            room_bonus = 1.5 if case.room_type == task_spec.room_type else 1.0
            candidate_tokens = _tokenize(
                " ".join([case.room_type, case.object, case.failure_type, case.bad_pattern])
            )
            score = _keyword_score(query_tokens, candidate_tokens) * room_bonus
            scored.append((score, case))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [case.to_hint_text() for _, case in scored[: self._max_failure] if _ > 0]

    def _retrieve_skills(
        self, task_spec: SceneTaskSpec, stage: str, query_tokens: set[str]
    ) -> list[str]:
        scored: list[tuple[float, Skill]] = []
        for skill in self._store.skills:
            if skill.stage != stage:
                continue
            room_bonus = 1.5 if task_spec.room_type in skill.room_types else 1.0
            candidate_tokens = _tokenize(
                " ".join(
                    [skill.skill_name, skill.stage]
                    + skill.room_types
                    + skill.preconditions
                    + skill.procedure
                )
            )
            score = _keyword_score(query_tokens, candidate_tokens) * room_bonus
            scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [skill.to_procedure_text() for _, skill in scored[: self._max_skills] if _ > 0]
