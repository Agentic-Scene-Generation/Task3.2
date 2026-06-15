"""Hybrid vector retriever for SceneExpert fast memory."""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Any

import numpy as np

from scenesmith.scene_expert.memory.embedding import SceneMemoryEmbedder
from scenesmith.scene_expert.memory.index import NumpyMemoryIndex
from scenesmith.scene_expert.memory.schemas import FailureCase, Skill, SuccessCase
from scenesmith.scene_expert.memory.scoring import (
    HybridScoreWeights,
    hybrid_score,
    object_overlap,
    record_required_objects,
    record_room_compatible,
    task_required_objects,
)
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.schemas import MemoryPack, SceneTaskSpec

console_logger = logging.getLogger(__name__)
MemoryRecord = SuccessCase | FailureCase | Skill


class HybridMemoryRetriever:
    """Retrieve success/failure/skill memory with vector recall + hybrid rerank."""

    def __init__(
        self,
        store: FastMemoryStore,
        memory_dir: str,
        embedder: SceneMemoryEmbedder,
        index_dir: str | Path | None = None,
        max_success: int = 3,
        max_failure: int = 3,
        max_skills: int = 2,
        recall_top_k: int = 30,
        sim_threshold: float = 0.0,
        object_overlap_threshold: float = 0.15,
        weights: HybridScoreWeights | None = None,
        require_indexes: bool = True,
    ) -> None:
        self._store = store
        self._memory_dir = Path(memory_dir)
        self._index_dir = Path(index_dir) if index_dir else self._memory_dir / "indexes"
        self._embedder = embedder
        self._max_success = max_success
        self._max_failure = max_failure
        self._max_skills = max_skills
        self._recall_top_k = recall_top_k
        self._sim_threshold = sim_threshold
        self._object_overlap_threshold = object_overlap_threshold
        self._weights = weights or HybridScoreWeights()
        self._index_cache: dict[tuple[str, str], NumpyMemoryIndex | None] = {}

        if require_indexes:
            self._validate_required_indexes()

    def retrieve(self, task_spec: SceneTaskSpec, stage: str) -> MemoryPack:
        query_text = build_query_text(task_spec, stage)
        query_vec = self._embedder.encode([query_text])
        if query_vec.ndim == 2:
            query_vec = query_vec[0]

        success = self._retrieve_bank(
            "success",
            stage,
            query_vec,
            task_spec,
            final_top_k=self._max_success,
        )
        failure = self._retrieve_bank(
            "failure",
            stage,
            query_vec,
            task_spec,
            final_top_k=self._max_failure,
        )
        skills = self._retrieve_bank(
            "skill",
            stage,
            query_vec,
            task_spec,
            final_top_k=self._max_skills,
        )

        placement_reference = ""
        for _, record in success:
            if isinstance(record, SuccessCase):
                placement_reference = record.to_placement_text()
                if placement_reference:
                    break

        return MemoryPack(
            success_hints=[
                record.to_positive_guidance() if isinstance(record, SuccessCase) else ""
                for _, record in success
                if isinstance(record, SuccessCase)
            ],
            failure_hints=[
                record.to_negative_constraint() if isinstance(record, FailureCase) else ""
                for _, record in failure
                if isinstance(record, FailureCase)
            ],
            skill_texts=[
                record.to_procedure_text() if isinstance(record, Skill) else ""
                for _, record in skills
                if isinstance(record, Skill)
            ],
            placement_reference=placement_reference,
        )

    def _retrieve_bank(
        self,
        memory_type: str,
        stage: str,
        query_vec: np.ndarray,
        task_spec: SceneTaskSpec,
        final_top_k: int,
    ) -> list[tuple[float, MemoryRecord]]:
        if final_top_k <= 0:
            return []

        index = self._load_index(memory_type, stage)
        if index is None:
            return []

        scored: list[tuple[float, MemoryRecord]] = []
        for emb_score, metadata in index.search(query_vec, top_k=self._recall_top_k):
            if emb_score < self._sim_threshold:
                continue
            record = self._record_from_metadata(memory_type, metadata)
            if record is None:
                continue
            if not self._structured_filter(record, task_spec, stage, memory_type):
                continue
            score = hybrid_score(
                embedding_similarity=emb_score,
                record=record,
                task_spec=task_spec,
                stage=stage,
                memory_type=memory_type,
                weights=self._weights,
            )
            scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:final_top_k]

    def _structured_filter(
        self,
        record: MemoryRecord,
        task_spec: SceneTaskSpec,
        stage: str,
        memory_type: str,
    ) -> bool:
        if record.stage != stage:
            return False

        if memory_type == "failure" and isinstance(record, FailureCase):
            if record.scope in ("global", "stage") or record.is_deterministic:
                return True
            if not record_room_compatible(record, task_spec):
                return False
            record_objects = record_required_objects(record)
            if not record_objects:
                return True
            return (
                object_overlap(record_objects, task_required_objects(task_spec, stage))
                >= self._object_overlap_threshold
            )

        if not record_room_compatible(record, task_spec):
            return False

        record_objects = record_required_objects(record)
        if not record_objects:
            return True
        task_objects = task_required_objects(task_spec, stage)
        if not task_objects:
            return True
        return object_overlap(record_objects, task_objects) >= self._object_overlap_threshold

    def _load_index(self, memory_type: str, stage: str) -> NumpyMemoryIndex | None:
        key = (memory_type, stage)
        if key in self._index_cache:
            return self._index_cache[key]

        index = NumpyMemoryIndex.for_bank(self._index_dir, memory_type, stage)
        if not index.vectors_path.exists() or not index.metadata_path.exists():
            console_logger.warning(
                "HybridMemoryRetriever: missing index for %s/%s under %s",
                memory_type,
                stage,
                self._index_dir,
            )
            self._index_cache[key] = None
            return None

        index.load()
        self._index_cache[key] = index
        return index

    def _record_from_metadata(
        self,
        memory_type: str,
        metadata: dict[str, Any],
    ) -> MemoryRecord | None:
        records: list[MemoryRecord]
        if memory_type == "success":
            records = self._store.success_cases
        elif memory_type == "failure":
            records = self._store.failure_cases
        elif memory_type == "skill":
            records = self._store.skills
        else:
            return None

        memory_id = str(metadata.get("memory_id", ""))
        record_index = metadata.get("record_index")
        if isinstance(record_index, int) and 0 <= record_index < len(records):
            record = records[record_index]
            if _record_id(record) == memory_id:
                return record

        for record in records:
            if _record_id(record) == memory_id:
                return record
        console_logger.warning(
            "HybridMemoryRetriever: stale index entry not found: %s/%s",
            memory_type,
            memory_id,
        )
        return None

    def _validate_required_indexes(self) -> None:
        missing: list[str] = []
        for memory_type, records in (
            ("success", self._store.success_cases),
            ("failure", self._store.failure_cases),
            ("skill", self._store.skills),
        ):
            for stage in sorted({record.stage for record in records}):
                index = NumpyMemoryIndex.for_bank(self._index_dir, memory_type, stage)
                if not index.vectors_path.exists() or not index.metadata_path.exists():
                    missing.append(f"{memory_type}/{stage}")
        if missing:
            raise FileNotFoundError(
                "Hybrid memory index is missing for non-empty memory banks: "
                + ", ".join(missing)
                + ". Run scripts/build_memory_index.py before setting "
                "SCENEEXPERT_MEMORY_RETRIEVER_TYPE=hybrid."
            )


def _record_id(record: MemoryRecord) -> str:
    if isinstance(record, SuccessCase):
        return record.case_id
    if isinstance(record, FailureCase):
        return record.failure_id
    return record.skill_name


def build_query_text(task_spec: SceneTaskSpec, stage: str) -> str:
    """Build the structured query text embedded for hybrid memory retrieval."""
    lines = [
        f"stage={stage}",
        f"room_type={task_spec.room_type}",
        f"style={task_spec.style}",
    ]
    required = task_required_objects(task_spec, stage)
    if required:
        lines.append("required_objects=" + ", ".join(required))
    if task_spec.functional_zones:
        lines.append("functional_zones=" + ", ".join(task_spec.functional_zones))
    if task_spec.interaction_constraints:
        lines.append(
            "interaction_constraints="
            + "; ".join(task_spec.interaction_constraints)
        )
    if task_spec.aesthetic_constraints:
        lines.append(
            "aesthetic_constraints=" + "; ".join(task_spec.aesthetic_constraints)
        )
    return "\n".join(lines)
