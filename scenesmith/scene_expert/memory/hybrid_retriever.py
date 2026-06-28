"""Hybrid vector retriever for SceneExpert fast memory."""

from __future__ import annotations

import json
import logging
import time

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
        auto_build_indexes: bool = False,
        timing_path: str | Path | None = None,
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
        self._timing_path = Path(timing_path) if timing_path else None
        self._auto_build_indexes = auto_build_indexes

        if require_indexes:
            self._ensure_required_indexes()

    def retrieve(self, task_spec: SceneTaskSpec, stage: str) -> MemoryPack:
        total_start = time.perf_counter()
        query_text = build_query_text(task_spec, stage)
        encode_start = time.perf_counter()
        query_vec = self._embedder.encode([query_text])
        embedding_encode_sec = time.perf_counter() - encode_start
        if query_vec.ndim == 2:
            query_vec = query_vec[0]

        bank_timings: list[dict[str, Any]] = []
        success = self._retrieve_bank(
            "success",
            stage,
            query_vec,
            task_spec,
            final_top_k=self._max_success,
            bank_timings=bank_timings,
        )
        failure = self._retrieve_bank(
            "failure",
            stage,
            query_vec,
            task_spec,
            final_top_k=self._max_failure,
            bank_timings=bank_timings,
        )
        skills = self._retrieve_bank(
            "skill",
            stage,
            query_vec,
            task_spec,
            final_top_k=self._max_skills,
            bank_timings=bank_timings,
        )

        placement_reference = ""
        for _, record in success:
            if isinstance(record, SuccessCase):
                placement_reference = record.to_placement_text()
                if placement_reference:
                    break

        total_sec = time.perf_counter() - total_start
        self._record_timing(
            stage=stage,
            query_text=query_text,
            embedding_encode_sec=embedding_encode_sec,
            bank_timings=bank_timings,
            success_count=len(success),
            failure_count=len(failure),
            skill_count=len(skills),
            total_sec=total_sec,
        )

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
        bank_timings: list[dict[str, Any]],
    ) -> list[tuple[float, MemoryRecord]]:
        bank_timing: dict[str, Any] = {
            "memory_type": memory_type,
            "stage": stage,
            "requested_top_k": final_top_k,
            "index_cache_hit": (memory_type, stage) in self._index_cache,
            "index_found": False,
            "candidate_count": 0,
            "below_threshold_count": 0,
            "stale_count": 0,
            "structured_filtered_count": 0,
            "accepted_count": 0,
            "returned_count": 0,
            "index_load_sec": 0.0,
            "vector_search_sec": 0.0,
            "rerank_sec": 0.0,
        }
        if final_top_k <= 0:
            bank_timings.append(bank_timing)
            return []

        load_start = time.perf_counter()
        index = self._load_index(memory_type, stage)
        bank_timing["index_load_sec"] = time.perf_counter() - load_start
        if index is None:
            bank_timings.append(bank_timing)
            return []
        bank_timing["index_found"] = True

        scored: list[tuple[float, MemoryRecord]] = []
        search_start = time.perf_counter()
        search_results = index.search(query_vec, top_k=self._recall_top_k)
        bank_timing["vector_search_sec"] = time.perf_counter() - search_start
        bank_timing["candidate_count"] = len(search_results)

        rerank_start = time.perf_counter()
        for emb_score, metadata in search_results:
            if emb_score < self._sim_threshold:
                bank_timing["below_threshold_count"] += 1
                continue
            record = self._record_from_metadata(memory_type, metadata)
            if record is None:
                bank_timing["stale_count"] += 1
                continue
            if not self._structured_filter(record, task_spec, stage, memory_type):
                bank_timing["structured_filtered_count"] += 1
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
        output = scored[:final_top_k]
        bank_timing["rerank_sec"] = time.perf_counter() - rerank_start
        bank_timing["accepted_count"] = len(scored)
        bank_timing["returned_count"] = len(output)
        bank_timings.append(bank_timing)
        return output

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

    def _missing_required_indexes(self) -> list[tuple[str, str]]:
        missing_keys: list[tuple[str, str]] = []
        for memory_type, records in (
            ("success", self._store.success_cases),
            ("failure", self._store.failure_cases),
            ("skill", self._store.skills),
        ):
            for stage in sorted({record.stage for record in records}):
                index = NumpyMemoryIndex.for_bank(self._index_dir, memory_type, stage)
                if not index.vectors_path.exists() or not index.metadata_path.exists():
                    missing_keys.append((memory_type, stage))
        return missing_keys

    def _ensure_required_indexes(self) -> None:
        missing = self._missing_required_indexes()
        if missing and self._auto_build_indexes:
            self._auto_build_missing_indexes(missing)
            self._index_cache.clear()
            missing = self._missing_required_indexes()
        if missing:
            missing_text = ", ".join(f"{memory_type}/{stage}" for memory_type, stage in missing)
            raise FileNotFoundError(
                "Hybrid memory index is missing for non-empty memory banks: "
                + missing_text
                + ". Run scripts/build_memory_index.py before setting "
                "SCENEEXPERT_MEMORY_RETRIEVER_TYPE=hybrid, or set "
                "scene_expert.memory.index.auto_build_missing=true."
            )

    def _auto_build_missing_indexes(self, missing: list[tuple[str, str]]) -> None:
        """Build missing numpy indexes in-process before hybrid retrieval starts."""
        stages = tuple(sorted({stage for _, stage in missing}))
        memory_types = tuple(sorted({memory_type for memory_type, _ in missing}))
        console_logger.info(
            "HybridMemoryRetriever: auto-building missing memory indexes for %s "
            "under %s",
            ", ".join(f"{memory_type}/{stage}" for memory_type, stage in missing),
            self._index_dir,
        )
        try:
            from scripts.build_memory_index import build_memory_indexes
        except Exception as e:
            raise RuntimeError(
                "Cannot auto-build SceneExpert hybrid memory indexes because "
                "scripts/build_memory_index.py is not importable."
            ) from e

        model_dir = getattr(self._embedder, "model_dir", None)
        build_memory_indexes(
            memory_dir=self._memory_dir,
            embedding_model_dir=Path(model_dir) if model_dir else None,
            embedding_model_id=str(getattr(self._embedder, "model_id", "BAAI/bge-m3")),
            index_dir=self._index_dir,
            index_backend="numpy",
            device=str(getattr(self._embedder, "device", "cpu")),
            batch_size=int(getattr(self._embedder, "batch_size", 8)),
            max_length=int(getattr(self._embedder, "max_length", 512)),
            stages=stages,
            memory_types=memory_types,
            embedder=self._embedder,
        )

    def _record_timing(
        self,
        *,
        stage: str,
        query_text: str,
        embedding_encode_sec: float,
        bank_timings: list[dict[str, Any]],
        success_count: int,
        failure_count: int,
        skill_count: int,
        total_sec: float,
    ) -> None:
        index_load_sec = sum(float(x.get("index_load_sec", 0.0)) for x in bank_timings)
        vector_search_sec = sum(
            float(x.get("vector_search_sec", 0.0)) for x in bank_timings
        )
        rerank_sec = sum(float(x.get("rerank_sec", 0.0)) for x in bank_timings)
        payload = {
            "schema_version": "1.0",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage,
            "retriever_type": "hybrid",
            "memory_dir": str(self._memory_dir),
            "index_dir": str(self._index_dir),
            "query_text": query_text,
            "embedding_encode_sec": round(embedding_encode_sec, 6),
            "index_load_sec": round(index_load_sec, 6),
            "vector_search_sec": round(vector_search_sec, 6),
            "rerank_sec": round(rerank_sec, 6),
            "total_sec": round(total_sec, 6),
            "returned": {
                "success": success_count,
                "failure": failure_count,
                "skill": skill_count,
            },
            "banks": [
                {
                    **bank,
                    "index_load_sec": round(float(bank["index_load_sec"]), 6),
                    "vector_search_sec": round(float(bank["vector_search_sec"]), 6),
                    "rerank_sec": round(float(bank["rerank_sec"]), 6),
                }
                for bank in bank_timings
            ],
        }
        console_logger.info(
            "[SceneExpertTiming] stage=%s module=hybrid_memory_retrieval "
            "embedding_encode=%.3fs index_load=%.3fs vector_search=%.3fs "
            "rerank=%.3fs total=%.3fs returned=%s/%s/%s",
            stage,
            embedding_encode_sec,
            index_load_sec,
            vector_search_sec,
            rerank_sec,
            total_sec,
            success_count,
            failure_count,
            skill_count,
        )
        if self._timing_path is None:
            return
        self._timing_path.parent.mkdir(parents=True, exist_ok=True)
        with self._timing_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


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
