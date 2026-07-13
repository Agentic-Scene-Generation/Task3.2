"""Build SceneExpert memory vector indexes from JSONL memory banks.

This script is for PR-2 offline preparation only. The runtime hook still uses
the lexical retriever until the hybrid retriever is introduced.

Example:
    python scripts/build_memory_index.py \
      --memory-dir "$SCENEEXPERT_MEMORY_DIR/ablation_4" \
      --embedding-model-dir "$SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR" \
      --index-backend numpy \
      --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenesmith.scene_expert.memory.embedding import (
    DEFAULT_EMBEDDING_MODEL_ID,
    SceneMemoryEmbedder,
    resolve_memory_embedding_model_dir,
)
from scenesmith.scene_expert.memory.index import NumpyMemoryIndex
from scenesmith.scene_expert.memory.schemas import FailureCase, Skill, SuccessCase
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.memory.text_builder import build_embedding_text

STAGES = ("floor_plan", "furniture", "wall_mounted", "ceiling_mounted", "manipuland")
MEMORY_TYPES = ("success", "failure", "skill")


def default_memory_dir() -> Path:
    return (
        Path(os.environ.get("SCENEEXPERT_MEMORY_DIR", "outputs/scene_expert_memory"))
        / "ablation_4"
    )


def _source_file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def _source_files(memory_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "success": _source_file_info(memory_dir / "success_cases.jsonl"),
        "failure": _source_file_info(memory_dir / "failure_cases.jsonl"),
        "skill": _source_file_info(memory_dir / "skills.jsonl"),
    }


def _record_id(record: SuccessCase | FailureCase | Skill) -> str:
    if isinstance(record, SuccessCase):
        return record.case_id
    if isinstance(record, FailureCase):
        return record.failure_id
    return record.skill_name


def _record_required_objects(record: SuccessCase | FailureCase | Skill) -> list[str]:
    if isinstance(record, SuccessCase):
        return record.required_objects or record.task_signature
    return record.required_objects


def _record_metadata(
    record: SuccessCase | FailureCase | Skill,
    memory_type: str,
    record_index: int,
) -> dict[str, Any]:
    return {
        "memory_type": memory_type,
        "stage": record.stage,
        "memory_id": _record_id(record),
        "record_index": record_index,
        "room_type": getattr(record, "room_type", ""),
        "style": getattr(record, "style", ""),
        "required_objects": _record_required_objects(record),
        "functional_zones": getattr(record, "functional_zones", []),
        "quality_score": getattr(record, "quality_score", 0.5),
        "confidence": getattr(record, "confidence", 0.5),
        "trace_ref": getattr(record, "trace_ref", ""),
    }


def _records_by_type(
    store: FastMemoryStore,
) -> dict[str, list[SuccessCase | FailureCase | Skill]]:
    return {
        "success": store.success_cases,
        "failure": store.failure_cases,
        "skill": store.skills,
    }


def _encode_texts(
    texts: list[str],
    embedder: SceneMemoryEmbedder | None,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    if embedder is None:
        raise ValueError("embedder is required when building a non-empty index")
    return embedder.encode(texts)


def build_memory_indexes(
    memory_dir: Path,
    embedding_model_dir: Path | None = None,
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
    index_dir: Path | None = None,
    index_backend: str = "numpy",
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 512,
    stages: tuple[str, ...] = STAGES,
    memory_types: tuple[str, ...] = MEMORY_TYPES,
    dry_run: bool = False,
    embedder: SceneMemoryEmbedder | None = None,
) -> list[dict[str, Any]]:
    """Build per-bank/per-stage memory indexes.

    Returns a list of build summaries for logging/tests.
    """
    if index_backend != "numpy":
        raise NotImplementedError(
            f"index_backend={index_backend!r} is not implemented in PR-2. "
            "Use index_backend='numpy'."
        )

    memory_dir = Path(memory_dir)
    index_dir = Path(index_dir) if index_dir is not None else memory_dir / "indexes"
    model_dir = resolve_memory_embedding_model_dir(
        str(embedding_model_dir) if embedding_model_dir else None
    )

    store = FastMemoryStore(str(memory_dir))
    banks = _records_by_type(store)
    source_files = _source_files(memory_dir)
    needs_embedder = any(
        record.stage in stages
        for memory_type in memory_types
        for record in banks[memory_type]
    )
    if needs_embedder and embedder is None:
        embedder = SceneMemoryEmbedder(
            model_dir=str(model_dir),
            model_id=embedding_model_id,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            normalize=True,
        )

    summaries: list[dict[str, Any]] = []
    for memory_type in memory_types:
        records = banks[memory_type]
        for stage in stages:
            stage_records = [
                (idx, record)
                for idx, record in enumerate(records)
                if record.stage == stage
            ]
            texts = [
                record.embedding_text or build_embedding_text(record)
                for _, record in stage_records
            ]
            vectors = _encode_texts(texts, embedder)
            metadata = [
                _record_metadata(record, memory_type, idx)
                for idx, record in stage_records
            ]
            manifest = {
                "built_at": datetime.now(timezone.utc).isoformat(),
                "embedding_model_id": embedding_model_id,
                "embedding_model_dir": str(model_dir),
                "memory_type": memory_type,
                "stage": stage,
                "normalize": True,
                "source_files": source_files,
            }
            index = NumpyMemoryIndex.for_bank(index_dir, memory_type, stage)
            summary = {
                "memory_type": memory_type,
                "stage": stage,
                "count": len(stage_records),
                "vectors_path": str(index.vectors_path),
                "metadata_path": str(index.metadata_path),
                "manifest_path": str(index.manifest_path),
            }
            if not dry_run:
                index.build(vectors=vectors, metadata=metadata, manifest=manifest)
            summaries.append(summary)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-dir",
        default=str(default_memory_dir()),
        help="SceneExpert memory directory, e.g. $SCENEEXPERT_MEMORY_DIR/ablation_4",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Output index directory. Defaults to <memory-dir>/indexes.",
    )
    parser.add_argument(
        "--embedding-model-id",
        default=os.environ.get(
            "SCENEEXPERT_MEMORY_EMBEDDING_MODEL_ID",
            DEFAULT_EMBEDDING_MODEL_ID,
        ),
        help="Semantic model ID stored in manifest. Does not affect loading path.",
    )
    parser.add_argument(
        "--embedding-model-dir",
        default=os.environ.get("SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR"),
        help=(
            "Local BGE-M3 directory. Defaults to "
            "$SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR or "
            "$SCENEEXPERT_MODELS_DIR/bge-m3."
        ),
    )
    parser.add_argument(
        "--index-backend",
        default=os.environ.get("SCENEEXPERT_MEMORY_INDEX_BACKEND", "numpy"),
        choices=("numpy", "faiss"),
        help="Index backend. PR-2 implements numpy only; faiss is reserved.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get(
            "SCENEEXPERT_MEMORY_EMBEDDING_INDEX_DEVICE",
            os.environ.get("SCENEEXPERT_MEMORY_EMBEDDING_DEVICE", "cpu"),
        ),
        help="Embedding device for offline index build. Online default should stay cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("SCENEEXPERT_MEMORY_EMBEDDING_BATCH_SIZE", "8")),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=int(os.environ.get("SCENEEXPERT_MEMORY_EMBEDDING_MAX_LENGTH", "512")),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=list(STAGES),
        choices=STAGES,
        help="Stages to index.",
    )
    parser.add_argument(
        "--memory-types",
        nargs="+",
        default=list(MEMORY_TYPES),
        choices=MEMORY_TYPES,
        help="Memory banks to index.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.index_backend != "numpy":
        raise SystemExit("PR-2 only supports --index-backend numpy.")

    summaries = build_memory_indexes(
        memory_dir=Path(args.memory_dir),
        embedding_model_dir=(
            Path(args.embedding_model_dir) if args.embedding_model_dir else None
        ),
        embedding_model_id=args.embedding_model_id,
        index_dir=Path(args.index_dir) if args.index_dir else None,
        index_backend=args.index_backend,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stages=tuple(args.stages),
        memory_types=tuple(args.memory_types),
        dry_run=args.dry_run,
    )

    action = "Would build" if args.dry_run else "Built"
    print(
        f"{action} {len(summaries)} SceneExpert memory index file groups "
        f"under {args.index_dir or Path(args.memory_dir) / 'indexes'}"
    )
    for item in summaries:
        print(f"  {item['memory_type']}/{item['stage']}: {item['count']} records")


if __name__ == "__main__":
    main()
