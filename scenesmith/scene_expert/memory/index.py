"""Numpy vector index for SceneExpert memory retrieval.

This is the PR-2 storage/search backend. It intentionally avoids FAISS so the
offline memory pipeline can run on locked-down clusters with minimal optional
dependencies.
"""

from __future__ import annotations

import json

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

INDEX_SCHEMA_VERSION = 1


class NumpyMemoryIndex:
    """A small cosine/IP index backed by a normalized numpy matrix."""

    def __init__(
        self,
        vectors_path: Path,
        metadata_path: Path,
        manifest_path: Path | None = None,
    ) -> None:
        self.vectors_path = Path(vectors_path)
        self.metadata_path = Path(metadata_path)
        self.manifest_path = manifest_path or self.vectors_path.with_suffix(
            ".manifest.json"
        )
        self.vectors: np.ndarray | None = None
        self.metadata: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] = {}

    @classmethod
    def for_bank(
        cls,
        index_dir: Path,
        memory_type: str,
        stage: str,
    ) -> "NumpyMemoryIndex":
        stem = f"{memory_type}_{stage}"
        return cls(
            vectors_path=Path(index_dir) / f"{stem}.npy",
            metadata_path=Path(index_dir) / f"{stem}.metadata.jsonl",
            manifest_path=Path(index_dir) / f"{stem}.manifest.json",
        )

    def build(
        self,
        vectors: np.ndarray,
        metadata: list[dict[str, Any]],
        manifest: dict[str, Any] | None = None,
    ) -> None:
        """Write vectors, metadata, and manifest atomically where practical."""
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D vectors, got shape {matrix.shape}")
        if matrix.shape[0] != len(metadata):
            raise ValueError(
                "Vector row count does not match metadata count: "
                f"{matrix.shape[0]} vs {len(metadata)}"
            )

        self.vectors_path.parent.mkdir(parents=True, exist_ok=True)
        payload_manifest = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "index_backend": "numpy",
            "vector_count": int(matrix.shape[0]),
            "vector_dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            **(manifest or {}),
        }

        self._write_npy_atomic(self.vectors_path, matrix)
        self._write_jsonl_atomic(self.metadata_path, metadata)
        self._write_json_atomic(self.manifest_path, payload_manifest)

        self.vectors = matrix
        self.metadata = list(metadata)
        self.manifest = payload_manifest

    def load(self) -> None:
        """Load index files from disk."""
        if not self.vectors_path.exists():
            raise FileNotFoundError(f"Missing numpy index vectors: {self.vectors_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Missing numpy index metadata: {self.metadata_path}"
            )

        self.vectors = np.load(self.vectors_path).astype(np.float32, copy=False)
        self.metadata = []
        with self.metadata_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))
        if self.vectors.shape[0] != len(self.metadata):
            raise ValueError(
                "Loaded vector row count does not match metadata count: "
                f"{self.vectors.shape[0]} vs {len(self.metadata)}"
            )

        if self.manifest_path.exists():
            with self.manifest_path.open(encoding="utf-8") as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}

    def search(
        self, query_vec: np.ndarray, top_k: int
    ) -> list[tuple[float, dict[str, Any]]]:
        """Search by inner product. Use normalized vectors for cosine similarity."""
        if top_k <= 0:
            return []
        if self.vectors is None:
            self.load()
        assert self.vectors is not None

        if self.vectors.size == 0:
            return []

        query = np.asarray(query_vec, dtype=np.float32)
        if query.ndim == 2:
            if query.shape[0] != 1:
                raise ValueError(f"Expected one query vector, got shape {query.shape}")
            query = query[0]
        if query.ndim != 1:
            raise ValueError(f"Expected 1D query vector, got shape {query.shape}")
        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                "Query dimension does not match index dimension: "
                f"{query.shape[0]} vs {self.vectors.shape[1]}"
            )

        scores = self.vectors @ query
        limit = min(top_k, len(scores))
        if limit == 0:
            return []
        top_ids = np.argpartition(-scores, limit - 1)[:limit]
        top_ids = top_ids[np.argsort(-scores[top_ids])]
        return [(float(scores[idx]), self.metadata[int(idx)]) for idx in top_ids]

    @staticmethod
    def _write_npy_atomic(path: Path, matrix: np.ndarray) -> None:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            np.save(tmp, matrix)
        tmp_path.replace(path)

    @staticmethod
    def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
        with NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp_path = Path(tmp.name)
            for row in rows:
                tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp_path.replace(path)

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        with NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp_path = Path(tmp.name)
            json.dump(payload, tmp, ensure_ascii=False, indent=2, sort_keys=True)
            tmp.write("\n")
        tmp_path.replace(path)
