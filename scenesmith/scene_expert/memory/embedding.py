"""Local embedding wrapper for SceneExpert vector memory.

PR-2 only provides an offline/standalone embedding API. The online hook path
still uses the lexical retriever until the hybrid retriever is introduced.
"""

from __future__ import annotations

import os

from pathlib import Path

import numpy as np

DEFAULT_EMBEDDING_MODEL_ID = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DIRNAME = "bge-m3"


def resolve_memory_embedding_model_dir(model_dir: str | None = None) -> Path:
    """Resolve the local BGE-M3 path used for actual model loading.

    Loading path priority:
      1. explicit ``model_dir`` argument;
      2. ``SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR``;
      3. ``${SCENEEXPERT_MODELS_DIR}/bge-m3``.

    ``model_id`` is intentionally not used as a filesystem path; it is metadata
    only. This keeps the runtime aligned with the cluster layout where BGE-M3 is
    downloaded directly under ``SCENEEXPERT_MODELS_DIR/bge-m3``.
    """
    if model_dir:
        return Path(model_dir).expanduser()

    env_model_dir = os.environ.get("SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR")
    if env_model_dir:
        return Path(env_model_dir).expanduser()

    models_dir = Path(os.environ.get("SCENEEXPERT_MODELS_DIR", "models")).expanduser()
    return models_dir / DEFAULT_EMBEDDING_DIRNAME


def require_local_embedding_model_dir(model_dir: str | None = None) -> Path:
    """Resolve and validate the local embedding model directory."""
    path = resolve_memory_embedding_model_dir(model_dir)
    if not path.exists():
        raise FileNotFoundError(
            "SceneExpert memory embedding model directory does not exist: "
            f"{path}. Set SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR to the local "
            "BGE-M3 directory, usually ${SCENEEXPERT_MODELS_DIR}/bge-m3."
        )
    if not path.is_dir():
        raise NotADirectoryError(
            f"SceneExpert memory embedding model path is not a directory: {path}"
        )
    return path


class SceneMemoryEmbedder:
    """Thin local wrapper around BGE-M3 dense embeddings."""

    def __init__(
        self,
        model_dir: str | None = None,
        model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
        normalize: bool = True,
        use_fp16: bool | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_dir = require_local_embedding_model_dir(model_dir)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        if use_fp16 is None:
            use_fp16 = device.startswith("cuda")
        self.use_fp16 = use_fp16

        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise ImportError(
                "FlagEmbedding is required for SceneExpert memory embeddings. "
                "Install optional memory dependencies with: "
                "python -m pip install -r requirements-memory.txt"
            ) from e

        self._model = BGEM3FlagModel(
            str(self.model_dir),
            use_fp16=self.use_fp16,
            device=self.device,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to a float32 dense matrix."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        outputs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        if isinstance(outputs, dict):
            vecs = outputs.get("dense_vecs")
        else:
            vecs = outputs
        if vecs is None:
            raise ValueError("BGE-M3 encode output did not contain dense_vecs")

        matrix = np.asarray(vecs, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D embedding matrix, got shape {matrix.shape}")

        if self.normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)
        return matrix.astype(np.float32, copy=False)
