"""Zvec-backed semantic similarity search for HSSD meshes."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

from typing import Any

from scenesmith.agent_utils.hssd_retrieval.config import HssdZvecConfig
from scenesmith.agent_utils.hssd_retrieval.data_loader import HssdPreprocessedData

console_logger = logging.getLogger(__name__)


def _extract_embedding(response: Any) -> list[float]:
    """Handle llama.cpp embedding response variants."""
    if isinstance(response, list) and response and isinstance(response[0], dict):
        embedding = response[0].get("embedding") or response[0].get("embeddings")
    elif isinstance(response, list):
        embedding = response
    elif isinstance(response, dict):
        embedding = (
            response.get("embedding")
            or response.get("embeddings")
            or (response.get("data") or [{}])[0].get("embedding")
        )
    else:
        embedding = None

    while (
        isinstance(embedding, list)
        and len(embedding) == 1
        and isinstance(embedding[0], list)
    ):
        embedding = embedding[0]

    if not isinstance(embedding, list):
        return []
    return [float(value) for value in embedding]


class LlamaTextEmbeddingClient:
    """Small client for llama.cpp's native /embeddings endpoint."""

    def __init__(self, config: HssdZvecConfig) -> None:
        self.base_url = config.base_url.rstrip("/")
        self.timeout_seconds = config.timeout_seconds
        self.embd_normalize = config.embd_normalize
        self.request_retries = config.request_retries
        self.retry_sleep_seconds = config.retry_sleep_seconds

    def embed_text(self, text: str) -> list[float]:
        """Embed text with llama.cpp, tolerating a few payload variants."""
        payloads = [
            {"content": text, "embd_normalize": self.embd_normalize},
            {
                "content": {"prompt_string": text, "multimodal_data": []},
                "embd_normalize": self.embd_normalize,
            },
            {"input": text, "embd_normalize": self.embd_normalize},
        ]

        last_error: Exception | None = None
        for payload in payloads:
            try:
                body = self._post_embeddings(payload)
                embedding = _extract_embedding(json.loads(body))
                if embedding:
                    return embedding
                last_error = RuntimeError("Empty embedding response")
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to embed text query")

    def _post_embeddings(self, payload: dict[str, Any]) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.request_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"Embedding request failed: HTTP {exc.code}: {detail[:500]}"
                )
                if 400 <= exc.code < 500 and exc.code not in {408, 429}:
                    raise last_error from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = RuntimeError(f"Embedding request failed: {exc}")

            if attempt < self.request_retries:
                time.sleep(self.retry_sleep_seconds * (attempt + 1))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Embedding request failed")


class HssdZvecSearcher:
    """Query a local Zvec collection and return HSSD candidate ids."""

    def __init__(self, config: HssdZvecConfig) -> None:
        self.config = config
        self._client = LlamaTextEmbeddingClient(config)
        self._collection: Any | None = None

    def _get_collection(self) -> Any:
        # Keep CLIP-only deployments importable without the optional native module.
        import zvec

        if self._collection is None:
            option = zvec.CollectionOption(read_only=True, enable_mmap=True)
            self._collection = zvec.open(
                path=str(self.config.collection_path), option=option
            )
        return self._collection

    def get_top_k_similar_meshes(
        self,
        text_description: str,
        preprocessed_data: HssdPreprocessedData,
        category: str | None,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Retrieve top-k HSSD ids from Zvec with optional category filtering."""
        import zvec

        collection = self._get_collection()
        query_embedding = self._client.embed_text(text_description)

        candidate_pool = max(top_k, top_k * self.config.top_k_factor)
        results = collection.query(
            queries=zvec.Query(
                field_name=self.config.embedding_field,
                vector=query_embedding,
            ),
            topk=candidate_pool,
            output_fields=["asset_id", "wordnet_key", "object_groups"],
        )

        category_wordnets = None
        if category is not None:
            category_wordnets = set(
                preprocessed_data.object_categories.get(category, [])
            )

        filtered: list[tuple[str, float]] = []
        seen_ids: set[str] = set()
        for doc in results:
            fields = getattr(doc, "fields", {}) or {}
            mesh_id = fields.get("asset_id") or getattr(doc, "id", None)
            if not mesh_id or mesh_id in seen_ids:
                continue

            if category_wordnets is not None:
                wordnet_key = fields.get("wordnet_key")
                if wordnet_key not in category_wordnets:
                    continue

            if preprocessed_data.get_metadata(mesh_id) is None:
                continue

            seen_ids.add(mesh_id)
            filtered.append((mesh_id, float(doc.score or 0.0)))
            if len(filtered) >= top_k:
                break

        console_logger.info(
            "Top-%d embedding candidates: %s",
            len(filtered),
            [(mesh_id[:8], f"{score:.3f}") for mesh_id, score in filtered],
        )
        return filtered
