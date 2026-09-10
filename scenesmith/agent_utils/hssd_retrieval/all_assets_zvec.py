"""All-source Zvec retrieval that exports meshes through the HSSD server API."""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from scenesmith.agent_utils.hssd_retrieval.config import HssdZvecConfig
from scenesmith.agent_utils.hssd_retrieval.zvec_similarity import (
    LlamaTextEmbeddingClient,
)
from scenesmith.agent_utils.mesh_frame import gltf_y_up_dimensions_to_scene_z_up

console_logger = logging.getLogger(__name__)


@dataclass
class AllAssetsRetrievalCandidate:
    """One loadable candidate from the namespaced all-source collection."""

    mesh_id: str
    mesh: trimesh.Trimesh
    clip_score: float
    bbox_score: float


class AllAssetsZvecRetriever:
    """Retrieve source-independent mesh paths stored in the all-assets index.

    The server API deliberately keeps the historical ``hssd_id`` response field
    for backwards compatibility.  In this mode it carries the namespaced
    ``asset_id`` (for example ``3dfuture:<uuid>``), not an HSSD SHA-1.
    """

    def __init__(self, config: HssdZvecConfig, top_k: int) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.config = config
        self.top_k = top_k
        self._client = LlamaTextEmbeddingClient(config)
        self._collection: Any | None = None
        self._unit_scales = self._load_unit_scales(config.all_assets_manifest_path)

    @staticmethod
    def _load_unit_scales(manifest_path: Path | None) -> dict[str, float]:
        """Load audited source-unit scales keyed by namespaced asset ID.

        The all-assets index deliberately stores only retrieval fields. Its
        paired shared manifest is the source of truth for mesh unit conversion;
        source-wide defaults are invalid because individual assets may differ.
        """
        if manifest_path is None:
            return {}

        scales: dict[str, float] = {}
        with manifest_path.open(encoding="utf-8") as manifest_file:
            for line_number, line in enumerate(manifest_file, start=1):
                try:
                    record = json.loads(line)
                    mesh_id = str(record["asset_uid"])
                    raw_scale = (record.get("geometry_ref") or {}).get(
                        "unit_scale_to_m", 1.0
                    )
                    scale = float(raw_scale if raw_scale is not None else 1.0)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        "Invalid all-assets manifest record at "
                        f"{manifest_path}:{line_number}"
                    ) from exc
                if not np.isfinite(scale) or scale <= 0.0:
                    raise ValueError(
                        "All-assets unit_scale_to_m must be finite and positive "
                        f"at {manifest_path}:{line_number}"
                    )
                scales[mesh_id] = scale
        return scales

    def _get_collection(self) -> Any:
        import zvec

        if self._collection is None:
            self._collection = zvec.open(
                str(self.config.collection_path),
                option=zvec.CollectionOption(read_only=True, enable_mmap=True),
            )
        return self._collection

    @staticmethod
    def _bbox_score(
        desired_dimensions: np.ndarray | None, mesh: trimesh.Trimesh
    ) -> float:
        if desired_dimensions is None:
            return 0.0
        scene_extents = np.asarray(
            gltf_y_up_dimensions_to_scene_z_up(mesh.extents), dtype=float
        )
        return float(np.sum(np.abs(desired_dimensions - scene_extents)))

    def retrieve_multiple(
        self,
        description: str,
        object_type: str,
        desired_dimensions: np.ndarray | None = None,
        max_candidates: int | None = None,
    ) -> list[AllAssetsRetrievalCandidate]:
        """Return loadable candidates, ranked by requested dimension fit.

        ``object_type`` remains part of the shared retrieval-server contract but
        is intentionally not mapped to HSSD WordNet categories: the all-source
        collection has heterogeneous taxonomies.
        """
        del object_type
        import zvec

        requested = max_candidates or self.top_k
        candidate_pool = max(self.top_k, requested * self.config.top_k_factor)
        query_embedding = self._client.embed_text(description)
        results = self._get_collection().query(
            queries=zvec.Query(
                field_name=self.config.embedding_field, vector=query_embedding
            ),
            topk=candidate_pool,
            output_fields=["asset_id", "asset_path"],
        )

        candidates: list[AllAssetsRetrievalCandidate] = []
        seen_ids: set[str] = set()
        for doc in results:
            fields = getattr(doc, "fields", {}) or {}
            mesh_id = str(fields.get("asset_id") or getattr(doc, "id", ""))
            asset_path = Path(str(fields.get("asset_path") or ""))
            if not mesh_id or mesh_id in seen_ids or not asset_path.is_file():
                continue
            try:
                mesh = trimesh.load(asset_path, force="mesh")
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError(f"expected Trimesh, got {type(mesh)}")
                unit_scale_to_m = self._unit_scales.get(mesh_id, 1.0)
                if unit_scale_to_m != 1.0:
                    mesh.apply_scale(unit_scale_to_m)
            except Exception as exc:
                console_logger.warning(
                    "Skipping unreadable all-assets candidate %s at %s: %s",
                    mesh_id,
                    asset_path,
                    exc,
                )
                continue
            seen_ids.add(mesh_id)
            candidates.append(
                AllAssetsRetrievalCandidate(
                    mesh_id=mesh_id,
                    mesh=mesh,
                    clip_score=float(getattr(doc, "score", 0.0) or 0.0),
                    bbox_score=self._bbox_score(desired_dimensions, mesh),
                )
            )

        candidates.sort(key=lambda candidate: candidate.bbox_score)
        if max_candidates is not None:
            candidates = candidates[:max_candidates]
        if not candidates:
            raise ValueError(f"No loadable all-assets candidates for {description!r}")
        return candidates
