"""All-source Zvec retrieval that exports meshes through the HSSD server API."""

from __future__ import annotations

import json
import logging
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from scenesmith.agent_utils.hssd_retrieval.alignment import (
    apply_hssd_alignment_transform,
)
from scenesmith.agent_utils.hssd_retrieval.config import HssdZvecConfig
from scenesmith.agent_utils.hssd_retrieval.data_loader import (
    HssdMeshMetadata,
    load_hssd_metadata_by_wordnet,
)
from scenesmith.agent_utils.hssd_retrieval.zvec_similarity import (
    LlamaTextEmbeddingClient,
)
from scenesmith.agent_utils.mesh_frame import gltf_y_up_dimensions_to_scene_z_up

console_logger = logging.getLogger(__name__)

_PLANAR_WALL_DECOR_TOKENS = frozenset(
    {
        "art",
        "artwork",
        "blackboard",
        "canvas",
        "chalkboard",
        "clock",
        "frame",
        "map",
        "mirror",
        "painting",
        "panel",
        "photo",
        "picture",
        "poster",
        "print",
        "screen",
        "sign",
        "television",
        "tv",
    }
)
_MAX_PLANAR_WALL_DEPTH_RATIO = 0.5


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

    def __init__(
        self,
        config: HssdZvecConfig,
        top_k: int,
        *,
        hssd_preprocessed_path: Path | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.config = config
        self.top_k = top_k
        self._client = LlamaTextEmbeddingClient(config)
        self._collection: Any | None = None
        self._unit_scales = self._load_unit_scales(config.all_assets_manifest_path)
        self._hssd_metadata_by_id = self._load_hssd_orientation_metadata(
            hssd_preprocessed_path
        )

    @staticmethod
    def _load_hssd_orientation_metadata(
        preprocessed_path: Path | None,
    ) -> dict[str, HssdMeshMetadata]:
        if preprocessed_path is None:
            return {}
        metadata_by_wordnet = load_hssd_metadata_by_wordnet(
            preprocessed_path / "hssd_wnsynsetkey_index.json"
        )
        return {
            metadata.mesh_id: metadata
            for entries in metadata_by_wordnet.values()
            for metadata in entries
        }

    def _align_source_mesh(
        self, mesh_id: str, mesh: trimesh.Trimesh
    ) -> trimesh.Trimesh:
        """Restore source-specific mesh-frame normalization before export."""
        if not mesh_id.startswith("hssd:"):
            return mesh

        raw_hssd_id = mesh_id.removeprefix("hssd:")
        metadata = self._hssd_metadata_by_id.get(raw_hssd_id)
        if metadata is None:
            raise ValueError(
                "HSSD orientation metadata is required in all-assets mode: "
                f"{mesh_id}"
            )
        return apply_hssd_alignment_transform(mesh, metadata)

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

    @staticmethod
    def _is_candidate_geometry_compatible(
        description: str,
        object_type: str,
        mesh: trimesh.Trimesh,
    ) -> bool:
        """Reject deep meshes retrieved for explicitly planar wall decor.

        Cross-source embeddings can match an object depicted *in* an artwork
        instead of an artwork asset itself (for example, a game controller for
        a controller poster). Keep protruding wall lights and shelves valid by
        applying this guard only when the request names a planar decor type.
        """
        if object_type.upper() != "WALL_MOUNTED":
            return True
        tokens = set(re.findall(r"[a-z0-9]+", description.lower()))
        if not tokens.intersection(_PLANAR_WALL_DECOR_TOKENS):
            return True

        width, depth, height = (
            abs(float(axis))
            for axis in gltf_y_up_dimensions_to_scene_z_up(mesh.extents)
        )
        planar_span = max(width, height)
        return planar_span > 0.0 and depth <= (
            planar_span * _MAX_PLANAR_WALL_DEPTH_RATIO
        )

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
                mesh = self._align_source_mesh(mesh_id, mesh)
                if not self._is_candidate_geometry_compatible(
                    description, object_type, mesh
                ):
                    console_logger.warning(
                        "Skipping non-planar all-assets candidate %s for planar "
                        "wall request %r (scene dimensions=%s)",
                        mesh_id,
                        description,
                        gltf_y_up_dimensions_to_scene_z_up(mesh.extents),
                    )
                    continue
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
