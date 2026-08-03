"""Deterministic structural admission for retrieved HSSD furniture.

Semantic VLM validation answers whether an asset *looks* like the requested
object.  It cannot reliably detect a room or staging fragment hidden behind a
recognizable bed/sofa in every view.  This module therefore inspects source
submeshes before semantic-cache reuse and rejects only high-confidence,
wall-like architectural components on furniture families where such geometry
is never intrinsic.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh


ASSET_STRUCTURE_CONTRACT_VERSION = "2.0"

# Wardrobes/cabinets legitimately contain tall thin panels.  Keep the hard
# structural rule limited to families for which a near-wall-height backdrop is
# unambiguously foreign geometry.  Other families retain semantic/dimension
# admission without this specialized rejection.
_BACKDROP_SENSITIVE_FAMILIES = frozenset({"bed", "sofa"})


@dataclass(frozen=True)
class MeshComponentEvidence:
    """Bounds evidence for one transformed source submesh."""

    name: str
    dimensions: tuple[float, float, float]


@dataclass(frozen=True)
class AssetStructureCheck:
    """Versioned structural evidence bound to one source geometry payload."""

    status: str
    reason: str
    geometry_fingerprint: str
    contract_version: str = ASSET_STRUCTURE_CONTRACT_VERSION
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return self.status == "reject"

    @property
    def cacheable(self) -> bool:
        return self.status == "pass" and bool(self.geometry_fingerprint)

    def to_cache_payload(self) -> dict[str, Any]:
        return asdict(self)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _axis_index(axis: str | None) -> int:
    normalized = str(axis or "+Z").strip().upper().lstrip("+-")
    return {"X": 0, "Y": 1, "Z": 2}.get(normalized, 2)


def evaluate_component_extents(
    *,
    family: str,
    components: Iterable[MeshComponentEvidence],
    up_axis: str | None,
    geometry_fingerprint: str = "",
) -> AssetStructureCheck:
    """Evaluate transformed component extents using a conservative hard rule.

    A rejected component must simultaneously be near wall height, broad, and
    thin along a horizontal axis.  The up-axis requirement distinguishes an
    architectural backdrop from a horizontally thin mattress or seat cushion.
    """

    normalized_family = str(family or "").strip().casefold()
    component_list = list(components)
    base_evidence: dict[str, Any] = {
        "family": normalized_family,
        "up_axis": str(up_axis or "+Z").upper(),
        "component_count": len(component_list),
        "components": [asdict(component) for component in component_list],
    }
    if normalized_family not in _BACKDROP_SENSITIVE_FAMILIES:
        base_evidence["rule_applied"] = False
        return AssetStructureCheck(
            status="pass",
            reason="No backdrop-specific structural rule applies to this family",
            geometry_fingerprint=geometry_fingerprint,
            evidence=base_evidence,
        )

    base_evidence["rule_applied"] = True
    up_index = _axis_index(up_axis)
    horizontal_indices = [index for index in range(3) if index != up_index]
    minimum_vertical = 1.75 if normalized_family == "bed" else 1.55

    ambiguous_panels: list[dict[str, Any]] = []
    for component in component_list:
        dimensions = np.asarray(component.dimensions, dtype=float)
        if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions)):
            continue
        vertical = float(dimensions[up_index])
        horizontal = [float(dimensions[index]) for index in horizontal_indices]
        span = max(horizontal)
        thickness = min(horizontal)
        reference = max(1e-6, min(vertical, span))
        thin_enough = thickness <= 0.22 and thickness / reference <= 0.14
        wall_sized = vertical >= minimum_vertical and span >= 1.20
        if thin_enough and wall_sized:
            base_evidence["rejected_component"] = asdict(component)
            base_evidence["vertical_extent"] = round(vertical, 6)
            base_evidence["horizontal_span"] = round(span, 6)
            base_evidence["horizontal_thickness"] = round(thickness, 6)
            return AssetStructureCheck(
                status="reject",
                reason=(
                    f"{normalized_family} contains a near-wall-height vertical "
                    f"thin component '{component.name}' with source dimensions "
                    f"{np.round(dimensions, 3).tolist()}"
                ),
                geometry_fingerprint=geometry_fingerprint,
                evidence=base_evidence,
            )

        # A broad, shallow panel below wall height is not sufficient evidence
        # for a deterministic reject: it may be a real headboard or sofa
        # backrest.  It is still material structural evidence, however, and
        # must prevent unqualified positive-cache reuse.  The live multi-view
        # validator can close this ambiguity explicitly, after which its result
        # is safe to cache against this exact geometry fingerprint.
        ambiguous_vertical = (normalized_family == "bed" and vertical > 1.20) or (
            normalized_family == "sofa" and vertical >= 0.70
        )
        if (
            len(component_list) > 1
            and ambiguous_vertical
            and span >= 1.20
            and thickness <= 0.40
            and thickness / max(1e-6, span) <= 0.25
        ):
            ambiguous_panels.append(asdict(component))

    if ambiguous_panels:
        base_evidence["ambiguous_panel_components"] = ambiguous_panels
        return AssetStructureCheck(
            status="inconclusive",
            reason=(
                f"{normalized_family} contains broad shallow source submesh(es) "
                "that require live standalone-object validation"
            ),
            geometry_fingerprint=geometry_fingerprint,
            evidence=base_evidence,
        )

    return AssetStructureCheck(
        status="pass",
        reason="No high-confidence architectural backdrop component was detected",
        geometry_fingerprint=geometry_fingerprint,
        evidence=base_evidence,
    )


def _transformed_components(mesh_path: Path) -> list[MeshComponentEvidence]:
    loaded = trimesh.load(str(mesh_path), force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        extents = tuple(float(value) for value in loaded.extents)
        return [MeshComponentEvidence(name=mesh_path.stem, dimensions=extents)]
    if not isinstance(loaded, trimesh.Scene):
        raise ValueError(f"Unsupported mesh payload: {type(loaded).__name__}")

    components: list[MeshComponentEvidence] = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph[node_name]
        geometry = loaded.geometry.get(geometry_name)
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.vertices):
            continue
        transformed = geometry.copy()
        transformed.apply_transform(transform)
        try:
            connected = list(transformed.split(only_watertight=False))
        except Exception:
            # ``trimesh.split`` may require an optional graph backend. The
            # original transformed source node remains deterministic evidence
            # when that optimization is unavailable.
            connected = [transformed]
        # Pathological meshes can contain thousands of tiny islands. Their
        # aggregate source node remains useful evidence without allowing the
        # structural precheck to dominate retrieval latency.
        parts = connected if 1 <= len(connected) <= 128 else [transformed]
        for part_index, part in enumerate(parts):
            extents = tuple(float(value) for value in part.extents)
            components.append(
                MeshComponentEvidence(
                    name=(f"{node_name}:{geometry_name}:component_{part_index:03d}"),
                    dimensions=extents,
                )
            )
    if not components:
        raise ValueError("Mesh scene contains no transformed geometry components")
    return components


def inspect_hssd_candidate_structure(
    *,
    mesh_path: str | Path,
    family: str,
    up_axis: str | None = "+Z",
) -> AssetStructureCheck:
    """Inspect one raw HSSD candidate without turning parser errors into rejects."""

    path = Path(mesh_path)
    try:
        fingerprint = _file_sha256(path)
        if str(family or "").strip().casefold() not in _BACKDROP_SENSITIVE_FAMILIES:
            return evaluate_component_extents(
                family=family,
                components=[],
                up_axis=up_axis,
                geometry_fingerprint=fingerprint,
            )
        components = _transformed_components(path)
    except Exception as exc:
        return AssetStructureCheck(
            status="inconclusive",
            reason=f"Structural mesh inspection unavailable: {type(exc).__name__}: {exc}",
            geometry_fingerprint="",
            evidence={
                "family": str(family or "").casefold(),
                "mesh_path": str(path),
                "up_axis": str(up_axis or "+Z").upper(),
            },
        )
    return evaluate_component_extents(
        family=family,
        components=components,
        up_axis=up_axis,
        geometry_fingerprint=fingerprint,
    )
