#!/usr/bin/env python3
"""Merge replay-backed open-mesh policy and optional GLB topology scans."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


POLICIES = {
    "mesh_mass_properties",
    "bbox_inertia",
    "simplified_proxy",
    "weld_or_static",
    "reject",
}
COLLISION_POLICIES = {"convex_decomposition", "simplified_proxy", "none"}


def load_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_lookup(path: Path, lookup: dict[str, dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(
            lookup,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_mesh(path: Path) -> dict[str, Any]:
    try:
        import numpy as np
        import trimesh
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("--geometry-root requires numpy and trimesh") from exc

    loaded = trimesh.load(path, force="scene", process=False)
    geometries = (
        list(loaded.geometry.values())
        if isinstance(loaded, trimesh.Scene)
        else [loaded]
    )
    meshes = [mesh for mesh in geometries if isinstance(mesh, trimesh.Trimesh)]
    if not meshes:
        raise ValueError(f"no triangle mesh in {path}")
    mesh = trimesh.util.concatenate(meshes)
    edge_counts = np.bincount(
        mesh.edges_unique_inverse, minlength=len(mesh.edges_unique)
    )
    boundary_edges = int(np.count_nonzero(edge_counts == 1))
    non_manifold_edges = int(np.count_nonzero(edge_counts > 2))
    areas = np.asarray(mesh.area_faces, dtype=float)
    area_tolerance = max(float(mesh.scale) ** 2 * 1e-12, 1e-16)
    degenerate_faces = int(
        np.count_nonzero(~np.isfinite(areas) | (areas <= area_tolerance))
    )
    components = mesh.split(only_watertight=False)

    bounds = np.asarray(mesh.bounds, dtype=float)
    volume = float(mesh.volume)
    center = np.asarray(mesh.center_mass, dtype=float)
    inertia = np.asarray(mesh.moment_inertia, dtype=float)
    tolerance = np.maximum(0.01, (bounds[1] - bounds[0]) * 0.02)
    center_in_bounds = bool(
        center.shape == (3,)
        and np.all(np.isfinite(center))
        and np.all(center >= bounds[0] - tolerance)
        and np.all(center <= bounds[1] + tolerance)
    )
    inertia_valid = False
    if inertia.shape == (3, 3) and np.all(np.isfinite(inertia)):
        eigenvalues = np.linalg.eigvalsh((inertia + inertia.T) / 2.0)
        inertia_valid = bool(np.all(eigenvalues > 0.0))
    mass_properties_trustworthy = bool(
        mesh.is_watertight
        and np.isfinite(volume)
        and volume > 0.0
        and center_in_bounds
        and inertia_valid
    )
    return {
        "watertight": bool(mesh.is_watertight),
        "connected_component_count": len(components),
        "boundary_edge_count": boundary_edges,
        "non_manifold_edge_count": non_manifold_edges,
        "degenerate_face_count": degenerate_faces,
        "topology_detail_status": "resolved_glb_scan_complete",
        "measurement_source": "resolved_glb",
        "measurement_method": "trimesh_topology_and_mass_properties",
        "resolved_glb_sha256": sha256(path),
        "mesh_volume_m3_unscaled": volume if np.isfinite(volume) else None,
        "center_of_mass_in_bounds": center_in_bounds,
        "inertia_positive_definite": inertia_valid,
        "mesh_mass_properties_trustworthy": mass_properties_trustworthy,
    }


def validate_annotation(asset_id: str, annotation: dict[str, Any]) -> None:
    topology = annotation.get("mesh_topology") or {}
    proxy = annotation.get("physics_proxy") or {}
    stability = annotation.get("support_stability") or {}
    if topology.get("watertight") not in {True, False, None}:
        raise ValueError(f"{asset_id}: invalid watertight value")
    if topology.get("is_open_by_design") not in {True, False, None}:
        raise ValueError(f"{asset_id}: invalid is_open_by_design value")
    if proxy.get("policy") not in POLICIES:
        raise ValueError(f"{asset_id}: invalid physics proxy policy")
    if proxy.get("collision_proxy_policy") not in COLLISION_POLICIES:
        raise ValueError(f"{asset_id}: invalid collision proxy policy")
    if (
        proxy.get("policy") == "reject"
        and proxy.get("is_usable_in_physics") is not False
    ):
        raise ValueError(f"{asset_id}: rejected assets cannot be physics-usable")
    if stability.get("stable_with_recommended_proxy") not in {True, False, None}:
        raise ValueError(f"{asset_id}: invalid stability result")


def merge_annotation(record: dict[str, Any], annotation: dict[str, Any]) -> None:
    quality = record.setdefault("asset_quality", {})
    topology = copy.deepcopy(annotation["mesh_topology"])
    proxy = copy.deepcopy(annotation["physics_proxy"])
    stability = copy.deepcopy(annotation["support_stability"])
    quality["mesh_topology"] = topology
    quality["physics_proxy"] = proxy
    quality["support_stability"] = stability
    quality["watertight"] = topology.get("watertight")
    quality["stable_on_support"] = stability.get("stable_with_recommended_proxy")

    warnings = [
        str(tag)
        for tag in quality.get("warning_tags", [])
        if str(tag) != "watertight_not_measured"
    ]
    for tag in proxy.get("warning_tags", []):
        if tag not in warnings:
            warnings.append(tag)
    quality["warning_tags"] = warnings
    evidence = quality.setdefault("evidence", {})
    evidence["watertight_status"] = (
        f"measured_{str(topology.get('watertight')).lower()}_via_"
        f"{topology.get('measurement_source')}"
    )
    evidence["physics_proxy_policy"] = proxy["policy"]
    if annotation.get("render_evidence"):
        evidence["open_mesh_render_review"] = copy.deepcopy(
            annotation["render_evidence"]
        )
    provenance = quality.setdefault("provenance", {})
    provenance["open_mesh_method_version"] = "hssd-open-mesh-v1"
    provenance["quality_does_not_modify_runtime_without_consumer"] = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    lookup = load_json(args.lookup)
    audit = load_json(args.audit)
    assets = audit.get("assets") or {}
    missing = sorted(set(assets) - set(lookup))
    if missing:
        raise KeyError(f"audit assets missing from lookup: {missing}")

    merged_rows = []
    for asset_id, annotation in assets.items():
        annotation = copy.deepcopy(annotation)
        record = lookup[asset_id]
        if annotation.get("category") != record.get("category"):
            raise ValueError(f"{asset_id}: category mismatch")
        geometry_path = None
        if args.geometry_root:
            relative = (record.get("geometry_ref") or {}).get("path")
            if relative:
                geometry_path = args.geometry_root / relative
            if geometry_path and geometry_path.is_file():
                measured = scan_mesh(geometry_path)
                topology_updates = {
                    key: value
                    for key, value in measured.items()
                    if key not in {
                        "center_of_mass_in_bounds",
                        "inertia_positive_definite",
                        "mesh_mass_properties_trustworthy",
                    }
                }
                annotation["mesh_topology"].update(topology_updates)
                for key in (
                    "center_of_mass_in_bounds",
                    "inertia_positive_definite",
                    "mesh_mass_properties_trustworthy",
                ):
                    annotation["physics_proxy"][key] = measured[key]
        validate_annotation(asset_id, annotation)
        merge_annotation(record, annotation)
        merged_rows.append(
            {
                "hssd_id": asset_id,
                "category": record.get("category"),
                "watertight": annotation["mesh_topology"].get("watertight"),
                "topology_detail_status": annotation["mesh_topology"].get(
                    "topology_detail_status"
                ),
                "physics_proxy_policy": annotation["physics_proxy"]["policy"],
                "stable_with_recommended_proxy": annotation["support_stability"].get(
                    "stable_with_recommended_proxy"
                ),
                "geometry_path": str(geometry_path) if geometry_path else None,
            }
        )

    output = args.output or args.lookup
    write_lookup(output, lookup)
    summary = {"merged": len(merged_rows), "output": str(output), "assets": merged_rows}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
