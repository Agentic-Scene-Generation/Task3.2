#!/usr/bin/env python3
"""Measure every HSSD GLB and merge a conservative physics policy into lookup."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


METHOD_VERSION = "hssd-mesh-physics-full-v2-runtime-aligned-20260802"
MOUNTED_ANCHORS = {"wall", "ceiling"}


def _load(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_one(task: tuple[str, str, str]) -> dict[str, Any]:
    asset_id, category, raw_path = task
    path = Path(raw_path)
    base = {"hssd_id": asset_id, "category": category, "geometry_path": str(path)}
    try:
        import numpy as np
        import trimesh
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        # Match SceneSmith's runtime loader, which enables trimesh's standard
        # vertex/face cleanup before testing watertightness and mass properties.
        loaded = trimesh.load(path, force="mesh", process=True)
        if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
            raise ValueError("resolved GLB contains no triangle mesh")

        vertices = np.asarray(loaded.vertices, dtype=float)
        faces = np.asarray(loaded.faces, dtype=np.int64)
        bounds = np.asarray(loaded.bounds, dtype=float)
        extents = bounds[1] - bounds[0]
        if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
            raise ValueError("mesh bounds are not finite 3D bounds")
        if np.any(~np.isfinite(extents)) or np.any(extents <= 0.0):
            raise ValueError("mesh extents are not finite and positive")

        edge_counts = np.bincount(
            loaded.edges_unique_inverse, minlength=len(loaded.edges_unique)
        )
        adjacency = np.asarray(loaded.face_adjacency, dtype=np.int64)
        if len(faces) == 1 or not len(adjacency):
            component_count = len(faces)
        else:
            graph = coo_matrix(
                (
                    np.ones(len(adjacency) * 2, dtype=np.uint8),
                    (
                        np.r_[adjacency[:, 0], adjacency[:, 1]],
                        np.r_[adjacency[:, 1], adjacency[:, 0]],
                    ),
                ),
                shape=(len(faces), len(faces)),
            ).tocsr()
            component_count = int(
                connected_components(graph, directed=False, return_labels=False)
            )

        areas = np.asarray(loaded.area_faces, dtype=float)
        area_tolerance = max(float(np.max(extents)) ** 2 * 1e-12, 1e-16)
        degenerate_faces = int(
            np.count_nonzero(~np.isfinite(areas) | (areas <= area_tolerance))
        )
        volume = float(loaded.volume)
        center = np.asarray(loaded.center_mass, dtype=float)
        inertia = np.asarray(loaded.moment_inertia, dtype=float)
        tolerance = np.maximum(0.01, extents * 0.02)
        center_in_bounds = bool(
            center.shape == (3,)
            and np.all(np.isfinite(center))
            and np.all(center >= bounds[0] - tolerance)
            and np.all(center <= bounds[1] + tolerance)
        )
        inertia_positive = False
        if inertia.shape == (3, 3) and np.all(np.isfinite(inertia)):
            eigenvalues = np.linalg.eigvalsh((inertia + inertia.T) / 2.0)
            inertia_positive = bool(np.all(eigenvalues > 0.0))
        watertight = bool(loaded.is_watertight)
        volume_positive = bool(np.isfinite(volume) and volume > 0.0)
        trustworthy = bool(
            watertight and volume_positive and center_in_bounds and inertia_positive
        )
        issue_reasons = []
        if not watertight:
            issue_reasons.append("open_mesh")
        if not volume_positive:
            issue_reasons.append("invalid_volume")
        if not center_in_bounds:
            issue_reasons.append("center_of_mass_out_of_bounds")
        if not inertia_positive:
            issue_reasons.append("inertia_not_positive_definite")
        if degenerate_faces:
            issue_reasons.append("degenerate_faces")

        return {
            **base,
            "scan_status": "complete",
            "file_size_bytes": path.stat().st_size,
            "resolved_glb_sha256": _sha256(path),
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "watertight": watertight,
            "connected_component_count": component_count,
            "boundary_edge_count": int(np.count_nonzero(edge_counts == 1)),
            "non_manifold_edge_count": int(np.count_nonzero(edge_counts > 2)),
            "degenerate_face_count": degenerate_faces,
            "bounds_min": bounds[0].tolist(),
            "bounds_max": bounds[1].tolist(),
            "mesh_volume_m3_unscaled": volume if np.isfinite(volume) else None,
            "center_of_mass_in_bounds": center_in_bounds,
            "inertia_positive_definite": inertia_positive,
            "mesh_mass_properties_trustworthy": trustworthy,
            "issue_reasons": issue_reasons,
        }
    except Exception as exc:  # Keep a complete inventory instead of dropping failures.
        return {**base, "scan_status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _geometry_path(object_root: Path, asset_id: str) -> Path:
    return object_root / asset_id[0].lower() / f"{asset_id}.glb"


def _reviewed_assets(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return (_load(path).get("assets") or {})


def _build_annotation(
    record: dict[str, Any], result: dict[str, Any], reviewed: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    anchor = str(
        (record.get("placement_dof") or {}).get("support_anchor")
        or (record.get("asset_quality") or {}).get("evidence", {}).get("support_anchor")
        or "unresolved"
    )
    complete = result.get("scan_status") == "complete"
    trustworthy = bool(result.get("mesh_mass_properties_trustworthy")) if complete else False
    mounted = anchor in MOUNTED_ANCHORS
    if mounted:
        policy = "weld_or_static"
    elif not complete:
        policy = "reject"
    elif trustworthy:
        policy = "mesh_mass_properties"
    else:
        policy = "bbox_inertia"

    reviewed_topology = (reviewed or {}).get("mesh_topology") or {}
    reviewed_proxy = (reviewed or {}).get("physics_proxy") or {}
    if reviewed_proxy.get("policy"):
        policy = reviewed_proxy["policy"]
        trustworthy = bool(reviewed_proxy.get("mesh_mass_properties_trustworthy"))
    topology = {
        "measured": complete,
        "watertight": result.get("watertight") if complete else None,
        "is_open_by_design": reviewed_topology.get("is_open_by_design"),
        "open_mesh_reason": reviewed_topology.get("open_mesh_reason"),
        "connected_component_count": result.get("connected_component_count"),
        "boundary_edge_count": result.get("boundary_edge_count"),
        "non_manifold_edge_count": result.get("non_manifold_edge_count"),
        "degenerate_face_count": result.get("degenerate_face_count"),
        "self_intersection_status": "unknown",
        "topology_detail_status": (
            "resolved_glb_scan_complete" if complete else "resolved_glb_scan_failed"
        ),
        "measurement_source": "hssd_models_objects_glb",
        "measurement_method": "trimesh_topology_and_mass_properties",
        "measurement_version": METHOD_VERSION,
        "resolved_glb_sha256": result.get("resolved_glb_sha256"),
        "vertex_count": result.get("vertex_count"),
        "face_count": result.get("face_count"),
        "mesh_volume_m3_unscaled": result.get("mesh_volume_m3_unscaled"),
        "scan_error": result.get("error"),
        "runtime_watertight_observation": reviewed_topology.get("watertight"),
        "runtime_measurement_source": reviewed_topology.get("measurement_source"),
    }
    warning_tags = list(result.get("issue_reasons") or [])
    warning_tags.extend(reviewed_proxy.get("warning_tags") or [])
    if policy == "bbox_inertia":
        warning_tags.append("do_not_use_mesh_volume")
    if mounted:
        warning_tags.append("mounted_asset")
    if not complete:
        warning_tags.append("geometry_scan_failed")
    proxy = {
        "policy": policy,
        "mesh_mass_properties_trustworthy": trustworthy,
        "center_of_mass_in_bounds": result.get("center_of_mass_in_bounds"),
        "inertia_positive_definite": result.get("inertia_positive_definite"),
        "bbox_inertia_positive_definite": complete,
        "collision_proxy_policy": "none" if policy == "reject" else "convex_decomposition",
        "is_usable_in_physics": policy != "reject",
        "warning_tags": list(dict.fromkeys(warning_tags)),
        "reason": reviewed_proxy.get("reason") or {
            "mesh_mass_properties": "Resolved GLB passed closed-volume, bounded-COM, and positive-inertia checks.",
            "bbox_inertia": "Resolved GLB mass properties are unsafe; retain collision pieces and use bounded visual-bbox COM/inertia.",
            "weld_or_static": f"Asset support anchor is {anchor}; attach it instead of free-body simulation.",
            "reject": "Resolved GLB could not be measured and must not silently enter physics.",
        }[policy],
    }
    reviewed_stability = (reviewed or {}).get("support_stability")
    if reviewed_stability:
        stability = reviewed_stability
    elif mounted:
        stability = {
            "support_axis": "+Y_hssd",
            "support_footprint_area_m2": None,
            "support_contact_component_count": None,
            "static_simulation_seconds": None,
            "max_translation_m": None,
            "max_tilt_deg": None,
            "stable_with_recommended_proxy": True,
            "validation_status": "policy_stable_when_attached",
            "evidence": f"Attachment policy derived from support_anchor={anchor}; no free-body claim.",
        }
    else:
        stability = {
            "support_axis": "+Y_hssd",
            "support_footprint_area_m2": None,
            "support_contact_component_count": None,
            "static_simulation_seconds": None,
            "max_translation_m": None,
            "max_tilt_deg": None,
            "stable_with_recommended_proxy": None,
            "validation_status": "pending_static_simulation",
            "evidence": "Topology and mass properties measured; dynamic support trajectory not inferred.",
        }
    return topology, proxy, stability


def _merge_record(
    record: dict[str, Any], result: dict[str, Any], reviewed: dict[str, Any] | None
) -> None:
    topology, proxy, stability = _build_annotation(record, result, reviewed)
    quality = record.setdefault("asset_quality", {})
    quality["mesh_topology"] = topology
    quality["physics_proxy"] = proxy
    quality["support_stability"] = stability
    quality["watertight"] = topology["watertight"]
    warnings = [
        str(tag)
        for tag in quality.get("warning_tags", [])
        if str(tag) != "watertight_not_measured"
    ]
    for tag in proxy["warning_tags"]:
        if tag not in warnings:
            warnings.append(tag)
    quality["warning_tags"] = warnings
    evidence = quality.setdefault("evidence", {})
    evidence["watertight_status"] = (
        f"measured_{str(topology['watertight']).lower()}_via_resolved_glb"
        if topology["measured"]
        else "resolved_glb_scan_failed"
    )
    evidence["physics_proxy_policy"] = proxy["policy"]
    if reviewed and reviewed.get("render_evidence"):
        evidence["open_mesh_render_review"] = reviewed["render_evidence"]
    provenance = quality.setdefault("provenance", {})
    provenance["open_mesh_method_version"] = METHOD_VERSION
    provenance["resolved_glb_sha256"] = topology["resolved_glb_sha256"]
    provenance["quality_does_not_modify_runtime_without_consumer"] = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--reviewed-audit", type=Path)
    parser.add_argument("--output-lookup", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--issues-csv", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    args = parser.parse_args()

    lookup: dict[str, dict[str, Any]] = _load(args.lookup)
    reviewed = _reviewed_assets(args.reviewed_audit)
    completed: dict[str, dict[str, Any]] = {}
    if args.checkpoint.exists():
        with args.checkpoint.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                completed[row["hssd_id"]] = row

    tasks = []
    for asset_id, record in lookup.items():
        if asset_id in completed:
            continue
        tasks.append(
            (asset_id, str(record.get("category") or ""), str(_geometry_path(args.object_root, asset_id)))
        )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if tasks:
        with args.checkpoint.open("a", encoding="utf-8") as checkpoint:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for index, result in enumerate(pool.map(_scan_one, tasks, chunksize=1), 1):
                    checkpoint.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                    checkpoint.flush()
                    completed[result["hssd_id"]] = result
                    if index % 100 == 0:
                        print(f"scanned {len(completed)}/{len(lookup)}", flush=True)

    missing = sorted(set(lookup) - set(completed))
    if missing:
        raise RuntimeError(f"checkpoint is incomplete: {len(missing)} assets missing")

    ordered_results = [completed[asset_id] for asset_id in lookup]
    for result in ordered_results:
        record = lookup[result["hssd_id"]]
        _merge_record(record, result, reviewed.get(result["hssd_id"]))

    output_lookup = args.output_lookup or args.lookup
    _write_json(output_lookup, lookup)
    _write_json(args.results, {"method_version": METHOD_VERSION, "assets": ordered_results})

    scan_status = Counter(row["scan_status"] for row in ordered_results)
    watertight = Counter(str(row.get("watertight")) for row in ordered_results if row["scan_status"] == "complete")
    policies = Counter(
        record["asset_quality"]["physics_proxy"]["policy"] for record in lookup.values()
    )
    reasons = Counter(
        reason for row in ordered_results for reason in (row.get("issue_reasons") or [])
    )
    summary = {
        "schema_version": "hssd_mesh_physics_full_audit@1.0",
        "method_version": METHOD_VERSION,
        "asset_count": len(lookup),
        "scan_status_counts": dict(sorted(scan_status.items())),
        "watertight_counts": dict(sorted(watertight.items())),
        "physics_proxy_policy_counts": dict(sorted(policies.items())),
        "issue_reason_counts": dict(sorted(reasons.items())),
        "reviewed_evidence_preserved_count": len(set(reviewed) & set(lookup)),
        "object_root_layout": "objects/{first_character}/{hssd_id}.glb",
    }
    _write_json(args.summary, summary)

    args.issues_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.issues_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "hssd_id", "category", "scan_status", "watertight",
                "physics_proxy_policy", "issue_reasons", "error",
            ],
        )
        writer.writeheader()
        for row in ordered_results:
            policy = lookup[row["hssd_id"]]["asset_quality"]["physics_proxy"]["policy"]
            if policy == "mesh_mass_properties":
                continue
            writer.writerow(
                {
                    "hssd_id": row["hssd_id"],
                    "category": row["category"],
                    "scan_status": row["scan_status"],
                    "watertight": row.get("watertight"),
                    "physics_proxy_policy": policy,
                    "issue_reasons": "|".join(row.get("issue_reasons") or []),
                    "error": row.get("error", ""),
                }
            )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
