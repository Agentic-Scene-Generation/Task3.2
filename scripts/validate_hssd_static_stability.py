#!/usr/bin/env python3
"""Validate HSSD free-body toppling with a reproducible MuJoCo proxy test.

This is a screening validator, not a replacement for scene-level replay. It
extracts a bottom support footprint from the resolved mesh, applies graded tilt
perturbations, and records both analytic and dynamic evidence.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METHOD_VERSION = "hssd-static-stability-support-footprint-v2-20260802"


def load(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _quat(axis: str, degrees: float) -> list[float]:
    angle = math.radians(degrees) / 2.0
    s = math.sin(angle)
    return {"x": [math.cos(angle), s, 0.0, 0.0], "y": [math.cos(angle), 0.0, s, 0.0]}[axis]


def _simulate_box(
    *, extents_hssd: list[float], mass: float, friction: float,
    support: dict[str, Any], seconds: float, timestep: float,
    perturb_deg: float, axis: str,
) -> dict[str, Any]:
    import mujoco

    # HSSD is Y-up; MuJoCo/SceneSmith proxy is Z-up. x stays x, HSSD z is
    # depth/y, and HSSD y is height/z.
    x, y_hssd, z_hssd = [float(v) for v in extents_hssd]
    half = np.array([x / 2.0, z_hssd / 2.0, y_hssd / 2.0], dtype=float)
    if not np.all(np.isfinite(half)) or np.any(half <= 0.0):
        raise ValueError("non-positive or non-finite AABB extent")
    support_half = np.asarray(support["half_extents_z_up_m"], dtype=float)
    support_center = np.asarray(support["center_relative_to_bbox_z_up_m"], dtype=float)
    inertia = np.array([
        mass * ((2 * half[1]) ** 2 + (2 * half[2]) ** 2) / 12.0,
        mass * ((2 * half[0]) ** 2 + (2 * half[2]) ** 2) / 12.0,
        mass * ((2 * half[0]) ** 2 + (2 * half[1]) ** 2) / 12.0,
    ])
    xml = f"""
    <mujoco model="hssd_static_stability">
      <option gravity="0 0 -9.81" timestep="{timestep:.6f}" integrator="implicitfast"/>
      <default><geom condim="3" friction="{max(0.01, friction):.5f} 0.5 0.5" solref="0.003 1"/>
      </default>
      <worldbody>
        <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1"/>
        <body name="asset" pos="0 0 0">
          <freejoint/>
          <inertial pos="0 0 0" mass="{max(1e-5, mass):.8f}"
            diaginertia="{max(1e-9, inertia[0]):.9f} {max(1e-9, inertia[1]):.9f} {max(1e-9, inertia[2]):.9f}"/>
          <geom name="support_proxy" type="box"
            pos="{support_center[0]:.8f} {support_center[1]:.8f} {support_center[2]:.8f}"
            size="{support_half[0]:.8f} {support_half[1]:.8f} {support_half[2]:.8f}" mass="0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    # Place the tilted proxy on the floor by its rotated lowest point. Using
    # half[2] here would start tall/thin objects deeply inside the floor and
    # create a false "topple" through contact stabilization.
    data.qpos[:3] = [0.0, 0.0, 0.0]
    data.qpos[3:7] = _quat(axis, perturb_deg)
    mujoco.mj_forward(model, data)
    initial_rotation = data.xmat[1].reshape(3, 3)
    local_corners = np.array([
        support_center + np.array([sx, sy, sz]) * support_half
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ])
    rotated_corners = (initial_rotation @ local_corners.T).T
    data.qpos[2] = float(-rotated_corners[:, 2].min() + 0.005)
    mujoco.mj_forward(model, data)

    max_tilt = 0.0
    max_horizontal = 0.0
    min_bottom = float("inf")
    steps = int(math.ceil(seconds / timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        z_axis = np.asarray(data.xmat[1].reshape(3, 3)[:, 2], dtype=float)
        tilt = math.degrees(math.acos(float(np.clip(z_axis[2], -1.0, 1.0))))
        max_tilt = max(max_tilt, tilt)
        max_horizontal = max(max_horizontal, float(np.linalg.norm(data.qpos[:2])))
        rotation = data.xmat[1].reshape(3, 3)
        world_corners = (rotation @ local_corners.T).T + np.asarray(data.qpos[:3])
        bottom = float(world_corners[:, 2].min())
        min_bottom = min(min_bottom, bottom)

    final_tilt = math.degrees(
        math.acos(float(np.clip(data.xmat[1].reshape(3, 3)[2, 2], -1.0, 1.0)))
    )
    fell = bool(
        max_tilt > 15.0
        or final_tilt > 15.0
        or max_horizontal > max(0.05, float(np.linalg.norm(half[:2])) * 0.75)
        or min_bottom < -0.025
    )
    return {
        "axis": axis,
        "initial_perturbation_deg": perturb_deg,
        "max_tilt_deg": max_tilt,
        "final_tilt_deg": final_tilt,
        "max_horizontal_translation_m": max_horizontal,
        "minimum_support_bottom_m": min_bottom,
        "fell": fell,
        "steps": steps,
    }


def _support_profile(path: Path) -> dict[str, Any]:
    import trimesh
    from scipy.spatial import ConvexHull

    mesh = trimesh.load(path, force="mesh", process=True)
    vertices = np.asarray(mesh.vertices, dtype=float)
    bounds = np.asarray(mesh.bounds, dtype=float)
    extents = bounds[1] - bounds[0]
    band = max(0.005, float(extents[1]) * 0.03)
    bottom = vertices[vertices[:, 1] <= bounds[0, 1] + band]
    if len(bottom) < 3:
        raise ValueError("fewer than three vertices in support slice")
    projected = bottom[:, [0, 2]]
    planar_min = projected.min(axis=0)
    planar_max = projected.max(axis=0)
    planar_extents = planar_max - planar_min
    if np.any(planar_extents <= 1e-6):
        raise ValueError("support slice has no 2D footprint")
    try:
        hull = ConvexHull(projected)
        area = float(hull.volume)
    except Exception:
        area = float(np.prod(planar_extents))
    center_hssd = bounds.mean(axis=0)
    support_center_xz = (planar_min + planar_max) / 2.0
    support_height = min(band, max(0.002, float(extents[1]) * 0.05))
    return {
        "method": "lowest_3_percent_vertex_slice_convex_hull",
        "source": "resolved_official_glb",
        "slice_vertex_count": int(len(bottom)),
        "slice_height_m": band,
        "footprint_area_m2": area,
        "footprint_extents_xz_m": planar_extents.tolist(),
        "half_extents_z_up_m": [
            max(0.001, float(planar_extents[0]) / 2.0),
            max(0.001, float(planar_extents[1]) / 2.0),
            support_height / 2.0,
        ],
        "center_relative_to_bbox_z_up_m": [
            float(support_center_xz[0] - center_hssd[0]),
            float(support_center_xz[1] - center_hssd[2]),
            float(bounds[0, 1] + support_height / 2.0 - center_hssd[1]),
        ],
    }


def validate_asset(
    asset_id: str, record: dict[str, Any], scan: dict[str, Any],
    *, object_root: Path, seconds: float, timestep: float,
    perturb_degrees: list[float],
) -> dict[str, Any]:
    quality = record.get("asset_quality") or {}
    proxy = quality.get("physics_proxy") or {}
    policy = proxy.get("policy")
    common = {
        "hssd_id": asset_id,
        "category": record.get("category"),
        "policy": policy,
        "collision_source": "mesh_bottom_support_footprint_proxy",
        "method_version": METHOD_VERSION,
    }
    if policy == "weld_or_static":
        return {
            **common,
            "validation_status": "policy_stable_when_attached",
            "stable_with_recommended_proxy": True,
            "free_body_toppling": False,
            "evidence": "Asset is wall/ceiling anchored and is not a free rigid body.",
        }
    if policy == "reject":
        return {
            **common,
            "validation_status": "rejected_before_simulation",
            "stable_with_recommended_proxy": None,
            "free_body_toppling": None,
            "evidence": "Physics policy is reject; no free-body result is claimed.",
        }
    if scan.get("scan_status") != "complete":
        return {
            **common,
            "validation_status": "missing_geometry_measurement",
            "stable_with_recommended_proxy": None,
            "free_body_toppling": None,
            "evidence": "Resolved GLB scan failed; no proxy dimensions are available.",
        }
    extents = np.asarray(scan.get("bounds_max"), dtype=float) - np.asarray(scan.get("bounds_min"), dtype=float)
    physics = record.get("asset_physics") or {}
    mass = float(physics.get("mass_kg") or 1.0)
    friction = float(physics.get("friction_coefficient") or 0.5)
    try:
        support = _support_profile(object_root / asset_id[0].lower() / f"{asset_id}.glb")
    except Exception as exc:
        return {
            **common,
            "validation_status": "support_footprint_extraction_failed",
            "stable_with_recommended_proxy": None,
            "free_body_toppling": None,
            "evidence": f"{type(exc).__name__}: {exc}",
        }
    support_half = np.asarray(support["half_extents_z_up_m"], dtype=float)
    support_center = np.asarray(support["center_relative_to_bbox_z_up_m"], dtype=float)
    margins = support_half[:2] - np.abs(support_center[:2])
    com_height = max(1e-6, -(support_center[2] - support_half[2]))
    critical_angles = [
        0.0 if margin <= 0.0 else math.degrees(math.atan2(float(margin), com_height))
        for margin in margins
    ]
    analytic = {
        "assumed_center_of_mass": "visual_bbox_center",
        "com_height_above_support_m": com_height,
        "com_projection_inside_support": bool(np.all(margins > 0.0)),
        "support_margin_x_m": float(margins[0]),
        "support_margin_y_m": float(margins[1]),
        "estimated_critical_tilt_about_y_deg": critical_angles[0],
        "estimated_critical_tilt_about_x_deg": critical_angles[1],
    }
    trials = []
    try:
        for perturb_deg in perturb_degrees:
            for axis in ("x", "y"):
                trials.append(_simulate_box(
                    extents_hssd=extents.tolist(), mass=mass, friction=friction,
                    support=support, seconds=seconds, timestep=timestep,
                    perturb_deg=perturb_deg, axis=axis,
                ))
    except Exception as exc:
        return {
            **common,
            "validation_status": "proxy_simulation_failed",
            "stable_with_recommended_proxy": None,
            "free_body_toppling": None,
            "support_footprint": support,
            "analytic_stability": analytic,
            "evidence": f"{type(exc).__name__}: {exc}",
        }
    failed = [trial for trial in trials if trial["fell"]]
    first_failure = min(
        (float(trial["initial_perturbation_deg"]) for trial in failed),
        default=None,
    )
    if not analytic["com_projection_inside_support"] or first_failure == 0.0:
        risk_class = "static_unstable"
    elif first_failure is not None and first_failure <= 2.0:
        risk_class = "fragile"
    elif first_failure is not None:
        risk_class = "vulnerable"
    else:
        risk_class = "robust_at_tested_perturbations"
    return {
        **common,
        "validation_status": "support_footprint_proxy_simulation",
        "stable_with_recommended_proxy": not failed,
        "free_body_toppling": bool(failed),
        "mass_kg": mass,
        "friction_coefficient": friction,
        "bbox_extents_hssd_y_up_m": extents.tolist(),
        "support_footprint": support,
        "analytic_stability": analytic,
        "risk_class": risk_class,
        "first_failing_perturbation_deg": first_failure,
        "seconds": seconds,
        "timestep": timestep,
        "trials": trials,
        "evidence": (
            "At least one orthogonal graded-tilt trial exceeded the toppling threshold."
            if failed else
            "All orthogonal graded-tilt trials remained within the toppling thresholds."
        ),
    }


def _validate_job(job: tuple[Any, ...]) -> dict[str, Any]:
    asset_id, record, scan, object_root, seconds, timestep, perturb_degrees = job
    return validate_asset(
        asset_id, record, scan, object_root=Path(object_root), seconds=seconds,
        timestep=timestep, perturb_degrees=perturb_degrees,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--perturb-deg", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 5.0],
        help="One or more initial tilt perturbations in degrees.",
    )
    args = parser.parse_args()
    lookup = load(args.lookup)
    scan_rows = {row["hssd_id"]: row for row in load(args.scan)["assets"]}
    ids = args.ids or list(lookup)
    if args.limit is not None:
        ids = ids[:args.limit]
    perturb_degrees = sorted(set(args.perturb_deg))
    jobs = [
        (
            asset_id, lookup[asset_id], scan_rows[asset_id], str(args.object_root),
            args.seconds, args.timestep, perturb_degrees,
        )
        for asset_id in ids
        if asset_id in lookup and asset_id in scan_rows
    ]
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.workers == 1:
        results = [_validate_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(_validate_job, jobs, chunksize=1))
    topple = sum(bool(row.get("free_body_toppling")) for row in results)
    simulated = sum(
        row.get("validation_status") == "support_footprint_proxy_simulation"
        for row in results
    )
    write(args.output, {
        "schema_version": "hssd_static_stability_validation@1.0",
        "method_version": METHOD_VERSION,
        "collision_source": "mesh_bottom_support_footprint_proxy",
        "perturbation_degrees": perturb_degrees,
        "thresholds": {"max_tilt_deg": 15.0, "max_horizontal_translation_m": 0.05},
        "asset_count": len(results),
        "simulated_free_body_count": simulated,
        "free_body_toppling_count": topple,
        "assets": results,
    })
    print(json.dumps({"asset_count": len(results), "simulated": simulated, "toppling": topple}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
