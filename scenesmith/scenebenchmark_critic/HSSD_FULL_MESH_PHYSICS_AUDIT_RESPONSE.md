# HSSD Full Mesh Physics Audit Response

Date: 2026-08-02

Branch: `dev_yz_from_hrk_week31`, based on `dev_hrk_week31@687800b`

## Request

The Week 31 plant failure showed that open HSSD meshes can produce unsafe
volume-derived COM and inertia. This response extends the six replay samples to
the complete 10,963-asset lookup and makes SceneSmith execute the selected
policy during SDF generation.

## Complete Geometry Audit

| Result | Count |
| --- | ---: |
| Lookup assets | 10,963 |
| Successful resolved-GLB scans | 10,962 |
| Explicit scan errors | 1 |
| Non-watertight | 10,256 |
| Watertight | 706 |
| `bbox_inertia` | 7,889 |
| `mesh_mass_properties` | 691 |
| `weld_or_static` | 2,383 |

Each successful scan records GLB SHA-256, vertex/face counts, connected face
components, boundary/non-manifold/degenerate faces, volume validity, bounded
COM, and inertia positive-definiteness.

Geometry cannot determine whether every opening is semantically intentional.
`is_open_by_design` therefore remains `null` unless reviewed evidence exists.

## Edge Cases

- Plant `cb9b5a9...ad54f`: 16,796 boundary edges and 484 face components;
  reviewed thin-leaf semantics and Week 31 stability evidence are preserved.
- Spoon `3b63f0...535634`: official GLB is watertight, while the Week 31 runtime
  mesh was observed non-watertight. Both observations are stored and the safer
  runtime-backed `bbox_inertia` policy wins.
- Wall art `775a3c...172e1f`: one zero-thickness extent prevents a 3D mass scan.
  Its wall anchor selects `weld_or_static`; it is not incorrectly rejected.

## Runtime Execution

`generate_drake_sdf(..., physics_proxy_policy=...)` now supports:

- `bbox_inertia`: skip mesh volume moments and use per-instance visual-bbox COM
  and positive box inertia;
- `mesh_mass_properties`: retain numerical validity checks and fall back if the
  measured runtime mesh is unsafe;
- `weld_or_static`: emit a static SDF model;
- `reject`: stop SDF generation explicitly.

Collision pieces are unchanged. A bbox mass-property policy does not replace
the visual or convex collision geometry.

## Deliverables

- `asset_annotation_data/hssd_annotation_lookup.json.gz`
- `asset_annotation_data/HSSD_MESH_PHYSICS_FULL_SUMMARY.json`
- `asset_annotation_data/HSSD_MESH_PHYSICS_ISSUES.csv`
- `asset_annotation_data/HSSD_MESH_PHYSICS_FULL_AUDIT.json.gz`
- `scripts/audit_hssd_mesh_physics_full.py`
- `tests/unit/test_hssd_open_mesh_annotations.py`
- `tests/unit/test_sdf_generator.py`

The scanner is parallel and checkpoint-resumable. Dynamic support metrics are
not fabricated from topology: assets without replay or static-simulation
evidence remain `pending_static_simulation`.
