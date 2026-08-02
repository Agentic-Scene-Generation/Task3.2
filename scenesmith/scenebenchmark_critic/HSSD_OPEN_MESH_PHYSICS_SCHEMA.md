# HSSD Open-Mesh Physics Annotation Schema

Version: `hssd-open-mesh-v1`

## Purpose

Open meshes are not automatically defective. Plant leaves, vessel openings,
lamp shades, textiles, and thin dishes may be open by design, while missing
faces and export damage are not. These annotations separate topology evidence
from the physics policy SceneSmith should use.

The fields live under `asset_quality`:

```text
asset_quality.mesh_topology
asset_quality.physics_proxy
asset_quality.support_stability
```

## `mesh_topology`

| Field | Type | Meaning |
| --- | --- | --- |
| `measured` | bool | At least one topology property was measured from resolved geometry. |
| `watertight` | bool/null | Direct closed-surface result. |
| `is_open_by_design` | bool/null | Whether an open mesh is semantically expected. |
| `open_mesh_reason` | string/null | Human-readable structural reason. |
| `connected_component_count` | int/null | Resolved GLB scan result. |
| `boundary_edge_count` | int/null | Edges belonging to exactly one face. |
| `non_manifold_edge_count` | int/null | Edges belonging to more than two faces. |
| `degenerate_face_count` | int/null | Near-zero or non-finite triangle areas. |
| `self_intersection_status` | string | `unknown`, `not_detected`, or `detected`. |
| `topology_detail_status` | string | Completeness of the measurement. |
| `measurement_source` | string | Geometry/evidence source. |
| `measurement_method` | string | Tool or runtime predicate. |
| `resolved_glb_sha256` | string/null | Geometry binding when a full scan is available. |

`watertight=false` does not imply `asset_quality.is_acceptable=false`.

## `physics_proxy`

Allowed `policy` values:

| Policy | Runtime behavior |
| --- | --- |
| `mesh_mass_properties` | Mesh volume, COM, and inertia are trusted. |
| `bbox_inertia` | Skip mesh volume moments and use bounded visual-bbox COM/inertia. |
| `simplified_proxy` | Use a reviewed simplified physics/collision representation. |
| `weld_or_static` | Do not free-simulate; attach or keep static. |
| `reject` | Exclude from retrieval and choose a reserve candidate. |

`collision_proxy_policy` is separate because `bbox_inertia` changes mass
properties only. It does not replace the visual mesh or the convex collision
pieces.

## `support_stability`

This block records support axis, footprint/contact evidence, static simulation
duration, translation, tilt, and the final stability decision. Unknown metrics
must be `null`; they must never be fabricated as zero.

`stable_with_recommended_proxy=true` may be based on a scene replay when the
provenance says so, but an exact five-second stability benchmark requires
exported trajectory metrics.

## Consumer Contract

SceneSmith may read the policy through
`AssetLibraryAnnotationStore.get_physics_proxy_policy(hssd_id)`. Runtime code
must retain its numerical fallback when annotations are missing. Annotation
data does not silently alter collision geometry or scene behavior.

The complete-library consumer passes `physics_proxy.policy` to
`generate_drake_sdf(..., physics_proxy_policy=...)`. `bbox_inertia` forces
bounded per-instance COM/inertia, `weld_or_static` emits a static model, and
`reject` stops generation. Runtime numerical checks remain active even when a
record says `mesh_mass_properties`.

Official-GLB and runtime-resolved watertight observations are separate fields.
When they conflict, the more conservative reviewed runtime policy wins.

## Reproduction

Apply replay-backed evidence without a local geometry mount:

```bash
python scripts/enrich_hssd_open_mesh_physics.py \
  --lookup scenesmith/scenebenchmark_critic/asset_annotation_data/hssd_annotation_lookup.json.gz \
  --audit scenesmith/scenebenchmark_critic/asset_annotation_data/HSSD_OPEN_MESH_PHYSICS_AUDIT.json
```

Add detailed topology measurements when the root containing
`materialized_assets/.../model.glb` is mounted:

```bash
python scripts/enrich_hssd_open_mesh_physics.py \
  --lookup scenesmith/scenebenchmark_critic/asset_annotation_data/hssd_annotation_lookup.json.gz \
  --audit scenesmith/scenebenchmark_critic/asset_annotation_data/HSSD_OPEN_MESH_PHYSICS_AUDIT.json \
  --geometry-root /path/to/resolved/hssd/root
```
