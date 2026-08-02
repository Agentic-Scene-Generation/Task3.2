# HSSD Open-Mesh Physics Annotation Response

Date: 2026-08-02

Base: `dev_hrk_week31@687800bb909a50afcdf95b89f629fc79bd76b4d8`

Target branch: `dev_yz_from_hrk_week31`

## Decision for the Falling Plant

Asset `cb9b5a9ee8e0eb6cacd1eb98cfe65cced77ad54f` is not rejected.

- Category: `potted plant`
- Runtime-resolved mesh: `watertight=false`
- Open by design: `true`
- Reason: thin leaf surfaces
- Recommended mass-property policy: `bbox_inertia`
- Collision policy: retain convex decomposition
- Physics usable: `true`
- Stability: verified in the Week 31 scene replay with two repeated instances

The annotation deliberately does not claim boundary-edge counts, contact area,
translation, or tilt values that were not exported. Those values remain null
with `pending_resolved_glb_scan` or replay-specific validation status.

## Why This Is Not an Asset-ID Runtime Hack

The runtime already detects invalid/open visual meshes and safely falls back to
bounded box inertia. The annotation records the reason and recommended policy
before SDF generation. It does not add a plant-only branch to the physics code.

The same schema covers five additional Week 31 open-mesh observations:

- monitor `d160e323f1d5a92394de74865aef8938f986d344`
- pendant lamp `19c7d98ce8c620cd916b4fe978c565b065edcea1`
- spoon `3b63f000011404011c4aeb3f19bb04a112535634`
- glass `xxxx82261fccxf2d8x4aacxbf4axb3a0a222dbad`
- plate `c717fbe8f10ad354c83f9cfc7ce154ae2fcdf96c`

Pendant lamps use `weld_or_static`; the remaining free rigid bodies use
`bbox_inertia` until full topology and static-stability scans are available.

## Delivered Files

- `HSSD_OPEN_MESH_PHYSICS_SCHEMA.md`
- `asset_annotation_data/HSSD_OPEN_MESH_PHYSICS_AUDIT.json`
- `scripts/enrich_hssd_open_mesh_physics.py`
- enriched `asset_annotation_data/hssd_annotation_lookup.json.gz`
- annotation-store policy access and functional-hint passthrough
- `tests/unit/test_hssd_open_mesh_annotations.py`

## Remaining Measurement Work

The current machine does not expose the resolved GLB root used by the Week 31
runtime. The enrichment script is ready to fill component, boundary-edge,
non-manifold-edge, degenerate-face, COM, inertia, and SHA-256 fields once that
root is mounted. Runtime `trimesh.is_watertight` evidence is retained as a real
measurement, while unavailable detailed measurements are explicitly null.
