# HSSD Static Stability Validator Receipt

Date: 2026-08-02  
Branch: `dev_yz_from_hrk_week31`

## Purpose

`scripts/validate_hssd_static_stability.py` screens HSSD assets for free-body
toppling risk. It is designed to catch the failure mode reported for open-mesh
plants without trusting invalid mesh volume, center of mass, or inertia.

This is a candidate detector, not a substitute for replaying the final asset in
its generated scene. Wall- and ceiling-mounted assets are evaluated through
their `weld_or_static` policy rather than as unsupported free bodies.

## Method

1. Resolve the official HSSD GLB by unique asset ID.
2. Convert HSSD Y-up geometry to the SceneSmith/MuJoCo Z-up convention.
3. Extract vertices in the lowest 3% of mesh height (at least 5 mm), project
   them to the floor, and form a support-footprint proxy.
4. Assume the conservative reviewed fallback center of mass at the visual bbox
   center and use bbox inertia, never open-mesh mass properties.
5. Compute COM projection margins and estimated critical tilt angles.
6. Simulate orthogonal `0, 0.5, 1, 2, 5` degree perturbations in MuJoCo.
7. Mark a trial as toppled at more than 15 degrees tilt, excessive translation,
   or invalid floor penetration.

Risk classes are `static_unstable`, `fragile` (failure at 2 degrees or below),
`vulnerable` (failure above 2 degrees through the tested maximum), and
`robust_at_tested_perturbations`. Attached assets retain the separate
`policy_stable_when_attached` status.

## Reproduction

```bash
python3 scripts/validate_hssd_static_stability.py \
  --lookup scenesmith/scenebenchmark_critic/asset_annotation_data/hssd_annotation_lookup.json.gz \
  --scan scenesmith/scenebenchmark_critic/asset_annotation_data/HSSD_MESH_PHYSICS_FULL_AUDIT.json.gz \
  --object-root /data/task3_2/share_data/hsm/hssd-models/objects \
  --output scenesmith/scenebenchmark_critic/asset_annotation_data/HSSD_STATIC_STABILITY_VALIDATION.json \
  --perturb-deg 0 0.5 1 2 5 \
  --workers 8
```

The committed `HSSD_STATIC_STABILITY_VALIDATION_FULL.json.gz` run contains all
10,963 records: 8,574 simulated free bodies, 2,383 attached/static policy
records, and 6 support-extraction unknowns. It reports 1,008 toppling
candidates: 631 static-unstable, 166 fragile, and 211 vulnerable. The
`HSSD_STATIC_STABILITY_VALIDATION_REPRESENTATIVE_17.json` file remains as a
small, human-readable evidence set with 17 real assets: 13 simulated free
bodies and 4 attached assets. It detects:

- `cb9b5a9...ad54f` (the Week31 potted plant): `fragile`, first failure at 1 degree.
- `3f42e7d0...1ff8` (plant with invalid raw COM/inertia): `static_unstable`, failure at 0 degrees.
- 11 other free-body examples: robust through 5 degrees in this proxy test.

The full-library candidate counts are screening output, not automatic reject
labels. Soft goods, decorative surfaces, and assets whose lowest slice is not
their intended placement base require category-aware review or scene replay.

## Interpretation And Limits

- A positive result means the asset needs reviewed collision/support geometry,
  attachment, or scene-level replay. It does not mean the visual mesh must be
  deleted.
- A negative result only covers the tested perturbations and assumed bbox-center
  COM. Impacts, articulated motion, stacking, and contacts with other objects
  require scene-level validation.
- The lowest-slice footprint is intentionally recorded in every result so
  reviewers can detect a bad support extraction instead of accepting it silently.
- Validator output is stored separately and must not overwrite manually reviewed
  `support_stability` evidence.
