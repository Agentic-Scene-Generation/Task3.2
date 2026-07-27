# HSSD Semantic Direction Audit

Date: 2026-07-27

This release completes the semantic-direction pass over all 10,963 HSSD
assets. A semantic direction is a face that has a stable functional, display,
access, receiving, support, output, or light-emission meaning for the asset
category. It is independent of SceneSmith placement yaw.

## Result

- Assets with at least one semantic direction: **10,781 / 10,963**
- Strict positive horizontal fronts: **6,734**
- Primary horizontal semantic fronts: **6,902**
- Primary upward semantic faces: **2,937**
- Primary downward functional faces: **942**
- Explicitly no stable category direction: **182**
- Categories audited: **469 / 469**

The 182 explicit `none` assets are mostly flexible textiles, generic
decorative objects, balls, rocks, nets, and other objects whose front cannot be
defined from the asset library alone without an instance-specific human
decision. They are not silently assigned a fake `+Z` semantic front.

## Coordinate contract

HSSD source assets use asset-local **Y-up** coordinates:

| Semantic face | Asset-local axis | Existing render evidence |
| --- | --- | --- |
| Horizontal front | `+Z` (or an already audited asset-specific horizontal axis) | `back.png` for `+Z` |
| Opposite horizontal face | `-Z` | `front.png` |
| Horizontal side | `+X` / `-X` | `right.png` / `left.png` |
| Upward face | `+Y` | `top.png` |
| Downward functional face | `-Y` | No bottom image is present in the shared six-view bundle |

The `front_view_image_index` field remains the week27 normalized horizontal
index and is `null` for `up` and `down` directions. Downward faces are recorded
as functional evidence rather than represented by an invented render.

## Data fields

Every record now contains `canonical_front.semantic_directions`, an array of
one or more entries with:

- `kind`: `front`, `up`, or `down`
- `axis`: asset-local unit axis
- `axis_frame`: `asset_local_hssd_y_up`
- `is_primary`
- `is_strict_positive_front`
- `render_evidence_view`

The legacy `canonical_orientation_is_semantic_front` field remains the
backward-compatible existence flag for a canonical semantic face. For upward
and downward primary faces it is true, while
`is_strict_positive_front` is false.

Machine-readable outputs:

- `data/SEMANTIC_DIRECTION_AUDIT.json`
- `data/SEMANTIC_DIRECTION_ASSETS.csv`
- `data/hssd_annotation_lookup.json.gz`

The deterministic builder is
`build/annotate_semantic_directions.py`. It emits an explicit category policy
for all 469 categories so future changes are reviewable rather than hidden in
the generated lookup.
