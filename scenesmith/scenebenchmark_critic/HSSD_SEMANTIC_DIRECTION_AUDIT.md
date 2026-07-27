# HSSD Semantic Direction Audit

Date: 2026-07-27

This release completes the semantic-direction pass over all 10,963 HSSD
assets. Project policy requires canonical front to be horizontal. Vertical
functional directions are therefore stored separately and can never make
`canonical_orientation_is_semantic_front` true.

## Result

- Assets with a horizontal semantic front: **6,902 / 10,963**
- Strict positive horizontal fronts: **6,734**
- Assets with an upward functional direction: **2,939**
- Assets with a downward functional direction: **942**
- Assets using a non-semantic horizontal fallback: **4,061**
- Categories audited: **469 / 469**

Fallback assets still have a horizontal `+Z` canonical orientation axis for
placement descriptions, but it is explicitly non-semantic. Bowls and ceiling
lamps are examples: their up/down function is annotated without pretending
that the vertical axis is a front.

## Coordinate contract

HSSD source assets use asset-local **Y-up** coordinates:

| Semantic face | Asset-local axis | Existing render evidence |
| --- | --- | --- |
| Horizontal front | `+Z` (or an already audited asset-specific horizontal axis) | `back.png` for `+Z` |
| Opposite horizontal face | `-Z` | `front.png` |
| Horizontal side | `+X` / `-X` | `right.png` / `left.png` |
| Upward functional direction (not front) | `+Y` | `top.png` |
| Downward functional direction (not front) | `-Y` | No bottom image is present in the shared six-view bundle |

The `front_view_image_index` field remains the week27 normalized horizontal
index and is `null` for `up` and `down` directions. Downward faces are recorded
as functional evidence rather than represented by an invented render.

## Data fields

`canonical_front.semantic_directions` contains horizontal front entries only.
Vertical entries are stored in the record-level `functional_directions` array.
Both entry types include:

- `kind`: `front`, `up`, or `down`
- `axis`: asset-local unit axis
- `axis_frame`: `asset_local_hssd_y_up`
- `is_primary`
- `is_strict_positive_front`
- `render_evidence_view`

`canonical_orientation_is_semantic_front` is true only for horizontal semantic
fronts. An up/down functional direction never changes that flag and always has
`direction_role == non_front_functional_direction`.

Machine-readable outputs:

- `data/SEMANTIC_DIRECTION_AUDIT.json`
- `data/SEMANTIC_DIRECTION_ASSETS.csv`
- `data/hssd_annotation_lookup.json.gz`

The deterministic builder is
`build/annotate_semantic_directions.py`. It emits an explicit category policy
for all 469 categories so future changes are reviewable rather than hidden in
the generated lookup.
