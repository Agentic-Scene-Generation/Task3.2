# HSSD Horizontal Front Schema Correction Response

Date: 2026-07-27

Branch: `dev_yz_from_hrk`

## Requirement

The project requires every asset `front` to lie in the horizontal plane.
Upward and downward directions may be annotated when they carry functional
meaning, but they are not fronts.

## Correction

The HSSD annotation layer was corrected across all 10,963 assets:

- `canonical_front.canonical_orientation_axis` is horizontal for every asset.
- `canonical_front.semantic_directions` contains horizontal `front` entries
  only.
- Up/down annotations are stored in record-level `functional_directions`.
- Up/down entries use
  `direction_role = non_front_functional_direction`.
- Up/down entries always use `is_strict_positive_front = false`.
- Up/down entries never set
  `canonical_orientation_is_semantic_front = true`.
- Assets without a real horizontal semantic front retain a horizontal `+Z`
  canonical fallback and explicitly set
  `canonical_orientation_is_semantic_front = false`.

For example, a ceiling lamp now has:

```text
canonical_front.canonical_orientation_axis = [0, 0, 1]
canonical_front.canonical_orientation_is_semantic_front = false

functional_directions[0].kind = down
functional_directions[0].axis = [0, -1, 0]
functional_directions[0].direction_role = non_front_functional_direction
```

## Coverage

- total assets: 10,963
- categories audited: 469
- horizontal semantic fronts: 6,902
- strict positive horizontal fronts: 6,734
- horizontal non-semantic fallbacks: 4,061
- upward functional directions: 2,939
- downward functional directions: 942
- non-horizontal canonical front axes: 0

## SceneSmith Integration

The `scenebenchmark_critic` adapter and annotation store now pass through:

- `semantic_direction_kind`
- `semantic_directions`
- `functional_directions`
- `canonical_orientation_is_semantic_front`

SceneSmith consumers must use `canonical_front` only for horizontal orientation
and front-facing relations. Vertical function should be read from
`functional_directions` and must not be converted into a front relation.

## Artifacts

- `asset_annotation_data/hssd_annotation_lookup.json.gz`
- `asset_annotation_data/SEMANTIC_DIRECTION_AUDIT.json`
- `asset_annotation_data/SEMANTIC_DIRECTION_ASSETS.csv`
- `annotate_semantic_directions.py`
- `HSSD_SEMANTIC_DIRECTION_AUDIT.md`

## Verification

- lookup records checked: 10,963
- horizontal-axis invariant violations: 0
- bowl: horizontal fallback front plus separate `up` function
- ceiling lamp: horizontal fallback front plus separate `down` function
- wardrobe: horizontal semantic front preserved
- adapter modules compile successfully

The corrected annotation release was first published on this branch as commit
`92acd5c`.
