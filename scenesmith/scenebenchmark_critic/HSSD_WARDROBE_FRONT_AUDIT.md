# HSSD Wardrobe Canonical-Front Completion

## Finding

All 17 HSSD assets whose normalized category is `wardrobe` still had a stable
fallback direction but no asset-specific front validation:

```text
canonical_orientation_is_semantic_front = false
canonical_orientation_confidence = 0.2
front_view_image_index = null
validation_status = pending_asset_front_audit
```

This was incomplete because every wardrobe has a functional semantic front:
the door, drawer, handle, mirror, or open-storage access face.

## Review method

For every wardrobe ID, the six-view bundle under
`/data/task3_2/share_data/scenesmith/hssd_rendered_assets/<hssd_id>/` was
reviewed. The selected `back.png` view shows the asset-local `+Z` access face.
Under the week27 HSSD four-horizontal-view convention, this maps to
`front_view_image_index=3`.

The correction does not define a SceneSmith placement direction and does not
write a world-frame front. It validates the existing asset-library local axis.

## Result

- wardrobe assets reviewed: 17
- semantic fronts: 17
- strict-positive fronts: 16
- semantic but non-unique strict-positive front: 1
- selected asset-local axis: `[0, 0, 1]`
- selected week27 view index: `3`
- validation status: `render_multiview_verified`

`2373e13c60de2cadf4f4d9c62760b9a169e25ef9` is a corner/two-access-face
wardrobe. It receives a stable semantic canonical front, but
`is_strict_positive_front=false` records that this front is not unique.

The mirror-like `644aecdd8cb3efa146aa57a77d6fe832d952d56e` has a distinct dark
facade but limited visible hardware, so its confidence remains conservatively
set to `0.78`. Other assets use per-ID confidence from `0.82` to `0.96`.

## Reproduce

```bash
python scripts/enrich_hssd_wardrobe_front.py \
  --source /path/to/hssd-annotations/data/hssd_annotation_lookup.json.gz \
  --target scenesmith/scenebenchmark_critic/asset_annotation_data/hssd_annotation_lookup.json.gz \
  --annotations scenesmith/scenebenchmark_critic/asset_annotation_data/wardrobe_front_annotations.json
```

Only the 17 listed assets' `canonical_front` objects are replaced. Physics,
quality, clearance, relations, affordances, and all unlisted assets are left
unchanged.
