# HSSD Asset Physics and Quality Integration

## Branch baseline

`dev_yz_from_hrk` was created directly from
`Agentic-Scene-Generation/Task3.2:dev_hrk` commit
`b26591b06e16b3702fed00831c368868b8b1b385`. It does not merge the older `yz`
branch. This keeps all other planner, critic, and front-axis changes identical
to the colleague branch at handoff.

## Added annotation fields

The bundled lookup contains `asset_physics` and `asset_quality` for all 10,963
HSSD IDs. The enrichment script copies only those two fields into the existing
critic lookup, preserving `scenebenchmark_fd_sa`, functional hints, front,
relations, clearance, DOF, and every other current field.

`MeshPhysicsAnalysis.friction_coefficient` is optional. HSSD annotations
override front, material, mass/range, and friction after per-field validation.
Missing or invalid values retain the previous deterministic/VLM result.
`generate_drake_sdf()` consumes annotated friction when present and otherwise
continues to call `get_friction(material)`.

Quality remains advisory: unacceptable records emit a warning but are not
silently removed from retrieval. This prevents a metadata rollout from
changing scene-generation recall.

## Data provenance and reproduction

- Method: `category-dimension-render-v2-20260720`
- Render evidence: `/data/task3_2/share_data/scenesmith/hssd_rendered_assets/`
- Physics vocabulary: Task3.2 `materials.yaml` at baseline commit above
- Generator and standalone API: `K-Chronofox/hssd-annotations`

To refresh only the new fields:

```bash
python scripts/enrich_hssd_asset_physics_quality.py \
  --source /path/to/standalone/hssd_annotation_lookup.json.gz \
  --target scenesmith/scenebenchmark_critic/asset_annotation_data/hssd_annotation_lookup.json.gz
```

See `asset_annotation_data/ASSET_PHYSICS_QUALITY_SUMMARY.json` and
`asset_annotation_data/asset_physics_profiles.json` for full-run counts and
machine-readable assumptions.

## 2026-07-20 mismatch and mass audit

The v2 refresh replaced raw substring category matching with ordered complete
word/phrase rules and corrected compound categories such as `tablet computer`,
`desk calendar`, `toilet paper holder`, `table runner`, and `pet bed`. Vehicle
references were also corrected to full-scale weights, with a dimension-based
override for one miniature motorcycle asset.

The standalone and bundled SceneSmith annotations were compared across all
10,963 IDs. There are zero ID mismatches, zero physics/quality payload
mismatches, zero formula mismatches, zero invalid mass intervals, and zero
annotated IDs without a non-empty render directory. The full report and the
469-category mass table are bundled as `ASSET_PHYSICS_QUALITY_AUDIT.json` and
`ASSET_PHYSICS_CATEGORY_AUDIT.csv`.

`asset_physics.audit` distinguishes rule-check passes from bounded estimates.
It does not claim direct weighing, material measurement, or pixel-level proof
that a render depicts the intended mesh.
