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

- Method: `category-dimension-render-v1-20260719`
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
