# SceneExpert run metrics

Every top-level `scripts/run_parallel_critic_on.sh` execution now attempts to
write an independent metrics bundle after all batches finish. Collection also
runs when a batch fails and never changes the generation process exit code.

The bundle is stored below the run output root:

```text
<OUTPUT_ROOT>/metrics/
├── run_metrics.json       # run KPIs, quality warnings, and all scene rows
├── run_metrics.md         # short human-readable summary
├── scene_metrics.jsonl    # one lossless row per expected scene
└── scene_metrics.csv      # flat table for paired analysis
```

The collector uses `critic_on/batch_*/batch_cases.csv` as the expected-scene
denominator. Failed and missing scenes therefore remain visible instead of
silently disappearing from quality averages. It reads generation artifacts,
but writes only to `<OUTPUT_ROOT>/metrics`.

## Metric interpretation

- `completion_rate` and `trace_coverage` are run-integrity guardrails.
- `critic_mean_score` and `critic_zero_fail_rate` use the current main
  SceneBenchmark final-scene reports. Runs without those reports are marked as
  incomplete evidence.
- `memory_retrieval_scene_coverage` measures whether compatible cross-task
  memory was returned; same-task records remain excluded by the retriever.
- `memory_injection_delivery_rate` measures retrieved stages whose memory text
  or placement reference was verified at the designer prompt boundary.
- `memory_writer_*` distinguishes model failures, valid no-op writes,
  promotions, store adds/merges, and forbidden fallback writes.
- `scenesmith_repair_events` counts the main pipeline's deterministic repair
  events. `sceneexpert_repair_plans` and `sceneexpert_repairs_executed` count
  only wrapper repair decisions, preventing main hard-gate failures from being
  attributed to SceneExpert.
- `quality_comparison_ready` is true only when every expected scene reached a
  terminal successful state and has both trace and final critic evidence.

An empty hybrid retrieval includes a `zero_result_reason`, such as
`no_active_stage_records`, `all_candidates_same_task`,
`no_structurally_compatible_memory`, or `below_similarity_threshold`. These
reasons diagnose memory coverage without relaxing the compatibility filters.

## Manual collection

The collector can be rerun safely for an existing output root:

```bash
python -m scenesmith.scene_expert.run_metrics \
  --output-root /absolute/path/to/outputs/critic_probe/<RUN_ID> \
  --run-id <RUN_ID>
```

For a cold/warm comparison, keep the case set, shared base, model, component
flags, and runtime settings fixed. Use a new empty memory root for the cold
start, then reuse exactly that root for later warm runs. Join the resulting
`scene_metrics.csv` files on `case_id`; do not compare aggregate scores when
either run has `quality_comparison_ready=false`.
