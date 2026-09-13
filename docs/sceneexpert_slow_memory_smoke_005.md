# Qwen3.8 Slow Memory smoke 005 review

Reviewed 2026-09-13. Source runtime revision:
`2d2aef809df943b3c676a36c681a7a1140e6d55a` (clean).
Input archive: `tmp/results/recollect_qwen38_critic_smoke_005/output_root`.

## Decision and evidence

The generation and observer-capture smoke passed. This is not a successful DPO
collection or training preflight. Preserve this run; do not rerun scene 40 merely
to recover its export, and do not scale the current observer to 12 scenes expecting
it to generate decision-level preference pairs.

| Check | Observed result |
| --- | --- |
| Completed / missing / failed | 1 / 0 / 0 |
| Degraded scenes | 1, `completed_with_quality_issues` |
| Completed stages | All five, including the shared floor plan |
| Required-object coverage | 1.0 |
| Final hard success | `pass_scene=false`, `deterministic_pass=false` |
| Final native Critic checklist | 53/58 passing; five functional checks failed |
| Observed model IDs | All 26 records use `unsloth/Qwen3.8-27B-GGUF` |
| Capture | 26 complete observations: 12 Designer, 12 Critic, 2 deterministic repair |
| Designer verdicts | 2 accepted, 8 rejected, 2 unlabeled |
| Designer decisions | 6 initial, 6 repair; 12 distinct contexts |
| Multiple candidates in one context | 0 |
| Materialized media | 116 files; both returned copies verified, 232 file/hash checks |
| Persisted Writer updates | 1 |
| Eligible DPO pairs | 0; all train/validation/test splits empty |

The previous picture-frame ownership/stage-readiness fix allows the scene to
complete. No grammar parse failure remains in this run. Final quality issues
include ambiguous bindings for multiple lamps, books, blankets and windows, plus
blocked armchair access. Keep these failures as evidence; do not rewrite hard
passes or expand cross-team Critic/Designer/repair changes to make this smoke green.

One fresh scene cannot demonstrate cross-scene Fast Memory benefit. Writer may
remain enabled between canonical completed scenes. Different injected memory or
repair history means different DPO conditioning, even when the later result is
better. The current native initial and repair decisions cannot be paired just
because they belong to the same stage.

## Root causes fixed in this delivery

1. The DPO loader resolved relative image paths before comparing records with the
   same ID. The identical `hydra` and `latest-run` copies therefore generated 22
   false collisions. Deduplication now compares serialized payloads first. Real
   conflicting IDs quarantine all versions instead of keeping the first one.
2. The observer wrapper wrote its DPO probe beside `runs`, outside the canonical
   ACP archive's `output_root` link. The returned package therefore could not show
   export status. Review artifacts now live under `runs/collection`, including
   the probe, audit, wrapper manifest, entrypoint and final collection exit code.
   The absence of the old probe does not establish that export never ran remotely.
3. The Full launcher was an ignored local dependency. It is now tracked, and it
   preserves the outer entrypoint identity passed by the collection wrapper.

Generation failure still triggers offline audit, preserves diagnostic artifacts,
and retains its failure exit code. Python import preflight happens before GPU
generation. Existing collection/output roots are rejected to avoid mixing batches.

The new audit separates:

- `generation_complete`: the expected scenes completed without missing/failed jobs.
- `observer_collection_ready`: trace/capture/model/media checks passed. Hard scene
  failures remain valid raw observations.
- `dpo_export_ready`: the existing DPO validator accepts a nonempty package.
- `training_preflight_status`: `not_run`; even a valid one-pair export does not
  satisfy the separate training data-size, coverage and base-model requirements.
- `gate_passed`: observer checks plus the explicitly requested minimum pair count.

`--min-pairs 0` permits an honest observer-only pilot. `--min-pairs 1` additionally
requires a real valid pair and exits 2 on this archive. Neither setting creates
candidates or starts training. The generation wrapper exposes the same setting as
`MIN_DPO_PAIRS`.

After deduplication, the export diagnostics are 14 excluded non-Designer task
types, 10 missing exact-context counterparts and 2 non-authoritative observations.
There are no ID collisions. These filters are retained.

## Next ACP: audit existing 005 without GPU generation

Run in the same ACP project mount. This writes derived review artifacts while
preserving the scene output and trajectories.

```bash
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
git pull --ff-only origin dev_lwz_maintain

PYTHON_BIN="$PWD/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=/mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-main/.venv/bin/python
fi

"$PYTHON_BIN" scripts/audit_sceneexpert_collection.py \
  --run-root outputs/slow_memory/recollect_qwen38_critic_smoke_005/runs \
  --expected-model unsloth/Qwen3.8-27B-GGUF \
  --min-pairs 0
```

If auditing a downloaded package rather than the original run, set `--run-root`
to that package's `output_root`. Expected result: generation and observer gates
true, `dpo_export_ready=false`, zero pairs, exit 0. Return the new `collection/`
directory. Running the same audit with `--min-pairs 1` must exit 2; that is the
expected rejection of an empty training package, not a new generation failure.

## Next experiment sequence

1. **Immediate:** retain 005 as a development pilot and run the offline audit.
   Reserve separate evaluation families before building training memory banks.
2. **Next development milestone:** implement the opt-in decision-level candidate
   adapter described in `sceneexpert_qwen_paired_collection.md`. Start with two
   to four furniture initial-decision groups, two Qwen executions per group.
   Capture the same effective request, scene/assets, session, tool schema, images,
   and injected memory; isolate candidate side effects and obtain evidence from
   each candidate's own execution. Keep candidate A as canonical continuation.
   Retain Writer only in the canonical completed-scene lifecycle. The paired
   worker is still unimplemented in this delivery; the observer ACP cannot be
   turned into this worker through an environment variable.
3. **First pair gate:** prove isolation/context equality and export at least one
   genuine non-tied preference. Zero pairs remains a failed pairing milestone;
   diagnose ties or candidate coverage before spending on more full scenes.
4. **Coverage pilot:** expand to at most 12 decision groups, targeting six initial
   and six naturally occurring repair groups across at least three placement
   stages. Record actual coverage instead of manufacturing repair requests.
5. **Scale and train:** scale by useful pairs per GPU-hour after that pilot passes.
   Run the existing training preflight with the matching Qwen3.8 safetensors base;
   keep held-out memory frozen and evaluate the adapter separately from online
   memory gains. The current archive is not a training dataset.

No additional full-scene generation is required to validate this delivery. When
new raw observer runs are needed, use the existing recollection wrapper with a
fresh `RUN_ID`; it now emits the complete audit automatically. Do not describe
such runs as paired collection.

## Validation scope

- Offline archive audit: observer gate passes; zero eligible preference pairs.
- Unit regression and Bash orchestration tests cover copied media, real ID
  conflicts, missing/corrupted media, mixed models, trace copies, empty runs,
  positive/empty pair gates, error propagation, artifact packaging and run reuse.
- Full-launcher tests use stubs; no new GPU or simulator execution is claimed.
- The older combined Slow Memory test module cannot collect on this Windows
  environment because its existing import chain requires `pydrake`.
