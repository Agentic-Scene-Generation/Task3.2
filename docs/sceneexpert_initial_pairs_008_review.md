# Qwen3.8 initial pairs 008: stale physics labels and recovery

Current result: 010 successfully rescored both candidates; both passed and no
eligible pair was produced. Continue with [011 fresh collection](sceneexpert_initial_pairs_010_review.md).
The rescore commands in this document are historical.

Update: run 009 hit a native serialization equality failure. Use the
[010 recovery plan](sceneexpert_initial_pairs_009_recovery.md) instead of the
historical 009 command below. Scoring protocol v3 adds explicit verified
restoration to the original v2 raw-evidence contract described here.

Reviewed on 2026-09-15 from
`tmp/results/slow_memory/qwen38_initial_pairs_008_review_20260914_150046/qwen38_initial_pairs_008_review/`.
Paths below are relative to the extracted review root unless stated otherwise.

## Verdict

The execution and isolation pilot succeeded. Preference collection is not ready:
the original export contains zero pairs, and both rejection labels used stale
physics observations. Refresh evidence before interpreting candidate quality or
expanding generation. Do not train on the original 008 labels.

| Check | Observed result |
| --- | --- |
| Package integrity | 498 included files matched recorded sizes and SHA256; 1523 omitted entries; no package warnings |
| Canonical Full scene | 1/1 complete; all five stages present; deterministic/full pass |
| Fast Memory | 4 persisted MemoryWriter records |
| Observer data | 19 Qwen trajectories; 8 Designer contexts, each with one candidate |
| Isolated initial pair | 1 completed group, 2 Qwen candidates; shadow exit 0 |
| Input/isolation | First HTTP request bodies identical; canonical state, assets and memory unchanged by B |
| Original preference export | 0 pairs; `minimum_pair_gate_failed`; no training |

The observer's `gate_passed=true` uses `min_pairs=0` and only certifies ordinary
collection. It does not override the separate initial-pair gate.

Evidence locations:

- `outputs/slow_memory/qwen38_initial_pairs_008/runs/collection/collection_audit.json`
- `outputs/slow_memory/qwen38_initial_pairs_008/runs/paired_initial/pair_audit.json`
- `outputs/slow_memory/qwen38_initial_pairs_008/runs/paired_initial/group_000/`
- `_package/manifest.json`

Full scene assets were intentionally omitted from the review package. The local
review verifies included bytes and serialized evidence, not a new Drake simulation
or a complete rerun of the asset-integrity gate.

## Root cause

Before this fix, SceneExpert's `InitialPair.capture_raw()` called Main's room
evaluator directly. That evaluator converts the scene into a case pack and reads
`scene.metadata.scenebenchmark_physics_evidence`; it does not recompute collisions.

Both Designer trajectories moved furniture after their last successful physics
tool check. A later tool check was blocked by the native per-call budget. Capturing
the final raw state therefore preserved the geometry after those moves together
with physics observations from before them. Scoring was temporally misbound even
though the raw-state hash itself matched.

| Candidate | Original score | Cached collision failures | Collisions after native safety refreshed evidence |
| --- | --- | --- | --- |
| A | 0.8269230769 | 3 | 0 |
| B | 0.9200000000 | 1 | 0 |

For both candidates, `raw_state.json` and `returned_state.json` differ only in
`metadata`: serialized objects and room geometry are identical. Both `safety.json`
files report that deterministic hard checks passed after the initial Designer
call. This corroborates stale observations, rather than repaired furniture being
mistaken for the original raw candidate. It does not replace an independently
recomputed raw-state report: fresh scores/verdicts still require server validation.

The pair's shared context hash is
`5e6fd4940430a693aa1be08a26e410027776efad2e60effaf6f4ec641d31eb46`.
The old `missing_exact_context_counterpart` diagnostic means neither response was
accepted here; B was present and the effective requests matched. New diagnostics
distinguish this from a genuinely missing execution.

## Implemented correction

All runtime changes remain in SceneExpert's opt-in collection/scoring layer.
Native Designer, Critic, repair policy, tool budgets and prompts are unchanged.

- `paired_scoring.py` recomputes collisions from the candidate's final raw state
  using the existing native collision routine and frozen tolerances. It supplies
  fresh physics to a detached case pack and uses the existing Main checks and
  aggregation. It makes no model calls and cannot consume Designer tool budget.
- State and content hashes must remain unchanged across evaluation. The raw state,
  copied asset hashes, fresh `physics.json` and `report.json` are bound in
  `evaluation_proof.json` using `sceneexpert.raw_candidate_scoring.v2`.
- Normal pair export requires that fresh proof. Historical cached reports cannot
  silently pass the new audit. Missing assets, unknown checks, simulation errors,
  state mutation or broken hashes fail closed; none become synthetic negatives.
- `paired_rescore.py` validates completed legacy executions, restores their raw
  states with retained private assets, and applies the same acceptance thresholds.
  It writes fresh evaluations into a new collection. Original reports are retained
  in `rescore_origin.json`, and revised trajectories receive new IDs with explicit
  provenance. Prompts, responses, actions and contexts are preserved.
- `pair_audit.json` now separates `execution_integrity_passed` from
  `preference_gate_passed`. Overall success still requires a real eligible pair.
  Two rejected responses produce `no_accepted_candidate`; two accepted responses
  without supported relative judging produce `no_eligible_preference_contrast`.

## Next ACP: rescore retained 008 candidates without generation

Manually synchronize the updated source files first. Run this on the server that
still has the complete original `outputs/slow_memory/qwen38_initial_pairs_008/`
tree. The downloaded lightweight review bundle is insufficient for reconstruction.
Use the existing scene runtime Python with Drake; no Qwen service, rendering
service, model path or Git command is needed.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_008_rescore_009 \
SOURCE_RUN_ID=qwen38_initial_pairs_008 \
bash tmp/acp/acp_qwen38_initial_pairs_rescore.sh
```

The launcher uses the project `.venv/bin/python`, falling back to the existing
`Task3.2-main/.venv/bin/python`. Set `PYTHON_BIN` explicitly if the scene runtime
lives elsewhere. `SOURCE_PAIR_ROOT` can override the original pair directory.
An existing output RUN_ID is rejected, preserving previous attempts.

Success criteria for rescoring:

1. `runs/paired_initial/rescore_status.json` has `status=completed`.
2. `pair_audit.json` has `execution_integrity_passed=true`, `valid_group_count=1`
   and `candidate_count=2`; both candidates carry fresh scoring proofs.
3. `rescore_manifest.json` records `model_calls=0`, old/new verdicts, original
   source paths and the new scoring code fingerprint.
4. Original 008 files remain intact. Revised evidence and exports are only under
   `outputs/slow_memory/qwen38_initial_pairs_008_rescore_009/`.

Exit 0 additionally requires at least one eligible preference pair. Exit 2 with
completed rescoring, intact executions and zero pairs can be a correct lack of
preference contrast; inspect the diagnostic. A traceback or failed rescore status
is an execution problem and must be resolved first.

After reviewing 009, expand to two to four fresh furniture initial groups under a
new RUN_ID if fresh evidence and isolation pass. Track valid-group rate, actual
pair yield and cost per valid pair. If both candidates repeatedly pass, choose
more demanding development contexts or add independently validated relative
quality evidence; do not label an arbitrary response as rejected. Repair and
multistage expansion still require their own restoration support. A zero-pair
artifact does not justify training or a 12-group expansion.

## Matching lightweight packaging command

Run separately after the rescoring process stops, including after exit 2 or a
runtime error. Do not append this after a failed command in a `set -e` shell.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_008_rescore_009 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
```

Download the printed `tmp/downloads/<RUN_ID>_review_<timestamp>.tar.gz` and
`.tar.gz.sha256` paths. The archive preserves project-relative organization and
keeps audit reports, old/new evidence, trajectories, relevant media and logs.
Models, mesh trees, databases and duplicate raw snapshots remain on the server.
The package manifest lists omissions and verifies included bytes. Preserve the
full server directories until evidence review and dataset validation finish.

## Validation boundary

Local regression tests cover stale-positive and stale-negative caches, evaluator
failures/mutations, mandatory scoring proofs, isolated rescoring, original-file
preservation, new trajectory provenance, refusal of tampered/existing outputs,
diagnostics, native adapter ordering through doubles, and shell exit propagation.
The Windows environment lacks Drake, so native collision recomputation and legacy
raw-scene restoration remain the purpose of server run 009. No new physical
scores or usable preference pairs are claimed before that run completes.

Validated locally: 101 tests passed across initial-pair scoring, execution,
snapshot, provenance, collection, launcher and review-package suites. Bash syntax
checks passed for the rescoring and packaging launchers. New scoring/rescoring
modules passed Ruff; modified existing code passed focused fatal-error checks.
Pytest emitted two configuration warnings from the local environment.
