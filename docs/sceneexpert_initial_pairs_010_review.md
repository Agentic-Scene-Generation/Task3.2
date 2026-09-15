# Initial-pair rescore 010: completed, no eligible preference contrast

## Result and cause of exit 2

Reviewed the supplied `tmp/results/slow_memory/qwen38_initial_pairs_008_rescore_010/`.
Unlike 009, 010 has no exception traceback. Both native restorations and fresh
Drake evaluations completed, and the server's execution-integrity audit passed.

| Candidate | Fresh score | Collision failures | All check failures | Unknown checks | Verdict |
| --- | --- | --- | --- | --- | --- |
| A | 0.9375 | 0 | 0 | 0 | accepted |
| B | 0.9600 | 0 | 0 | 0 | accepted |

`outputs/runs/paired_initial/rescore_status.json` says `status=completed`.
`pair_audit.json` records one valid group, two candidates,
`execution_integrity_passed=true`, `preference_gate_passed=false`, and zero pairs.
The sole pairing diagnostic is `no_eligible_preference_contrast`.

The prior CLI returned exit 2 whenever the **dataset** gate was false, even when
the requested **rescoring operation** had succeeded. That operational ambiguity
made a valid no-pair result appear to be another restoration failure.

Both candidates have deterministic evidence. Neither is an authoritative rejected
response, and the exporter has no supported independent relative-critic ranking
for this pair. Their score difference is only 0.0225, also below the existing
0.05 quality-margin requirement; that is an additional observation, not the
diagnostic branch that rejected this pair. Do not flip A's label or reinterpret
the aggregate scores merely to produce a training example.

The three missing-clearance-index warnings did not stop this run. They do not
establish complete clearance-index coverage, but the supplied Main summaries have
zero unknown checks. No Critic or clearance-rule change is justified by this exit.

## Evidence reviewed

- Logs show 0 collisions for A and B, native collision computation taking about
  2.55 and 2.58 seconds, and successful independent candidate results.
- Revalidated serialized restoration/scoring proofs, source/restored state hashes,
  trajectory byte hashes and all 15 referenced media entries (A: 6; B: 9).
- Actual first HTTP request bodies match. Context hashes match, and the server's
  continuation proof records unchanged canonical state, assets and memory.
- Fresh proof protocol is v3. A has no quaternion roundoff; B has the previously
  verified single machine-precision difference. Both use explicit asset relocation.

The provided folder reorganizes server output under `outputs/runs/`; it is not a
complete original asset tree. These local checks validate retained evidence and
server audit records, not another full native scene replay. Original 010 results
were not rewritten or relabeled.

## Implemented operational correction

Pair audits now include `status` and `candidate_verdict_counts`:

- `execution_failed`: execution, scope, state, assets or evidence integrity failed.
- `completed_no_pairs`: intact candidate execution but no eligible preference pair.
- `completed_dataset_gate_failed`: pairs exist but the requested dataset gate fails.
- `completed_with_pairs`: execution integrity and the requested pair gate pass.

The rescore-only CLI now defaults to exit 0 when execution/evidence integrity passes,
while `gate_passed` and `preference_gate_passed` remain false for a no-pair dataset.
It logs that the dataset is not ready. `--require-pairs` (launcher environment
`REQUIRE_PAIRS=true`) explicitly restores strict pair-gate exit behavior. Exceptions
and invalid execution/evidence remain failures under either policy.

Normal generation and `--audit-root` keep their strict pair gate: no synthetic
negative, lowered margin or weakened export validator was introduced. The initial
collection launcher now saves the final audit console output to `pair_audit.log`
and explains a no-pair exit separately from `execution_failed`. All runtime changes
are confined to SceneExpert collection/audit entrypoints.

Validation: 121 focused tests passed, covering restoration/scoring proofs, isolated
candidate execution, rescoring exit policy, launcher exit propagation, provenance,
collection and review packaging. Python formatting/static checks and Bash syntax
checks passed. Pytest emitted two existing asyncio-configuration warnings.

**Do not repeat 010 just to change its exit code.** Its scientific result is already
available. The next step is new candidate collection, not further rescoring of 008.

## Next ACP: 011, four fresh furniture decision groups

Manually synchronize the latest source first. The verified `legacy8` registry
contains all four IDs below. Keep one canonical scene at a time; each selected
scene supplies A plus an isolated B from the same furniture decision context.
The four tasks cover a living room, mixed-edge meeting-room seating, a study and
dining-room access constraints. They are development collection tasks, not a
held-out test set. Canonical MemoryWriter stays enabled between completed scenes.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_011 \
PAIR_GROUPS=4 \
CASE_SET=legacy8 \
SCENE_SELECTION=default_living_room,meeting_room_mixed_edge_seating,study_desk_access_crunch,dining_room_service_squeeze \
CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE=false \
MODEL_DIR=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF \
MODEL=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf \
MMPROJ=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf \
bash tmp/acp/acp_qwen38_initial_pairs.sh
```

Inspect final generation completion (4/4 canonical scenes), pair integrity
(`valid_group_count=4`, `candidate_count=8`), candidate verdict counts, actual pair
yield and exclusion reasons. At least one independently supported eligible pair
is required for the collection's strict gate. Zero pairs can still be a completed
sampled collection: inspect `status=completed_no_pairs`; it must not be confused
with `execution_failed`. Pair yield is an experimental outcome, not guaranteed by
increasing the group count. Do not train until the separate dataset/training
preflight passes. Review 011 before choosing any further scale or judging changes.

## Matching lightweight packaging command

Run separately after the process stops, including after exit 2:

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_011 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
```

Download the printed `tmp/downloads/<RUN_ID>_review_<timestamp>.tar.gz` and its
`.tar.gz.sha256`. Preserve the archive's project-relative directory hierarchy when
extracting it. The package keeps logs (including `pair_audit.log`), run configuration,
metrics, traces, status/audit/scoring proofs, trajectories and relevant media.
Models, meshes, databases, replay asset copies and redundant output remain on the
server. Defaults are 32 MiB per file and 512 MiB selected bytes; the package manifest
records checksums, omissions and truncations. Full server artifacts remain required
for replay and training.
