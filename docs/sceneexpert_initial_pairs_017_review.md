# 017 review: complete collection, zero preferences, two label-contract defects

## Decision and evidence

017 passes the execution-integrity target, but does not meet the preference-yield
target. Both Qwen candidates completed, the canonical scene reached its terminal
stage, and the final audit ran. Both candidates have real hard violations and
remain rejected. Do not export B as chosen merely because its aggregate score is
higher, and do not restart the same two-hour generation to validate a scoring fix.

Source archive:
`tmp/results/slow_memory/qwen38_initial_pairs_long_living_017_review_20260921_080238.tar.gz`.
Its SHA-256 is
`1e489b1ffb5154d69f35e794604643ac8190bf4e5422dff874695f0dffc0ff5c`.
The archive is 28,492,653 bytes (27.2 MiB); all 495 manifest-listed included files
pass their hashes. Selected uncompressed files total 87,024,536 bytes (83.0 MiB).
There are 2,503 documented omissions. One detailed timing JSONL exceeds the
32 MiB per-file limit; the manifest flags it as omitted structured evidence.
Candidate reports, trajectories, scoring proofs and all finalization markers are
present. Replay assets stay on the server, as intended.

| Evidence | Result |
| --- | --- |
| `group_000/status.json` | completed; shadow exit 0 |
| Candidate A/B | two complete `unsloth/Qwen3.8-27B-GGUF` executions |
| First serialized requests | identical |
| Local original raw-state, trajectory and scoring-proof hashes | all pass |
| Server pair audit | one valid group; execution integrity passed |
| Candidate verdicts / eligible preferences | rejected 2 / pairs 0 |
| Pair audit status | `completed_no_pairs` |
| Pair wrapper exits | `generation_exit=0`, `pair_audit_exit=2` |
| DPO exclusion | `no_accepted_candidate` |
| Canonical trace | completed, all five stages accounted for, degraded |
| Canonical full report | generation complete; requirements partial; `pass_scene=false` |
| MemoryWriter | one success case persisted/promoted; bank revision 0 to 1 |

The pair audit's exit 2 records the unmet minimum-pair gate, not a generation
crash. The two trace locations contain copies of the same MemoryWriter result;
they are not two updates. No `Requests list cannot be empty` error appears in
either retained initial trajectory. These two trajectories do not contain the
new argument-error response either, so they show an unblocked end-to-end run,
not a fresh exercise of the malformed-input recovery branch.

The previously audited 014/015 office preference remains one sample. 017 adds
no usable preference; its complete failures are useful diagnostic evidence.
Full server-side raw-asset auditing cannot be repeated from a lightweight archive.

## Real failures and incorrect constraints

| Candidate | Original score | Original core failures | Recorded hard collisions |
| --- | ---: | ---: | ---: |
| A | 0.734848 | 17 | 3 |
| B | 0.798387 | 12 | 3 |

A has a 7.02 cm sofa/plant penetration, additional wall/plant and TV-stand/plant
collisions, a television outside the room and unsupported by its stand, and
incorrect dining-chair facing. B has a 7.76 cm TV/stand penetration, a 1.40 cm
table/chair penetration and a wall/stand collision, with blocked cabinet access
and a dining group intersecting the living-area view corridor. These failures
remain after the semantic correction below. Later canonical repairs do not
relabel the original furniture decisions.

Two contract problems also polluted both original labels:

1. The explicit phrase **floor plants** did not produce floor-support constraints.
   Generic asset annotations instead required some of those plants to be on a
   table. Calibration v1 recognized the verb phrase **plants on the floor**,
   but missed the compound noun in this prompt.
2. The five-chair layout contained a full 5-chair edge-distribution rule plus an
   inferred 4-chair long-side rule. The evaluator binds all matching chairs for
   each rule, so the partial duplicate could not match five chairs. An additional
   inferred `centered_on_wall` rule incorrectly represented a chair centered on
   the table's short side. The full compiled rule also lost the explicit facing
   and short-side centering requirements. v1 only joined disjoint partitions
   with equal orientation fields; it could not repair this full-plus-partial case.

## Repair and validation

The runtime change is confined to
`scenesmith/scene_expert/slow_memory/paired_contract.py`. Calibration v2:

- Normalizes positive floor-plant noun phrases through the existing deterministic
  parser while retaining the original evidence span. Negative and conflicting
  plant-support instructions do not authorize this normalization. Generic table
  support is auxiliary only when a real core floor-support check is present.
- Restores the complete prompt-derived edge topology, including chair count,
  empty opposite side, equal spacing, short-side centering and facing. It removes
  only matching covered parts and the specific inferred wall-centering error.
  Conflicting counts/orientations and explicit or actual-wall constraints remain.
- Preserves before/after contract hashes and removed rules in calibration
  metadata, which remains bound by the scoring proof. No geometry, prompt,
  candidate context, thresholds or online critic/repair logic is changed.

24 native deterministic-parser/evaluator tests pass, including correct layouts,
wrong facing, missing/extra chairs, off-center short-side seating, lifted plants,
negative/mixed instructions and constraints that must not be suppressed.
107 portable pairing, capture, scoring, restoration, exit-policy and packaging
tests pass. The SDK mock test now bypasses environment proxy mounts so that its
mocked transport cannot accidentally send its synthetic request to a proxy.
No production HTTP-client behavior was changed.

Black/Ruff pass for the calibration module and native contract tests. The existing
SDK test file's five lint diagnostics are unchanged. Both published launchers
pass Bash syntax checks. No new model or full scene-generation run was executed
on the local review machine.

The retained A/B case packs were also evaluated locally with native rules.
The unmodified baseline exactly reproduces the original check labels and scene
summaries. v2 restores floor and complete chair-layout constraints, demotes 4/2
generic support checks, and leaves 12/9 core failures for A/B respectively. Both
still have all three recorded hard collisions. This is a diagnostic replay of
retained geometry and physics evidence, not freshly computed mesh physics or a
replacement training export. Original archive files were not rewritten.

The known missing interaction-clearance index warnings remain. Matching the
baseline verifies this comparison; it does not establish complete asset-specific
interaction coverage. Restore the real index data before final paper evaluation.
No fabricated index records or changes to the other team's evaluator are made.

## Next experiment: 018, one model-free authoritative rescore

Manually synchronize the changed source and retain 017's full server output,
including `input_scene` and both `raw_scene` trees. Use the existing raw-snapshot
rescore entrypoint. It validates the source, copies the selected group to a new
run, restores each exact raw candidate, recomputes mesh physics, and performs
the pair audit. It starts no Qwen service and calls no model.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_long_living_017_calibrated_018 \
SOURCE_RUN_ID=qwen38_initial_pairs_long_living_017 \
RESCORE_GROUPS=group_000 \
REQUIRE_PAIRS=false \
bash tmp/acp/acp_qwen38_initial_pairs_rescore.sh
```

Acceptance criteria:

- `rescore_status.json`: `operation_succeeded=true`, `candidate_count=2`;
  `pair_audit.json`: execution integrity passed and one valid group.
- Both reports contain calibration v2, floor-support and complete edge-layout
  replacements, and fresh scoring/restoration proofs. Neither source assets nor
  the original 017 results are changed.
- Expected substantive outcome: both candidates remain rejected, with zero pairs
  and `completed_no_pairs`. With `REQUIRE_PAIRS=false`, a successful rescore exits
  0 even though the preference/training gates remain false. This setting changes
  only the rescore exit policy; it does not authorize an invalid preference.

After one successful validation, close this initial-context investigation.
Repeated rescoring or new random initial generations of the same hard task are
not the next priority. The next development priority is paired **repair**
decisions: capture the same failed decision-time state and full session, execute
two Qwen repairs in isolation, and attach each label to its own executed result.
That collector still needs implementation; no existing ACP flag enables it.
Do not start large-batch collection or training on the current one-pair dataset.

## Matching lightweight package

Run separately after 018 ends, including if it exits nonzero:

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_long_living_017_calibrated_018 \
PACKAGE_MAX_FILE_MIB=32 \
PACKAGE_MAX_TOTAL_MIB=512 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
```

Download the printed `.tar.gz` and `.tar.gz.sha256` from `tmp/downloads/`.
The archive preserves project-relative directories, with logs, source/run
configuration, traces, candidate reports/proofs, rescore/pair audits and bounded
evidence media. Models, meshes, databases, replay asset copies and redundant
payloads remain on the server. Its manifest records selected hashes, omissions,
truncation and finalization markers. Limits are for uncompressed selected files.
The package does not replace the full server dataset or asset tree.

## Elapsed-time estimate

| Phase | Range | Operator involvement |
| --- | --- | --- |
| Manually synchronize and launch | 3-5 min | hands-on |
| Source validation/copy, restore and rescore two candidates | 3-15 min | unattended |
| Final audit and quick report checks | 1-3 min | mostly automatic |
| Package | 1-3 min | launch, then unattended |
| Download | 1-5 min | network dependent |

Reserve approximately **10-35 minutes**, including **5-10 minutes hands-on**.
017's complete collection ran from 09:26:00 to its pair audit at 11:32:03 UTC
on September 17 (about 126 minutes). That generation is not repeated in 018.
The six native physics evaluations in the retained 013 rescore log took
2.57-6.82 seconds each; source hashing, AFS I/O and asset copies dominate the
rescore estimate. The 017 manifest lists about 2.2 GB of omitted files, including
assets retained only on the server. These are planning ranges, not a guaranteed
runtime or a substitute for checking the final exit and audit records.
