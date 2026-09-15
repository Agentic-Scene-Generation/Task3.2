# Initial-pair audit 012 review and calibration 013 plan

## Result: the audit behaved as expected

Reviewed package:
`tmp/results/slow_memory/qwen38_initial_pairs_011_audit_012_review_20260915_143914/qwen38_initial_pairs_011_audit_012_review/`.
All 15 included-file checksums match, with 85,253 selected bytes and no omissions.

012 completed its read-only audit with zero model calls and no native rescoring.
Its exit 2 is the predicted source-integrity and preference-gate finding, not a
new restoration or scoring exception. Do not repeat 012.

| Item | Observed |
| --- | --- |
| Valid decision groups | 3: group_000, group_002, group_003 |
| Quarantined group | group_001: execution incomplete, B/result.json missing |
| Admitted candidates | 6: 4 rejected, 2 accepted |
| Eligible preference pairs | 0 |
| Exclusions | no_accepted_candidate: 2; no_eligible_preference_contrast: 1 |
| Audit status | execution_failed |
| Training splits | train/val/test empty; training preflight not run |

The meeting-room A record remains part of the historical capture but cannot enter
pair training without a valid B execution. Its missing counterpart is not fixed
by selecting the other groups. The source failure is preserved in the new
rescoring manifest rather than silently hidden.

## Label corrections implemented in SceneExpert

Candidate scoring now calibrates a detached case pack against the existing
deterministic parser of the original user prompt. This changes neither the live
scene nor the designer context, native Critic implementation, online repair loop
or MemoryWriter. There is no model-identity preference or reduced pairing margin.

1. Restore an explicit floor-support constraint omitted by the compiled contract.
   For the same bound objects, generic asset annotations asking for table support
   become auxiliary evidence only if a replacement core floor check was actually
   materialized. Native floor-support checks remain mandatory, so a
   floating plant still fails. Explicit table-placement requirements stay strict.
2. Restore the parser's seating alignment relation for "tucked under" when the
   compiled contract instead introduced an axial front/rear separation. The native
   seating/work-surface checks retain distance and orientation requirements. A
   separately grounded front/rear requirement prevents this normalization.
3. Reconstitute an exact prompt-defined edge partition when split cohort selectors
   cannot identify the 6+1 chair groups. Category, target, count, orientation and
   the full per-edge distributions must agree. No cohort is chosen from candidate
   positions. The native evaluator checks all seven chairs, including wrong facing.

The label lineage has its own version,
`sceneexpert.candidate_contract_calibration.v1`, with original/calibrated contract
hashes, replaced constraints and superseded check IDs. The scoring proof binds this
lineage. Fresh physics still uses `sceneexpert.raw_candidate_scoring.v3`.
Historical v3 evidence remains auditable, while rescored trajectories have new
IDs and source-content provenance. Source results are never overwritten.

Offline rescoring also accepts explicit group selection. By default every group
must still be valid. Unknown, duplicate, empty or invalid selections fail; selection
cannot admit a partial A/B group. The manifest records selected groups, excluded
groups and the original audit errors. Each selected group still undergoes asset,
request, trajectory, restoration, physics and final pairing validation.

## Local evidence and its limits

Re-evaluating the seven saved 011 case packs with the real deterministic parser
and rule evaluator produced the following results. This local diagnostic reused
the previously saved physics evidence because review packages omit replay assets;
it is not a fresh native-physics evaluation of those seven scenes.
The local evaluator also warned that three external interaction-clearance index
files were absent. These diagnostics do not establish full benchmark coverage.

| Task / candidate | Old score / failed checks | Calibrated local score / failed checks |
| --- | --- | --- |
| Living room A and B | 0.92308 / 2 each | 1.00000 / 0 each |
| Study A and B | 0.97143 / 1 each | 1.00000 / 0 each |
| Meeting room A | 0.94118 / 2 | 1.00000 / 0; group still quarantined |
| Dining room A | 0.97143 / 0 | unchanged |
| Dining room B | 1.00000 / 0 | unchanged |

Restoring the complete chair partition also lets the native evaluator recognize
the intended seating companions in its access checks. This accounts for the
meeting-room A access result changing; collision and orientation failures are not
blanket suppressed. That A-only finding is a regression case, not training data.

These changes remove common false negatives. They do not create a defensible
negative candidate: the three complete groups are still expected to yield zero
eligible preferences. Do not train, reduce the margin, or arbitrarily choose a
winner to fill the dataset.

Validation: 144 focused portable tests passed, including group selection,
source preservation, calibration-proof tampering, launcher arguments, exit
propagation and packaging. Another 13 tests passed in the Linux Drake runtime:
nine native contract/rule cases and four scene-restoration/physics cases. The
tests reported two existing unrecognized pytest asyncio configuration warnings.

## Next ACP: one model-free calibration rescore

Manually synchronize the updated code to the server first. Use the complete
original **011** collection, not the 012 audit directory or a downloaded review
package. Keep the original collection stopped. This job requires its retained
raw assets and the existing scene runtime with Drake, but starts no Qwen server
and makes no model calls.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_011_calibrated_013 \
SOURCE_RUN_ID=qwen38_initial_pairs_011 \
RESCORE_GROUPS=group_000,group_002,group_003 \
REQUIRE_PAIRS=false \
bash tmp/acp/acp_qwen38_initial_pairs_rescore.sh
```

Acceptance for this calibration job:

- `rescore_status.json`: `operation_succeeded=true`, `candidate_count=6`.
- `rescore_manifest.json`: exactly groups 000/002/003 selected, 001 explicitly
  excluded, `model_calls=0`, six completed old/new result comparisons.
- Three groups pass execution integrity; all six candidates have fresh native
  physics/restoration proofs and hashed calibration lineage.
- Living-room floor support and study tucked seating are evaluated with corrected
  constraints. Dining-room behavior should stay stable. Any changed physics
  outcome must be reviewed, not forced to match the local diagnostic.
- With zero preferences, expect `outcome=completed_no_pairs`,
  `preference_gate_passed=false`, and launcher exit **0** because
  `REQUIRE_PAIRS=false`. This means calibration succeeded, not that DPO is ready.
  Actual execution, restoration or proof errors still fail the job.

Run the matching packaging command separately, even if rescoring fails:

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_011_calibrated_013 \
PACKAGE_MAX_FILE_MIB=32 \
PACKAGE_MAX_TOTAL_MIB=512 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
```

Download the printed `.tar.gz` and `.tar.gz.sha256` under `tmp/downloads/`.
The existing packer preserves project-relative paths and includes logs, reports,
proofs, audit/export diagnostics, metadata and size-bounded evidence media. It
omits large replay asset trees, models, meshes and databases, records omissions,
truncation and hashes in `_package/manifest.json`, and leaves server originals
untouched. Its 512 MiB limit covers selected file bytes, not the exact compressed
archive size. The review archive is not a complete training or replay dataset.

## Decision after calibration

Do not repeat these same four initial tasks hoping for a different pairing gate.
If the fresh server evidence confirms the local findings, the next development
target is a decision-level repair-pair entrypoint: preserve one failing state and
its actual context, independently execute two Qwen responses, and bind each label
to its own effects. Current initial-pair parameters cannot enable repair capture.
That entrypoint needs implementation and isolation validation before a new GPU
collection command is issued. Preserve structured tool-failure evidence in that
work so a failed shadow can be diagnosed instead of guessed from nearby logs.

013 is solely a bounded verification of trustworthy labels and replay evidence;
it is not a preference-yield experiment or a reason to scale training.
