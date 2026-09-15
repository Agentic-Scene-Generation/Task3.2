# Initial-pair 011 review and 012 audit plan

## Verdict

011 has not met the collection acceptance criteria. The supplied review snapshot
contains four decision groups, seven scored candidate records, three completed
A/B groups and one failed shadow. Reapplying the existing pairing policy to the
six records in the completed groups yields zero pairs. No training is justified.

Reviewed package:
`tmp/results/slow_memory/qwen38_initial_pairs_011_review_20260915_125701/qwen38_initial_pairs_011_review/`.
Its manifest is byte-identical to the earlier `124955` package's manifest.
All 1,573 included-file checksums passed (259,981,947 selected bytes). Missing
final artifacts are not listed as package omissions, and no size-limit truncation
was recorded. The package is a valid transfer of an incomplete run snapshot.

The terminal reports three completed critic-on batches. The dining-room log ends
at 12:31:31 UTC during manipuland work. Its final scene status, the collection exit
record, pair exit record, final pair audit and DPO export are absent. This cannot
establish whether the server is still running, was interrupted, or has completed
since packaging. Do not infer a final process exit code from this snapshot.

## Candidate results

| Group / task | A | B | Existing preference eligibility |
| --- | --- | --- | --- |
| 000 / living room | rejected, 0.92308, 2 failed checks | rejected, 0.92308, 2 failed checks | no accepted candidate |
| 001 / meeting room | rejected, 0.94118, 2 failed checks | shadow failed; no scored record | quarantine whole group |
| 002 / study | rejected, 0.97143, 1 failed check | rejected, 0.97143, 1 failed check | no accepted candidate |
| 003 / dining room | accepted, 0.97143 | accepted, 1.00000 | no eligible preference contrast |

All seven records use `unsloth/Qwen3.8-27B-GGUF` and fresh v3 scoring proofs. All
seven retained scoring proofs, raw-state hashes, trajectory hashes and 45 media
references passed local verification. Actual first-request bodies match in all
four groups, and all four continuation proofs report unchanged canonical state,
assets and memory. Context hashes match within the three complete pairs.
The two tied score pairs have different executions; equal aggregate scores do not
mean duplicate model outputs. These checks do not revalidate omitted server asset
trees or constitute a full native replay.

The pairing-only check loaded six records without errors and produced two
`no_accepted_candidate` diagnostics and one `no_eligible_preference_contrast`.
The meeting-room A is a seventh scored record but is not exportable by itself.
A full audit may therefore report six admitted candidates, not seven captured
records. Neither count indicates a missing valid preference pair.

## Problems to resolve before more GPU collection

1. **Explicit floor placement conflicts with generic asset dependencies.** The
   living-room prompt explicitly asks for two large plants on the floor. Both
   candidates instead receive hard failures for not placing those plants on the
   coffee table, from generic `explicit_target_relation` asset annotations
   (`watering_can`, `plant_stand`, `table`). These are not defensible negative
   training labels for obeying the prompt. Reconcile prompt-authorized support
   relations with asset defaults, with floor-plant and table-plant counterexamples.
2. **Tucked seating is compiled into a conflicting axial relation.** Study failures
   come from `in_front_of(office_chair, desk)` compiled from “tucked under the desk”.
   The evaluator requires 0.71 m / 0.64 m forward distances, while A/B have 0.57 m /
   0.60 m. Calibrate the intended tucked/support/clearance contract before deciding
   whether either candidate is actually wrong. Do not simply lower its threshold.
3. **Meeting-room cohorts are unresolved.** The prompt has six long-side chairs and
   one short-side chair. Its two `edge_distribution` constraints fail strict
   category/cohort binding. A also has an access-zone failure that may be genuine.
   Check the 3+3+1 contract and retain independent true layout failures. The shadow
   also fails the unhandled-tool/transport guard. Its log includes an HSSD whiteboard
   proportion failure, but the rejected tool output was not retained; the package
   does not prove that this asset message was the guard's exact match. Persist
   structured tool-failure evidence before quarantine in the next collection change.
4. **External SDK tracing adds noise.** All shadow logs show non-fatal tracing
   network errors and Timeout JSON serialization errors. The three successful
   shadows prove this is not the universal execution blocker. Disable external
   telemetry for the offline worker while preserving local trajectories and actual
   HTTP timeout settings; do not confuse telemetry with model/tool execution.

Prioritize SceneExpert compilation and collection adapters. Where a shared
relation evaluator itself contradicts the explicit task, a narrowly scoped shared
contract correction is justified because it directly contaminates Slow Memory
labels. No broad Designer, Critic, repair, memory or model architecture change is
proposed. Keep these groups as regression cases; do not silently relabel originals.

## Immediate ACP: audit retained full evidence, zero model calls

This turn adds `--audit-output` and an audit launcher. They read the retained
011 evidence and export to a fresh 012 directory. Existing output or overlapping
source/output paths are refused. The launcher uses the existing collection lock
without changing its bytes and refuses an active collector. Pair/evidence gates
remain strict. The new audit records its own source-content fingerprint separately
from the candidates' original provenance. This does not rescore or repair any labels.

After the original 011 launcher has stopped, manually synchronize the code and run:

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_011_audit_012 \
SOURCE_RUN_ID=qwen38_initial_pairs_011 \
EXPECTED_GROUPS=4 \
bash tmp/acp/acp_qwen38_initial_pairs_audit.sh
```

If the server retains the supplied state, expect three valid groups, six admitted
candidate records, zero pairs and a failed integrity/pair gate due to group 001.
Exit 2 is then an expected audit finding, not another restoration crash. Do not
rerun generation or native physics to alter this exit. Inspect `pair_audit.json`
and its detailed quarantine reasons. If the lock is busy, allow the original run
to finish; never start overlapping work against it.

Run the matching packaging command separately, including after audit exit 2:

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_011_audit_012 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
```

Download the printed `.tar.gz` and `.tar.gz.sha256`. This small package contains
the new audit, export diagnostics, source-run reference and audit logs; it does
not recopy the already reviewed 011 media. The packer preserves project-relative
paths, uses 32 MiB per-file / 512 MiB selected-byte limits, and records omissions,
truncation and hashes. Original 011 files remain on the server untouched.

## Following experimental decision

Complete label-contract regression cases and diagnostic capture first. Re-evaluate
retained usable candidates under an explicitly versioned corrected protocol where
necessary; fixing incorrect common failures still does not guarantee a contrast.
Then run a bounded two-to-four-group candidate pilot with at least one
independently supported eligible preference pair as its collection acceptance
criterion. Select wider stages/repair contexts only after that entrypoint supports
their frozen state and independent execution. Review the small pilot before
expanding to 12 groups; do not launch a blind repetition of the current four tasks.
Each later GPU run needs a new run ID, runnable ACP and matching review package.

Validation of the audit-output change: 74 focused unit/launcher/packaging tests
passed, including source preservation, output collision rejection, failed-group
quarantine and active-collector refusal. Launcher lock outcomes are stubbed in
portable tests; full evidence-tree validation remains the server-side 012 job.
