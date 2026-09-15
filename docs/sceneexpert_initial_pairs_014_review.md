# 014 review: one supported preference, incomplete batch evidence

## Finding

The supplied package contains the first locally exportable preference from this
pilot sequence, but does not establish completion of the requested two-group run.
Do not repeat the original generation or claim that all of 014 passed.

Reviewed package:
`tmp/results/slow_memory/qwen38_initial_pairs_new3_014_review_20260915_175349/qwen38_initial_pairs_new3_014_review/`.
All 472 included-file SHA256 checks pass. Selected source bytes total 98,631,898
(about 94.1 MiB); 2,101 intentional omissions cover large assets/databases,
intermediate media, symlinks, replay trees and redundant payloads. No size-limit
truncations are recorded. None of the missing finalization files is listed as an
omitted file; this is not evidence of a final audit being dropped by the packer.

Only `group_000` (office) is present and completed. The long-living-room shared
floor plan exists, but its candidate group and critic-on batch are absent. The
package lacks final `pair_audit.json`, pair exit status, collection exit status
and ACP exit status. Its office scene log ends during manipuland work. These
observations cannot distinguish a running job, interruption, or a later server
completion. There is no evidence-backed overall exit code or root cause for the
missing completion; do not infer an OOM or a crash from this snapshot.

## Candidate comparison

| Office candidate | Score | Failed core checks | Fresh collisions | Verdict |
| --- | --- | --- | --- | --- |
| A, canonical | 0.98387097 | 1 | 0 | rejected |
| B, isolated shadow | 1.00000000 | 0 | 0 | accepted |

A fails the explicit prompt requirement to keep the water dispenser's front
accessible. The native `clear_access` check identifies `office_chair_3` as the
blocker of `water_dispenser_0`, with a required depth of 0.8 m. A puts the dispenser
at approximately (3.24, -2.00), facing west, and the chair at (2.50, -2.00), in that
front access zone. B locates the dispenser near (3.20, 2.72), facing south, with
the corresponding chair away from that zone. This is an explicit task-grounded
functional difference, not a generic asset support prior or a model-identity label.

Both sides use the pinned Qwen3.8 model and the same first request. Snapshot,
raw-state, physics, calibration, report and trajectory hashes passed retained-file
checks. The continuation proof records unchanged canonical state, assets and
memory. B has `shadow_exit=0`; both candidate result files say `completed`.

The unchanged local exporter loaded two records without diagnostics, selected
**B as chosen and A as rejected**, exported one tool-bearing initial/furniture
pair, and passed its dataset-format validation. The local diagnostic output is
`tmp/validation/initial_pairs_014/local_dpo_probe/`; it is not a substitute for
the full server-side audit. Raw asset trees were omitted from the review package
and were not independently rehashed locally.

The raw aggregate score difference is 0.01612903. Pairing is allowed by the
existing `hard_first_dominance=true` rule because B passes hard constraints and A
fails one. The serialized `quality_margin=0.05` is the existing policy floor, not
an observed 0.05 score improvement; provenance preserves the raw difference.
No threshold was reduced and neither label was manufactured. This differs from
013, where all six deterministic candidates were accepted and no contrast existed.

One development-task pair, one stage and no validation/test groups do not meet
the full training requirements. Do not train or claim effectiveness from this
result. The previously noted missing asset-level interaction-clearance indexes
also remain a coverage limitation; the positive finding here is bounded to the
explicit access check and other recorded evidence.

## Runtime evidence

All times below are UTC, as recorded by the server:

| Event | Time | Elapsed context |
| --- | --- | --- |
| ACP startup | 15:45:05 | start |
| Two shared floor plans complete | by 15:56:18 | about 11 minutes |
| Office initial Designer starts | 15:59:04 | about 14 minutes after startup |
| A trajectory captured | 16:03:45 | initial Designer timing: 265.839 seconds |
| B trajectory captured | 16:11:01 | about 7 minutes after A capture |
| Office furniture post-stage | 16:20:46 | about 22 minutes after stage entry |
| Office manipuland stage starts | 16:36:15 | canonical continuation still running |
| Last retained office log | 16:55:46 | at least 70 minutes after startup |

The first pair is available roughly 26 minutes after startup, but the collection
launcher waits for the full scene sequence before its final audit. The 70-minute
log span is a lower bound on the observed run, not a measured completed duration
or an estimate of remaining time. Shadow logs still contain a non-fatal external
SDK tracing Timeout serialization error; B completion and later canonical work
show that this message did not prevent this pair from being captured.

## Delivery improvement

Review packages now inventory finalization markers in their manifest and CLI
summary. A pair snapshot with no end markers produces
`finalization_not_established_by_package`. Rescore and separate-audit workflows
have their own marker sets. Marker presence does not certify a successful exit,
dataset readiness or process termination; process state remains explicitly unknown.
Diagnosis packages can still be created for incomplete or failed runs, with source
files untouched. This fixes the review ambiguity, not an unproven server failure.

The project delivery guide now requires per-phase time ranges, operator versus
unattended time, and estimation evidence. Validation: 47 packaging/launcher tests
pass, including incomplete snapshots, failed-but-finalized runs, short audit/rescore
jobs and original file preservation. Scoring and training eligibility are unchanged.

## Next experiment: 015 full retained-evidence audit, no model calls

Confirm that 014 has stopped before synchronizing runtime source files: changing
source contents during an active pair run can trip its provenance guard. The audit
entrypoint already exists and can be run with the server's present code. Its source
lock refuses an active source without changing it. Synchronize the packaging
improvement only after the source job has stopped. Do not overwrite or regenerate
014, and keep the expected
group count at two so the missing long-living-room group cannot be hidden.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_new3_014_audit_015 \
SOURCE_RUN_ID=qwen38_initial_pairs_new3_014 \
EXPECTED_GROUPS=2 \
bash tmp/acp/acp_qwen38_initial_pairs_audit.sh
```

Interpretation:

- If the source lock is busy, the command exits before creating audit outputs.
  Wait for the source to finish; do not package a nonexistent 015 directory.
- If the server has only the supplied one completed group, expect one eligible
  pair if full asset checks pass, but an overall failed gate/exit 2 for
  `group_count_mismatch: 1 != 2`. Preserve that distinction rather than changing
  `EXPECTED_GROUPS` to one.
- If the second group completed after packaging, audit both actual groups. Full
  acceptance is two valid groups, four candidates and at least one eligible pair.
- Any asset, request, trajectory or scoring-proof mismatch requires investigation
  before accepting the local preference result. An execution failure is never a
  negative training example.

Run the matching packaging command separately after the audit has actually run,
including after an audit finding returns exit 2:

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_new3_014_audit_015 \
PACKAGE_MAX_FILE_MIB=32 \
PACKAGE_MAX_TOTAL_MIB=512 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
```

Download the printed `.tar.gz` and `.tar.gz.sha256`. This new package contains
the audit/export diagnostics and logs, with original project-relative paths. It
does not recopy the large original scene tree. Checksums, omissions, truncation
and finalization-marker availability are recorded; full replay/training data
remain on the server.

### Estimated time

| Step | Estimate | Operator involvement |
| --- | --- | --- |
| Manual code synchronization and launch | 3-5 min | hands-on |
| Full source hashes and pair/export audit | 2-10 min | unattended; no model calls or new physics |
| Small 015 review package | 1-3 min | launch, then unattended |
| Download/review handoff | 1-3 min | size and network dependent |

Budget about **10-25 minutes**, reserving 30 minutes for slow shared storage. This
is an engineering estimate based on one or two retained groups and the small
audit/export workload, not a measured full 015 run. Waiting for an active 014 job
is additional and cannot be estimated reliably from the current incomplete logs.

Once 015 confirms full evidence, decide whether only the missing scene needs a
new collection. The first pair means the zero-contrast stopping condition has not
been reached. Repair-pair support remains useful, but neither broad initial
resampling nor an unimplemented repair flag is the immediate next experiment.
