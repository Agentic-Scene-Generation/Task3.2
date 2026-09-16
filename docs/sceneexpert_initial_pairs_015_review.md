# 015 review: verified office preference, missing second collection group

## Decision

015 produced the expected strict audit result for an incomplete two-group source:
one valid office group and one eligible preference, with exit 2 because the
second group does not exist. Preserve the office pair and collect only the missing
long-living-room task in a new single-group run. No scoring, pairing or runtime
change is justified by this audit. Do not repeat office, rerun 015, or lower the
expected group count of the historical two-group experiment.

Reviewed package:
`tmp/results/slow_memory/qwen38_initial_pairs_new3_014_audit_015_review_20260916_115247/qwen38_initial_pairs_new3_014_audit_015_review/`.

## Evidence

All 15 manifest-listed file hashes pass, including the generated package note.
The manifest records 369,249 selected source bytes, zero omissions and zero
warnings. Both finalization markers are present. The recorded exit is 2 at
`2026-09-15T18:49:08+00:00`; marker presence does not mean the gate passed.

The authoritative outputs are under
`outputs/slow_memory/qwen38_initial_pairs_new3_014_audit_015/runs/paired_initial/`:

| Check | Observed result |
| --- | --- |
| `pair_audit.json` status | `execution_failed` |
| Only audit error | `group_count_mismatch: 1 != 2` |
| `group_000` individual integrity | valid, no errors |
| Candidates | 2: one accepted, one rejected |
| Eligible preferences | 1 |
| Preference gate | passed |
| Batch execution-integrity gate | failed: missing second group |
| DPO format validation | valid, no errors |
| Splits | train 1, validation 0, test 0 |
| Training preflight | not run |

015 acquired the inactive-source lock and inspected the original server trees,
including the input snapshot and candidate assets omitted from the earlier review
download. The missing second group is therefore confirmed at audit time; it was
not merely omitted from the 014 archive. The small 015 archive contains the audit
report, not the original asset trees, so this local review validates the delivered
server audit and its exported records rather than rerunning asset checks locally.
It does not establish why 014 stopped before the second group. There is no new
exception or evidence supporting an OOM, a model crash or an audit implementation
defect.

The exported preference is
`dpo_90e6fba9da2b406e7f4c611e7321df06`: B chosen, A rejected. Both use
`unsloth/Qwen3.8-27B-GGUF`. The complete exported record is identical to the local
014 diagnostic export; 015 validates the same pair, not an additional sample.
It preserves exact prompt, tool, media and spatial-context matches.

As documented in the 014 review, A violates the explicit water-dispenser front
access requirement, while B passes the recorded core checks. Scores are
0.9838709677 and 1.0 respectively. The observed aggregate difference is
0.0161290323; the serialized `quality_margin=0.05` is the existing hard-first
policy floor, not a measured five-point gain. No model-identity preference or
new threshold was introduced.

This is one tool-bearing `designer_initial` / `furniture` preference from one
independent task. It is useful collection evidence, not a training-ready dataset
or evidence of Slow Memory improvement. Keep 014's full source tree and 015's
audit/export together; do not count the same pair from both locations twice.
The previously identified asset-level interaction-index coverage limitation
remains unchanged.

## Next experiment: 016, one long-living-room group

Use the existing initial-pairs entrypoint for one new task and two independently
executed Qwen candidates. Explicitly keep the terminal stage at `manipuland` so
the canonical scene can finish and the normal MemoryWriter lifecycle remains
eligible. The preference is still scored at the raw furniture decision boundary;
later repairs or final-scene scores do not relabel it.

The new run uses its own default cold memory directory. This bounds task coverage
and preserves the old evidence; it is not a measurement of cross-scene Fast Memory
gain. Group numbering restarts at `group_000` within 016. The one-group target is
specific to this new experiment and does not retroactively make 014 complete.

Manually synchronize source before launching, and keep source files unchanged
while the job runs. No new runtime patch is required by this review.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_long_living_016 \
PAIR_GROUPS=1 \
CASE_SET=new3 \
SCENE_SELECTION=long_living_room \
DIFFICULTY_SELECTION=all \
ACP_PARALLELISM=1 \
PIPELINE_STOP_STAGE=manipuland \
CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE=false \
MODEL_NAME=unsloth/Qwen3.8-27B-GGUF \
MODEL_DIR=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF \
MODEL=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf \
MMPROJ=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf \
bash tmp/acp/acp_qwen38_initial_pairs.sh
```

Acceptance has two separate levels:

- Execution: one valid group, two complete candidates, no request/source/asset/
  scoring-proof mismatch, complete canonical generation, and generation exit 0.
  Inspect the terminal reports and MemoryWriter lifecycle, not only candidate
  files that can appear much earlier than scene completion.
- Preference yield: `eligible_pair_count >= 1`, format validation passes, final
  `pair_audit_exit=0`. If generation and integrity pass but the two candidates
  have no eligible contrast, `status=completed_no_pairs` and pair audit exit 2
  are an informative zero-yield result. Retain and review it; do not weaken
  labels or repeat sampling until a preferred answer appears. Infrastructure
  failures are quarantined rather than converted into rejected training answers.

Review 016 together with the retained office pair before increasing collection
volume. A successful second task establishes broader initial-decision coverage;
it still does not justify training or imply that repair-pair collection exists.

## Lightweight result packaging

Run this separately after the ACP finishes, including a nonzero exit. A separate
invocation ensures that the collection command's `set -e` does not skip packaging.
If the run is interrupted, the same command can collect a diagnostic snapshot;
check its finalization warning before calling the experiment complete.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_long_living_016 \
PACKAGE_MAX_FILE_MIB=32 \
PACKAGE_MAX_TOTAL_MIB=512 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
```

Download the printed `tmp/downloads/...tar.gz` and its `.tar.gz.sha256` companion.
The package preserves project-relative paths and includes logs, configuration,
metrics, traces, audit/export diagnostics and selected evidence media. Models,
large meshes, databases, replay asset copies and redundant outputs are excluded.
The limits apply to selected uncompressed source files; archive size may differ.
The manifest records exclusions, truncation, hashes and available finalization
markers. Server originals remain untouched and are still required for full
replay, re-audit and training preparation.

## Estimated time

| Phase | Elapsed estimate | Operator time |
| --- | --- | --- |
| Manual synchronization and launch | 3-5 min | hands-on |
| Service startup and one shared floor plan | 8-15 min | unattended |
| Furniture A/B capture plus canonical furniture continuation | 20-40 min | unattended |
| Wall, ceiling, manipuland and normal finalization | 45-100 min | unattended |
| Automatic audits and export | 2-10 min | unattended |
| Review packaging | 2-5 min | launch, then unattended |
| Download and handoff | 2-10 min | network dependent |

Reserve about **1.5-3 hours**, with roughly **5-15 minutes hands-on**. These are
planning estimates, not measured long-living-room completion times or an imposed
job timeout. Slow model/asset services and repair iterations can extend them.
The basis is 014: two floor plans and startup took about 11 minutes; the office
pair was captured about 26 minutes after startup; canonical furniture took about
22 minutes; the retained full-scene log spanned at least 70 minutes and ended
before completion. The new scene's five dining place settings make manipuland
duration a material uncertainty. Candidate files appearing after roughly
25-50 minutes do not mean the full ACP is finished.

## Local validation

The review checked every included-file SHA256, the sole audit error, individual
group validity, model identity, preference direction, format validation and exact
equality with the earlier local export. The existing launcher chain supports the
selected task, group count and stop-stage setting. Both published Bash snippets
and their entrypoint scripts pass shell syntax checking. This delivery changes
only this review/runbook; no GPU experiment was run locally.
