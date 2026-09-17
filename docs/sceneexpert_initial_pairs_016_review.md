# 016 review: malformed asset arguments blocked candidate capture

## Result

016 does not meet the single-group collection target. The initial Designer ran,
but candidate A was quarantined before raw-state scoring/capture; candidate B
was never launched. There are no usable new preferences. Preserve the verified
office preference from 014/015; it remains one sample, not two.

Reviewed package:
`tmp/results/slow_memory/qwen38_initial_pairs_long_living_016_review_20260917_081202/qwen38_initial_pairs_long_living_016_review/`.

All 199 manifest-listed file hashes pass. Selected source bytes are 55,784,088
(about 53.2 MiB). The 1,657 intentional omissions comprise 1,589 asset/database/
nonreview files, 64 intermediate images, three symlinks and one replay subtree.
There is no recorded size-limit truncation explaining the missing final outputs.

| Evidence | Result |
| --- | --- |
| `paired_initial/group_000/status.json` | failed, phase `canonical_evidence` |
| `A/tool_execution_failure.json` | exact call and error retained |
| A raw candidate state, scoring proof and result | absent |
| Candidate B | absent |
| Final pair audit / pair exit / collection exit / ACP exit | all absent |
| Canonical SceneExpert trace | partial, status `running` |
| Last retained scene log | wall_mounted, 2026-09-16 15:37:36 UTC |

The operator reports that ACP displayed normal completion. The archive does not
corroborate full-scene completion or an overall exit code. The checked launcher
waits for its scene runner, writes its exit record, and then runs the collection
and pair audits on the normal exit path. The absence of all those records is an
unresolved disagreement between the UI report and archived evidence, not proof
of a particular timeout, OOM, or manual interruption. The package already flags
`finalization_not_established_by_package`. Do not label this `completed_no_pairs`:
the candidate group has an explicit execution-evidence failure.

## Root cause and repair

The retained initial-Designer payload is
`runs/critic_on/batch_003/hydra/scene_002/scene_expert/audit/llm_payloads/1789571634388_designer_request_initial_design.json`.
Its structured trace has 63 tool calls and their 63 results. The failure diagnostic
points to result index 7, call `r5Wqi5GOI0W44X8C67i3nSFU1m8Ilmo0`:

```json
{
  "object_descriptions": ["Flat screen 55 inch television on a low rectangular stand base"],
  "short_names": ["television"],
  "desired_dimensions": []
}
```

The causal chain is:

1. An earlier asset batch partially succeeded; the television failed the existing
   asset-proportion check. That ordinary acquisition failure made a bounded
   recovery attempt available.
2. The model retried with a description and name but no dimensions. The furniture
   tool lacked batch validation before safety accounting, so the malformed call
   consumed the available recovery attempt.
3. In the direct HSSD path, `zip(descriptions, dimensions, asset_paths)` produced
   no requests. The HSSD client rejected the empty request list locally, before
   issuing that batch's HTTP request. The SDK exposed the exception as
   `An error occurred while running the tool. Please try again. Error: Requests list cannot be empty`.
4. The paired-candidate guard correctly refused an unhandled-tool exception.
   Later model retries with `[[]]` and then valid dimensions were blocked by the
   already-consumed recovery allowance. Canonical safety and later stages still
   continued, but the failed pair was not recreated.

The repair adds a pure SceneExpert argument validator and a small guard at the
furniture `generate_assets` boundary, before safety accounting, size policy or
asset acquisition. It checks nonempty aligned lists, nonblank descriptions/names,
and exactly three finite positive dimensions per item. Invalid input produces the
existing structured unsuccessful asset response with correction instructions.
It does not invent missing dimensions, silently drop requests, consume the asset
recovery allowance or call a backend. Corrected valid requests follow the original
path and still obey the safety controller.

This small furniture-tool change is necessary because the demonstrated defect
directly prevents Slow Memory pair capture. Designer/critic loops, prompts,
retrieval ranking, size policy, safety limits and preference scoring are unchanged.
The quarantine rule is unchanged: genuine transport and unhandled backend errors
remain excluded. Recoverable argument feedback stays visible in the trajectory;
it does not itself select a winner. The eventual executed scene determines the
preference through the existing evidence gates.

Do not rescore 016 using its final or post-repair furniture checkpoint. A was
rejected before its raw candidate state was preserved, and B does not exist.
An independent fresh pair is required.

## Validation

63 portable tests passed: the new argument/SDK boundary tests plus the existing
pairing and failure-evidence suites. They cover empty and mismatched lists,
malformed/nonfinite/nonpositive dimensions, valid batches, correction without
consuming the safety allowance, unchanged valid-request denial, and quarantine
of a real backend exception. The SDK tests execute the actual furniture closure
with native acquisition replaced by spies.

The exact recorded failing call was also replayed against the repaired closure:
structured correction feedback, zero safety calls, zero size-policy calls and
zero acquisition calls. Its public tool schema and description exactly match the
recorded 016 first request. Both changed runtime files are covered by the source
fingerprint. New helper/tests pass Black and Ruff checks; the furniture file's
16 existing lint diagnostics are unchanged, with no additions. Both published
Bash commands pass syntax checks. No model, GPU or native scene-generation
validation was performed locally; 017 is that integration check.

## Next experiment: 017, repeat only the failed long-living-room task

Confirm 016 is inactive before manually synchronizing source. The check below
also refuses a still-locked 016 source before starting another model job. Use a
new run and cold memory directory. The canonical scene still runs through
manipuland, with MemoryWriter at its normal completed-scene position. The raw
furniture candidates retain their own labels before canonical repair.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

previous_lock=outputs/slow_memory/qwen38_initial_pairs_long_living_016/runs.lock
if [[ -f "$previous_lock" ]]; then
  (
    exec 9<"$previous_lock"
    flock -n 9 || { echo '016 is still active; wait before starting 017.' >&2; exit 2; }
  )
fi

RUN_ID=qwen38_initial_pairs_long_living_017 \
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

Acceptance criteria:

- One valid group and two complete candidates, with raw scoring proofs, matching
  effective inputs and valid source/asset hashes. Empty argument lists must no
  longer become the observed unhandled-tool exception. A corrected call must
  remain subject to the existing real-acquisition budget.
- Complete canonical generation, terminal trace and exit records, with
  `generation_exit=0`. Check the four finalization markers listed above, even if
  the ACP UI reports normal completion. If they are still missing, preserve the
  ACP platform job log as well as the review archive before another GPU run.
- At least one eligible preference and `pair_audit_exit=0` meet the yield goal.
  If execution integrity passes but `status=completed_no_pairs`, pair exit 2 is a
  zero-yield observation, not a reason to alter labels. A missing/failed candidate
  is not a zero-contrast outcome. Review 017 before increasing collection volume
  or attempting training.

## Lightweight packaging

Run separately after ACP ends, including after a nonzero exit. The package can
also retain an interrupted run for diagnosis without claiming it is complete.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_long_living_017 \
PACKAGE_MAX_FILE_MIB=32 \
PACKAGE_MAX_TOTAL_MIB=512 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
```

Download the printed `.tar.gz` and `.tar.gz.sha256` under `tmp/downloads/`. The
archive preserves original project-relative hierarchy, includes diagnostics,
configuration, metrics, traces, pair/audit records and selected evidence media,
and omits models, large asset/mesh files, databases, replay copies and redundant
outputs. Limits apply to selected uncompressed files. The manifest records all
omissions, truncation, hashes and finalization-marker availability. Keep the full
server results for later replay/auditing/training preparation.

## Estimated time

| Phase | Elapsed estimate | Operator involvement |
| --- | --- | --- |
| Verify inactivity, manually synchronize and launch | 3-8 min | hands-on |
| Services and shared floor plan | 8-15 min | unattended |
| Furniture A/B and canonical furniture continuation | 30-60 min | unattended |
| Remaining stages and normal finalization | 45-100 min | unattended |
| Automatic audit and export | 2-10 min | unattended |
| Packaging | 2-5 min | launch, then unattended |
| Download and handoff | 2-10 min | network dependent |

Budget **1.5-3.5 hours**, about **5-15 minutes hands-on**; reserve a four-hour
availability window where practical. No timeout setting is inferred from ACP's
completion display. The basis is 016's observed 14:52 startup, furniture rendering
by 15:04, A evidence failure at 15:13:56, wall rendering at 15:35, and last retained
log at 15:37:36 (roughly 45 minutes, not a completed duration). A successful B
adds work absent from 016. 014 previously needed at least 70 minutes without
finishing the canonical scene. Manipuland, acquisition latency and repair work
remain the main uncertainties, so these estimates are not guaranteed bounds.
