# SceneExpert autonomous experiment loop

The user authorized ongoing ACP submission, CCI result review, scoped code repairs, commits, remote synchronization and follow-up experiments on 2026-10-12. Use one persistent heartbeat for this conversation and one active SceneExpert GPU experiment at a time.

## Starting evidence

ACP 028 (`pt-7xj3dctv`) ended with collection exit 2 and package exit 0. Ten of its twelve selected tasks yielded complete, independent A/B groups. Including 027, the retained dataset contains twelve audited groups and twenty-four trajectories, with two direct-action training pairs and one validation pair. The review archive and server artifacts remain preserved.

SceneEval 477 failed because seven deterministic intent-extension checks had unresolved entity bindings. SceneEval 336 failed IntentCompiler schema validation because an inventory requirement lacked an inventory `subject_ref`. Two otherwise eligible pairs were excluded because the sampled B first turn contained TODO management without a direct design tool. These are distinct mechanisms; a later B action must not be relabeled as its first response.

The first next experiment is `qwen38_dpo_direct_smoke_029`, using `PROFILE=pipeline_smoke` and the verified-relative dataset under `qwen38_design_pairs_027`. It runs two optimizer steps. Both the training-request validator and the actual training CLI CPU dry-run passed. This is an infrastructure check; two train pairs cannot establish effectiveness.

## State and lifecycle

Maintain an atomic journal at `tmp/sceneexpert_loop/state.json` on CCI and mirror it locally. Record the phase, RUN_ID, exact ACP job ID/UID, submission time, command, source HEAD/fingerprint, dataset identity, terminal records, audit counts, package/checksum, next plan, and whether a submission is pending or verified. Runtime journal files are not repository source files.

Before submitting, inspect the ACP list for the exact planned display name and inspect the CCI run directory and journal. Reuse an existing task when it is already submitted. A submission timeout is an unknown outcome: refresh and find the existing job before any new submit click. Persist its identity promptly.

During an active task, only inspect its console and server records. Do not edit code used by that task, re-submit it, or start a second SceneExpert GPU task. Queueing and scheduling errors do not establish script failure. Keep routine progress quiet; report meaningful completion, failure, or required user action.

After the task truly ends, inspect workflow exits, supervisor finalization and actual artifacts. Collection requires complete A/B executions, request equality, unchanged canonical state/assets/memory, fresh candidate-bound scoring and valid media/trajectory hashes. Training requires completed optimizer steps, final manifest, adapter and finite metrics. ACP status alone does not satisfy these checks.

Run lightweight CPU audit and packaging on CCI. Source results remain intact. Final audit, failure-analysis and training-preflight summaries have priority over bulky logs and raw payloads, so the transfer cap cannot silently remove the current conclusion. Record omissions and verify each archive. Download a review package when helpful.

## Continuing decisions

1. Inspect 029's finalization and training diagnostics. On success, proceed to a small collection validation with the intended action scope. On failure, diagnose the first causal failure before changing implementation.
2. Address task/intent contract generation and direct-action sampling scope with bounded, opt-in changes. Prefer SceneExpert wrappers and adapters. Modify another group's code only for a demonstrated blocker. Never replace missing evidence with invented labels or silently loosen request/isolation checks.
3. Freeze a new campaign identity after collection-code changes; an existing frozen source bundle cannot be resumed under changed code. Retain deterministic task partitions, frozen memory and all historical audited data. Merge only sources that passed full server-side pair validation, retaining their original task splits.
4. Validate a small batch before expansion. Continue bounded SceneEval batches toward the explicit pilot gate (at least eight independent train pairs and four validation pairs), then regular furniture-initial training (at least sixteen train pairs) and reserved scene-level base/adapter evaluation. Report preference yield and missing tasks, including zero-yield batches.
5. Each completed experiment produces a root-cause analysis, a concrete next command, its matching lightweight packaging command and staged runtime estimates. Evidence of method effectiveness requires a reserved paired scene evaluation; neither infrastructure smoke nor training loss is sufficient.

Source changes made on CCI must be committed in English on `dev_lwz_maintain`, mirrored to the local branch via an offline bundle when needed, pushed to `origin/dev_lwz_maintain`, and verified at matching HEADs with clean intended tracked files before the next experiment. Preserve unrelated user changes. Do not force-push. ACP commands contain no Git operations.

CCI autonomous execution is limited to one GPU and estimated elapsed time at most one hour. Longer or multi-GPU work uses ACP. Estimate 029 at 2–5 minutes preparation, 40–80 minutes single-H100 execution plus queue, 5–10 minutes validation and 1–5 minutes packaging. Operator involvement is about 5–15 minutes. These estimates include uncertainty from model loading, compilation and AFS storage.

## Next ACP command

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
CUDA_VISIBLE_DEVICES=0 \
CAMPAIGN_ID=qwen38_design_pairs_027 \
RUN_ID=qwen38_dpo_direct_smoke_029 \
PROFILE=pipeline_smoke \
BASE_MODEL=/mnt/afs/task3_2/share_model/Qwen/Qwen3.8-27B \
PACKAGE_MAX_FILE_MIB=16 PACKAGE_MAX_TOTAL_MIB=128 \
bash tmp/acp/acp_qwen38_dpo_train.sh
```

After actual finalization:

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
RUN_ID=qwen38_dpo_direct_smoke_029 \
PACKAGE_PATH="$PWD/tmp/results/slow_memory/qwen38_dpo_direct_smoke_029_review_final.tar.gz" \
PACKAGE_MAX_FILE_MIB=16 PACKAGE_MAX_TOTAL_MIB=128 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
.venv/bin/python scripts/package_sceneexpert_results.py \
  --verify tmp/results/slow_memory/qwen38_dpo_direct_smoke_029_review_final.tar.gz
```
