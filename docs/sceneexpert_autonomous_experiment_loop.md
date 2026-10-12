# SceneExpert autonomous experiment loop

The user authorized ongoing ACP submission, CCI result review, scoped code repairs, commits, remote synchronization and follow-up experiments on 2026-10-12. Use one persistent heartbeat for this conversation and one active SceneExpert GPU experiment at a time.

## Starting evidence

ACP 028 (`pt-7xj3dctv`) ended with collection exit 2 and package exit 0. Ten of its twelve selected tasks yielded complete, independent A/B groups. Including 027, the retained dataset contains twelve audited groups and twenty-four trajectories, with two direct-action training pairs and one validation pair. The review archive and server artifacts remain preserved.

SceneEval 477 failed because seven deterministic intent-extension checks had unresolved entity bindings. SceneEval 336 failed IntentCompiler schema validation because an inventory requirement lacked an inventory `subject_ref`. Two otherwise eligible pairs were excluded because the sampled B first turn contained TODO management without a direct design tool. These are distinct mechanisms; a later B action must not be relabeled as its first response.

The first direct-action smoke, `qwen38_dpo_direct_smoke_029` (`pt-gm0e8ehj`), failed before an optimizer step. Historical OpenAI tool arguments were strings while direct-action completions used mappings; the Qwen template requires mappings. Arrow mixed-type inference also changed nested schemas, and default JSON feature decoding changed floating-point render values. The loader now parses object arguments at the training boundary, preserves heterogeneous columns with explicit JSON storage, and uses standard-library decoding for exact values. Source JSONL files remain unchanged.

`qwen38_dpo_direct_smoke_030` (`pt-uks57dzt`) completed all two optimizer steps. The supervisor ran for 600 seconds, training took 315 seconds, peak CUDA allocation was 53.49 GiB, and all 512 saved adapter tensors were finite with 256 nonzero LoRA B tensors. Dataset checksums and runtime source hashes matched. The one validation pair had loss 0.668 and preference accuracy 1.0; this verifies infrastructure only, and the promotion gate correctly rejected deployment.

The next experiment is the new frozen campaign `qwen38_design_pairs_031`. A four-task mechanism pilot uses existing SceneEval assignments for 336, 477, 340 and 273, retaining three train tasks and one validation task. Neither the reserved test split nor the old campaign artifacts are modified. Portable known assets mistakenly classified as architecture are returned to their owning inventory; specific folding/privacy/bamboo screen phrases use the shared room-divider identity, while electronic/projection screens retain their previous identities. A contradictory Main IntentCompiler instruction is corrected to require inventory subject_ref; this small shared-contract change addresses the observed schema blocker.

The default B sampling protocol remains a single native draw. This pilot explicitly freezes `sceneexpert.shadow_direct_action_sampling.v1` with a maximum of three draws from the exact same request. Every rejected response and every wire request is saved, no rejected tools execute, and selection depends only on first-response design scope. The first eligible draw is executed through the native SDK; if all draws are out of scope, the last draw is executed normally and remains subject to the existing export exclusion. No later action is relabeled as the first turn, no quality score influences sampling, and all existing isolation/physics/request gates remain active.

Acceptance requires four complete audited A/B groups, no systematic compiler/runtime failure, matching requests, preserved canonical assets/memory and candidate-bound fresh evidence. Preference yield is measured independently; at least one valid direct-action pair is the pilot target, not an invented guarantee. A zero-yield but intact batch is retained and analyzed before further sampling. After mechanism validation, continue bounded SceneEval batches toward the pilot training gate and reserved base/adapter scene evaluation.


## State and lifecycle

Maintain an atomic journal at `tmp/sceneexpert_loop/state.json` on CCI and mirror it locally. Record the phase, RUN_ID, exact ACP job ID/UID, submission time, command, source HEAD/fingerprint, dataset identity, terminal records, audit counts, package/checksum, next plan, and whether a submission is pending or verified. Runtime journal files are not repository source files.

Before submitting, inspect the ACP list for the exact planned display name and inspect the CCI run directory and journal. Reuse an existing task when it is already submitted. A submission timeout is an unknown outcome: refresh and find the existing job before any new submit click. Persist its identity promptly.

During an active task, only inspect its console and server records. Do not edit code used by that task, re-submit it, or start a second SceneExpert GPU task. Queueing and scheduling errors do not establish script failure. Keep routine progress quiet; report meaningful completion, failure, or required user action.

After the task truly ends, inspect workflow exits, supervisor finalization and actual artifacts. Collection requires complete A/B executions, request equality, unchanged canonical state/assets/memory, fresh candidate-bound scoring and valid media/trajectory hashes. Training requires completed optimizer steps, final manifest, adapter and finite metrics. ACP status alone does not satisfy these checks.

Run lightweight CPU audit and packaging on CCI. Source results remain intact. Final audit, failure-analysis and training-preflight summaries have priority over bulky logs and raw payloads, so the transfer cap cannot silently remove the current conclusion. Record omissions and verify each archive. Download a review package when helpful.

## Continuing decisions

1. Inspect 031's execution, sampling proofs and preference yield before expanding the frozen campaign. Diagnose the first causal failure if a mechanism remains broken.
2. Address task/intent contract generation and direct-action sampling scope with bounded, opt-in changes. Prefer SceneExpert wrappers and adapters. Modify another group's code only for a demonstrated blocker. Never replace missing evidence with invented labels or silently loosen request/isolation checks.
3. Freeze a new campaign identity after collection-code changes; an existing frozen source bundle cannot be resumed under changed code. Retain deterministic task partitions, frozen memory and all historical audited data. Merge only sources that passed full server-side pair validation, retaining their original task splits.
4. Validate a small batch before expansion. Continue bounded SceneEval batches toward the explicit pilot gate (at least eight independent train pairs and four validation pairs), then regular furniture-initial training (at least sixteen train pairs) and reserved scene-level base/adapter evaluation. Report preference yield and missing tasks, including zero-yield batches.
5. Each completed experiment produces a root-cause analysis, a concrete next command, its matching lightweight packaging command and staged runtime estimates. Evidence of method effectiveness requires a reserved paired scene evaluation; neither infrastructure smoke nor training loss is sufficient.

Source changes made on CCI must be committed in English on `dev_lwz_maintain`, mirrored to the local branch via an offline bundle when needed, pushed to `origin/dev_lwz_maintain`, and verified at matching HEADs with clean intended tracked files before the next experiment. Preserve unrelated user changes. Do not force-push. ACP commands contain no Git operations.

CCI autonomous execution is limited to one GPU and estimated elapsed time at most one hour. Longer or multi-GPU work uses ACP. For the four-task 031 pilot, estimate 5–10 minutes preparation, 60–120 minutes execution plus queue, 5–15 minutes server audit and 1–5 minutes packaging. The reference is 028's twelve tasks in about 4 hours 47 minutes with four-way concurrency; task complexity and AFS/rendering latency can widen this range. These estimates are not guarantees. The user has no required hands-on submission work because the loop is authorized to submit; manual review would take roughly 5–15 minutes.

## Next ACP command

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
RUN_ID=qwen38_design_pairs_031 \
MAX_TASKS=4 CHUNK_SIZE=4 ACP_PARALLELISM=4 \
CAMPAIGN_SCENEEVAL_IDS=336,477,340,273 \
SHADOW_SCOPE_SAMPLES=3 \
MEMORY_SEED_DIR="$PWD/outputs/slow_memory/qwen38_design_pairs_027/frozen_memory/ablation_4c" \
PACKAGE_MAX_FILE_MIB=16 PACKAGE_MAX_TOTAL_MIB=128 \
bash tmp/acp/acp_qwen38_design_pairs.sh
```

After real finalization:

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
RUN_ID=qwen38_design_pairs_031 \
PACKAGE_PATH="$PWD/tmp/results/slow_memory/qwen38_design_pairs_031_review_final.tar.gz" \
PACKAGE_MAX_FILE_MIB=16 PACKAGE_MAX_TOTAL_MIB=128 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
.venv/bin/python scripts/package_sceneexpert_results.py \
  --verify tmp/results/slow_memory/qwen38_design_pairs_031_review_final.tar.gz
```
