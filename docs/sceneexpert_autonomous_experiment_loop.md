# SceneExpert autonomous experiment loop

The user authorized ongoing ACP submission, CCI result review, scoped code repairs, commits, remote synchronization and follow-up experiments on 2026-10-12. Use one persistent heartbeat for this conversation and one active SceneExpert GPU experiment at a time.

## Starting evidence

ACP 028 (`pt-7xj3dctv`) ended with collection exit 2 and package exit 0. Ten of its twelve selected tasks yielded complete, independent A/B groups. Including 027, the retained dataset contains twelve audited groups and twenty-four trajectories, with two direct-action training pairs and one validation pair. The review archive and server artifacts remain preserved.

SceneEval 477 failed because seven deterministic intent-extension checks had unresolved entity bindings. SceneEval 336 failed IntentCompiler schema validation because an inventory requirement lacked an inventory `subject_ref`. Two otherwise eligible pairs were excluded because the sampled B first turn contained TODO management without a direct design tool. These are distinct mechanisms; a later B action must not be relabeled as its first response.

The first direct-action smoke, `qwen38_dpo_direct_smoke_029` (`pt-gm0e8ehj`), failed before an optimizer step. Historical OpenAI tool arguments were strings while direct-action completions used mappings; the Qwen template requires mappings. Arrow mixed-type inference also changed nested schemas, and default JSON feature decoding changed floating-point render values. The loader now parses object arguments at the training boundary, preserves heterogeneous columns with explicit JSON storage, and uses standard-library decoding for exact values. Source JSONL files remain unchanged.

`qwen38_dpo_direct_smoke_030` (`pt-uks57dzt`) completed all two optimizer steps. The supervisor ran for 600 seconds, training took 315 seconds, peak CUDA allocation was 53.49 GiB, and all 512 saved adapter tensors were finite with 256 nonzero LoRA B tensors. Dataset checksums and runtime source hashes matched. The one validation pair had loss 0.668 and preference accuracy 1.0; this verifies infrastructure only, and the promotion gate correctly rejected deployment.

The four-task mechanism pilot uses existing SceneEval assignments for 336, 477, 340 and 273, retaining three train tasks and one validation task. Neither the reserved test split nor the old campaign artifacts are modified. Portable known assets mistakenly classified as architecture are returned to their owning inventory; specific folding/privacy/bamboo screen phrases use the shared room-divider identity, while electronic/projection screens retain their previous identities. A contradictory Main IntentCompiler instruction is corrected to require inventory subject_ref; this small shared-contract change addresses the observed schema blocker.

`qwen38_design_pairs_031` (`pt-mim9x5g2`) exited with collection code 2 before any model or scene process started. The generation launcher required a CUDA 13 directory in an obsolete `Task3.2` checkout. Export-only CPU preparation had checked serialization and campaign configuration, without exercising the native serving runtime. No A/B group, trajectory or preference pair was generated. The retained failure audit and verified final package distinguish this startup failure from scheduling wait and from low preference yield.

The launcher now resolves all three required CUDA 13 SONAMEs from this project's managed environments, validates explicit overrides without silent replacement, and checks both serving binaries with `ldd`. It exports the resolved directory to the embedding helper, which otherwise has its own stale default. `SCENEEXPERT_RUNTIME_PREFLIGHT_ONLY=true` exercises the actual generation entrypoint on CPU and exits before model startup or asset mirroring. The real CCI preflight selected `.venv_dpo`'s CUDA 13 libraries and resolved both llama-server binaries; 60 focused runtime, campaign and packaging tests passed. This is dependency evidence, not a GPU/model execution result. The retry uses a new frozen campaign `qwen38_design_pairs_032`; 031's source and zero-yield results remain untouched.

The default B sampling protocol remains a single native draw. This pilot explicitly freezes `sceneexpert.shadow_direct_action_sampling.v1` with a maximum of three draws from the exact same request. Every rejected response and every wire request is saved, no rejected tools execute, and selection depends only on first-response design scope. The first eligible draw is executed through the native SDK; if all draws are out of scope, the last draw is executed normally and remains subject to the existing export exclusion. No later action is relabeled as the first turn, no quality score influences sampling, and all existing isolation/physics/request gates remain active.

Acceptance requires four complete audited A/B groups, no systematic compiler/runtime failure, matching requests, preserved canonical assets/memory and candidate-bound fresh evidence. Preference yield is measured independently; at least one valid direct-action pair is the pilot target, not an invented guarantee. A zero-yield but intact batch is retained and analyzed before further sampling. After mechanism validation, continue bounded SceneEval batches toward the pilot training gate and reserved base/adapter scene evaluation.

032 produced three complete audited groups and one relative training preference, with no validation preference. Its fourth group was excluded after an HSSD cushion retrieval failed. The 033 continuation produced no new groups: SceneEval 129's floor-plan planner twice finished without calling Designer, while 127, 360 and 371 completed their floor plans. The launcher required every base to succeed before starting any furniture branch, stranding those three valid checkpoints; the campaign's zero-progress guard then stopped the remaining chunks. This exposed batch failure coupling rather than a new GPU dependency failure.

The collection-only `isolated` policy now audits the original assignment, completion status/marker, scene identity, readable layout, declared local geometry and SDFs. Only complete checkpoints continue, at their original case and scene indices; excluded cases and the original nonzero exit remain visible. The standard runner still defaults to strict behavior, and partial continuation requires one scene per native batch. The opt-in policy is frozen as `sceneexpert.collection_fault_isolation.v1`; no failed base becomes a preference. CPU tests exercise the actual Bash branch controller with mixed, empty, nominal-zero and failed branches. The real 033 checkpoints were inspected on CPU and selected exactly 127, 360 and 371, excluding 129 without changing their files. Eighty-two focused CCI regressions passed.

Because launcher source changed, the next pilot uses fresh campaign `qwen38_design_pairs_034`, covering four train tasks (360, 127, 129, 371) and four validation tasks (273, 201, 473, 353). The frozen SceneEval partition, memory seed and three-draw scope policy remain the same. The failed 033 source/results and both 032 review packages remain preserved. For a partial batch, a nonzero workflow exit remains correct even when valid pairs were collected; assess branch completeness and preference yield separately. After confirming task isolation on actual ACP execution, expand the same campaign without a mechanism filter.


## State and lifecycle

Maintain an atomic journal at `tmp/sceneexpert_loop/state.json` on CCI and mirror it locally. Record the phase, RUN_ID, exact ACP job ID/UID, submission time, command, source HEAD/fingerprint, dataset identity, terminal records, audit counts, package/checksum, next plan, and whether a submission is pending or verified. Runtime journal files are not repository source files.

Before submitting, inspect the ACP list for the exact planned display name and inspect the CCI run directory and journal. Reuse an existing task when it is already submitted. A submission timeout is an unknown outcome: refresh and find the existing job before any new submit click. Persist its identity promptly.

During an active task, only inspect its console and server records. Do not edit code used by that task, re-submit it, or start a second SceneExpert GPU task. Queueing and scheduling errors do not establish script failure. Keep routine progress quiet; report meaningful completion, failure, or required user action.

After the task truly ends, inspect workflow exits, supervisor finalization and actual artifacts. Collection requires complete A/B executions, request equality, unchanged canonical state/assets/memory, fresh candidate-bound scoring and valid media/trajectory hashes. Training requires completed optimizer steps, final manifest, adapter and finite metrics. ACP status alone does not satisfy these checks.

Run lightweight CPU audit and packaging on CCI. Source results remain intact. Final audit, failure-analysis and training-preflight summaries have priority over bulky logs and raw payloads, so the transfer cap cannot silently remove the current conclusion. Record omissions and verify each archive. Download a review package when helpful.

## Continuing decisions

1. Inspect 034's checkpoint isolation, A/B execution, sampling proofs and preference yield before expanding the frozen campaign. Diagnose the first causal failure if a mechanism remains broken.
2. Address task/intent contract generation and direct-action sampling scope with bounded, opt-in changes. Prefer SceneExpert wrappers and adapters. Modify another group's code only for a demonstrated blocker. Never replace missing evidence with invented labels or silently loosen request/isolation checks.
3. Freeze a new campaign identity after collection-code changes; an existing frozen source bundle cannot be resumed under changed code. Retain deterministic task partitions, frozen memory and all historical audited data. Merge only sources that passed full server-side pair validation, retaining their original task splits.
4. Validate a small batch before expansion. Continue bounded SceneEval batches toward the explicit pilot gate (at least eight independent train pairs and four validation pairs), then regular furniture-initial training (at least sixteen train pairs) and reserved scene-level base/adapter evaluation. Report preference yield and missing tasks, including zero-yield batches.
5. Each completed experiment produces a root-cause analysis, a concrete next command, its matching lightweight packaging command and staged runtime estimates. Evidence of method effectiveness requires a reserved paired scene evaluation; neither infrastructure smoke nor training loss is sufficient.

Source changes made on CCI must be committed in English on `dev_lwz_maintain`, mirrored to the local branch via an offline bundle when needed, pushed to `origin/dev_lwz_maintain`, and verified at matching HEADs with clean intended tracked files before the next experiment. Preserve unrelated user changes. Do not force-push. ACP commands contain no Git operations.

CCI autonomous execution is limited to one GPU and estimated elapsed time at most one hour. Longer or multi-GPU work uses ACP. For the eight-task 034 isolation pilot, estimate 5–10 minutes preparation, 2–4 hours execution plus queue, 10–20 minutes server audit and 2–10 minutes packaging/download. The references are 032's four tasks in about 51–52 minutes and 028's twelve tasks in about 4 hours 47 minutes with four-way concurrency; task complexity and AFS/rendering latency can widen this range. These estimates are not guarantees. No user submission work is required because the loop is authorized to submit.

## Next ACP command

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
RUN_ID=qwen38_design_pairs_034 \
MAX_TASKS=8 CHUNK_SIZE=4 ACP_PARALLELISM=4 \
CAMPAIGN_SCENEEVAL_IDS=360,127,129,371,273,201,473,353 \
SHADOW_SCOPE_SAMPLES=3 \
SHARED_BASE_FAILURE_POLICY=isolated \
MEMORY_SEED_DIR="$PWD/outputs/slow_memory/qwen38_design_pairs_027/frozen_memory/ablation_4c" \
PACKAGE_MAX_FILE_MIB=16 PACKAGE_MAX_TOTAL_MIB=128 \
bash tmp/acp/acp_qwen38_design_pairs.sh
```

After real finalization:

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
RUN_ID=qwen38_design_pairs_034 \
PACKAGE_PATH="$PWD/tmp/results/slow_memory/qwen38_design_pairs_034_review_final.tar.gz" \
PACKAGE_MAX_FILE_MIB=16 PACKAGE_MAX_TOTAL_MIB=128 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
.venv/bin/python scripts/package_sceneexpert_results.py \
  --verify tmp/results/slow_memory/qwen38_design_pairs_034_review_final.tar.gz
```
