# Qwen3.8 furniture initial-pair pilot

Implementation: 2026-09-14. Local contract tests pass; native Linux/Drake/GPU
execution must be validated by the ACP pilot below. Smoke 005 need not be rerun.

## What this entrypoint executes

`tmp/acp/acp_qwen38_initial_pairs.sh` adds one isolated furniture Designer
execution to each canonical Full scene. Default scope is two single-room legacy
development tasks, `default_bedroom` and `default_living_room`. The legacy registry
uses these names, not SceneEval numeric IDs. Maximum scope is four decision groups;
scene parallelism must stay at one.

At the first furniture `request_initial_design` boundary, the SceneExpert adapter:

1. Requires an unfurnished state, first stage attempt, pinned Qwen alias, and empty
   Designer/Critic histories. It records the resolved agent configuration, clean
   Git revision, complete serialized room/house state, assets, scene attributes,
   actual instruction, system prompt, tool declarations, safety-controller state,
   placement-noise profile, renderer caches/counters, and local Python/NumPy RNG.
   Live SQLite handles are not copied; verified-empty candidate sessions are fresh.
2. Executes A using the original native Runner. It captures the actual first
   serialized HTTP request after SDK input filters, including model settings,
   messages, images and tool schemas. Headers are not captured.
3. Saves and scores A's **raw post-Runner state before native end-of-call safety**.
   It then allows the native safety transaction to finish and separately journals
   the returned state and safety message.
4. Starts B in a separate process from a physical copy of the input snapshot.
   B reconstructs its own sessions, tools, renderer and collision server. It
   restores the frozen prompt, injected memory, noise, caches and safety state;
   scene, tools and settings must match. Shared retrieval/model services remain
   shared services; writable scene assets, session files and output directories
   are private. B does not invoke the outer pipeline, Planner, or MemoryWriter.
5. Independently scores B's raw state by the same procedure, journals its native
   safety result, and closes its private servers. A always continues the main
   scene regardless of which candidate scores better. No best-of-two policy is
   introduced into online evaluation.
6. Verifies that B left A's serialized state, scene asset bytes and active public
   memory bank unchanged. Canonical MemoryWriter runs at its normal scene-final
   position; later scenes can therefore benefit from updated Fast Memory.

The only shared-code seam is the optional adapter around the initial Designer
call and its SDK client setup. Native critic, repair, designer tools and prompts
are not redesigned. With the collection environment variable absent, the native
execution and default HTTP client remain unchanged.

## What labels mean

The pilot uses the existing read-only Main deterministic room evaluator on each
raw furniture result. A report must contain effective scored checks and no unknown
checks. Zero failures means accepted; one or more failures means rejected.
This first pilot measures deterministic execution quality, not learned visual or
style preference. Native end-of-call corrections/rollback are recorded separately;
their repaired score never labels the raw candidate.

The exporter retains its existing exact-context and authoritative-evidence rules.
The complete initial snapshot hash is part of the pairing context. A/B first HTTP
request bodies must be identical; different output trajectories are expected.
Missing, failed, truncated or corrupt candidates are quarantined. A missing B is
never a synthetic negative. Ties, two rejected candidates, and two candidates
without an eligible preference margin do not create a fabricated pair.

One exported pair proves the collection/export mechanism works. It does not
establish training readiness, sample efficiency, visual quality, or publishable
improvement. No training is launched by this script.

## Run on ACP

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
git pull --ff-only origin dev_lwz_maintain

RUN_ID=qwen38_initial_pairs_006 \
PAIR_GROUPS=2 \
CASE_SET=legacy8 \
SCENE_SELECTION=default_bedroom,default_living_room \
CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE=false \
MODEL_DIR=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF \
MODEL=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf \
MMPROJ=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf \
bash tmp/acp/acp_qwen38_initial_pairs.sh
```

Use a new RUN_ID if this one exists. The launcher rejects overwrites. It reuses
the validated Full startup/server orchestration. It generates fresh floor plans,
then runs normal full canonical scenes with the extra furniture B at each supported
boundary. This is not a two-call-only GPU workload.

Optional: `SCENEEXPERT_PAIR_TIMEOUT=3600` bounds each B worker. Failed workers
produce diagnostics; cancellation attempts cleanup and then terminates the private
process group. An interrupted B directory is retained and never silently reused;
use a fresh run/group to retry. Auditing finished artifacts is repeatable.
Snapshots are limited to 2 GiB and 20,000 files per copied tree; unsupported links
and oversized snapshots are rejected rather than partially replayed.

## Acceptance and returned artifacts

Results live under `outputs/slow_memory/$RUN_ID/runs/paired_initial/`:

- `pair_audit.json`: must have `gate_passed=true`, two valid groups, four candidate
  records, no errors, and `eligible_pair_count >= 1`.
- `dpo/manifest.json`, `all.jsonl`, `images/`, and diagnostics: portable pair export.
- `group_*/snapshot.json`, `input_scene/`: common initial state and input assets.
- `group_*/A/` and `B/`: first HTTP request, raw state/assets, independent Main
  report, full candidate trajectory/media, native safety message and returned state.
- `group_*/continuation_proof.json`: unchanged canonical state/assets/memory hashes.
- `group_*/status.json`, `shadow.log`, and optional `B/failure.json`: failure phase
  and subprocess diagnostics.
- `exit_status.env`, `pair_manifest.env`, `entrypoint.sh`: execution/audit results.

Also return the ordinary `runs/collection/`, `runs/metrics/`, runtime model/server
logs and canonical scene traces. The observer audit excludes paired snapshots and
candidate copies, so they cannot inflate completed-scene counts. The observer can
succeed with zero pairs; this new wrapper's final pair gate cannot.

If both candidates execute but no eligible pair is exported, the wrapper exits 2.
Inspect the candidate reports and DPO rejection reasons before changing anything.
Do not lower the pair gate or turn infrastructure failures into rejected answers.

## Next experiment decisions

1. Run the two-group pilot above; inspect the first exported tool trajectories,
   images, state hashes and raw Main report for one accepted/rejected pair.
2. If isolation and pairing pass, run four fresh furniture initial groups with a
   new RUN_ID and four explicit legacy task names. Measure valid-group rate,
   eligible-pair yield, failure causes and cost per valid pair. These are development
   tasks, not held-out evaluation results.
3. Only then add verified restoration for repair/nonempty histories and additional
   stages. A 12-group multistage run requires that extension; changing this pilot's
   `PAIR_GROUPS` alone cannot provide it. Add independently isolated VLM judging
   before claiming visual/style preference coverage.
4. Keep training separate. Record the actual compatible Qwen safetensors base and
   processor revision, deduplicate task families, establish held-out splits and run
   the existing training preflight before DPO/QLoRA experiments.

## Local validation

```bash
python -m pytest --confcutdir=tests/unit \
  tests/unit/test_initial_pairs.py \
  tests/unit/test_initial_pair_execution.py \
  tests/unit/test_slow_memory_collection.py \
  tests/unit/test_slow_memory_collection_launcher.py -q
bash -n tmp/acp/acp_qwen38_initial_pairs.sh
```

These tests cover proof tampering, files/media, native seam ordering with simulator
doubles, an actual OpenAI HTTP client with mocked transport, and shell orchestration
with stub workers. They do not substitute for the Linux/Drake/Qwen execution above.
