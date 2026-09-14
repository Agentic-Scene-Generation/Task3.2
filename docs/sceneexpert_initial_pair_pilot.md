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
   Designer/Critic histories. It records the resolved agent configuration, source
   content fingerprint, complete serialized room/house state, assets, scene attributes,
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

Manually synchronize the latest development code to the server before running.
No Git executable, repository metadata, clean-worktree check or Git network access
is required. The launcher disables optional Git queries in canonical trace logging
and fingerprints source files, prompts, configurations, launchers and dependency
manifests. That fingerprint is pinned from startup through both candidates. Source
changes still fail the gate; output/log/cache changes do not.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
RUN_ID=qwen38_initial_pairs_007 \
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

The 006 launch reported `detected dubious ownership` from the old Git-dependent
preflight and exited before generation. This was a deployment assumption in the
collector, not a Qwen inference failure. Do not change global Git ownership
exceptions to run this experiment. Version 2 snapshots use `code_provenance` with
`identity_kind=source_bundle_sha256`; a digest is never mislabeled as a Git commit.
Older Git-only snapshots can still be audited but cannot be resumed under the new
content-identity contract. Use a fresh RUN_ID after synchronizing the fix.

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

The lightweight packaging command below selects the ordinary `runs/collection/`,
`runs/metrics/`, runtime model/server logs, canonical scene traces and pair evidence
for download. Full assets remain on the server. The observer audit excludes paired
snapshots and candidate copies, so they cannot inflate completed-scene counts. The
observer can succeed with zero pairs; this new wrapper's final pair gate cannot.

If both candidates execute but no eligible pair is exported, the wrapper exits 2.
Inspect the candidate reports and DPO rejection reasons before changing anything.
Do not lower the pair gate or turn infrastructure failures into rejected answers.

## Lightweight results download (success or failure)

After the ACP process has stopped, run the matching packaging script. It does not
start models, change experiment verdicts, use Git or modify source results. Python
3.11+ with the standard library suffices, even if the GPU environment is broken.
If 007 is already running, wait for it to stop before synchronizing any new source
files: the active run pins its source fingerprint. Packaging needs no rerun.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
RUN_ID=qwen38_initial_pairs_007 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
```

Download only the generated `tmp/downloads/<RUN_ID>_review_<timestamp>.tar.gz` and
its `.sha256` sidecar. The script prints both exact paths and the compressed size.
It verifies every archived file before returning success. Existing output names
are never overwritten. After extraction the layout is:

```text
<RUN_ID>_review/
  outputs/slow_memory/<RUN_ID>/    # Original project-relative paths
    runs/paired_initial/          # Audits, reports, trajectories, evidence media
    runs/collection/              # Collection diagnostics
    runs/metrics/                 # Metrics and summaries
    memory/                       # Fast Memory records
  tmp/acp_logs/<RUN_ID>/          # Runtime/model/service logs and run configuration
  _package/
    manifest.json                # Included SHA256 hashes and omitted files/subtrees
    README.txt
    log_excerpts/                # Explicit derivatives, only for oversized logs
```

Defaults: at most 32 MiB per original file and 512 MiB total selected, uncompressed
file content. Tar headers and package metadata are additional; gzip determines the
actual download size. Structured evidence is selected first, then logs, evidence
images, and optional raw debug payloads. Ordinary included files retain their exact
bytes. Slow Memory `media/` and DPO `images/` are retained within these limits.

Models, meshes, SQLite databases, intermediate render textures, caches and duplicate
`input_scene`/`raw_scene`/private `scene` asset trees stay on the server. Byte-identical
`latest-run` copies are omitted only when the corresponding `hydra` file is already
included; symlinks are never followed. Oversized logs receive labelled head/tail
excerpts with original byte ranges under `_package/log_excerpts/`; the original
log path is listed as omitted. Other oversized evidence and files exceeding the
total budget are listed explicitly, with warnings. Check those warnings before
assuming all evidence images or reports are present.

For a custom result location or limits, set `COLLECTION_ROOT`, `ACP_LOG_DIR`,
`PACKAGE_PATH`, `PACKAGE_MAX_FILE_MIB` and/or `PACKAGE_MAX_TOTAL_MIB` on the same
command. Source roots must be inside the project, and the archive must be outside
those roots. Missing one source root is reported but does not prevent packing the
other. If a startup failure occurred before project logging began, save the ACP
platform's exported console log into `tmp/acp_logs/<RUN_ID>/` before packing; the
script cannot recover console output that was never saved.

Downloaded archive integrity can also be checked locally without extraction:

```bash
python scripts/package_sceneexpert_results.py --verify /path/to/downloaded.tar.gz
```

This check verifies transfer/package integrity only. Copied `pair_audit.json` and
other experiment reports keep the server's results, and are not recomputed by the
packer. Full pair audits, scene replay and training still require the complete
server artifacts. Omitted files must never be treated as evidence of a failed
candidate or a passing experiment.

## Next experiment decisions

1. Run the two-group pilot above; inspect the first exported tool trajectories,
   images, state hashes and raw Main report for one accepted/rejected pair. Package
   the stopped run using the matching command above, even if execution failed.
2. If isolation and pairing pass, run four fresh furniture initial groups with a
   new RUN_ID and four explicit legacy task names. Measure valid-group rate,
   eligible-pair yield, failure causes and cost per valid pair. These are development
   tasks, not held-out evaluation results.
   Use the same packaging script with that run's new RUN_ID after it stops.
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
  tests/unit/test_pair_code_provenance.py \
  tests/unit/test_initial_pairs.py \
  tests/unit/test_initial_pair_execution.py \
  tests/unit/test_slow_memory_collection.py \
  tests/unit/test_slow_memory_collection_launcher.py \
  tests/unit/test_trace_logger.py -q
bash -n tmp/acp/acp_qwen38_initial_pairs.sh
python -m pytest --confcutdir=tests/unit tests/unit/test_review_package.py -q
bash -n tmp/acp/pack_qwen38_initial_pairs.sh
```

These tests cover proof tampering, files/media, native seam ordering with simulator
doubles, an actual OpenAI HTTP client with mocked transport, and shell orchestration
with stub workers. The code-identity regressions cover absent/unusable Git, manual
copies, Windows/Linux newlines, ignored outputs and rejected source changes. They
do not substitute for the Linux/Drake/Qwen execution above.
Packaging regressions additionally exercise original paths/bytes, relevant images,
bounded log excerpts, quotas, missing roots, symlink exclusion, duplicate copies,
source mutation, archive integrity and a standard-library-only CLI/Bash entrypoint.
