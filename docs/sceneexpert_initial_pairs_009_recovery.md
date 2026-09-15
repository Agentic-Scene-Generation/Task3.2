# Initial-pair rescore 009: native restoration fix and run 010

Update: 010 completed successfully with two accepted candidates and no eligible
preference contrast. Follow [the 011 collection plan](sceneexpert_initial_pairs_010_review.md);
do not rerun the historical 010 command below. Rescore-only commands now return
operation success separately from the dataset gate, unless `REQUIRE_PAIRS=true`.

## Confirmed failure

The ACP log `logs-acp-20260915T164150.txt` terminates in `restore_raw_scene()`:
`ValueError: restored raw state differs; refuse offline relabeling`.
The preceding missing-clearance-index messages are warnings; they are not the
exception that stopped this run. No new training result is established by 009.

The previous restoration gate incorrectly required bit-identical JSON after a
native quaternion-to-matrix-to-quaternion round trip. Reproducing the actual
008/B `nightstand_0` transform with Drake 1.49.0 changes its last quaternion
component from `-0.025281540604135816` to `-0.02528154060413582`, a difference of
`3.469446951953614e-18`. This is conversion roundoff, not a moved object. Previous
portable tests replaced native restoration with a double and missed this boundary.

## Correction within SceneExpert

- Keep original `raw_state.json`, original hashes, trajectories and all original
  008/009 result directories intact. The retry writes a new run directory.
- Serialize the actual reconstructed scene separately as `restored_state.json`.
  `restoration_proof.json` binds original, relocated and reconstructed state
  hashes, explicit asset relocations, and each accepted quaternion difference.
- Only native object/support-surface quaternion fields permit unit-quaternion
  sign equivalence and a maximum component error of `64 * ulp(1.0)` (about
  `1.42e-14`). Translation, object identity, bounds, metadata, prompts, lists and
  other state fields still require exact serialized equality. There is no global
  float rounding, score relaxation or tolerance for physical repairs.
- Rebase typed absolute SDF/geometry paths into verified private `raw_scene`
  copies. Original absolute references previously pointed outside the private
  room directory. Missing physical assets and uncaptured external geometry fail.
- 008's captured file manifests lack five A and six B legacy relative asset
  preview-image paths. These unused references are preserved and listed in
  `unavailable_reference_images`; deterministic geometry scoring does not read
  them. Actual model-input and tool-output media keep their existing mandatory
  byte/hash checks. Missing training images are not exempted.
- Scoring protocol v3 binds fresh physics to the **actual reconstructed state**,
  and that state to the original candidate through the verified restoration proof.
  Export revalidates the chain. It does not pretend different serialized hashes
  are equal. Subsequent geometry edits or altered proofs invalidate the candidate.
- Per-candidate progress is logged. Failures identify candidate and phase;
  restoration mismatches also save bounded field paths and the reconstructed
  state for review, instead of returning an unexplained equality failure.

Designer, Critic rules, native physics code, repair policy, tool budgets and
MemoryWriter behavior are unchanged. No LLM calls are made by the rescore entry.

## Validation

Native tests run in an isolated Linux environment with real Drake 1.49.0, native
`RoomScene` / `RoomGeometry` serialization, native collision computation and Main
deterministic aggregation. They cover both collision-free and colliding primitive
scenes, original assets being unavailable, expected numerical roundoff, missing
or external physical assets, and rejection of subsequent geometry/proof changes.

A serialization-only probe also restored the complete original 008 A/B JSON:
A had zero quaternion differences; B had exactly the single difference above.
Both had two absolute asset relocations. That probe used placeholder files only
to exercise the codec; it did not recompute 008 physics or produce labels.
The real 008 asset meshes remain on the server and must be scored by run 010.

Validation results: 110 portable regression tests and four native Linux/Drake
tests passed. Ruff checks and Bash syntax checks passed. Both pytest environments
reported two pre-existing configuration warnings about unavailable asyncio options.

## Next ACP: retry from original 008, use a new output ID

Manually synchronize the latest code, including new `paired_restoration.py`.
Preserve failed 009 for diagnostics. The source is still the complete original
008 directory, not the incomplete 009 output or the lightweight review package.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_008_rescore_010 \
SOURCE_RUN_ID=qwen38_initial_pairs_008 \
bash tmp/acp/acp_qwen38_initial_pairs_rescore.sh
```

The existing scene Python environment with Drake is required; use `PYTHON_BIN`
only if it differs from the launcher's project/Main venv paths. No model server
or GPU generation is started. Existing output IDs are refused.

Inspect `runs/paired_initial/rescore_status.json` for `status=completed` and
`pair_audit.json` for `execution_integrity_passed=true`, `valid_group_count=1`,
`candidate_count=2`. Both candidates must have verified `restoration_proof.json`
and v3 `evaluation_proof.json` with fresh physics. Original and restored hashes
may differ only through that explicit verified chain.

Exit 0 additionally requires a real eligible preference pair. Exit 2 with
completed rescoring and valid evidence can correctly mean zero preference pairs;
two accepted candidates without an eligible relative ranking remain unpaired.
A traceback or failed status is an execution failure. After this evidence gate
passes, review pair yield before expanding to new furniture decision groups.

## Matching lightweight results package

After the process stops, run this command separately even if rescoring exited 2:

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_008_rescore_010 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
```

Download the printed `tmp/downloads/<RUN_ID>_review_<timestamp>.tar.gz` and
`.tar.gz.sha256`. The existing packer retains project-relative directory layout,
run configuration/logs, old/new reports, raw/restored JSON states, proofs,
trajectories and relevant evidence media. It omits meshes, model files, databases
and raw asset copies; limits remain 32 MiB per file and 512 MiB selected bytes.
The manifest records checksums, omissions and truncations. Keep the original
server asset directories intact: the review archive cannot replace full replay.
