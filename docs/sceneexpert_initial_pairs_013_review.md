# Calibration 013 review and bounded 014 collection plan

## 013 passed its calibration acceptance criteria

Reviewed package:
`tmp/results/slow_memory/qwen38_initial_pairs_011_calibrated_013_review_20260915_152332/qwen38_initial_pairs_011_calibrated_013_review/`.
All 145 included-file SHA256 values match. Selected file bytes total 13,092,993;
the 15 intentional omissions are nine replay-asset directories and six redundant
candidate payloads. There are no packaging warnings or size-limit truncations.

The server completed at `2026-09-15T15:19:42+00:00` with exit 0, zero model calls,
`operation_succeeded=true`, and `outcome=completed_no_pairs`. Groups 000/002/003
are valid; group 001 remains explicitly excluded because its B result is missing.
This matches the 013 command and its declared calibration objective. Neither
013 nor its predecessor needs to be repeated.

| Task | A score | B score | Calibrated result |
| --- | --- | --- | --- |
| Living room | 1.00000 | 1.00000 | both accepted; explicit floor support restored |
| Study | 1.00000 | 1.00000 | both accepted; tucked seating alignment restored |
| Dining room | 0.97143 | 1.00000 | both accepted; no contract changes |

All six candidates have zero failed checks, zero unknown emitted checks, and zero
freshly measured collisions. The retained raw-state, restoration, physics,
calibration, report and trajectory hashes passed local consistency checks. Actual
first requests match within all three groups, and the three continuation proofs
show unchanged canonical state, assets and memory. Full native asset trees remain
on the server; this review did not independently rerun Drake over omitted meshes.

## Why there are still no preference pairs

All six trajectories use `unsloth/Qwen3.8-27B-GGUF`, `designer_initial`, and
`evidence.kind=deterministic`. The current exporter requires either an accepted
versus rejected outcome or two independently Critic-scored accepted responses.
It does not turn one passing deterministic response into a rejected response just
because another passing response has a higher aggregate score.

Therefore the direct exclusion is `no_eligible_preference_contrast` in each of
the three groups. The dining-room difference of approximately 0.02857 is also below
the configured 0.05 quality margin, but the evidence-kind/outcome gate rejects
these records before a margin comparison. Lowering the margin would not resolve
this result. Relabeling deterministic evidence as Critic evidence would be invalid.

Loading the six packaged trajectories and applying the unchanged pairing code
locally reproduces zero pairs and the same three exclusions without load errors.
All training splits remain empty. This establishes the calibration/replay chain;
it establishes neither training readiness nor the effectiveness of Slow Memory.

## Remaining evidence limitation

The server still reports missing `functional_partners_index.json`,
`nonartic_clearance_index.json` and `artic_clearance_index.json` under the native
interaction-clearance metric's data directory. The native loader substitutes empty
indexes, so asset-specific clearance coverage is incomplete. Zero unknown emitted
checks does not prove that all applicable checks were generated.

This warning did not stop 013 and is not a reason to repeat identical candidate
rescoring. Keep the next pilot's interpretation limited to observed constraints
and recorded physics. Restore and validate the real clearance data before claiming
complete interaction coverage or using this configuration as the final paper
evaluation. Do not invent index entries or treat the existing scores as complete
scene-quality certification. No other group's evaluator or dataset is changed here.

## Small implementation improvement for the next collection

Previously, a candidate could fail the transport/unhandled-tool guard before its
trace was saved. This left a generic error and made the missing meeting-room B
difficult to diagnose. SceneExpert now writes `tool_execution_failure.json` before
quarantining such A or B candidates. It records the exact matched phrase, call ID,
result index, full-output hash/length and a bounded 4,096-character excerpt with
explicit offsets and truncation metadata. Fatal asset failures are recorded too.

The detector and its eligibility decision are unchanged. A transport failure is
still excluded, and a recoverable placement rejection remains normal tool evidence.
The lightweight packer includes the new diagnostic. This does not reconstruct the
already lost 011 B trace or claim to fix its unknown underlying execution failure.

Validation: 102 focused tests passed, covering the runtime capture seam for both
candidate sides, bounded excerpts, ordinary placement rejections, existing audit
and export gates, launcher exit behavior and packaging. Initial collection and
packaging launchers also pass Bash syntax checking. No new GPU run was executed
on the local review machine.

## 014: two complex initial contexts, one bounded preference-yield probe

Repair-pair collection remains the next interface-development priority. Its
nonempty sessions, decision-time tool state and private replay still need an
implemented and tested entrypoint. The existing initial collector cannot capture
repair merely by changing an environment flag.

To continue collection using the now-validated entrypoint, run one two-group
probe on different, more densely constrained tasks already present in `new3`:

- `office`: four separate desk/chair workstations, exact one-to-one relationships,
  shared aisle and access requirements.
- `long_living_room`: distinct living/dining zones, five-chair edge distribution,
  media alignment, four distinct-corner plants and circulation.

These tasks are selected from their prompt constraints, not an observed winning
label. Record them as development/collection tasks, not independent held-out paper
tests. Keep the model, sampling settings, scoring rules, pairing margin and two
candidates per context unchanged. Writer continues at normal completed-scene
boundaries; the sibling execution stays isolated as in the validated collector.
The expected benefit is a better chance of observing a genuine accepted/rejected
contrast, not a promise of a pair. Do not expand to a large batch yet.

After manually synchronizing the code, run:

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_new3_014 \
PAIR_GROUPS=2 \
CASE_SET=new3 \
SCENE_SELECTION=office,long_living_room \
DIFFICULTY_SELECTION=all \
ACP_PARALLELISM=1 \
CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE=false \
MODEL_DIR=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF \
MODEL=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf \
MMPROJ=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf \
bash tmp/acp/acp_qwen38_initial_pairs.sh
```

The target is two valid groups/four candidates and at least one independently
supported preference pair. Audit first-request equality, immutable context and
continuation, candidate-specific fresh physics/calibration and causal tool effects.
Even one eligible pair only demonstrates pilot yield; it is insufficient for the
full training coverage and leakage gates.

The collection launcher's exit policy differs intentionally from 013 rescoring:
it requires at least one eligible pair. If all executions are valid but no pair
exists, expect `completed_no_pairs` and exit 2. This is a yield failure, not by
itself an execution crash. If a group fails, inspect its status, shadow log and
new tool-failure diagnostic; never use the failure as a negative training example.

**Stopping rule:** if this bounded probe again produces no valid contrast, stop
repeating initial-pair generation and prioritize the repair-pair implementation.
If a contrast exists, inspect the two actual executions for label correctness
before deciding expansion. No automatic expansion or training is requested here.

Run the matching packaging command separately, including after exit 2:

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_new3_014 \
PACKAGE_MAX_FILE_MIB=32 \
PACKAGE_MAX_TOTAL_MIB=512 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
```

Download the printed `.tar.gz` and `.tar.gz.sha256` from `tmp/downloads/`. The
archive preserves project-relative directories and records checksums, omissions
and truncation. Logs, configuration, traces, pairing reports and bounded evidence
media are included; models, meshes, databases and replay copies remain on the
server. The selected-file limits are 32 MiB each and 512 MiB total. Originals stay
untouched, and the review package does not replace full server-side training data.
