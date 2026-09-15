# Source-bound spatial methods

Fast Memory adds advisory context; it does not alter the native stage order,
designer/critic decisions, asset validation, or scoring. A source-backed method
is a **transfer hypothesis**, not proof of improvement on another scene.

## Stored contract

New spatial records use `placement_experience.schema_version = placement-experience.v2`
inside the existing memory v3 JSONL format. The bank layout is unchanged.

| Field | Meaning |
| --- | --- |
| `episodes` | Immutable source object-pair geometry, actions, native checks and source references. Numeric observations remain here. |
| `critic_advice` | Exact whole source-report excerpts, report/quote hashes, trace, stage-entry index, unambiguous snapshot fingerprint, and mentioned object roles. Opinions are not deterministic spatial verdicts. |
| `method_steps[].instruction` | How to adapt/recompute placement using current assets. No source constants, IDs-only summaries, or claims of guaranteed improvement. |
| `method_steps[].episode_ids` / `critic_refs` | Sources for this specific step, selected only from evidence visible to the Writer. |
| `method_steps[].evidence_kinds` | Source observation, critic advice, or both. |
| `method_steps[].verification_status` | `transfer_unverified`; source stage success never upgrades this to a demonstrated transfer benefit. |
| `method_steps[].relation_bindings` | Code-bound exact episode, subject/anchor instance IDs, metric and source value. Writer declarations cannot change IDs or measurements. |
| `method_steps[].relation_binding_version` | `1` for the additive pair/metric contract; `0` reads older records without inventing new evidence. |
| `source_context_episode_ids` | Same-stage provenance for critic-only methods, NOT method geometry or downstream observation targets. |
| `procedure`, `successful_pattern`, `positive_guidance` | One canonical, readable set of surviving instructions; raw unbound model summaries cannot bypass the source contract. |

For example, a bed-wall instruction must cite the bed-wall episode. A
bed-nightstand episode alone cannot support a bed-wall claim. A critic comment
about lighting may be cited as an opinion, but source fixture spacing does not
establish a universal minimum or prove non-overlapping light pools.

The binder checks source identity, visibility, stage/snapshot consistency,
explicit numeric/guarantee leakage and known object-role scope. This is not a
natural-language entailment prover. It does not infer unrecorded before/after
improvements, prove every proposed action occurred, or treat final AABB gaps as
walkable clearance. Quoted repair advice remains advice, not a verified repair.

Unusable steps are logged and excluded, not silently repaired by guessing a
source. A spatial candidate needs at least one substantive supported step, an
applicability description and the existing source-outcome eligibility checks.
Fewer/no records can therefore be a valid result when evidence is insufficient.
There is no reason to invent a second filler step. A purpose clause such as
"to ensure nightstands flank the bed" is not itself a universal guarantee.
Fixed numerical targets and explicit claims of guaranteed benefits remain gated.

Each newly proposed measured step declares `relations` with `episode_id`,
`subject_id`, `anchor_id`, and `metric`; source values are computed by Python.
`bbox_center_distance_m` describes bounding-box centers, not optical centers,
light-pool overlap, or navigable clearance. Old episode measurements/hashes are
unchanged. Distinct fixtures cannot be replaced by one fixture and two walls.
Only exact critic support may survive without a matching pair; it is explicitly
`critic_advice` only and produces no measured geometric success. No replacement
pair is silently guessed, even if a plausible pair exists elsewhere in the input.

## Retrieval and delivery

Both embedding text and the planner input use the canonical method. Source
observations and critic excerpts are labelled separately. The Writer packs
whole evidence units within its request budget; full audits remain on disk.

The planner may select a subset using `source_method_step_indices` (zero-based).
All episode references required by those steps must remain in
`source_relation_indices`. `null` means all steps; it is not automatic filtering.
Current object IDs are allowed. Source numeric thresholds must not be recast as
new task requirements. Named roles from selected critic excerpts may be bound to
current objects but do not create a measured historical object pair.

Accepted designer context contains the adapted actions, bindings and read-only
observation checks, not a second copy of raw historical instructions. Main's
quality criteria remain unchanged.
Declared method metrics must be represented in `advice_checks`; a center-distance
method cannot be certified by a wall gap or even by the same pair's AABB gap.
Critic-only steps may be delivered without fabricated observation targets; their
geometric result is `unknown` and never an automatic positive gain.

## Compatibility

- Existing v1 placement records and older banks remain readable without a rewrite.
- Legacy procedures lack explicit per-step provenance and remain labelled as such.
  Shared-reference legacy model responses are also labelled and lower-priority
  than valid explicitly bound methods; they are not retroactively certified.
- A v2 record with missing/broken step references is not injected. It cannot
  silently downgrade to the legacy path.
- Preserve existing banks and frozen experiment snapshots. Validate the new
  Writer in a fresh isolated directory before building another collection bank.

## Writer-only server acceptance

After synchronizing the code, run from the project root:

```bash
bash scripts/run_qwen38_memory_writer_replay.sh
```

The existing launcher starts/checks Qwen in the current CCI instance, reuses
batch_091 evidence, and writes a new isolated `tmp/writer_replay_091_*` directory.
It does not regenerate scenes, edit the source bank, or run training. Set
`SCENE_EXPERT_DIR` only when replaying another source scene. For an intentionally
managed existing model service, use `REUSE_EXISTING_MODEL_SERVICES=true`.

Check these files under the new replay directory:

- `replay_summary.json`: completed service/call/persistence status; do not equate
  model-call success or candidate count with records added to the bank.
- `audit/memory_writer_debug.json`: `result_status.placement_method_decisions`
  and `placement_candidate_decisions` explain each bound/rejected step/candidate.
  Inspect `accepted_episode_ids`, `relation_bindings`, `warnings`, and
  `evidence_scope` to distinguish actual method geometry from critic-only advice.
- `audit/memory_writer_prompt_*.json`: the exact request; every cited ID must
  have been visible in its corresponding request.
- `bank/success_cases.jsonl` / `failure_cases.jsonl`: inspect the fields above,
  source-only measurements and exact critic excerpts, not just `status=active`.

Local regression tests use synthetic fixtures and captured responses. They do
not constitute a new Qwen run. After live Writer acceptance, use a few held-out
tasks to confirm actual delivery, matching spatial observations, rework and full
cost. Only a frozen matched comparison can establish a transfer gain; `active`
means retrieval-eligible, not causally beneficial.
