# Qwen3.8-only paired recollection

Status: proposed implementation and experiment protocol, 2026-09-11.
The decision-level collection runner described below is **not implemented** by
this document. Existing Full capture is an observer, not a paired sampler.

## Decision

Start a new Qwen3.8-only dataset. Do not make restoration of old Terra contexts a
prerequisite. Keep the Terra dataset separately for possible later research;
exclude its trajectories from this collection's export sources.

Keep generic model fallback configuration. The new collection entrypoint must
explicitly pin the served Qwen3.8 alias and record the model build, sampling
settings, code revision, and resolved configuration. A fallback used by unrelated
launchers is not a reason to rewrite them. Training will use the corresponding
open Qwen3.8 Transformers/safetensors checkpoint; record its actual path or Hub
revision and processor before training. A missing training path need not delay
collection once the inference model identity is recorded.

Keep Fast Memory learning enabled for the canonical online scene lifecycle.
Freeze actual model-visible input within each candidate group; allow later groups
to benefit from memory updates. Standard DPO compares two responses to the same
input. A response before a memory update and another response with different
injected memory are separate contexts, even if the second scene is better.

## Verified integration boundaries

- `scenesmith/agent_utils/base_stateful_agent.py`: both
  `_request_initial_design_impl` and `_request_design_change_impl` construct the
  prompt/context and call `Runner.run` with the persistent Designer session.
- `scenesmith/agent_utils/room.py`: `to_state_dict` and
  `restore_from_state_dict` serialize scene content and relative asset paths.
  They do not serialize the complete agent, sessions, tool caches, or counters.
- `scenesmith/agent_utils/turn_trimming_session.py`: session processing can alter
  effective history. Copying only the new instruction is insufficient for repair.
- `scenesmith/scene_expert/hooks.py`: `finalize` performs the eligible terminal
  MemoryWriter update. Retain that lifecycle; do not add Writer calls per decision.
- `scenesmith/scene_expert/slow_memory/trajectory.py`: current `capture_stage`
  labels the final Designer from the stage report and earlier revised Designers
  from downstream revision requests. It does not independently judge sibling
  candidates from a common initial state.
- `scenesmith/scene_expert/slow_memory/dpo.py`: exact-context grouping and
  authoritative evidence checks already exist. Keep their meaning unchanged.

The existing furniture transaction restores scene state for safety. It is not a
general multi-stage candidate sandbox. The Full shared-floor-plan launcher and
Fast Memory OFF/ON pair launcher are not substitutes for decision-level sampling.
Writer replay samples Writer proposals, not Designer tool trajectories.

## Smallest intended implementation

Add an opt-in collection adapter under `scenesmith/scene_expert/slow_memory/` and
a tracked collection script under `scripts/`. Keep the normal execution path
unchanged when collection is disabled. A narrow dispatch seam at the two native
Designer call sites is justified because stage-only hooks cannot recover repair
session state; do not rewrite planner/designer/critic control logic.

Initially support furniture, wall-mounted, ceiling-mounted, and manipuland room
stages. Floor-plan overrides have a different execution path and are outside the
first collector patch.

1. Prepare the decision input once. Snapshot scene/assets, effective Designer
   history, relevant Critic history, working memory, accepted stage context,
   image bytes, tool schemas, safety/checkpoint state, budgets, and relevant tool
   state. Persist immutable group metadata before executing any candidate.
   Verify the actual request at the model boundary after history processing;
   do not infer equality from a shared prompt or a scene content hash alone.
   Include the full pre-decision scene identity and effective conversation in the
   canonical pairing context, not only in provenance. A matching bounded spatial
   summary must not hide a different executable scene state or session history.
2. Execute two stochastic Qwen candidates from independent restorations of that
   snapshot. Each candidate needs separate writable assets, sessions, render and
   audit directories. Reconstruct workers rather than blindly deep-copying live
   clients, renderer handles, or database connections. A group is unsupported
   until its stage-specific mutable state can be restored and verified.
3. For each candidate, run the same native evaluation procedure on its own result.
   Bind raw candidate state, safety actions, resulting state, report, tool trace,
   and media to its candidate ID. Never label one candidate using another
   candidate's final report or a previous checkpoint's scores. If native safety
   rolls back, record both states and the rollback; do not call restored success
   evidence of a successful proposed action. Infrastructure failures stay audit
   failures rather than synthetic low-quality completions.
   Extract the reusable evaluation portion of `hooks.py:post_stage` if needed;
   candidate evaluation must not advance the stage, consume Planner repair
   budgets, invoke outer repair, or update long-term memory. Any additional
   candidate Critic calls must use isolated sessions and leave the canonical
   Planner-visible critique cadence unchanged.
4. Persist candidate-level authoritative evidence explicitly. Extend capture for
   this path rather than sending sibling logs through the chronological
   `capture_stage` final/revised heuristic. Keep labels outside the model input.
5. Preserve the predeclared canonical candidate (A) as the online continuation;
   B is a shadow execution. Do not introduce best-of-K selection into the native
   scene generation policy. Only the canonical completed scene owns the normal
   Writer update. Shadow branches must not mutate the shared bank or utility
   counters. Groups already opened retain their captured context even if another
   scene finishes and updates the bank. Initially serialize group creation to
   avoid ambiguous bank-read/update ordering.
6. Export each completed group immediately using the existing exact-context and
   quality-margin rules. Two accepted outcomes may yield a relative preference
   under the existing rules. Identical completions, ties, two rejected outcomes,
   and insufficient evidence need not produce a pair. Do not force labels.

Use a fresh batch namespace, for example
`outputs/slow_memory/qwen38_pairs_<timestamp>/`, containing a batch manifest,
group snapshots, per-candidate state and evidence, status journals, and exports.
Candidate IDs differ; task IDs and group conditioning remain identical. Resume
only unfinished candidates after validating snapshot/model/config hashes. A
crash after writing an artifact must not create a duplicate candidate or a second
Writer update. Never silently reuse an existing output directory for a new batch.

The collection manifest must reject unexpected model IDs and disallow old Terra
sources. Record actual resolved model identity, not merely a model name chosen
for the output directory. Reuse the current ACP Qwen service profile through a
tracked launch interface; do not rely on ignored local scripts as the only
reproducible experiment specification.

## Memory and held-out tasks

Assign scenario families to train/validation/test before collection. The training
bank may evolve from training tasks only. Validation/test use a frozen train-only
bank and must not write experience back into it. Audit the seed bank's provenance;
if it contains held-out experience, use a verified train-only bank instead.

The current exporter assigns group splits at export time. Add an optional explicit
family-split manifest for this collection and make collector and exporter consume
the same assignment. Retain legacy splitting when that option is absent. Missing
or conflicting assignments must not silently fall back to a random split.

## Experiments and acceptance

1. **Branch correctness:** deterministic test fixtures verify equal effective
   request/history/images/tools, isolated mutations, candidate-specific labels,
   interruption/resume idempotency, and unchanged normal execution when disabled.
   Include rollback and a repair session with trimmed history. Reuse existing
   Slow Memory tests for exact grouping, accepted-vs-accepted ranking, media and
   leakage checks; add tests for the new split/model controls.
2. **First real batch:** cap at 12 fresh decision groups and two executions per
   group. Target six initial and six naturally occurring repair groups across at
   least three placement stages and multiple scenario families. Report actual
   coverage; do not manufacture repairs to satisfy the quota. Keep this a bounded
   sampler validation, not a claim of sufficient training data.
3. **Immediate export:** require exact equality for every purported pair,
   candidate-specific Main evidence, valid media, Qwen-only provenance, and at
   least one real exported pair. Inspect both initial/repair coverage before
   scaling. Zero pairs is a collection result to diagnose, not a reason to weaken
   the exporter. An optional later round can sample up to four candidates under
   the same original group snapshot, retaining all outcomes and attempt counts.
4. **Scale by useful data:** track pairs per candidate execution and per GPU-hour,
   context mismatch reasons, runtime failures, ties, stage/task-type coverage,
   hard-constraint outcomes, and bank revision. Increase contexts/cases only after
   the first batch proves genuine pairing. Do not optimize for trajectory count.
5. **Train:** run the existing preflight against the matching safetensors base.
   Current minima (16 training pairs, eight train groups, three stages, both task
   types with at least four pairs each) are engineering gates, not paper-scale
   evidence. Keep DPO plus the existing small SFT term initially; avoid changing
   several training variables before establishing a baseline.
6. **Evaluate:** compare base Qwen3.8 and its adapter with identical held-out
   prompts, shared bases, a frozen train-only bank, code, and runtime budgets.
   Use the existing scene promotion gate (at least 20 paired cases) in addition
   to offline preference accuracy. Report held-out test results once after model
   selection; do not tune using test feedback. Online memory-evolution gains and
   adapter gains require separate controlled comparisons.

## Delivery status

This change records the recollection decision and corrects the blanket
MemoryWriter-off guidance. No paired worker, simulator run, collected candidate,
training job, or performance improvement is claimed. The next executable work is
the decision snapshot/branch/evidence adapter and its bounded ACP launch, followed
immediately by the first real batch above.
