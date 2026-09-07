# Accepted Memory at design decisions

`memory-context.v1` adds a thin context path on top of the existing SceneExpert
hooks. It does not change native Designer/Critic tools, decisions, transactions,
scores, stage order, failure policies, or physics. It starts no extra repair
loop and no extra online Planner call.

## Flow and boundaries

1. Observe bounded current room/object state: IDs, names, available categories,
   translations, yaw, bounds, immutable flags, and explicitly available support
   surfaces. Missing measurements stay missing. Observation errors abstain
   instead of failing native generation.
2. Retrieve with the immutable task plus **observed**, not required, object
   roles. This allows an already-present optional chair to retrieve chair
   experience without adding a chair to the task's required inventory. No
   hypothetical optional inventory is fabricated before design.
3. The existing GlobalPlanner call sees typed source IDs, content hashes,
   complete advice, spatial parameters, effective evidence grades, and current
   state. It emits `memory_adaptations` with accepted/adapted/rejected decisions,
   role/current-object bindings, preconditions, actions, and checks. Missing or
   fallback decisions abstain; `recommended_skills` alone is not acceptance.
4. Validate identities, source hashes, role bindings, explicit spatial/count/
   frame conflicts, duplicate choices, and conflicts with already accepted
   sources. Budget complete adapted items, not sliced strings. Parameterized
   source templates refer to matching current intent rows, never old counts.
5. Only `accepted_items` produce cross-task advice. Raw hints and global layout
   strings are not appended after the Planner. Known verbatim memory in generic
   brief fields is removed to avoid duplicate/rejected-text backflow. This is
   not a semantic proof for every possible free-text LLM paraphrase.
6. For the four RoomScene stages (furniture, wall_mounted, ceiling_mounted,
   manipuland), keep only task-derived brief text in the scene description.
   Deliver accepted memory through the existing context-bundle interface at
   `request_initial_design` and `request_design_change`. Native Critic context
   gets no accepted-memory block. Each modification request filters cached
   advice against its current issue and object state; it does not call the
   Planner or retrieval again. Moved/missing bound objects or changed room
   geometry withhold stale advice. Advice already present in the request/native
   working memory is not duplicated when the full advice is an exact match.

The independent floor-plan subprocess retains its existing initial enhanced
prompt path, now using the same explicit acceptance rule. It has no new
per-repair delivery hook in this change. Reuse experiments skip floor_plan and
therefore test only the four RoomScene stages. No monkey-patching of native
agent loops is used to extend that boundary.

Bindings are suggestions for the current design, not new required assets.
`auto` remains required-first plus optional design freedom. Unrelated memory
can be absent at any stage without skipping or failing that stage.

## Runtime switches and compatibility

- Existing `fast_memory_retrieval`, `global_planner`, and `prompt_injection`
  gates remain authoritative. Planner unavailable/disabled means no accepted
  memory, even if recall produced candidates.
- `SCENEEXPERT_INJECT_STAGE_CONTEXT_BUNDLE=0` disables request-boundary delivery.
  The current ACP defaults leave this existing gate enabled. No new ACP command
  or Git check is required.
- Old memory databases remain readable; no source records or IDs are migrated.
  Semantic indexes are now `memory-text.v3`, excluding task/run IDs from the
  embedding text while retaining them in provenance. Existing automatic index
  refresh applies; frozen-bank caches remain outside the bank.
- GlobalPlanner output ceilings are 2048 tokens initially and 4096 on the
  existing structured retry. These allow complete adaptation JSON; they do not
  enable Harness budgets or more model calls. Explicit role configuration is
  respected. Compare actual output/latency costs, not these ceilings alone.

## Inspecting output

Under each scene's `scene_expert` directory:

- `memory_activity.json`: `stages.<stage>.retrieval.current_scene_state`, atomic
  candidate selections, `injection.adaptation_decisions`, and
  `injection.accepted_items` show recall, rejection reasons, source evidence,
  role bindings and prepared advice.
- `context_bundles/<stage>/*_request_initial_design.json` and
  `*_request_design_change.json`: `memory_delivery.text` is the block prepared
  for that particular request. `memory_delivery.decisions` identifies delivered,
  already-in-request, or withheld sources, with content hashes and reasons.
  These files are saved by the existing stage-working-memory audit when enabled.
- Native LLM payload logs remain the authority for what was actually sent to
  the model. A context prepared before a provider failure is not a completed
  model action; `application_observed` deliberately remains null.

Pre-stage `designer_prompt_contains_memory` can now be false because delivery
happens later. Do not interpret that old aggregate as retrieval being off or
use the prepared `prompt_delivered_skill_names` list as observed application.
Request-level reconciliation and all-attempt costs now live in
[EVALUATION.md](EVALUATION.md). Use the v9 metrics and per-item decision_usage,
not the pre-stage flag, for delivery auditing. These engineering checks do not
prove positive scene-generation gains or constitute full server validation.

## Checks

`tests/unit/test_memory_context.py` covers acceptance, ambiguity, atomic source
binding, source conflicts, observed optional retrieval (lexical and hybrid),
state/support snapshots, initial/repair delivery, duplicate suppression,
changed-object invalidation, native-Critic isolation, error isolation, and
Planner success/fallback without additional calls. It uses actual context and
Planner code with a synthetic provider and lightweight scene objects, not a
GPU/Blender simulation. The wider contract/writer/skill/index and metric tests
remain applicable. Run full native integration tests in a Drake/Blender-enabled
environment before interpreting large-scale paired results.
