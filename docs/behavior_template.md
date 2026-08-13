# Deterministic resident behavior template

SceneExpert can optionally expand a scene prompt into a resident persona, a
seven-day schedule, detailed action steps, and assets grouped by room and
SceneSmith stage.

This implementation intentionally preserves the latest template behavior from
the AgentSense integration. It is not an end-to-end LLM behavior generator:

- the resident persona uses the configured Qwen/OpenAI-compatible model and
  degrades to the deterministic `Maya` persona on provider or parsing failure;
- schedules and action steps are Python templates;
- asset needs come from room defaults plus explicit prompt object matching;
- grouped output covers furniture, wall-mounted, ceiling-mounted, and
  manipuland stages.

Enable it under the active `scene_expert` configuration:

```yaml
behavior:
  enabled: true
  planner: "template"
  horizon: "week"
  inferred_assets_are_required: true
```

Run the behavior planner directly without starting the 3D pipeline:

```bash
python -m scenesmith.scene_expert.behavior \
  --prompt "A bedroom with a bed, two nightstands, and a wardrobe." \
  --output behavior_spec.json
```

Without a configured model, this command uses the deterministic Maya persona.
Set `SCENEEXPERT_MODEL_ID`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY` to use the
same OpenAI-compatible Qwen service as Task3.2 for persona generation.

The default is disabled and therefore a strict no-op. When enabled, the full
typed result is written to
`scene_<id>/scene_expert/behavior_spec.json`. Required assets are merged into
`SceneTaskSpec`; existing prompt-derived quantities take precedence and the
merged quantity is the maximum rather than the sum. Because `SceneTaskSpec` is
a single-room contract, only the behavior assets for its primary room are
merged.

Behavior support and facing relations remain typed in the audit output. They
are not inserted into TaskCompiler's explicit-prompt constraints. Promoting
them to hard critic requirements requires a separate provenance-aware extension
to the v4 intent contract.

Set `inferred_assets_are_required: false` to retain the audit report while only
merging assets explicitly recognized in the input prompt.

The serialized contract preserves the latest AgentSense-facing fields such as
`target_rooms`, `weekly_schedule`, `detailed_routines`, `object_needs`,
`placement_relations`, `room_behavior_blocks`, and `enriched_prompt`. Task3.2
consumers can read the additional `assets_by_room_and_stage` mapping directly
without traversing the room records.

`object_needs` is the source-compatible view. It intentionally preserves the
latest integration's field structure and matching behavior. Task3.2 planning
and task merging consume `assets_by_room_and_stage`, which adds stage and
provenance metadata and uses whole-object matching so a room label such as
`bedroom` is not treated as an explicit request for a `bed`.

## Limitations

The templates currently support bedroom, kitchen, living room, and bathroom.
They encode a remote-worker lifestyle and do not reliably adapt activities to
age, disability, hobbies, minimalism, or other persona details in the prompt.
These inferred requirements should not be described as semantic LLM output.

## Attribution

The behavior-to-assets implementation is derived from AgentSense integration
work in `coolbeam/Lived-in-3D-Scene`.

Copyright (c) 2025 Zikang Leng. Licensed under the MIT License.
