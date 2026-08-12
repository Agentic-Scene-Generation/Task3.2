# SceneExpert — Claude Code Guide

## Project Goal

Build **SceneExpert MVP** on top of the existing SceneSmith codebase.

SceneSmith (existing) handles 3D scene generation via a 5-stage pipeline (floor_plan → furniture → wall_mounted → ceiling_mounted → manipuland), using GPT/Qwen as the backbone. SceneExpert is a **new outer wrapper layer** that adds:
- Structured task understanding (TaskCompiler)
- Deterministic stage control (Harness FSM)
- Experience-based memory retrieval (Fast Memory System)
- Expert planning hints per stage (StageBrief)
- Rule-based quality verification (Verifier)
- Automatic repair loops (Repair Controller)
- Full trace logging and memory updating after each run

The model backbone has already been switched from GPT to **Qwen3** (via vLLM, OpenAI-compatible endpoint). The goal is to use SceneExpert's design to compensate for the smaller open model's weaknesses.

### MVP Scope (Online Closed-Loop Only)

Implement **only** the following pipeline:
```
Text Prompt
  → TaskCompiler
  → Harness FSM
  → Fast Memory Retrieval
  → StageBrief (Global Planner)
  → SceneSmith Stage Execution   ← SceneSmith runs unchanged
  → Stage Verifier
  → Repair Controller
  → Trace Logger
  → Memory Writer
```

**Do NOT implement in MVP**: full KG memory, RL memory manager, policy curriculum, large-scale skill induction, multi-model ensemble, complex multi-agent debate, SFT/DPO offline training.

---

## Repository Layout

```
scenesmith-main/
├── main.py                          # Entry point (Hydra)
├── configurations/
│   ├── config.yaml                  # Root Hydra config
│   ├── experiment/
│   │   ├── base_experiment.yaml     # Shared experiment settings
│   │   └── indoor_scene_generation.yaml
│   ├── furniture_agent/
│   │   ├── base_furniture_agent.yaml   # Model, session, rendering config
│   │   └── stateful_furniture_agent.yaml
│   └── [floor_plan|wall|ceiling|manipuland]_agent/  (same pattern)
├── scenesmith/
│   ├── agent_utils/
│   │   ├── base_stateful_agent.py   # Core planner/designer/critic loop
│   │   ├── room.py                  # AgentType, RoomScene, HouseScene
│   │   ├── scoring.py               # CritiqueWithScores, score helpers
│   │   └── ...                      # servers, tools, physics utils
│   ├── experiments/
│   │   ├── base_experiment.py
│   │   └── indoor_scene_generation.py  # 5-stage pipeline orchestrator
│   ├── [floor_plan|furniture|wall|ceiling|manipuland]_agents/
│   │   ├── base_*_agent.py
│   │   ├── stateful_*_agent.py
│   │   └── tools/
│   └── prompts/
│       ├── manager.py
│       ├── registry.py              # Prompt enums + Jinja rendering
│       └── data/
│           └── [agent]/             # YAML prompt templates per stage
└── scripts/
    ├── start_vllm.sh                # Start Qwen3 vLLM server
    └── deploy_qwen.sh
```

### New Files to Create (SceneExpert MVP)

```
scenesmith/
└── scene_expert/
    ├── __init__.py
    ├── schemas.py              # Pydantic: SceneTaskSpec, StageBrief, VerifyReport, etc.
    ├── task_compiler.py        # Qwen3 call → SceneTaskSpec
    ├── harness.py              # Deterministic Harness: FSM, budget, repair control
    ├── global_planner.py       # Qwen3 call → StageBrief for each stage
    ├── verifier.py             # Rule-based + Qwen3 stage/full verifier
    ├── repair_controller.py    # Local repair / stage regen / rollback
    ├── trace_logger.py         # JSON trace writer
    ├── pipeline.py             # SceneExpertPipeline (wraps IndoorSceneGeneration)
    └── memory/
        ├── __init__.py
        ├── schemas.py          # Pydantic: SuccessCase, FailureCase, Skill
        ├── store.py            # JSON file-based memory storage
        ├── retriever.py        # BM25 + keyword retrieval (no heavy embeddings)
        └── writer.py           # Qwen3 memory_writer role → memory updates

configurations/
└── scene_expert/
    └── base_scene_expert.yaml  # SceneExpert-specific config
```

---

## Key Existing Code to Understand

### IndoorSceneGeneration (`scenesmith/experiments/indoor_scene_generation.py`)

The 5-stage pipeline. Key points:
- Stage order: `floor_plan → furniture → wall_mounted → ceiling_mounted → manipuland`
- `STAGE_CHECKPOINTS`: maps stage → required upstream checkpoint dir name
- `STAGE_ASSET_DIRS`: asset dirs needed when resuming
- Per-prompt: builds `HouseScene`, runs each `Stateful*Agent`, applies physics post-processing
- Supports `pipeline.resume_from_path` to branch from existing checkpoint

### BaseStatefulAgent (`scenesmith/agent_utils/base_stateful_agent.py`)

The planner/designer/critic multi-agent loop per stage. Key points:
- Uses **OpenAI Agents SDK** (`agents` package: `Agent`, `Runner`, `SQLiteSession`)
- Model is set via `cfg.openai.model` (already `"Qwen/Qwen3.5-35B-A3B"`)
- Planner orchestrates: `request_initial_design → request_critique → request_design_change`
- Checkpoint/rollback system based on `CritiqueWithScores` deltas
- Session state persisted in SQLite (`.db` files in output dir)

### Qwen3 / vLLM Integration

The model is already configured in YAML as `"Qwen/Qwen3.5-35B-A3B"`. The vLLM server provides an OpenAI-compatible API. The `agents` package (OpenAI SDK) connects to it via the `OPENAI_BASE_URL` and `OPENAI_API_KEY` environment variables (set in `.env`). All direct Qwen3 calls (TaskCompiler, GlobalPlanner, MemoryWriter) should use the same `openai` Python library with the same base URL.

---

## SceneExpert Data Schemas

All schemas are Pydantic models. Define them in `scenesmith/scene_expert/schemas.py` and `scenesmith/scene_expert/memory/schemas.py`.

### SceneTaskSpec
```python
class SceneTaskSpec(BaseModel):
    room_type: str
    style: str
    required_large_objects: list[str]
    required_wall_objects: list[str]
    required_ceiling_objects: list[str]
    required_small_objects: list[str]
    functional_zones: list[str]
    interaction_constraints: list[str]
    aesthetic_constraints: list[str]
```

### StageBrief
```python
class StageBrief(BaseModel):
    stage: str  # e.g., "furniture", "manipuland"
    stage_objective: str
    recommended_skills: list[str]
    constraints_for_designer: list[str]
    checks_for_critic: list[str]
    failure_patterns_to_avoid: list[str]
```

### StageVerifyReport
```python
class StageVerifyReport(BaseModel):
    stage: str
    pass_stage: bool
    scores: dict[str, float]  # semantic, aesthetic, physics, interaction (0-1)
    issues: list[dict]        # [{type, object, description}]
    repair_suggestions: list[str]
```

### FullVerifyReport
```python
class FullVerifyReport(BaseModel):
    semantic_score: float
    aesthetic_score: float
    style_consistency: float
    collision_free_rate: float
    stability_score: float
    walkable_area_ratio: float
    reachability_score: float
    support_relation_accuracy: float
    overall_score: float
    pass_scene: bool
```

### Memory Schemas (`memory/schemas.py`)
```python
class SuccessCase(BaseModel):
    case_id: str
    room_type: str
    style: str
    stage: str
    task_signature: list[str]
    successful_pattern: list[str]
    scores: dict[str, float]
    trace_ref: str

class FailureCase(BaseModel):
    failure_id: str
    room_type: str
    stage: str
    object: str
    failure_type: str
    bad_pattern: str
    failure_reason: str
    repair_action: str
    repair_verified: bool

class Skill(BaseModel):
    skill_name: str
    stage: str
    room_types: list[str]
    preconditions: list[str]
    procedure: list[str]
    failure_avoidance: list[str]
    postconditions: list[str]
```

---

## Module Implementation Guidelines

### Module 1: TaskCompiler (`task_compiler.py`)

- Single Qwen3 call with `role=task_compiler` system prompt
- Input: raw text prompt string
- Output: `SceneTaskSpec` (use Pydantic structured output or parse JSON response)
- Use `openai` library directly (not the `agents` SDK — no tool calls needed)
- System prompt: instruct Qwen3 to extract structured scene requirements

```python
class TaskCompiler:
    def __init__(self, model: str, api_base_url: str, api_key: str): ...
    def compile(self, prompt: str) -> SceneTaskSpec: ...
```

### Module 2: Deterministic Harness (`harness.py`)

Pure Python — **no LLM calls**. Controls the execution flow.

Key responsibilities:
- **Stage FSM**: enforces fixed stage order, Qwen3 cannot skip stages
- **Budget Controller**: tracks `max_designer_iterations` and `max_repair_steps` per stage
- **Repair Controller**: decides repair strategy based on verify report (see below)
- **Trace Logger**: delegates to `TraceLogger`

```python
class HarnessContext:
    stage: str
    task_spec: SceneTaskSpec
    memory_pack: MemoryPack
    stage_brief: StageBrief
    stage_budget: StageBudget  # {max_designer_iterations, max_repair_steps}

class Harness:
    STAGE_ORDER = ["floor_plan", "furniture", "wall_mounted", "ceiling_mounted", "manipuland"]

    def build_context(self, stage, task_spec, memory_pack, stage_brief) -> HarnessContext: ...
    def should_repair(self, verify_report: StageVerifyReport) -> bool: ...
    def select_repair_strategy(self, verify_report: StageVerifyReport, repair_attempt: int) -> str:
        # Returns: "local_repair", "stage_regeneration", or "rollback"
        # light to heavy: local_repair first, then stage_regeneration, then rollback
        ...
```

### Module 3: Fast Memory System (`memory/`)

**Storage**: JSON files in `memory_dir/` (configured path, e.g. `outputs/scene_expert_memory/`).
- `success_cases.jsonl` — one JSON per line
- `failure_cases.jsonl` — one JSON per line
- `skills.jsonl` — one JSON per line

**Retrieval** (`retriever.py`):
- MVP: keyword/BM25 matching — no heavy embedding model required
- Match on: `room_type`, `stage`, and keyword overlap with `task_signature` / object names
- Return top-3 success cases, top-3 failure cases, top-2 skills

```python
class MemoryPack(BaseModel):
    success_hints: list[str]   # compressed hint strings
    failure_hints: list[str]
    skills: list[Skill]

class MemoryRetriever:
    def retrieve(self, task_spec: SceneTaskSpec, stage: str) -> MemoryPack: ...
```

**Writer** (`writer.py`):
- Qwen3 call with `role=memory_writer` system prompt
- Input: trace summary + final report + related old memory
- Output: list of memory update ops `[{op: "ADD"|"UPDATE"|"NOOP", memory_type, content}]`
- MVP only uses `ADD`, `UPDATE`, `NOOP` — do not implement `DELETE`

### Module 4: Global Planner (`global_planner.py`)

Single Qwen3 call with `role=global_planner` system prompt.

Input: `HarnessContext` (stage, task_spec, scene_state_summary, memory_pack, stage_budget)
Output: `StageBrief`

Key: StageBrief is **injected into SceneSmith's stage prompt** as additional context.
The planner does NOT place objects, generate meshes, or modify the Drake scene directly.

```python
class GlobalPlanner:
    def generate_stage_brief(self, context: HarnessContext, scene_state_summary: str) -> StageBrief: ...
```

Scene state summary: extract key facts from SceneSmith's current `RoomScene`/`HouseScene` state
(present objects, their categories, approximate positions, support surfaces).

### Module 5: SceneSmith Stage Injection

This is the critical integration point. The injection approach:

1. Convert `StageBrief` to a structured text block (the "expert hint")
2. Inject the expert hint into the SceneSmith agent's **initial design prompt kwargs**

The cleanest injection mechanism: subclass or patch the `Stateful*Agent`'s
`_get_initial_design_prompt_kwargs()` to include `stage_brief` and `memory_pack` strings.
Alternatively, prepend the expert hint to the designer's system prompt at runtime.

**Do not modify the core SceneSmith agent logic**. Add the injection as a thin wrapper.

### Module 6: Verifier (`verifier.py`)

Two layers:
- `StageVerifier`: runs after each stage, text-only rule checks for MVP
- `FullVerifier`: runs at end, aggregates stage scores

**MVP verifier is primarily rule-based** (no expensive VLM calls):
- Read `scores.yaml` already produced by SceneSmith's own CritiqueWithScores
- Map SceneSmith's existing 6-category scores to the SceneExpert schema
- Add simple rule checks: object count vs task_spec, required objects present, etc.

```python
class Verifier:
    def verify_stage(self, stage: str, scene_state_path: str,
                     task_spec: SceneTaskSpec, stage_brief: StageBrief,
                     scenesmith_scores: dict) -> StageVerifyReport: ...

    def verify_full(self, final_scene_path: str,
                    stage_reports: list[StageVerifyReport]) -> FullVerifyReport: ...
```

### Module 7: Repair Controller (`repair_controller.py`)

Repair strategies (lightest to heaviest):
1. **local_repair**: generate a text instruction for the SceneSmith designer agent to fix specific objects
2. **stage_regeneration**: re-run the current stage from its checkpoint with updated StageBrief
3. **rollback**: use SceneSmith's `resume_from_path` to revert to previous stage checkpoint

For MVP: implement `local_repair` and `stage_regeneration`. Rollback is optional.

```python
class RepairController:
    def repair(self, repair_type: str, stage: str, verify_report: StageVerifyReport,
               scene_path: str, memory: FastMemory) -> RepairResult: ...
```

### Module 8: Trace Logger (`trace_logger.py`)

Writes a structured JSON trace file per run.

```python
class TraceLogger:
    def __init__(self, output_dir: str): ...
    def log_stage(self, stage: str, context: HarnessContext, scene_state_path: str,
                  verify_report: StageVerifyReport, repair_actions: list, cost: dict): ...
    def finalize(self, full_report: FullVerifyReport, exports: dict) -> dict: ...
    def save(self): ...  # writes trace_*.json to output_dir
```

### Module 9: SceneExpert Pipeline (`pipeline.py`)

The main orchestrator. Implements the full MVP algorithm:

```python
class SceneExpertPipeline:
    """Wraps IndoorSceneGeneration with SceneExpert pre/post stage hooks."""

    def __init__(self, cfg, base_pipeline: IndoorSceneGeneration): ...

    def run_scene(self, prompt: str) -> tuple[str, dict, FullVerifyReport]:
        """Run SceneExpert for one prompt. Returns (scene_path, trace, full_report)."""
        # 1. TaskCompiler: prompt → SceneTaskSpec
        # 2. For each stage in STAGE_ORDER:
        #    a. Load checkpoint
        #    b. Retrieve memory
        #    c. Generate StageBrief
        #    d. Inject StageBrief into SceneSmith stage
        #    e. Execute SceneSmith stage
        #    f. Verify stage
        #    g. If fail: Repair Controller loop (up to max_repair_steps)
        #    h. Log stage trace
        # 3. Full verifier
        # 4. Memory Writer → update fast memory
        # 5. Return results
```

---

## Configuration (`configurations/scene_expert/base_scene_expert.yaml`)

```yaml
# SceneExpert MVP configuration

# Memory system
memory:
  dir: "outputs/scene_expert_memory"
  retrieval:
    max_success_cases: 3
    max_failure_cases: 3
    max_skills: 2

# Harness budget per stage
stage_budget:
  default:
    max_designer_iterations: 2
    max_repair_steps: 1
  manipuland:
    max_designer_iterations: 2
    max_repair_steps: 2  # Manipuland placement most error-prone

# Verifier thresholds
verifier:
  stage_pass_threshold: 0.6   # Min score to consider stage passing
  full_pass_threshold: 0.7

# Qwen3 call settings for SceneExpert modules
# (uses same OPENAI_BASE_URL / OPENAI_API_KEY as SceneSmith)
qwen3:
  model: "Qwen/Qwen3.5-35B-A3B"
  task_compiler_max_tokens: 1024
  global_planner_max_tokens: 2048
  memory_writer_max_tokens: 1024
  temperature: 0.1   # Low temperature for structured outputs
```

---

## Prompt Design Guidelines

All SceneExpert prompts live in `scenesmith/prompts/data/scene_expert/`.
Create YAML files for each role:
- `task_compiler.yaml` — extract SceneTaskSpec from raw prompt
- `global_planner.yaml` — generate StageBrief given context + memory
- `memory_writer.yaml` — update memory given trace summary

Prompt principles for small Qwen3 models:
- **Structured output**: always ask for JSON output with explicit schema in prompt
- **Few-shot examples**: include 1-2 concrete examples in the prompt
- **Stage-specific context**: mention the current stage name prominently
- **Memory integration**: format success_hints, failure_hints, and skills as numbered lists
- **Constraint injection**: format StageBrief constraints as bullet points for easy parsing

---

## Integration with Existing SceneSmith  *(IMPLEMENTED)*

### Actual Integration Pattern

The integration is done via a **hook runner** (`scenesmith/scene_expert/hooks.py`)
that is created once per scene and called at fixed points inside `_generate_room`.

**Files modified** (surgical changes only):
- `configurations/config.yaml` — added `scene_expert: base_scene_expert` to defaults
- `scenesmith/experiments/indoor_scene_generation.py` — 3 types of changes:
  1. Added `scene_expert_hooks` parameter to `_generate_room` and `_run_sequential_room_generation`
  2. Added `pre_stage(stage, scene)` calls before each stage agent builds
  3. Added `post_stage(stage, scene, room_dir)` calls after each stage checkpoint saves
  4. Added `build_hook_runner(...)` call + `finalize()` call in `_generate_single_scene`
  5. Added `TYPE_CHECKING` import for `SceneExpertHookRunner`

**What NOT to modify** (still true):
- `BaseStatefulAgent` internals (planner/designer/critic loop)
- Individual `Stateful*Agent` implementations
- All server infrastructure
- Physics post-processing pipeline
- Existing prompt YAML templates

### StageBrief Injection Mechanism

`pre_stage(stage, scene)` appends the StageBrief as text to `scene.text_description`
before the stage agent is built. Each `Stateful*Agent` reads `self.scene.text_description`
inside `_get_initial_design_prompt_kwargs()` and passes it as `scene_description` to
the designer's initial instruction template.

`post_stage(stage, scene, room_dir)` restores the original `scene.text_description`
to keep it clean for the next stage.

---

## Ablation Experiment Interface  *(IMPLEMENTED)*

### The 5 Ablation Configurations

| # | Name | Config file | `scene_expert.mode` | Components active |
|---|------|-------------|---------------------|-------------------|
| 1 | SceneSmith Original | `ablation_1_scenesmith_original.yaml` | `disabled` | None — pure SceneSmith, GPT backbone |
| 2 | Qwen3 Naive | `ablation_2_qwen3_naive.yaml` | `disabled` | None — pure SceneSmith, Qwen3 backbone |
| 3 | Qwen3 + Harness | `ablation_3_qwen3_harness.yaml` | `harness_only` | TaskCompiler + Harness + GlobalPlanner |
| 4 | Qwen3 + Harness + Memory | `ablation_4_qwen3_harness_memory.yaml` | `harness_memory` | + FastMemory + MemoryWriter |
| 5 | Full SceneExpert | `ablation_5_qwen3_full.yaml` | `full` | + LoRA model (served via vLLM) |

### Running an Ablation

```bash
# Ablation 1 — SceneSmith original (set model via CLI override)
python main.py experiment=ablation_1_scenesmith_original \
  furniture_agent.openai.model="gpt-4o" \
  wall_agent.openai.model="gpt-4o" \
  ceiling_agent.openai.model="gpt-4o" \
  manipuland_agent.openai.model="gpt-4o" \
  floor_plan_agent.openai.model="gpt-4o"

# Ablation 2 — Qwen3 naive (default model is already Qwen3)
python main.py experiment=ablation_2_qwen3_naive

# Ablation 3 — Qwen3 + Harness
python main.py experiment=ablation_3_qwen3_harness

# Ablation 4 — Qwen3 + Harness + Memory (MVP default)
python main.py experiment=ablation_4_qwen3_harness_memory

# Ablation 5 — Full SceneExpert with LoRA
python main.py experiment=ablation_5_qwen3_full \
  furniture_agent.openai.model="Qwen/Qwen3-SceneExpert-LoRA" \
  [other agent model overrides]
```

### Mode-to-Component Mapping

`build_hook_runner()` in `hooks.py` reads `cfg_dict["scene_expert"]["mode"]` and
conditionally instantiates each component:

| Component | `disabled` | `harness_only` | `harness_memory` | `full` |
|-----------|-----------|---------------|-----------------|--------|
| TaskCompiler | ✗ | ✓ | ✓ | ✓ |
| Harness FSM | ✗ | ✓ | ✓ | ✓ |
| GlobalPlanner | ✗ | ✓ | ✓ | ✓ |
| Memory retrieval | ✗ | ✗ | ✓ | ✓ |
| FastMemoryStore | ✗ | ✗ | ✓ | ✓ |
| MemoryWriter | ✗ | ✗ | ✓ | ✓ |
| TraceLogger | ✗ | ✓ | ✓ | ✓ |
| Verifier | ✗ | ✓ | ✓ | ✓ |
| RepairController (log only) | ✗ | ✓ | ✓ | ✓ |
| LoRA model | ✗ | ✗ | ✗ | ✓ (via model ID) |

### Adding a New Ablation Dimension

To ablate a specific sub-component (e.g., "memory retrieval but no memory writer"):
1. Add a new mode string to `ABLATION_MODES` in `hooks.py`
2. Add a conditional branch in `build_hook_runner()` that sets `memory_writer=None` for that mode
3. Create a new `ablation_N_*.yaml` config file with `mode: "your_new_mode"`

---

## Qwen3 API Call Pattern

Use `openai` library directly for TaskCompiler, GlobalPlanner, MemoryWriter:

```python
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-35B-A3B",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ],
    temperature=0.6,
    max_tokens=2048,
    response_format={"type": "json_object"},  # enforce JSON output
)
result = json.loads(response.choices[0].message.content)
```

For structured Pydantic parsing, use `model.model_validate(result)` after JSON parse.

---

## File Naming and Coding Conventions

- All new files go under `scenesmith/scene_expert/`
- Use Pydantic v2 `BaseModel` for all data schemas
- Use `dataclasses.dataclass` for simple internal data structures (not serialized)
- Log with `logging.getLogger(__name__)` — do not use `print()`
- Trace files: `outputs/{experiment_dir}/traces/trace_{scene_idx:06d}.json`
- Memory files: at `cfg.scene_expert.memory.dir` (configured path)
- Follow existing code style: type hints on all function signatures, docstrings on public methods

---

## MVP Success Criteria

The implementation is complete when:
1. Qwen3 + SceneExpert Harness stably runs the full 5-stage SceneSmith pipeline
2. StageBrief is injected into each stage's designer prompt
3. Memory is retrieved and formatted into StageBrief for stages beyond the first scene
4. Stage verifier reports are generated and logged
5. At least `local_repair` strategy is functional
6. Full trace JSON is written after each run
7. Memory Writer updates `success_cases.jsonl` / `failure_cases.jsonl` after each run
8. SceneSmith pipeline remains fully functional with `use_scene_expert: false`

---

## Development Order

Implement in this order (each unblocks the next):
1. `schemas.py` — all data models
2. `memory/schemas.py` + `memory/store.py` — memory storage
3. `task_compiler.py` — simplest Qwen3 call
4. `trace_logger.py` — needed by pipeline
5. `verifier.py` — read existing SceneSmith scores + add rule checks
6. `memory/retriever.py` — BM25 retrieval
7. `global_planner.py` — StageBrief generation
8. `harness.py` — FSM + budget control
9. `repair_controller.py` — local repair first
10. `memory/writer.py` — Qwen3 memory update call
11. `pipeline.py` — wire everything together
12. Config + integration hooks in `indoor_scene_generation.py`
