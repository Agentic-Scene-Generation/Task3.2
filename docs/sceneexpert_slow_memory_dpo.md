# SceneExpert Slow Memory and DPO Workflow

Slow Memory is an offline observer of decisions already made by Main. Capture,
export, training, and model selection are independent gates. None of them changes
the online Designer, Critic, checkpoint, deterministic repair, or stage decision.

## 1. Capture replayable trajectories

The `ablation_4c_qwen3_hybrid_memory` preset keeps capture off so its online
generation remains the clean Harness + Hybrid Fast Memory ablation. The
`ablation_5_qwen3_full` preset inherits the complete 4c runtime and enables only
`scene_expert.components.slow_memory_capture.enabled`. An explicit component
override can still control capture for a dedicated diagnostic run.

When capture is disabled, the existing Main debug payload remains v1 and no
additional raw SDK/tool/media state is collected. When enabled, every scene writes:

```text
scene_XXX/scene_expert/slow_memory/
├── trajectories.jsonl
├── capture_manifest.json
├── media/
│   └── <content-sha256>.<ext>
└── evidence/
    └── <stage>_<evidence-hash>.json
```

Trajectory v2 retains model-visible messages, tool schemas, observed tool
calls/results, pre-decision spatial context, final scene context, model/config/code
provenance, and label-only outcome evidence. Input images and images returned by
tools are content-addressed under `media/`; trajectory JSON contains hashes and
stable references rather than inline Base64. Duplicate raw SDK payloads remain in
the normal audit tree and are represented by hashes, so training rows stay complete
without carrying multi-megabyte copies. Credentials are redacted before persistence.

The final Designer decision receives the authoritative post-stage verdict. An
earlier Designer decision is labeled `rejected` only when Main observably requested
a downstream revision; otherwise it remains `unlabeled`. Critic advice is retained
for audit but is excluded from the default training task types because current Main
logs do not prove advice-to-repair causality. Critic self-scoring is never used as
its own label. `capture_manifest.json` exposes incomplete rows, missing media,
embedded-data leaks, resolved mode/component flags, and code provenance; require
`quality_summary.default_training_payload_valid=true` before export.

## 2. Import external teacher candidates

One input JSONL row represents one observed candidate, not an already fabricated
pair. Required fields are:

```json
{
  "sample_id": "teacher-run-001-candidate-a",
  "task_id": "stable-task-id",
  "scenario_family_id": "leakage-group-id",
  "stage": "furniture",
  "agent_role": "designer",
  "event": "request_design_change",
  "task_type": "designer_repair",
  "model_id": "teacher-model-id",
  "messages": [{"role": "user", "content": "..."}],
  "tools": [{"type": "function", "function": {"name": "...", "parameters": {}}}],
  "image_paths": ["relative/or/absolute/render.png"],
  "spatial_context": {"objects": [], "relations": []},
  "completion_messages": [{"role": "assistant", "content": "..."}],
  "action_trace": [],
  "outcome": {
    "execution_complete": true,
    "tool_call_valid": true,
    "hard_passed": true,
    "hard_violation_count": 0,
    "relation_satisfaction": 1.0,
    "causal_link_verified": true
  },
  "evidence": {
    "kind": "critic_and_deterministic",
    "verdict": "accepted",
    "authoritative": true,
    "quality_score": 1.8,
    "source": "teacher_execution_plus_main_verifier",
    "details": {}
  }
}
```

Import without inferring missing labels:

```bash
python scripts/import_sceneexpert_teacher_data.py \
  --source outputs/teacher/candidates.jsonl \
  --output outputs/slow_memory/teacher_trajectories.jsonl
```

Invalid rows go to `teacher_trajectories.diagnostics.jsonl` and make the command
fail unless `--allow-rejected-rows` is explicitly set. Critic accepted/rejected
labels without `causal_link_verified=true` are rejected.

## 3. Export preference pairs

Repeat generation from a frozen shared base, memory snapshot, prompt, tool set, and
configuration. Disable MemoryWriter during repeated candidate collection so the
injected context does not drift.

```bash
python scripts/export_sceneexpert_dpo.py \
  --trajectory-source outputs/run_001 \
  --trajectory-source outputs/run_002 \
  --trajectory-source outputs/slow_memory/teacher_trajectories.jsonl \
  --output-dir outputs/slow_memory/dpo_dataset
```

The exporter requires exact messages, tool schemas, image hashes, spatial context,
task, stage, role, event, and task type. It ranks real outcomes in this order:

1. execution and tool-call validity;
2. hard-constraint pass and new/remaining hard violations;
3. stage pass and spatial-relation satisfaction;
4. deterministic and visual quality;
5. action count and latency.

It never synthesizes a rejected response. Critic advice, deterministic repair, and
legacy rows are audit-only by default; the default dataset contains only
`designer_initial` and `designer_repair`. Use repeated `--include-task-type` only
for a deliberate custom mix. Images are content-addressed and copied into the
exported package.
Splits use `scenario_family_id` (falling back to `task_id`), preventing related
prompts/scenes from leaking across train, validation, and test.

## 4. Validate and train Qwen

The llama.cpp Qwen3.8 GGUF is inference-only. Set the matching Hugging Face
safetensors checkpoint in the config or CLI. The default is Transformers QLoRA for
the VLM, freezes visual modules, trains language attention/MLP adapters, uses
DPO plus a small chosen-response SFT term, and sets `max_length=null` to preserve
image tokens.

```bash
python scripts/train_sceneexpert_dpo.py \
  --config configurations/slow_memory/qwen_dpo_qlora.yaml \
  --dataset-dir outputs/slow_memory/dpo_dataset \
  --model /mnt/afs/task3_2/share_model/<qwen-transformers-checkpoint> \
  --dry-run

uv run scripts/train_sceneexpert_dpo.py \
  --config configurations/slow_memory/qwen_dpo_qlora.yaml \
  --dataset-dir outputs/slow_memory/dpo_dataset \
  --model /mnt/afs/task3_2/share_model/<qwen-transformers-checkpoint> \
  --output-dir outputs/slow_memory/qwen38_dpo
```

Preflight blocks GGUF, mismatched text/VLM data, tool data without explicit tool
support, small or single-family datasets, missing Designer initial/repair coverage,
stage under-coverage, imbalanced task types, leakage, missing validation/test splits,
and unverifiable media.

Training writes `training_manifest.json`. An adapter is not promotable unless held-
out TRL preference accuracy passes the configured threshold. This offline gate is
necessary but not sufficient: serve the candidate under a new alias and run paired
SceneEval tests against the exact base model. Promote only if hard-pass rate and
spatial-relation metrics improve without regressing Main Critic success.

After collecting `run_metrics.json` for the base and adapter on the same frozen
SceneEval prompts/shared base/Memory snapshot, apply the final gate:

```bash
python scripts/evaluate_sceneexpert_dpo.py \
  --baseline outputs/base_sceneeval/metrics/run_metrics.json \
  --candidate outputs/dpo_sceneeval/metrics/run_metrics.json \
  --output outputs/slow_memory/qwen38_dpo/scene_promotion.json
```

The command returns exit code 3 and keeps the adapter non-promotable if controls do
not match, if any completion/hard-pass/scene-pass aggregate regresses, or if the
configured paired Critic-score and net case-win improvements are not reached.

## 5. Select a validated adapter

After the paired scene-level gate passes, serve the validated checkpoint under an
explicit alias and select that alias independently of the experiment mode:

```bash
export SCENEEXPERT_MODEL_ID="Qwen/SceneExpert-DPO"
python main.py experiment=ablation_5_qwen3_full
```

`ablation_5_qwen3_full` never starts DPO training and never selects a checkpoint on
the user's behalf. Running it with the base-model alias produces 4c-equivalent scene
generation plus capture; running it with a validated candidate alias evaluates that
candidate under the same online pipeline.
