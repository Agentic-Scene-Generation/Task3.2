# SceneExpert Slow Memory and DPO Workflow

Slow Memory is an offline learning path. It observes artifacts that SceneSmith
has already produced and never changes the online designer, critic, checkpoint,
or repair decisions.

## 1. Runtime trajectory capture

`harness_only`, `harness_memory`, and `full` modes enable the independent
`scene_expert.components.slow_memory_capture` gate by default. Set it to
`false` for a strict no-capture ablation.

Each scene writes:

```text
scene_XXX/scene_expert/slow_memory/
├── trajectories.jsonl
├── capture_manifest.json
└── evidence/
    └── <stage>_<evidence-hash>.json
```

The collector reads the existing `audit/llm_payloads` and
`timing/repair_events.jsonl` files after a stage finishes. Credentials are
redacted. Only the final designer call in a stage receives that stage's
accepted/rejected verdict; earlier calls remain `unlabeled` because a stage
verdict cannot safely be projected onto every intermediate candidate.

## 2. Export a DPO dataset

Combine several independent repetitions so the exporter can find different
responses to the exact same prompt:

```bash
python scripts/export_sceneexpert_dpo.py \
  --trajectory-source outputs/run_001 \
  --trajectory-source outputs/run_002 \
  --trajectory-source outputs/run_003 \
  --output-dir outputs/slow_memory/dpo_dataset
```

For DPO collection, freeze one verified memory-bank snapshot and disable only
`scene_expert.components.memory_writer.enabled`. Keep retrieval, prompts,
configuration, and the upstream shared base fixed while repeating stochastic
generations. If MemoryWriter keeps changing the injected context between runs,
those runs correctly fail the exact-context pairing gate and remain diagnostic
evidence rather than unsafe DPO samples.

The exporter requires:

- byte-identical model prompt/context after credential redaction and the same task,
  stage, role, and event;
- one accepted and one rejected response, or two real accepted responses with
  a sufficient main-critic quality margin;
- authoritative SceneBenchmark critic or deterministic verification evidence;
- non-truncated, different responses with a positive quality margin.

It never synthesizes a rejected answer. For two accepted outcomes, `rejected`
means only "the lower critic-ranked observed response for DPO"; its original
runtime verdict stays in the evidence record. A run with insufficient evidence may
produce zero pairs; use `--allow-empty` to materialize diagnostics without
treating that expected state as a job failure.

Output files:

```text
dpo_dataset/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
├── all.jsonl
├── rejected_pair_diagnostics.jsonl
├── stats.json
├── validation.json
└── manifest.json
```

Splits are grouped by `task_id`, so one scene prompt cannot leak across train,
validation, and test.

## 3. Validate and train

The current llama.cpp Qwen3.8 GGUF is an inference artifact and cannot be used
as the training base. Obtain the corresponding Hugging Face Transformers
checkpoint (`config.json` plus safetensors), then run a dry preflight first:

```bash
python scripts/train_sceneexpert_dpo.py \
  --config configurations/slow_memory/qwen_dpo_qlora.yaml \
  --dataset-dir outputs/slow_memory/dpo_dataset \
  --model /mnt/afs/task3_2/share_model/<qwen-transformers-checkpoint> \
  --dry-run
```

Start or resume training only after preflight reports `valid: true`:

```bash
uv run scripts/train_sceneexpert_dpo.py \
  --config configurations/slow_memory/qwen_dpo_qlora.yaml \
  --dataset-dir outputs/slow_memory/dpo_dataset \
  --model /mnt/afs/task3_2/share_model/<qwen-transformers-checkpoint> \
  --output-dir outputs/slow_memory/qwen38_dpo

uv run scripts/train_sceneexpert_dpo.py \
  --config configurations/slow_memory/qwen_dpo_qlora.yaml \
  --dataset-dir outputs/slow_memory/dpo_dataset \
  --model /mnt/afs/task3_2/share_model/<qwen-transformers-checkpoint> \
  --output-dir outputs/slow_memory/qwen38_dpo \
  --resume-from-checkpoint outputs/slow_memory/qwen38_dpo/checkpoint-25
```

The 27B default uses Unsloth QLoRA, saves resumable TRL checkpoints, evaluates
when a validation split exists, records Trackio metrics, and exports the final
adapter plus `training_manifest.json`. Hub upload is disabled unless explicitly
enabled in YAML and an `HF_TOKEN` is present.

## 4. Run the trained full experiment

Serve the adapter with its exact base model (or serve an offline merged model)
under a stable alias. Select it only for ablation 5:

```bash
export SCENEEXPERT_FULL_MODEL_ID="Qwen/SceneExpert-DPO"
python main.py experiment=ablation_5_qwen3_full
```

When `SCENEEXPERT_FULL_MODEL_ID` is unset, the full configuration falls back to
the existing `SCENEEXPERT_MODEL_ID`. Ablations 1-4 are unaffected.
