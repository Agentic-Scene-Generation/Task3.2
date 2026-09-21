# SceneEval Qwen3.8 DPO campaign — 2026-09-21

## Decision and scope

Move from repeated single-scene rescoring to a bounded, resumable collection campaign.
Keep the existing architecture and native Designer/Critic/repair behavior. Changes are
in SceneExpert's collection, data curation, training and operator entrypoints.

The first trained policy covers **the first assistant turn of furniture initial
design**, credited with its independently executed rollout outcome. It does not yet
train all placement turns, repair decisions or all five stages. Full trajectories
remain on the server. Initial observation-only actions can be weak credit signals;
track first-turn pair yield separately and establish scene-level gains before making
a method-effectiveness claim. Two training steps on one old pair only verify plumbing.

Hidden reasoning is not present in these captures. The initial-policy profile
therefore conditions visible-action loss on Qwen's closed thinking envelope
(`enable_thinking=false` in the training template), and checks both textual and
token-level prompt prefixes before loading weights. It does not fabricate reasoning
targets. Decoding/thinking settings must be controlled in the later base/adapter
comparison; this profile is not evidence of improved reasoning generation.

## Why the previous process produced little training data

| Cause | Change |
| --- | --- |
| DPO required an absolutely accepted scene | Add an explicitly versioned relative policy; keep strict export alongside it |
| Furniture capture continued through all five stages | Batch collection stops at furniture; normal pipelines remain unchanged |
| Pilot entrypoint allowed at most four groups | Support up to 512 reserved group slots and multiple audited attempts |
| Full training gate required repair and three stages | Add a furniture-initial profile matching the actual collector |
| Multi-turn completions contained environment/tool results | Export the first assistant turn, exclude tool responses and transport call IDs |
| Inference environment could not load the trainable architecture | Separate `.venv_dpo`, pin the training API versions and verify real optimizer updates |
| Full-vocabulary logits for long prompts exhausted 80 GB during DPO, including the default fused loss | Remove globally loss-masked hidden positions immediately before the fused loss's vocabulary projection; retain the complete transformer input and the same DPO objective |
| Collection/splits could drift during retries | Freeze source contents, SceneEval task partitions and a memory snapshot; resume only missing tasks |

`verified_relative_v1` adds pairs only when execution and tool calls completed,
authoritative deterministic evidence is present, constraint sets agree, there are no
unknown checks, chosen quality is at least 0.50 and improves by at least 0.03.
Hard failures cannot increase; an imperfect winner must have strictly fewer failures.
Collision count, maximum penetration and total penetration must not increase.
Original accepted/rejected verdicts are preserved. Relative winners receive pure DPO,
not an added unconditional SFT loss. Strict export retains its existing 0.05 gate.
017 still does not qualify: its score/violation improvement conflicts with worse
maximum penetration. No label is forced to make a training set nonempty.

Exact prompt, tool schema, images, state, execution proofs and raw physics are not
relaxed. This is a relative-quality policy, not permission to pair unrelated states.
The distinction follows the preference-learning objective in the
[DPO paper](https://arxiv.org/abs/2305.18290).

## Prepared server resources

- Host: the user-provided `slai_debug_workspace_1` SSH endpoint.
- One H100 80 GB was visible and idle before the training smoke.
- Trainable base: `/mnt/afs/task3_2/share_model/Qwen/Qwen3.8-27B`.
- All 32 non-hidden repository files were already present and verified against
  [ModelScope's Qwen repository](https://modelscope.cn/models/Qwen/Qwen3.8-27B).
  Existing shared weights were not replaced and shared directory permissions were
  not changed. The report is `tmp/slow_memory_setup_019/base_verification.json`.
- Training dependencies are installed in `.venv_dpo`. Compatible installed packages
  were copied from the adjacent environment without modifying it, then checked by
  pip against `configurations/slow_memory/requirements_train.txt`. Versions are
  recorded in `.venv_dpo/sceneexpert_requirements.lock.txt`.
- Training smoke uses an isolated source copy under `tmp/slow_memory_setup_019/source`;
  this does not synchronize the operator's production checkout. Manually synchronize
  the final delivered code before starting the campaign.
- The warm seed bank passed read-only loading and known source-task leakage checks:
  47 success cases, 46 failure cases and 23 skills (one active, 22 candidate skills).

Real H100 smoke `qwen38_dpo_runtime_smoke_019d` finished with `exit_code=0`:
two optimizer steps, finite loss 0.693147, 256 nonzero LoRA B tensors and a saved
318,843,864-byte adapter. Reloading its safetensors on CPU confirmed all 512 tensors
were finite and its base/rank metadata matched Qwen3.8 / rank 32. The unchanged loss
is expected for this two-step warmup check and is **not** evidence of learning gains.
Training itself took 282.9 seconds, plus about 2.8 minutes to load weights; peak CUDA
allocation was 73.03 GiB with an untruncated 17,976-token prompt. Longer contexts need
another memory check; this is not an unlimited-context capacity guarantee. The two
completion-projection tests passed, including equality of prompt gradients.

Reports and logs are in `outputs/slow_memory/qwen38_dpo_runtime_smoke_019d`.
The verified lightweight review archive is
`tmp/results/slow_memory/qwen38_dpo_runtime_smoke_019d_review.tar.gz` (77,429 bytes).
The infrastructure-only adapter is explicitly nonpromotable and is not deployed.
Validation also passed 130 portable collection/packaging tests, 54 native
Slow Memory/memory-evaluation tests, two Torch loss/gradient tests and Bash syntax
checks for all three new launchers. The server training environment passes
`pip check`.

For another server, preparation is reproducible without Git:

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
mkdir -p tmp/slow_memory_setup_019
.venv/bin/python scripts/download_sceneexpert_base.py \
  --destination /mnt/afs/task3_2/share_model/Qwen/Qwen3.8-27B \
  --report tmp/slow_memory_setup_019/base_verification.json
TRAIN_PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
  bash tmp/acp/setup_qwen38_dpo_training.sh
```

## Collection ACP

The frozen pool contains 128 train, 24 validation and 32 held-out test tasks, sampled
by difficulty from distinct single-room SceneEval IDs 100–499. IDs 0–99 are excluded
because they were used in previous development. Test tasks are never collected into
the preference dataset. This prevents exact-task leakage, not all possible semantic
similarity between natural-language tasks.

Start **72 groups / 144 candidate executions**, automatically processed in chunks of
12. Every chunk is audited and exports both data policies before the next begins;
there is no operator approval between chunks. Invalid evidence is quarantined.
The same command resumes missing tasks; an intentionally capped successful batch
returns zero even while the larger pool has pending tasks. A failed chunk with no
valid completed group stops instead of repeating a system failure across the pool.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_sceneeval_dpo_019 \
MAX_TASKS=72 CHUNK_SIZE=12 ACP_PARALLELISM=2 \
MEMORY_SEED_DIR="$PWD/outputs/scene_expert_memory/frozen/full_memory_pair_sceneeval100_hard_qwen38_v5/ablation_4c" \
MODEL_NAME=unsloth/Qwen3.8-27B-GGUF \
MODEL_DIR=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF \
MODEL=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf \
MMPROJ=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf \
bash tmp/acp/acp_qwen38_dpo_campaign.sh
```

The seed bank is copied and validated, including known source-task leakage checks.
This dedicated collection profile reads a frozen bank and disables MemoryWriter,
because it stops before full-scene completion and must support controlled evaluation.
The normal five-stage pipeline retains its existing MemoryWriter behavior.
Omitting `MEMORY_SEED_DIR` deliberately creates a cold-start bank; do not silently
substitute that condition when comparing to a warm-memory baseline.

The first chunk is also the throughput/yield checkpoint. Read
`outputs/slow_memory/qwen38_sceneeval_dpo_019/campaign_audit.json` and both
`datasets/*/stats.json` (`raw_preference_pair_count` versus `eligible_pair_count`).
Keep execution completeness, evidence validity, raw pair
yield and first-turn training yield separate. The data gate of 16 train groups is
an engineering floor, not a sufficient paper sample size. Aim for 64–128 usable
independent training pairs if observed yield and the one-week budget permit it.

After the first 72, change `MAX_TASKS=0` in the same ACP to collect the remaining
train/validation pool, with the same source/model/memory settings. This is a
planned 152-group pool, not a promise of 152 usable DPO pairs. Do not train while
the same single H100 is occupied by collection services.

Packaging after any collection invocation:

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
mkdir -p tmp/results/slow_memory
.venv/bin/python scripts/package_sceneexpert_results.py \
  --run-id qwen38_sceneeval_dpo_019 \
  --output "tmp/results/slow_memory/qwen38_sceneeval_dpo_019_review_$(date +%Y%m%d_%H%M%S).tar.gz" \
  --max-file-mib 32 --max-total-mib 512
```

The archive includes attempt service logs under their original `tmp/acp_logs/...`
paths, configuration, traces, reports, dataset statistics and bounded evidence.
It omits weights, meshes, databases and replay asset copies; its manifest records
checksums, omissions and truncation. Full replay data and training data stay on the
server. An archive taken during collection is a progress snapshot, not completion
evidence.

## Training ACP after collection services have stopped

Use a fresh training run ID. Preflight requires at least 16 distinct training groups
and nonempty validation. No held-out scene test task is used to select hyperparameters.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
CAMPAIGN_ID=qwen38_sceneeval_dpo_019 \
RUN_ID=qwen38_dpo_initial_020 \
PROFILE=furniture_initial \
bash tmp/acp/acp_qwen38_dpo_train.sh
```

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
mkdir -p tmp/results/slow_memory
.venv/bin/python scripts/package_sceneexpert_results.py \
  --run-id qwen38_dpo_initial_020 \
  --output "tmp/results/slow_memory/qwen38_dpo_initial_020_review_$(date +%Y%m%d_%H%M%S).tar.gz" \
  --max-file-mib 32 --max-total-mib 128
```

Exit 2 is preflight/data failure; exit 3 means training finished but the offline
validation promotion gate was not met. Both must retain their diagnostics. The
adapter is never automatically published or deployed. Check optimizer steps,
finite loss, nonzero LoRA updates and saved adapter files before claiming that the
post-training implementation works.

## One-week scheduling budget

These ranges use one H100 and two collection workers; two workers do not imply 2x
GPU throughput. Observed 017 took about 126 minutes in total, about 68 minutes in
furniture and roughly 46 minutes to capture both raw candidates. New tasks and
shared-GPU contention introduce substantial uncertainty.

| Work | Preparation / hands-on | Unattended server execution | Validation / hands-on | Packaging |
| --- | --- | --- | --- | --- |
| First 72 groups | 10–20 min code synchronization and configuration | 36–72 h, including chunk restarts; recalibrate after first 12 | 15–30 min per checkpoint, 1–2 h total | 2–10 min, up to 512 MiB before compression |
| Remaining pool, only if time/yield justify it | 5–10 min | another 40–80 h, do not automatically spend the remaining week | 30–60 min | 2–10 min |
| QLoRA-DPO on 16–128 eligible pairs | 10–20 min | about 1.5–12 h for two epochs plus validation, extrapolated from 141 s per pair-microstep at 17,976 prompt tokens | 20–40 min | 1–5 min, up to 128 MiB |
| Held-out furniture evaluation | 1–2 h deployment/control verification | budget 24–48 h for matched base/adapter runs, subject to measured throughput | 2–4 h paired analysis and confidence intervals | 5–15 min |

Day 1–3: collect, audit every chunk and measure yield. Day 3–4: train and validate;
if strict data suffice, compare strict and relative curation as an ablation. Day
4–6: evaluate the frozen held-out tasks with identical inputs, budgets and memory,
keeping the critic/evaluation policy fixed. Day 7: analyze paired completion,
constraint/physics outcomes, quality, time and token cost; report uncertainty and
failures as well as gains. Adapter serving and the controlled held-out execution
must be validated before starting this evaluation; the training smoke alone does
not establish them. The operational commands above cover the immediately executable
collection/training steps; deployment commands will be fixed to the actual trained
adapter rather than an invented future checkpoint.

If a second H100 becomes available, prefer separating collection and training or
evaluation resources, after assigning distinct service ports and output roots.
Do not merely duplicate the same fixed-port ACP process.
