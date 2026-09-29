# SceneEval 019 audit and next DPO pilot — 2026-09-29

## Decision

Proceed with an explicitly exploratory DPO pilot using the existing 15 training
pairs and seven validation pairs. Do not spend the next cycle retrying all failed
scenes or lower evidence requirements to force more pairs. This is enough to test
learning behavior and adapter plumbing, but not to establish scene-generation
effectiveness or a paper-scale dataset.

The regular `furniture_initial` profile still requires 16 independent training
groups. The opt-in `furniture_initial_pilot` requires eight training groups and
four validation pairs, retains the same exact-context/evidence/assistant-only
contracts, and is always nonpromotable. It uses pure sigmoid DPO, two epochs and
no SFT term. Its thresholds are operational pilot floors, not statistical power
claims. No labels, score gaps, split assignments or held-out tasks were changed.

## Verified execution, evidence and yield

The last ACP invocation ran from 2026-09-28 10:35:04 UTC through the final campaign
audit around 2026-09-29 03:16 UTC, approximately 16 hours 41 minutes. It processed
68 tasks after the original four completed CCI groups. Collection exited 2 and
packaging exited 0. No collection processes remained during SSH inspection.

| Measure | Observed result |
| --- | --- |
| Attempted task total | 72 |
| Valid complete independent decision groups | 55 / 72 (76.4%) |
| Reserved but invalid groups | 13 |
| Tasks failing before pair capture | 4 |
| Valid trajectories | 110 |
| Strict raw / first-turn eligible pairs | 12 / 11 |
| Strict train / validation | 7 / 4 |
| Relative-policy raw / first-turn eligible pairs | 27 / 22 |
| Relative-policy train / validation | 15 / 7 |
| Eligible yield per complete group | 22 / 55 (40.0%) |
| Eligible yield per attempted task | 22 / 72 (30.6%) |
| Test pairs used for training/selection | 0 |

The 22 include the 11 strict pairs; these are overlapping exports, not 33 pairs.
The 33 complete groups without eligible pairs consist of 21 without an accepted
candidate under the applicable curation policy, seven without eligible contrast,
and five whose first assistant turns are identical after normalization.

An independent server-side audit rechecked all retained groups and both datasets
in 411 seconds. Dataset validation passed; task assignments match the frozen split;
train/validation task intersection is empty. These checks address exact-task
leakage and evidence integrity, not semantic independence of all natural-language
tasks. Full replay assets remain on the server.

Sources: `outputs/slow_memory/qwen38_sceneeval_dpo_019/campaign_audit.json`,
`datasets/*/{manifest,stats}.json`, per-attempt `pair_audit.json`, `run_metrics.json`,
and `tmp/review_20260929/{independent_audit,training_context_audit}.json`.

## Root causes and scientific limits

- Ten groups failed the tool-execution gate after HSSD returned no candidates:
  five toilets and one each of speaker, floor pillow, metal tray, wooden tray and
  bronze statuette. Asset coverage/retrieval limits are distinct from preference
  score thresholds. Main-scene recovery does not repair a failed shadow candidate.
- Three groups lack complete deterministic evidence: required entities or
  relational roles could not be bound (circular rug, workstation, shoji screen).
  Unknown constraints remain unknown; they are not negative labels.
- Four tasks never reached capture. SceneEval 381 and 418 exhausted the intent
  output budget; 431 and 440 produced `edge_distribution` without orientation.
  These are compiler output/schema failures, not training or packaging failures.
- The current export learns only the first assistant turn. All 22 pairs call
  observation/todo tools, with one rejected alternative also listing assets.
  Todo arguments may encode a plan, but object placement and repair actions are
  not trained directly. Whole-rollout outcomes supply indirect credit to this
  initial decision. A validation gain alone cannot show better final scenes.

The next collection-design improvement should fork after shared observation and
before a consequential asset/layout decision, preserving the exact input state
for both candidates. This is a separate collector change; do not fabricate such
pairs from divergent historical prompts. Keep 019 immutable after this audit.
Its source fingerprint will differ from newly delivered code, so do not attempt
to resume it with the new checkout or bypass its provenance guard.

## Code and archive synchronization

The remote source bytes initially matched all 1,046 tracked local files, but its
HEAD/index still pointed at `5c88493` while the files and remote-tracking ref were
at `a4b71a8`. Prior file-only synchronization caused the apparent uncommitted
changes. The old Git state was backed up and aligned only after verifying the
complete tracked-file comparison. No independent remote edits were discarded.

Subsequent delivery uses a local English commit/push and an offline Git bundle to
fast-forward CCI, followed by full tracked-content hashes, matching HEADs and a
clean worktree check. Experiment commands contain no Git operations. These
synchronization and CCI resource limits are recorded in `AGENTS.md`.

The original review archive is 67,190,230 bytes and passes all 5,123 member checks
both remotely and locally. Its size cap omitted some later attempt summaries
after earlier raw records consumed the budget. Packaging now prioritizes global
and per-attempt summaries, exit markers, failures and ACP logs before bulky raw
records. Omissions remain explicit; a bounded review archive is not a complete
training dataset or a substitute for server-side replay auditing.

The refreshed `qwen38_sceneeval_dpo_019_review_complete_20260929.tar.gz` is
68,603,800 bytes (65.4 MiB), contains 5,219 verified members and includes all six
existing per-attempt `run_metrics.json` files. It was downloaded and independently
verified locally. The original CCI attempt stopped before producing its metrics
file; its four pair groups were audited directly. Refreshing the archive took
74.5 seconds and did not rerun collection or change source results.

## Next ACP: exploratory training

The longest-pair CCI probe `qwen38_dpo_capacity_019e` caught a backward-pass OOM
at 19,273 rendered tokens: the process occupied 78.37 GiB and could not allocate
another 1.25 GiB. The earlier 17,976-token smoke did not establish capacity for
this dataset. Initial-policy profiles now enable TRL's native activation
offloading to CPU alongside checkpointing and completion-only fused projection.
This changes activation storage, not the retained context or the DPO objective.
See the [TRL memory guide](https://huggingface.co/docs/trl/reducing_memory_usage)
and the pinned local DPOConfig implementation. The generic full profile retains
its prior setting. The next probe 019f confirmed a TRL composition issue: its fused loss uses
`torch.func`, which rejects active saved-tensor hooks. The SceneExpert adapter
pauses only hook registration around the compact fused loss while retaining the
offloader tracker and streams for model backward. It restores registration on
exceptions and leaves evaluation without an entered offloader unchanged. A fresh
longest-pair check is required before the pilot ACP.

Use one H100 80 GB. The pilot is estimated to exceed one hour and therefore must
run through ACP. The two-step capacity probe is a separate, nonpromotable CCI
check on the longest training pair; validation examples are never used by it.
The training pilot below always starts from the original safetensors base, not
from a capacity-probe adapter. Keep collection/inference services off this GPU.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

CUDA_VISIBLE_DEVICES=0 \
CAMPAIGN_ID=qwen38_sceneeval_dpo_019 \
RUN_ID=qwen38_dpo_pilot_020 \
PROFILE=furniture_initial_pilot \
BASE_MODEL=/mnt/afs/task3_2/share_model/Qwen/Qwen3.8-27B \
PACKAGE_MAX_FILE_MIB=32 PACKAGE_MAX_TOTAL_MIB=128 \
bash tmp/acp/acp_qwen38_dpo_train.sh
```

The launcher automatically packages and verifies the run on success or training
failure. Its printed `Verified review package` path is the download target under
`tmp/results/slow_memory/`. Inspect `workflow_exit_status.env`: training and
packaging exits must be interpreted separately. Adapter weights/checkpoints stay
on the server; logs, config, metrics, template audit and training manifest retain
their original project-relative paths. For packaging only, without retraining:

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
PACKAGE="tmp/results/slow_memory/qwen38_dpo_pilot_020_review_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
.venv_dpo/bin/python scripts/package_sceneexpert_results.py \
  --run-id qwen38_dpo_pilot_020 --output "$PACKAGE" \
  --max-file-mib 32 --max-total-mib 128
.venv_dpo/bin/python scripts/package_sceneexpert_results.py --verify "$PACKAGE"
```

Pilot completion requires finite training/validation loss, real optimizer steps,
nonzero LoRA updates, saved adapter and a complete manifest. Exit 0 means the pilot
completed, not that quality improved. `promotion_gate.status=pilot_only` and
`promotable=false` are intentional, including when offline validation passes.
With only seven validation tasks, report counts and uncertainty, not a strong
effectiveness conclusion. The next decision is controlled base/adapter serving
and scene evaluation if the pilot has a useful signal; broad collection resumes
only with a more consequential decision target. No checkpoint is deployed by
this command.

| Work | Preparation / hands-on | Unattended execution | Validation / hands-on | Packaging |
| --- | --- | --- | --- | --- |
| Pilot: 15 train, 7 validation, two epochs | 3–5 min to submit | 1.5–2.5 h | 10–20 min | 1–5 min automatically |

Runtime is extrapolated from the earlier 019d H100 smoke (282.9 seconds for two
pair microsteps, plus about 2.8 minutes loading), allowing for longer inputs,
gradient accumulation, two validation passes, saving and shared-filesystem I/O.
The current maximum prompt has 19,190 tokens, versus 17,976 in 019d. These are
estimates, not guarantees. The user need only submit this ACP and return the
lightweight archive; there is no need to repeat the completed collection batch.
