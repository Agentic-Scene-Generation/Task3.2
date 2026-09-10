# Bounded Fast Memory collection

This workflow replays at most eight archived inputs through the existing Writer.
It starts no scenes, changes no main critic/asset logic, and never automatically
selects a seed bank, starts OFF/ON runs, or starts training. Existing single-scene
`scripts/run_qwen38_memory_writer_replay.sh` usage remains unchanged.

## Input contract

Create a UTF-8 JSON manifest with `schema_version: bounded-writer-replay.v1`,
`cases` and `holdout_cases`. Each collection case includes a filesystem-safe
`case_id`, project-relative `scene_expert_dir`, `prompt_sha256` and `input_sha256`.
Compute the latter with `fingerprint(load_input(source))` from the batch module.
Holdouts include a distinct `case_id` and `prompt_sha256` computed with
`fingerprint(prompt)`. Prompt hashes reject exact duplicates, not paraphrases;
human review must check semantic overlap. Every input is checked before starting
the model. Changed evidence must be reviewed, not bypassed by disabling checks.
These are artifact checks, not Git checks.

The plan under ignored `tmp/experiments/fast_memory_bounded_v1/` fixes six source
batches (042, 043, 044, 047, 048, 095) and four holdouts (041, 045, 092, 093).
These holdouts were historically generated but are excluded from this new seed
collection. This does not certify complete independence from all historical
memory exposure. Diagnostic batch 091 is not a seed source.

## Execution

Copy `plan.json` and `run_collection.sh` from that tmp folder to the same relative
folder on the server; Git deliberately does not distribute tmp files. Then run:

```bash
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
bash tmp/experiments/fast_memory_bounded_v1/run_collection.sh
```

One Qwen service is owned for the whole collection and stopped on exit. An
existing occupied port is never replaced. Only set
`REUSE_EXISTING_MODEL_SERVICES=true` for a deliberately managed compatible
service in the same CCI instance. The existing launcher controls model settings;
no manual GPU labels, database replacement, or new source schema is required.

For local projection without an API/service:

```bash
python scripts/replay_sceneexpert_memory_batch.py \
  --manifest tmp/experiments/fast_memory_bounded_v1/plan.local.json \
  --output-dir tmp/memory_collection_dry_run --dry-run
```

`--validate-only` checks all inputs without creating output. Output must not
exist. A failed replay stops the remaining paid work, checkpoints the result and
lists `not_attempted`; it does not implicitly retry the collection. The underlying
Writer retains its existing bounded request-retry policy.

## Outputs and next gate

`tmp/<RUN_ID>/` contains the fixed `plan.json`, aggregate `replay_summary.json`,
`review_candidates.json` (full records, all pending review), and a per-case folder
with its own `audit/`, `bank/`, and summary. Keep all JSON/JSONL files; these contain
actual evidence, not merely references. Service/terminal logs are in
`tmp/acp_logs/<RUN_ID>/`. No bank is merged or marked as an approved experiment
seed automatically. Individual persisted record statuses still use the existing
Writer policy.

Review 3–5 source-grounded, nonredundant operational lessons before transfer.
Record why each candidate is included or excluded. Retain unresolved critic
advice as advice, not verified repair. Prefer meaningful decisions/conditions
over task restatements or copying numeric source extrema. If six sources are
insufficient, at most two additional preregistered sources and one focused prompt
revision are allowed before reconsidering this route.

Only after content approval, prepare a separate frozen `ablation_4c` bank and
four held-out furniture-stage OFF/ON pairs with Writer off, shared compiled inputs
and shared base. Assess actual delivery, relevant spatial quality, native rework
and total stage cost. Stage-only checks do not demonstrate full-pipeline quality;
do not weaken existing comparison gates to manufacture a ready result. Repeat
the same preselected tasks in reverse arm order if results are promising, then
perform limited full-pipeline regression. Otherwise preserve the evidence and
proceed with retrieval off rather than adding unbounded complexity.
