# SceneExpert Memory Bank Contract

The memory directory is a durable, inspectable database. It may start as an
empty directory; `FastMemoryStore` creates the files and manifest on first use.

## Durable files

- `manifest.json`: stable bank ID, schema version, monotonic revision, per-bank
  revisions, and record counts.
- `success_cases.jsonl`, `failure_cases.jsonl`, `skills.jsonl`: one validated
  record per line.
- `events.jsonl`: evidence journal. Events are not retrieved as long-term
  memory.
- `indexes/`: disposable derived vector indexes. They are rebuilt when the bank
  ID, per-bank revision, or indexed record IDs no longer match.

## Lifecycle

1. Stage and repair hooks append critic/repair evidence to `events.jsonl`.
2. The final MemoryWriter receives the untruncated structured critic reports.
3. Qwen returns compact, schema-constrained candidates only.
4. Deterministic code supplies IDs, task metadata, critic evidence, quality,
   provenance, and promotion status.
5. Only evidence-backed records are promoted with `status="active"`.

Model failure, invalid JSON, empty output, or an unsupported lesson produces a
diagnostic no-write result. No fallback record is inserted into the active bank.
`candidate` and `quarantined` are reserved lifecycle states and are excluded
from both lexical and hybrid retrieval.

Successful records additionally require final and stage pass evidence plus the
configured `memory.writer.success_min_overall_score` (environment override:
`SCENEEXPERT_MEMORY_SUCCESS_MIN_SCORE`, default `0.75`).

Retrieval excludes records supported only by the identical prompt task by
default (`memory.retrieval.exclude_same_task=true`). A record observed across
multiple task IDs remains eligible, which keeps shared knowledge available
while preventing same-prompt replay leakage.

## Concurrency and maintenance

ACP runs use a directory-level Linux file lock. Each mutation batch rewrites
only affected JSONL banks atomically and increments the manifest once. A
long-lived worker checks the manifest and file signatures before retrieval, so
records written by another worker become visible without restarting the job.

Back up or move the complete directory, including `manifest.json`. Vector
indexes may be deleted and rebuilt. Existing pre-v2 JSONL records remain
readable and load as active records with `source="legacy"`; use a fresh memory
directory for clean experiments when old fallback records should not carry
forward.
