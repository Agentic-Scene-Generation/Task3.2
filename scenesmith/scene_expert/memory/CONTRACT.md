# Fast Memory reader and evidence contract

This change implements `memory-contract.v1`. Persisted records remain backward
compatible with `sceneexpert.memory.v3`; older records load with conservative
defaults. It does not change the native stage order, Critic decisions, Designer
tools, physics, or stage failure policy.

## One record, one payload

`RetrievedMemorySelection` is the canonical retrieval unit. Its typed identity
(`memory_type`, `memory_id`), source path, source tasks/runs, content hash, text,
and optional placement text are derived from the same persisted record.

- Equal text from distinct records survives recall until applicability checks.
- Admission reconstructs the payload from the store and rejects a changed hash.
- A selected record cannot inherit the layout of a pruned record.
- Duplicate IDs within a record type are ambiguous and rejected by admission.
- Legacy parallel text/ID arrays are used only when aligned. Unbound global
  layout text is not accepted as a verified reference.
- The character budget admits or rejects a whole record, including its layout.
  No later 300-character slice removes a repair action or its safety checks.
- Per-selection provenance is authoritative. Legacy untyped provenance maps omit
  keys that collide across success, failure, and skill records.

This is the retrieval/admission contract, not evidence that a Designer actually
used an instruction. Planner acceptance and decision-time delivery are separate
follow-up work.

## Spatial evidence is local to a constraint

Spatial text preserves count, grouping, orientation, reference frame, offsets,
yaw, clearance, and explicit template parameters where present in the record.
New writer records retain each native check's ID, stage, label, evaluation state,
tier, constraint hash, result hash, object bindings, and original observations.
These observations are added to `verify_report.hard_check_report.constraint_evidence`;
existing score calculations and native check results are not altered.

The writer joins evidence by stage, constraint ID, and exact constraint content.

| State | Meaning |
| --- | --- |
| `requirement_only` | A task asks for the relation; there is no matching per-check evidence. |
| `verified_pass` | All matching checks passed, are core/evaluated, and the spatial claim is unchanged. |
| `verified_fail` | At least one matching core/evaluated check failed for the unchanged claim. |
| `inconclusive` | Evidence is unknown, deferred, auxiliary, malformed, or mismatched. |
| `unknown` | Legacy record without this evidence contract. |

A whole stage passing does **not** prove every requested relation. A whole stage
failing does not invalidate a separately verified relation. The legacy
`geometry_verified` flag alone is not authoritative; consumers use
`has_verified_geometry`. Positive and negative verification labels are bound to
the current spatial claim hash.

Bootstrap skills may explicitly parameterize counts/groups for the current
task. Their `template_parameters` disclose this transformation; full source
constraints/counts remain in the observations. A source example passing does
not prove that its generalized skill will pass in another room or cause a gain.

Raw evidence stays in the trace and persisted memory. The MemoryWriter model
receives a compact projection of check outcomes and object bindings, avoiding
repeated geometry payloads; the model does not assign verification status.

## Existing banks and indexes

Do not delete or reset an existing bank for this change. IDs, original records,
and legacy world coordinates remain readable and available for auditing.
Unframed legacy coordinates are no longer injected as transferable, verified
layouts. Semantic lessons remain eligible subject to the existing admission
policy; unverifiable relations are not silently upgraded. Explicit updates are
separate from accumulating observations: incompatible spatial claims are not
merged merely because their record IDs match.

Derived vector indexes use `memory-text.v2`. Index building always renders the
current canonical record instead of trusting stale stored `embedding_text`.
Old-format indexes are stale and rebuild when `auto_build_missing=true` (the
current ACP launcher enables this). The first use can incur embedding cost.

For read-only/frozen banks, an index path inside the bank is redirected to an
external temporary cache under `scenesmith-memory-index/<path-hash>/memory-text.v2`.
Record files, manifest, revision, and source evidence are not rewritten. An
explicit external index directory is also supported. If automatic rebuilding
is disabled, rebuild using the same embedding model as runtime:

```bash
python scripts/build_memory_index.py \
  --memory-dir "$MEMORY_DIR" \
  --index-dir "$EXTERNAL_INDEX_DIR" \
  --embedding-model-dir "$SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR" \
  --read-only-memory
```

Use a new pair identity for future OFF/ON runs after code/contract changes. Do
not compare an old-code OFF run with a new-code ON run. This contract repair
does not itself establish a performance improvement.

## Read-only audit

Set `MEMORY_DIR` to the directory containing the three JSONL bank files. For
example, after activating the project's Python environment on the server:

```bash
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2
python -m scenesmith.scene_expert.memory.audit_contract \
  --memory-dir "$MEMORY_DIR" \
  --output "tmp/memory_contract_audit_$(date +%Y%m%d_%H%M%S).json"
```

The audit reads raw JSONL without opening or initializing a store. Output must
be a new file outside the bank. It records source file hashes, payload hashes,
source locators, evidence eligibility, omitted-layout warnings, malformed rows,
and duplicate identities. `reader_eligible` means readable active content, not
task compatibility, actual use, verified geometry, or measured benefit.

## Regression checks

CPU contract, writer, store, index, skill, and paired-metric tests:

```bash
python -m pytest --confcutdir=tests/unit \
  tests/unit/test_memory_contract.py \
  tests/unit/test_sceneexpert_memory_selection_policy.py \
  tests/unit/test_sceneexpert_memory_v3.py \
  tests/unit/test_memory_writer_resilience.py \
  tests/unit/test_hybrid_memory_index_lifecycle.py \
  tests/unit/test_memory_store_v2.py \
  tests/unit/test_sceneexpert_skill_memory.py \
  tests/unit/test_sceneexpert_evaluation_inputs.py \
  tests/unit/test_sceneexpert_paired_metrics.py \
  tests/unit/test_sceneexpert_run_metrics.py -q
```

Also run `tests/unit/test_sceneexpert_verifier_evidence.py` and
`tests/unit/test_scene_expert_memory.py` in the full SceneSmith environment.
Their imports require Drake; the root test setup also requires Blender. CPU
contract checks are not a substitute for a server-side full scene run.
