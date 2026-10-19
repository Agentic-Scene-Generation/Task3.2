# Fixed four-candidate collection

The optional `--candidates-per-context 4` campaign argument freezes one natural
canonical A and three private continuations B, C and D from A's exact first
mutation request. The default remains two candidates. Four candidates are only
supported at the `first_mutation` boundary, with all original prompt, message,
tool, visual, scene, memory and remaining native-turn inputs preserved.

The worker schedule is committed before any shadow runs and never stops because
a preferred answer has been found. Every shadow uses the existing quality-blind
scope sampler, with at most three draws and no forced tool choice. Worker exits,
requests, responses, usage, selected scope draws, raw states, fresh physics,
assets, media, memory and canonical-invariance proofs remain auditable. B, C and
D have distinct private memory directories. Native failure and partial pools are
preserved and cannot pass the complete-pool audit.

The model never receives another candidate's score or response: quality does not
condition generation or the fixed execution schedule. Raw scoring follows each
native rollout; offline ranking happens after the pool has completed. This does
not promise four distinct answers. The audit reports completion diversity and
independent original task groups separately from candidate count.

All four requests must equal A's complete wire request. A changed request, missing
candidate, wrong candidate identity, nonzero worker exit, modified asset or
canonical/memory drift quarantines the pool. Export uses the existing strict and
verified-relative comparators and the existing quality margins. It produces at
most one pair for an exact context. A four-candidate scene remains one original
task group; the unchanged 8-train/4-validation pilot admission counts groups.
Existing semantic qualification is still required after the native audit.

The campaign records the candidate-pool policy in `collection_protocol.json` and
each frozen snapshot. Existing A/B campaigns cannot be silently reinterpreted as
four-candidate campaigns. Runtime content fingerprints identify the new source;
old cohorts and original labels remain separate. Legacy A/B rescoring explicitly
rejects a four-candidate pool instead of silently dropping C and D.

This extension changes data collection only. It leaves the DPO loss, model,
IntentCompiler budget, physics, SceneSmith agent loop, memory behavior, fixed task
partition and verifier thresholds unchanged. It does not implement the pending
workstation IR or semantic identity/applicability proposals.
