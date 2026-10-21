# Pair-local four-candidate collection

`--pool-completion-policy pair_local_v2` explicitly enables
`sceneexpert.fixed_candidate_pool.pair_local_v2` in
`scripts/run_sceneexpert_campaign.py`. The default remains `all_four_v1`.
The option requires four candidates and the `first_mutation` boundary. A new
campaign is required; an existing campaign cannot change its recorded policy.

The canonical A rollout and the native B/C/D worker bodies are unchanged. All
four slots are recorded. A has no independent process exit. B/C/D retain their
observed process exits, including timeout termination, or a null exit when no
process started. Candidate completion failure does not stop later slots.
Changed source, frozen request, native budget, base files, or canonical/memory
invariance aborts the group. A recorded schedule cannot run twice.

The portable audit admits at least two complete, direct, independently checked
candidates from the frozen request. It applies the same raw-state, fresh physics,
safety, trajectory, media, scope-draw and native verdict checks as v1. Partial,
unknown and nondirect candidates are never negative targets. The existing native
ranking produces at most one pair per original task. Independent semantic review
can quarantine that pair; it does not search for a different favorable pair.

`pool_complete` and `pair_candidate_pool_eligible` are separate audit fields.
An admitted partial pool retains its failure and makes collection/audit exit
nonzero even when a native pair can be exported. Native exports are research
evidence until the independent qualification and delivery checks finish.

The named `furniture_initial_pairlocal_pilot_v2` training profile retains the
8 original train / 4 independent validation floor, final-checkpoint QLoRA sigmoid
DPO recipe and pilot-only status. Its preflight also requires a byte-bound
`qualification.json`, the original partition and approved prefix, source bridge,
semantic decisions, native pair and full-review delivery proofs. A raw export or
the existing 3T3V cannot pass. The source bridge may inherit only the six already
certified 605 groups; a changed runtime is a separate source identity.

The approved bounded acquisition order is 205T/212T + 188V/229V, then
297T/120T/419T + 193V, then 358T/198T/224T + 257V. The implementation flag does
not itself grant collection permission. Each allocation needs its immutable
source/control guard and matching review pack. A zero-yield batch or global
integrity/bridge failure stops the gap-filling plan. Protocol 065/071, verifier
obligations, thresholds, model requests, stage budgets and frozen test tasks are
outside this change.
