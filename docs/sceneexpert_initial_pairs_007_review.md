# Initial pairs 007 review and 008 recovery pilot

Reviewed on 2026-09-14. Input is the extracted lightweight review package under
`tmp/results/slow_memory/qwen38_initial_pairs_007_review_20260914_084215/`.

## Verdict

Canonical generation completed, but the independent-pair experiment failed.
Do not expand collection or use this run as a paired DPO dataset.

| Check | Observed |
| --- | --- |
| Package integrity | All 840 manifest entries present; sizes and SHA256 hashes match |
| Package warnings | None; replay asset omissions are intentional |
| Canonical scenes | 2/2 complete; both have all five stages |
| Canonical final pass | Bedroom passes; living room has `deterministic_pass=false` and `pass_scene=false` |
| Observer trajectories | 36; all identify `unsloth/Qwen3.8-27B-GGUF` |
| Designer trajectories | 13 distinct contexts; zero contexts with multiple candidates |
| MemoryWriter | 9 persisted records reported by the collection audit |
| Independent decision groups | 2 reserved; both fail before snapshot commit |
| Eligible candidate records | 0 |
| DPO preference pairs | 0; exported JSONL splits are empty |
| Exit states | Generation 0; final pair audit 2; `gate_passed=false` |

These are the copied server audit results, not a new full replay audit. The local
review verifies package bytes and inspects logs; native asset trees remain on the
server. Canonical success is not evidence that independent A/B collection worked.
The living-room quality issue is separate from this startup failure. Its final
scene score must not label an earlier raw Designer candidate.

Evidence within the package's original paths:

- `outputs/slow_memory/qwen38_initial_pairs_007/runs/paired_initial/pair_audit.json`
- `outputs/slow_memory/qwen38_initial_pairs_007/runs/paired_initial/exit_status.env`
- `outputs/slow_memory/qwen38_initial_pairs_007/runs/paired_initial/group_000/status.json`
- `outputs/slow_memory/qwen38_initial_pairs_007/runs/paired_initial/group_001/status.json`
- `outputs/slow_memory/qwen38_initial_pairs_007/runs/collection/collection_audit.json`
- `outputs/slow_memory/qwen38_initial_pairs_007/runs/critic_on/batch_001/hydra/scene_000/room_bedroom/room.log`

## Root cause and repair

Both group status files report `unsupported initial snapshot value: Timeout`.
The bedroom traceback at room.log lines 47-73 identifies the failing field:
`json_value(agent.designer.model_settings)` in `open_initial_pair`.

The native model-settings factory inserts an `httpx.Timeout` object into the SDK
dataclass's `extra_args.timeout`. The snapshot serializer descended through the
dataclass but could not encode this transport object. It therefore failed before
writing `snapshot.json`, capturing A or starting B. This is a serializer defect,
not an HTTP timeout or model inference failure. The optional observer correctly
allowed canonical execution to continue; the final pair gate correctly failed.

The SceneExpert fix:

1. Preserve typed `httpx.Timeout` values, including separate connect/read/write/pool
   limits and disabled (`null`) dimensions. Native B still reconstructs real settings
   from the frozen configuration and must match before it runs. No timeout is
   removed, converted to a display string or injected into the model request body.
2. Serialize dataclass fields recursively without deep-copying SDK objects. Unknown
   values still fail with their exact field path. Validate runtime state before
   copying potentially large scene assets.
3. Record the failure phase/type and propagate the original group error into the
   pair audit, alongside any missing-file evidence.
4. Run a real-SDK codec preflight before model/service launch, saving
   `tmp/acp_logs/<RUN_ID>/pair_preflight.json` and `pair_preflight.log`. The preflight
   only certifies snapshot serialization; it does not certify GPU scene execution.

No native Designer/Critic/repair code or normal MemoryWriter behavior is changed.
The old groups cannot be repaired into pairs: their decision snapshots and B
executions never existed. Keep 007 as a diagnostic run.

Local validation: 57 relevant regression tests passed, including the real native
model-settings factory with SDK objects at snapshot creation, changed timeout
dimensions, failure diagnostics, and preflight exit propagation. Both archived
007 furniture configurations also passed model-settings serialization and
reconstruction comparison. These checks do not replace the native GPU pilot.

## Next ACP run

Manually synchronize the fix, then run one bedroom group with a fresh RUN_ID. This
limits exposure while the first complete A/B execution is validated.

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_008 \
PAIR_GROUPS=1 \
CASE_SET=legacy8 \
SCENE_SELECTION=default_bedroom \
CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE=false \
MODEL_DIR=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF \
MODEL=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf \
MMPROJ=/mnt/afs/task3_2/share_model/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf \
bash tmp/acp/acp_qwen38_initial_pairs.sh
```

After the process stops, whether it succeeded or failed, package the same RUN_ID:

```bash
set -euo pipefail
cd /mnt/afs/task3_2/L202500276_lwz/projects/Task3.2-dev_lwz_pre_merge_v2

RUN_ID=qwen38_initial_pairs_008 \
bash tmp/acp/pack_qwen38_initial_pairs.sh
```

Download the printed `tmp/downloads/qwen38_initial_pairs_008_review_<timestamp>.tar.gz`
and its `.sha256` sidecar. Original relative paths are preserved; exclusions and
any oversized log excerpts are recorded in the package manifest. A preflight-only
failure can be packaged from the saved logs even when no result directory exists.

Acceptance: preflight passed; a complete `snapshot.json`; one valid group with A
and B completed, two candidate records, matching first requests and unchanged
canonical continuation proof. Dataset acceptance additionally requires at least
one eligible pair and `gate_passed=true`. If both valid candidates tie, record a
zero-yield group; do not fabricate a rejection or lower the evidence gate.

After these checks, select the next small batch based on the actual pair yield.
Do not jump to 12 groups or training. Repair/multistage replay still requires its
own implementation and validation.
