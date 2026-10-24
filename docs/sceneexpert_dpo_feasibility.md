# DPO feasibility with the seven certified groups

The explicit `furniture_initial_feasibility_4t3v_v1` profile prepares a separate
small-data experiment. It does not change admission to the existing 8T4V pilot
or authorize any closed collection retry. GPU execution requires a new direct
scope approval bound to the immutable proposal and qualification manifest.

Use exactly original training IDs 368, 371, 238 and 243, and original validation
IDs 479, 302 and 266. The six certified source605 rows retain their original
source, labels and qualification. ID243 retains its separate source610 cohort.
One frozen pair represents each task. The strict set is a subset, not additional
data. Copy rows byte for byte; do not add quarantined pairs or move validation
or test tasks into training.

The recipe remains Qwen3.8-27B QLoRA, sigmoid DPO without an SFT term, seed42,
two epochs, learning rate 1e-5, beta0.1 and gradient accumulation4. With batch1
on one GPU this schedule gives eight pair presentations and approximately two
optimizer updates. This is the complete small dataset schedule, not a two-pair
capacity probe. Keep the final checkpoint without validation-based selection.

The experiment can establish whether training makes finite parameter updates,
saves optimizer/scheduler/RNG state, reloads the exact adapter and reproduces
per-pair diagnostics. Three validation tasks cannot establish generalization,
scene-generation improvement, statistical power or deployment readiness. The
adapter is never promotable from these results alone.

Prepare the dataset and run CPU admission and template checks before allocation.
The plan must bind source contents, dataset rows, proposal, control scripts and
base checkpoint identity. The training process requires
`SCENEEXPERT_FEASIBILITY_APPROVAL_PATH` and its SHA256 in
`SCENEEXPERT_FEASIBILITY_APPROVAL_SHA256`; missing or unrelated approval fails
before creating training output or loading model weights. Preserve all original
collection results. Store this experiment under a new run ID and package its
logs, effective configuration, manifests and pair diagnostics with checksums;
retain large adapter and optimizer files on the server and list their hashes.

Budget preparation at 20–45 minutes, training at 10–40 minutes and saved-adapter
diagnostics at 10–30 minutes, with 10–20 minutes for review and packaging.
Operator work overlaps these stages. Queue and full-context performance are
unknown; these are planning ranges without a measured run on this dataset.
Use an approved one-GPU ACP allocation if the combined server estimate exceeds
the autonomous one-hour limit.
