#!/usr/bin/env python3
"""Score a frozen pilot pair by pair without training or executing scene tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scenesmith.scene_expert.slow_memory.policy_diagnostics import (
    load_diagnostic_source, pair_score, summarize_scores,
)
from scenesmith.scene_expert.slow_memory.training_lifecycle import write_json


def main() -> int:
    """Validate the source, load the exact adapter and persist each evaluated pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-pairs", type=int, default=0, help="Smoke only: first N rows, never select by scores")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_pairs < 0:
        parser.error("--max-pairs must be nonnegative")
    started = time.monotonic()
    manifest, source_rows = load_diagnostic_source(args.training_run)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("choose a new diagnostic output directory; results are immutable")
    if output.is_relative_to(args.training_run.resolve()):
        raise ValueError("diagnostic output must be outside the source training run")
    config = json.loads((args.training_run / "effective_config.json").read_text())
    if config["training"].get("loss_type") != "sigmoid":
        raise ValueError("diagnostic reference recovery requires the original sigmoid objective")
    rows = source_rows[:args.max_pairs] if args.max_pairs else source_rows
    output.mkdir(parents=True)
    preflight = {
        "source_training_run": str(args.training_run.resolve()),
        "source_execution_id": manifest["execution_id"],
        "source_dataset_snapshot": manifest["dataset_snapshot"],
        "source_adapter_dir": manifest["adapter_dir"],
        "source_adapter_sha256": _adapter_sha256(Path(manifest["adapter_dir"])),
        "full_source_pair_count": len(source_rows),
        "selected_pair_count": len(rows),
        "selection": "all" if not args.max_pairs else "first_n_smoke",
        "target_scope": summarize_scores(rows),
        "dry_run": args.dry_run,
    }
    write_json(output / "diagnostic_preflight.json", preflight)
    print(json.dumps(preflight, indent=2), flush=True)
    if args.dry_run:
        return 0

    import torch
    from datasets import Dataset
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from trl import DPOTrainer
    from train_sceneexpert_dpo import (
        _audit_template_boundaries, _build_model_and_processor, _training_args,
    )
    from scenesmith.scene_expert.slow_memory.fused_policy import completion_only_fused_loss

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("select exactly one CUDA GPU for this diagnostic")
    # Retain the original context/template/loss. Only evaluation resource settings change.
    config = deepcopy(config)
    config["training"].update(
        gradient_checkpointing=False, activation_offloading=False,
        per_device_eval_batch_size=1, prediction_loss_only=True,
        precompute_ref_log_probs=False, report_to="none", use_liger_kernel=True,
    )
    write_json(output / "diagnostic_config.json", config)
    dataset = Dataset.from_list([
        {key: row[key] for key in ("prompt", "chosen", "rejected", "tools") if key in row}
        | {"chat_template_kwargs": {"enable_thinking": False}}
        for row in rows
    ], on_mixed_types="use_json")
    model, processor, peft_config, quantization = _build_model_and_processor(config, manifest["model_name_or_path"])
    write_json(output / "template_boundary_audit.json", _audit_template_boundaries(processor, [dataset]))
    trainer = DPOTrainer(
        model=model, args=_training_args(config, output, True),
        train_dataset=dataset, eval_dataset=dataset,
        processing_class=processor, peft_config=peft_config,
        quantization_config=quantization,
    )
    weights = load_file(str(Path(manifest["adapter_dir"]) / "adapter_model.safetensors"))
    if _adapter_sha256(Path(manifest["adapter_dir"])) != preflight["source_adapter_sha256"]:
        raise ValueError("source adapter changed during diagnostic loading")
    if not all(torch.isfinite(tensor).all().item() for tensor in weights.values()):
        raise ValueError("source adapter contains nonfinite tensors")
    result = set_peft_model_state_dict(trainer.model, weights, adapter_name="default")
    missing_lora = [name for name in result.missing_keys if "lora_" in name]
    if missing_lora or result.unexpected_keys:
        raise ValueError(f"adapter load mismatch: {missing_lora}, {result.unexpected_keys}")
    del weights
    trainer.model.eval()
    trainer.liger_loss = completion_only_fused_loss(trainer.liger_loss)
    beta = float(config["training"]["beta"])
    scored = []
    for row, batch in zip(rows, trainer.get_eval_dataloader(), strict=True):
        trainer._metrics["eval"] = defaultdict(list)
        counts = batch["completion_mask"][:, 1:].sum(dim=1).tolist()
        loss, _, _ = trainer.prediction_step(trainer.model, batch, True)
        metrics = {
            name: float(trainer._metrics["eval"][name][-1]) for name in (
                "logps/chosen", "logps/rejected", "rewards/chosen", "rewards/rejected",
            )
        }
        record = {key: row[key] for key in ("pair_id", "task_id", "split", "target_scope")}
        record["score"] = pair_score(dict(metrics, loss=float(loss)), beta=beta, token_counts=counts)
        scored.append(record)
        with (output / "pair_scores.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_json(output / "diagnostic_progress.json", {"scored_pairs": len(scored), "expected_pairs": len(rows)})
        print(f"[diagnostic] {len(scored)}/{len(rows)} {row['split']} margin={record['score']['shift_margin']:.6f}", flush=True)
    summary = summarize_scores(scored)
    validation = summary["splits"].get("validation", {})
    source_validation_count = sum(row["split"] == "validation" for row in source_rows)
    fidelity = {"checked": validation.get("pair_count") == source_validation_count and source_validation_count > 0}
    if fidelity["checked"]:
        original = manifest["evaluation_metrics"]
        fidelity["loss_difference"] = validation["mean_dpo_loss"] - original["eval_loss"]
        fidelity["accuracy_difference"] = validation["shift_accuracy"] - original["eval_rewards/accuracies"]
        fidelity["passed"] = abs(fidelity["loss_difference"]) <= 0.002 and abs(fidelity["accuracy_difference"]) < 1e-8
    completed = len(scored) == len(rows) and (not fidelity["checked"] or fidelity["passed"])
    write_json(output / "policy_diagnostics.json", {
        "schema_version": "sceneexpert.dpo_policy_diagnostic.v1",
        "completed": completed, "full_dataset_evaluated": len(rows) == len(source_rows),
        "summary": summary, "source_validation_reproduction": fidelity,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "scene_effectiveness_measured": False, "promotable": False,
    })
    return 0 if completed else 2


def _adapter_sha256(directory: Path) -> str:
    """Fingerprint the saved adapter without holding its entire file in memory."""
    digest = hashlib.sha256()
    with (directory / "adapter_model.safetensors").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
