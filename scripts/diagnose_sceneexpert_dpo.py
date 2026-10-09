#!/usr/bin/env python3
"""Score a frozen pilot pair by pair without training or executing scene tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scenesmith.scene_expert.slow_memory.policy_diagnostics import (
    diagnostic_outcome, evaluate_policy_pairs, load_diagnostic_source, summarize_scores,
    validation_reproduction,
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
        "source_validation_pair_scores": manifest.get("evaluation_pair_scores"),
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
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from trl import DPOTrainer
    from train_sceneexpert_dpo import (
        _audit_template_boundaries, _build_model_and_processor, _load_json_dataset,
        _training_args,
    )
    from scenesmith.scene_expert.slow_memory.fused_policy import (
        FusedLossOffloadContext, completion_only_fused_loss,
    )
    from scenesmith.scene_expert.trace_logger import collect_code_provenance

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("select exactly one CUDA GPU for this diagnostic")
    # Reconstruct the source trainer, then let its native evaluate() prepare the
    # model. A direct prediction_step loop bypasses Trainer/Accelerator setup.
    train_cfg = config["training"]
    if (train_cfg.get("per_device_eval_batch_size", 1) != 1
            or not train_cfg.get("prediction_loss_only")
            or not train_cfg.get("use_liger_kernel")
            or train_cfg.get("precompute_ref_log_probs")):
        raise ValueError("source pilot does not match the supported evaluation protocol")
    write_json(output / "diagnostic_config.json", config)
    train_dataset, eval_dataset, _ = _load_json_dataset(
        Path(manifest["dataset_manifest"]).parent, visible_action_only=True,
    )
    model, processor, peft_config, quantization = _build_model_and_processor(config, manifest["model_name_or_path"])
    write_json(output / "template_boundary_audit.json", _audit_template_boundaries(processor, [train_dataset, eval_dataset]))
    trainer = DPOTrainer(
        model=model, args=_training_args(config, output, eval_dataset is not None),
        train_dataset=train_dataset, eval_dataset=eval_dataset,
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
    saved_dtypes = dict(Counter(str(value.dtype) for value in weights.values()))
    del weights
    offload_context = None
    if trainer.args.activation_offloading:
        offload_context = FusedLossOffloadContext(trainer.maybe_activation_offload_context)
        trainer.maybe_activation_offload_context = offload_context
    trainer.liger_loss = completion_only_fused_loss(trainer.liger_loss, offload_context=offload_context)
    write_json(output / "evaluation_protocol.json", {
        "entry_point": "DPOTrainer.evaluate", "original_train_validation_splits": True,
        "source_training_settings_preserved": True,
        "per_pair_observer_mutates_trainer_metrics": False,
        "saved_adapter_dtypes": saved_dtypes,
        "loaded_adapter_dtypes": dict(Counter(
            str(value.dtype) for name, value in trainer.model.named_parameters() if "lora_" in name
        )),
        "torch_version": torch.__version__, "code_provenance": collect_code_provenance(include_git=False),
    })
    beta = float(config["training"]["beta"])
    scored = []

    def persist_score(record: dict) -> None:
        scored.append(record)
        with (output / "pair_scores.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_json(output / "diagnostic_progress.json", {"scored_pairs": len(scored), "expected_pairs": len(rows)})
        print(f"[diagnostic] {len(scored)}/{len(rows)} {record['split']} margin={record['score']['shift_margin']:.6f}", flush=True)

    native_metrics = {}
    for split in ("validation", "train"):
        selected = [row for row in rows if row["split"] == split]
        if not selected:
            continue
        dataset = trainer.eval_dataset if split == "validation" else trainer.train_dataset
        dataset = dataset.select(range(len(selected)))
        metrics, _ = evaluate_policy_pairs(
            trainer, selected, dataset=dataset, beta=beta, on_score=persist_score,
        )
        native_metrics[split] = metrics
        write_json(output / "native_evaluation_metrics.json", native_metrics)
    summary = summarize_scores(scored)
    source_validation_count = sum(row["split"] == "validation" for row in source_rows)
    fidelity = validation_reproduction(summary, manifest.get("evaluation_metrics", {}), source_validation_count)
    # Revalidate immutable inputs before finalizing. A completed audit may report
    # negative or nonreproduced historical metrics without becoming a failed job.
    load_diagnostic_source(args.training_run)
    if _adapter_sha256(Path(manifest["adapter_dir"])) != preflight["source_adapter_sha256"]:
        raise ValueError("source adapter changed during diagnostic evaluation")
    outcome = diagnostic_outcome(scored_count=len(scored), expected_count=len(rows), fidelity=fidelity)
    write_json(output / "policy_diagnostics.json", {
        "schema_version": "sceneexpert.dpo_policy_diagnostic.v3", **outcome,
        "full_dataset_evaluated": len(rows) == len(source_rows),
        "summary": summary, "source_validation_reproduction": fidelity,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "scene_effectiveness_measured": False, "promotable": False,
    })
    print("[diagnostic] " + json.dumps({
        **outcome, "source_validation_reproduction": fidelity,
    }), flush=True)
    return 0 if outcome["completed"] else 2


def _adapter_sha256(directory: Path) -> str:
    """Fingerprint the saved adapter without holding its entire file in memory."""
    digest = hashlib.sha256()
    with (directory / "adapter_model.safetensors").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
