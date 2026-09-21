#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "torch==2.11.0",
#   "transformers==5.15.0",
#   "trl==1.13.0",
#   "peft==0.20.0",
#   "datasets==5.0.1",
#   "accelerate==1.14.0",
#   "bitsandbytes==0.50.2",
#   "liger-kernel==0.8.3",
#   "pillow>=11,<13",
#   "pyyaml>=6,<7",
#   "pydantic>=2.10,<3",
# ]
# ///
"""Train and gate a Qwen LoRA/QLoRA adapter from SceneExpert DPO data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenesmith.scene_expert.slow_memory.training import (
    apply_training_profile,
    evaluate_training_promotion,
    load_training_config,
    validate_training_request,
)
from scenesmith.scene_expert.trace_logger import collect_code_provenance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configurations/slow_memory/qwen_dpo_qlora.yaml"),
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--model", default="")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument(
        "--profile",
        choices=("full", "furniture_initial", "pipeline_smoke"),
        default="full",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset without importing CUDA libraries.",
    )
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path, *, dataset_dir: Path) -> list[dict[str, Any]]:
    from PIL import Image

    def load_image(image_path: Path) -> Any:
        with Image.open(image_path) as image:
            return image.convert("RGB")

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        row = {
            key: payload[key]
            for key in ("prompt", "chosen", "rejected", "tools", "images")
            if key in payload
        }
        image_paths = row.get("images") or []
        row["images"] = [
            load_image(dataset_dir / str(image_path)) for image_path in image_paths
        ]
        rows.append(row)
    return rows


def _load_json_dataset(
    dataset_dir: Path,
    *,
    visible_action_only: bool = False,
) -> tuple[Any, Any | None, dict[str, bool]]:
    from datasets import Dataset

    train_rows = _read_rows(dataset_dir / "train.jsonl", dataset_dir=dataset_dir)
    validation_path = dataset_dir / "validation.jsonl"
    validation_rows = (
        _read_rows(validation_path, dataset_dir=dataset_dir)
        if validation_path.exists() and validation_path.stat().st_size
        else []
    )
    has_tools = any(row.get("tools") for row in [*train_rows, *validation_rows])
    has_images = any(row.get("images") for row in [*train_rows, *validation_rows])
    for row in [*train_rows, *validation_rows]:
        if visible_action_only:
            # Captures contain visible actions, not hidden reasoning. Condition
            # their loss on the template's closed thinking envelope, rather than
            # inventing reasoning targets or misaligning the first action tokens.
            row["chat_template_kwargs"] = {"enable_thinking": False}
        if not has_tools:
            row.pop("tools", None)
        if not has_images:
            row.pop("images", None)
    train = Dataset.from_list(train_rows, on_mixed_types="use_json")
    validation = (
        Dataset.from_list(validation_rows, on_mixed_types="use_json")
        if validation_rows
        else None
    )
    return train, validation, {"has_tools": has_tools, "has_images": has_images}


def _audit_template_boundaries(processor: Any, datasets: list[Any]) -> dict[str, int]:
    """Reject template/token prefix drift before allocating the training model."""
    tokenizer = getattr(processor, "tokenizer", processor)
    stats = {
        "pair_count": 0,
        "max_text_prompt_tokens": 0,
        "max_text_completion_tokens": 0,
    }
    for dataset in datasets:
        if dataset is None:
            continue
        for row in dataset:
            tools = row.get("tools")
            tools = json.loads(tools) if isinstance(tools, str) else tools
            kwargs = {
                "tokenize": False,
                "tools": tools,
                **(row.get("chat_template_kwargs") or {}),
            }
            prefix = processor.apply_chat_template(
                row["prompt"], add_generation_prompt=True, **kwargs
            )
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            completions = []
            for side in ("chosen", "rejected"):
                rendered = processor.apply_chat_template(
                    row["prompt"] + row[side], **kwargs
                )
                tokens = tokenizer.encode(rendered, add_special_tokens=False)
                if (
                    not rendered.startswith(prefix)
                    or tokens[: len(prefix_ids)] != prefix_ids
                ):
                    raise ValueError(
                        f"{side} has a chat-template/token prefix mismatch; refuse misaligned DPO"
                    )
                completions.append(tokens[len(prefix_ids) :])
            if not all(completions) or completions[0] == completions[1]:
                raise ValueError(
                    "preference has no distinct completion after applying the model template"
                )
            stats["pair_count"] += 1
            stats["max_text_prompt_tokens"] = max(
                stats["max_text_prompt_tokens"], len(prefix_ids)
            )
            stats["max_text_completion_tokens"] = max(
                stats["max_text_completion_tokens"], *(len(x) for x in completions)
            )
    return stats


def _build_model_and_processor(
    config: dict[str, Any], model_ref: str
) -> tuple[Any, Any, Any | None, Any | None]:
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    backend = str(model_cfg.get("backend", "unsloth")).lower()
    tuning_mode = str(model_cfg.get("tuning_mode", "qlora")).lower()
    multimodal = bool(model_cfg.get("multimodal", False))
    load_in_4bit = tuning_mode == "qlora"
    target_modules = list(lora_cfg.get("target_modules") or [])
    quantization_config = None
    if backend == "unsloth":
        from unsloth import FastLanguageModel

        model, processor = FastLanguageModel.from_pretrained(
            model_name=model_ref,
            max_seq_length=int(model_cfg.get("max_length", 8192)),
            dtype=None,
            load_in_4bit=load_in_4bit,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lora_cfg.get("rank", 32)),
            target_modules=target_modules,
            lora_alpha=int(lora_cfg.get("alpha", 64)),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            bias=str(lora_cfg.get("bias", "none")),
            use_gradient_checkpointing=lora_cfg.get(
                "use_gradient_checkpointing", "unsloth"
            ),
            random_state=int(config["training"].get("seed", 42)),
        )
        peft_config = None
    else:
        import torch
        from peft import LoraConfig
        from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig

        processor = (
            AutoProcessor.from_pretrained(
                model_ref,
                trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            )
            if multimodal
            else AutoTokenizer.from_pretrained(
                model_ref,
                trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            )
        )
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        peft_config = LoraConfig(
            r=int(lora_cfg.get("rank", 32)),
            lora_alpha=int(lora_cfg.get("alpha", 64)),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            bias=str(lora_cfg.get("bias", "none")),
            target_modules=target_modules,
            exclude_modules=lora_cfg.get("exclude_modules"),
            task_type="CAUSAL_LM",
        )
        # Let DPOTrainer infer the exact Qwen causal/VLM architecture from config.
        model = model_ref
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, processor, peft_config, quantization_config


def _verify_processing_contract(
    processor: Any,
    *,
    dataset_features: dict[str, bool],
) -> None:
    tokenizer = getattr(processor, "tokenizer", processor)
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("the training checkpoint has no chat template")
    apply_template = getattr(processor, "apply_chat_template", None) or getattr(
        tokenizer, "apply_chat_template", None
    )
    if not callable(apply_template):
        raise RuntimeError(
            "the training processor cannot apply conversational templates"
        )
    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image"}, {"type": "text", "text": "Inspect scene"}]
                if dataset_features["has_images"]
                else "Inspect scene"
            ),
        }
    ]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    expected_tool_name = "sceneexpert_contract_probe"
    if dataset_features["has_tools"]:
        kwargs["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": expected_tool_name,
                    "description": "Validate tool-template support.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    try:
        rendered = apply_template(messages, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            "the checkpoint chat template cannot render this Slow Memory dataset"
        ) from exc
    if dataset_features["has_tools"] and expected_tool_name not in str(rendered):
        raise RuntimeError(
            "model.tool_calling=true but the chat template discarded the tool schema"
        )
    if dataset_features["has_images"] and not hasattr(processor, "image_processor"):
        raise RuntimeError("multimodal preference data requires an image processor")


def _training_args(
    config: dict[str, Any],
    output_dir: Path,
    has_eval: bool,
    *,
    has_images: bool = False,
) -> Any:
    import torch
    from trl import DPOConfig

    model_cfg = config["model"]
    train = config["training"]
    report_to = train.get("report_to", "trackio")
    multimodal = bool(model_cfg.get("multimodal", False))
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": str(train.get("run_name", "sceneexpert-qwen-dpo")),
        "seed": int(train.get("seed", 42)),
        "num_train_epochs": float(train.get("num_train_epochs", 1.0)),
        "max_steps": int(train.get("max_steps", -1)),
        "learning_rate": float(train.get("learning_rate", 1e-5)),
        "beta": float(train.get("beta", 0.1)),
        "per_device_train_batch_size": int(train.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(train.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(
            train.get("gradient_accumulation_steps", 16)
        ),
        # Transformers 5.15 folds fractional warmup into warmup_steps.
        "warmup_steps": float(
            train.get("warmup_steps", train.get("warmup_ratio", 0.05))
        ),
        "lr_scheduler_type": str(train.get("lr_scheduler_type", "cosine")),
        "optim": str(train.get("optim", "adamw_8bit")),
        "max_grad_norm": float(train.get("max_grad_norm", 1.0)),
        "logging_steps": int(train.get("logging_steps", 1)),
        "eval_strategy": "steps" if has_eval else "no",
        "eval_steps": int(train.get("eval_steps", 25)) if has_eval else None,
        "save_strategy": "steps",
        "save_steps": int(train.get("save_steps", 25)),
        "save_total_limit": int(train.get("save_total_limit", 3)),
        "bf16": bool(train.get("bf16", True)),
        "tf32": bool(train.get("tf32", True)),
        "gradient_checkpointing": bool(train.get("gradient_checkpointing", True)),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_length": None if multimodal else int(model_cfg.get("max_length", 8192)),
        "truncation_mode": str(train.get("truncation_mode", "keep_start")),
        "loss_type": train.get("loss_type", ["sigmoid", "sft"]),
        "loss_weights": train.get("loss_weights", [1.0, 0.2]),
        # TRL 1.13 precomputes text reference scores only. Vision batches are
        # processed on the fly and retain their normal reference forward pass.
        "precompute_ref_log_probs": bool(train.get("precompute_ref_log_probs", True))
        and not has_images,
        "report_to": report_to,
        "project": "sceneexpert-slow-memory",
        "trackio_space_id": train.get("trackio_space_id"),
        "remove_unused_columns": True,
        "use_liger_kernel": bool(train.get("use_liger_kernel", False)),
        "prediction_loss_only": bool(train.get("prediction_loss_only", True)),
        "load_best_model_at_end": False,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "dataset_num_proc": int(config.get("data", {}).get("num_proc", 4)),
    }
    if str(model_cfg.get("backend", "unsloth")).lower() == "transformers":
        kwargs["model_init_kwargs"] = {"dtype": torch.bfloat16}
    return DPOConfig(**kwargs)


def main() -> int:
    args = _parse_args()
    config = apply_training_profile(load_training_config(args.config), args.profile)
    data_cfg = config.setdefault("data", {})
    train_cfg = config.setdefault("training", {})
    dataset_dir = args.dataset_dir or Path(str(data_cfg.get("dataset_dir", "")))
    if args.output_dir:
        train_cfg["output_dir"] = str(args.output_dir)
    output_dir = Path(str(train_cfg.get("output_dir", "")))
    preflight = validate_training_request(
        config,
        dataset_dir=dataset_dir,
        model_name_or_path=args.model,
    )
    print(json.dumps(preflight, indent=2, ensure_ascii=False))
    if not preflight["valid"]:
        return 2
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("effective_config.json", config),
        ("preflight.json", preflight),
    ):
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    train_dataset, eval_dataset, dataset_features = _load_json_dataset(
        dataset_dir, visible_action_only=args.profile != "full"
    )
    model, processor, peft_config, quantization_config = _build_model_and_processor(
        config, preflight["model_name_or_path"]
    )
    _verify_processing_contract(processor, dataset_features=dataset_features)
    template_audit = _audit_template_boundaries(
        processor, [train_dataset, eval_dataset]
    )
    (output_dir / "template_boundary_audit.json").write_text(
        json.dumps(template_audit, indent=2), encoding="utf-8"
    )
    print(json.dumps({"template_boundary_audit": template_audit}), flush=True)
    from trl import DPOTrainer

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": _training_args(
            config,
            output_dir,
            eval_dataset is not None,
            has_images=dataset_features["has_images"],
        ),
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": processor,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    if quantization_config is not None:
        trainer_kwargs["quantization_config"] = quantization_config
    trainer = DPOTrainer(**trainer_kwargs)
    if trainer.args.use_liger_kernel:
        from scenesmith.scene_expert.slow_memory.fused_policy import (
            completion_only_fused_loss,
        )

        trainer.liger_loss = completion_only_fused_loss(trainer.liger_loss)
    resume = args.resume_from_checkpoint or train_cfg.get("resume_from_checkpoint")
    train_result = trainer.train(resume_from_checkpoint=resume or None)
    train_metrics = dict(train_result.metrics)
    if trainer.state.global_step < 1 or not math.isfinite(
        float(train_metrics.get("train_loss", float("nan")))
    ):
        raise RuntimeError("DPO did not complete an optimizer step with finite loss")
    import torch

    changed_lora_tensors = sum(
        bool(torch.count_nonzero(value.detach()).item())
        for name, value in trainer.model.named_parameters()
        if "lora_B" in name
    )
    if not changed_lora_tensors:
        raise RuntimeError("optimizer completed but all LoRA B tensors remain zero")
    train_metrics.update(
        optimizer_steps=trainer.state.global_step,
        nonzero_lora_b_tensors=changed_lora_tensors,
        peak_cuda_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
    )
    trainer.save_metrics("train", train_metrics)
    evaluation_metrics: dict[str, Any] = {}
    if eval_dataset is not None:
        evaluation_metrics = dict(trainer.evaluate())
        trainer.save_metrics("eval", evaluation_metrics)

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(adapter_dir)
    promotion = evaluate_training_promotion(
        config,
        evaluation_metrics=evaluation_metrics,
    )
    publish_cfg = config.get("publish") or {}
    if publish_cfg.get("push_to_hub"):
        if not promotion["promotable"]:
            raise RuntimeError(
                "refusing to publish an adapter that failed quality gates"
            )
        if not publish_cfg.get("hub_model_id"):
            raise ValueError("publish.hub_model_id is required when push_to_hub=true")
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("HF_TOKEN is required for explicit Hub publishing")
        trainer.args.hub_model_id = str(publish_cfg["hub_model_id"])
        trainer.args.hub_private_repo = bool(publish_cfg.get("private", True))
        trainer.push_to_hub(token=os.environ["HF_TOKEN"])

    manifest = {
        "schema_version": "sceneexpert.dpo_training_run.v2",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_profile": args.profile,
        "code_provenance": collect_code_provenance(include_git=False),
        "config_path": str(args.config.resolve()),
        "config_fingerprint": preflight["config_fingerprint"],
        "dataset_manifest": str((dataset_dir / "manifest.json").resolve()),
        "dataset_snapshot": {
            name: _file_sha256(dataset_dir / name)
            for name in (
                "manifest.json",
                "train.jsonl",
                "validation.jsonl",
                "test.jsonl",
            )
        },
        "dataset_features": dataset_features,
        "template_boundary_audit": template_audit,
        "use_liger_kernel": trainer.args.use_liger_kernel,
        "loss_projection": (
            "completion_positions_only"
            if trainer.args.use_liger_kernel
            else "all_positions"
        ),
        "reasoning_target": (
            "visible_actions_after_closed_thinking_envelope"
            if args.profile != "full"
            else "template_default"
        ),
        "learning_scope": (
            "first_assistant_turn"
            if args.profile != "full"
            else "configured_completion"
        ),
        "model_name_or_path": preflight["model_name_or_path"],
        "backend": preflight["backend"],
        "tuning_mode": preflight["tuning_mode"],
        "multimodal": preflight["multimodal"],
        "adapter_dir": str(adapter_dir.resolve()),
        "train_metrics": train_metrics,
        "evaluation_metrics": evaluation_metrics,
        "promotion_gate": promotion,
        "served_model_note": (
            "Serve this adapter with its exact base checkpoint only after the "
            "paired SceneEval scene-level gate passes, then set "
            "SCENEEXPERT_MODEL_ID to the served alias. Experiment mode never "
            "selects a checkpoint implicitly."
        ),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if args.profile == "pipeline_smoke":
        print("Pipeline smoke completed; this adapter is not an effectiveness result.")
        return 0
    return 0 if promotion["promotable"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
