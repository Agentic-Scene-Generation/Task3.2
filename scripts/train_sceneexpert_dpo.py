#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "accelerate>=1.10.0",
#   "bitsandbytes>=0.46.1",
#   "datasets>=3.6.0",
#   "peft>=0.17.0",
#   "pyyaml>=6.0.2",
#   "trackio>=0.2.0",
#   "transformers>=4.55.4",
#   "trl>=0.21.0",
#   "unsloth>=2025.8.0",
# ]
# ///
"""Train a Qwen LoRA/QLoRA adapter from a validated SceneExpert DPO package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenesmith.scene_expert.slow_memory.training import (
    load_training_config,
    validate_training_request,
)


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
        "--dry-run",
        action="store_true",
        help="Validate config and dataset without importing CUDA libraries.",
    )
    return parser.parse_args()


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_dataset(dataset_dir: Path, num_proc: int) -> tuple[Any, Any | None]:
    from datasets import load_dataset

    train = load_dataset(
        "json",
        data_files=str(dataset_dir / "train.jsonl"),
        split="train",
        num_proc=max(1, num_proc),
    )
    keep = {"prompt", "chosen", "rejected"}
    remove = [column for column in train.column_names if column not in keep]
    if remove:
        train = train.remove_columns(remove)
    validation_path = dataset_dir / "validation.jsonl"
    validation = None
    if validation_path.exists() and validation_path.stat().st_size:
        validation = load_dataset(
            "json",
            data_files=str(validation_path),
            split="train",
            num_proc=max(1, num_proc),
        )
        remove = [column for column in validation.column_names if column not in keep]
        if remove:
            validation = validation.remove_columns(remove)
        if len(validation) == 0:
            validation = None
    return train, validation


def _build_model_and_tokenizer(
    config: dict[str, Any], model_ref: str
) -> tuple[Any, Any, Any | None]:
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    backend = str(model_cfg.get("backend", "unsloth")).lower()
    tuning_mode = str(model_cfg.get("tuning_mode", "qlora")).lower()
    load_in_4bit = tuning_mode == "qlora"
    target_modules = list(lora_cfg.get("target_modules") or [])
    if backend == "unsloth":
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_ref,
            max_seq_length=int(model_cfg.get("max_length", 4096)),
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
        from transformers import AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        )
        quantization_config = None
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
            target_modules=target_modules or "all-linear",
            task_type="CAUSAL_LM",
        )
        # Current TRL can own model loading and combine this quantization
        # config with PEFT. Returning the model ID avoids two base-model copies.
        model = model_ref
        peft_config = (peft_config, quantization_config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer, peft_config


def _training_args(config: dict[str, Any], output_dir: Path, has_eval: bool) -> Any:
    from trl import DPOConfig

    model_cfg = config["model"]
    train = config["training"]
    report_to = train.get("report_to", "trackio")
    return DPOConfig(
        output_dir=str(output_dir),
        run_name=str(train.get("run_name", "sceneexpert-qwen-dpo")),
        seed=int(train.get("seed", 42)),
        num_train_epochs=float(train.get("num_train_epochs", 1.0)),
        learning_rate=float(train.get("learning_rate", 5e-6)),
        beta=float(train.get("beta", 0.1)),
        per_device_train_batch_size=int(train.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train.get("gradient_accumulation_steps", 16)),
        warmup_ratio=float(train.get("warmup_ratio", 0.05)),
        lr_scheduler_type=str(train.get("lr_scheduler_type", "cosine")),
        optim=str(train.get("optim", "adamw_8bit")),
        max_grad_norm=float(train.get("max_grad_norm", 1.0)),
        logging_steps=int(train.get("logging_steps", 1)),
        eval_strategy="steps" if has_eval else "no",
        eval_steps=int(train.get("eval_steps", 25)) if has_eval else None,
        save_strategy="steps",
        save_steps=int(train.get("save_steps", 25)),
        save_total_limit=int(train.get("save_total_limit", 3)),
        bf16=bool(train.get("bf16", True)),
        tf32=bool(train.get("tf32", True)),
        gradient_checkpointing=bool(train.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=int(model_cfg.get("max_length", 4096)),
        truncation_mode="keep_start",
        report_to=report_to,
        project="sceneexpert-slow-memory",
        trackio_space_id=train.get("trackio_space_id"),
        remove_unused_columns=True,
        load_best_model_at_end=False,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )


def main() -> int:
    args = _parse_args()
    config = load_training_config(args.config)
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
    train_dataset, eval_dataset = _load_json_dataset(
        dataset_dir, int(data_cfg.get("num_proc", 4))
    )
    model, tokenizer, peft_bundle = _build_model_and_tokenizer(
        config, preflight["model_name_or_path"]
    )
    from trl import DPOTrainer

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": _training_args(config, output_dir, eval_dataset is not None),
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
    }
    if isinstance(peft_bundle, tuple):
        peft_config, quantization_config = peft_bundle
        trainer_kwargs["peft_config"] = peft_config
        trainer_kwargs["quantization_config"] = quantization_config
    trainer = DPOTrainer(**trainer_kwargs)
    resume = args.resume_from_checkpoint or train_cfg.get("resume_from_checkpoint")
    train_result = trainer.train(resume_from_checkpoint=resume or None)
    train_metrics = dict(train_result.metrics)
    trainer.save_metrics("train", train_metrics)
    evaluation_metrics: dict[str, Any] = {}
    if eval_dataset is not None:
        evaluation_metrics = dict(trainer.evaluate())
        trainer.save_metrics("eval", evaluation_metrics)

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    publish_cfg = config.get("publish") or {}
    if publish_cfg.get("push_to_hub"):
        if not publish_cfg.get("hub_model_id"):
            raise ValueError("publish.hub_model_id is required when push_to_hub=true")
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("HF_TOKEN is required for explicit Hub publishing")
        trainer.args.hub_model_id = str(publish_cfg["hub_model_id"])
        trainer.args.hub_private_repo = bool(publish_cfg.get("private", True))
        trainer.push_to_hub(token=os.environ["HF_TOKEN"])

    manifest = {
        "schema_version": "sceneexpert.dpo_training_run.v1",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_revision": _git_revision(),
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
        "model_name_or_path": preflight["model_name_or_path"],
        "backend": preflight["backend"],
        "tuning_mode": preflight["tuning_mode"],
        "adapter_dir": str(adapter_dir.resolve()),
        "train_metrics": train_metrics,
        "evaluation_metrics": evaluation_metrics,
        "served_model_note": (
            "Serve this adapter with its exact base checkpoint (or merge it "
            "offline), then set SCENEEXPERT_FULL_MODEL_ID to the served alias."
        ),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
