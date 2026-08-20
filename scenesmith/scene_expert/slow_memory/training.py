"""Lightweight preflight helpers for the optional SceneExpert DPO trainer."""

from __future__ import annotations

import hashlib
import json

from pathlib import Path
from typing import Any

import yaml

from scenesmith.scene_expert.slow_memory.dpo import validate_dataset_dir


def load_training_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training config must contain a YAML mapping")
    return payload


def config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_training_request(
    config: dict[str, Any],
    *,
    dataset_dir: Path,
    model_name_or_path: str = "",
) -> dict[str, Any]:
    """Fail before importing CUDA libraries or allocating a model."""
    errors: list[str] = []
    warnings: list[str] = []
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}
    train_cfg = (
        config.get("training") if isinstance(config.get("training"), dict) else {}
    )
    model_ref = str(model_name_or_path or model_cfg.get("name_or_path") or "").strip()
    if not model_ref:
        errors.append(
            "model.name_or_path is required and must point to a Hugging Face "
            "Transformers checkpoint"
        )
    if model_ref.lower().endswith(".gguf"):
        errors.append(
            "GGUF is an inference artifact, not a trainable Transformers base "
            "checkpoint; use the corresponding safetensors model directory or Hub ID"
        )
    if model_ref:
        local_model = Path(model_ref)
        if local_model.exists() and local_model.is_file():
            errors.append("model.name_or_path must be a model directory or Hub ID")
        elif local_model.exists() and not (local_model / "config.json").exists():
            errors.append("local model directory is missing config.json")

    backend = str(model_cfg.get("backend") or "unsloth").lower()
    if backend not in {"unsloth", "transformers"}:
        errors.append("model.backend must be 'unsloth' or 'transformers'")
    tuning_mode = str(model_cfg.get("tuning_mode") or "qlora").lower()
    if tuning_mode not in {"lora", "qlora"}:
        errors.append("model.tuning_mode must be 'lora' or 'qlora'")
    if tuning_mode == "qlora" and backend != "unsloth":
        warnings.append(
            "QLoRA is supported by the transformers backend, but unsloth is the "
            "project default for the 27B Qwen training target"
        )
    target_modules = (
        config.get("lora", {}).get("target_modules", [])
        if isinstance(config.get("lora"), dict)
        else []
    )
    if backend == "unsloth" and not target_modules:
        errors.append("lora.target_modules is required for the unsloth backend")
    if int(model_cfg.get("max_length", 0) or 0) < 256:
        errors.append("model.max_length must be at least 256 tokens")

    dataset_dir = Path(dataset_dir)
    validation = validate_dataset_dir(dataset_dir)
    if not validation["valid"]:
        errors.extend(f"dataset: {error}" for error in validation.get("errors", []))
    if validation.get("split_counts", {}).get("validation", 0) == 0:
        warnings.append(
            "validation split is empty; checkpoints will be saved but no held-out "
            "DPO evaluation will run"
        )
    if validation.get("split_counts", {}).get("test", 0) == 0:
        warnings.append(
            "test split is empty; reserve more task groups before reporting"
        )
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        errors.append("dataset manifest.json is missing")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"dataset manifest.json is invalid: {exc}")
        else:
            policy = manifest.get("pairing_policy") or {}
            if policy.get("synthetic_rejected_responses_allowed") is not False:
                errors.append("dataset does not prohibit synthetic rejected responses")
            if policy.get("same_context_required") is not True:
                errors.append("dataset does not require exact same-context pairing")

    output_value = str(train_cfg.get("output_dir") or "").strip()
    output_dir = Path(output_value) if output_value else Path()
    if not output_value:
        errors.append("training.output_dir is required")
    else:
        try:
            if output_dir.resolve() == dataset_dir.resolve():
                errors.append("training.output_dir must not overwrite the dataset")
        except OSError:
            pass
    if (
        data_cfg.get("dataset_dir")
        and Path(str(data_cfg["dataset_dir"])) != dataset_dir
    ):
        warnings.append("CLI --dataset-dir overrides data.dataset_dir from YAML")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "model_name_or_path": model_ref,
        "backend": backend,
        "tuning_mode": tuning_mode,
        "dataset_validation": validation,
        "config_fingerprint": config_fingerprint(config),
    }
