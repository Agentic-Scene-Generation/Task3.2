"""Preflight and promotion gates for SceneExpert preference training."""

from __future__ import annotations

import hashlib
import json
import math

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from scenesmith.scene_expert.slow_memory.dpo import validate_dataset_dir
from scenesmith.scene_expert.slow_memory.schemas import DPOPreferencePair


def load_training_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training config must contain a YAML mapping")
    return payload


def config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_split_stats(path: Path) -> dict[str, Any]:
    role_counts: Counter[str] = Counter()
    task_type_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    leakage_groups: set[str] = set()
    pair_count = 0
    if not path.exists():
        return {
            "pair_count": 0,
            "unique_leakage_groups": 0,
            "pairs_by_role": {},
            "pairs_by_task_type": {},
            "pairs_by_stage": {},
        }
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            pair = DPOPreferencePair.model_validate_json(line)
        except ValueError:
            continue
        role_counts[pair.agent_role] += 1
        task_type_counts[pair.task_type] += 1
        stage_counts[pair.stage] += 1
        leakage_groups.add(pair.leakage_group)
        pair_count += 1
    return {
        "pair_count": pair_count,
        "unique_leakage_groups": len(leakage_groups),
        "pairs_by_role": dict(sorted(role_counts.items())),
        "pairs_by_task_type": dict(sorted(task_type_counts.items())),
        "pairs_by_stage": dict(sorted(stage_counts.items())),
    }


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
    gate_cfg = (
        config.get("quality_gate")
        if isinstance(config.get("quality_gate"), dict)
        else {}
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
            "checkpoint; use the corresponding safetensors directory or Hub ID"
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
    multimodal = bool(model_cfg.get("multimodal", False))
    if multimodal and backend != "transformers":
        errors.append(
            "multimodal DPO currently requires model.backend=transformers; the "
            "text-only Unsloth path remains available for designer tool policies"
        )
    target_modules = (
        config.get("lora", {}).get("target_modules", [])
        if isinstance(config.get("lora"), dict)
        else []
    )
    if not target_modules:
        errors.append(
            "lora.target_modules is required for reproducible adapter training"
        )
    max_length = model_cfg.get("max_length")
    if multimodal:
        if max_length is not None and not bool(
            model_cfg.get("allow_verified_vision_truncation", False)
        ):
            errors.append(
                "set model.max_length=null for VLM DPO unless every sample was "
                "verified to retain all image tokens"
            )
    elif not isinstance(max_length, int) or max_length < 1024:
        errors.append("text-only model.max_length must be at least 1024 tokens")

    dataset_dir = Path(dataset_dir)
    validation = validate_dataset_dir(dataset_dir)
    if not validation["valid"]:
        errors.extend(f"dataset: {error}" for error in validation.get("errors", []))
    multimodal_pairs = int(validation.get("multimodal_pair_count", 0) or 0)
    tool_pairs = int(validation.get("tool_pair_count", 0) or 0)
    if multimodal_pairs and not multimodal:
        errors.append(
            f"dataset contains {multimodal_pairs} multimodal pairs but "
            "model.multimodal=false"
        )
    if multimodal and not multimodal_pairs:
        warnings.append("model.multimodal=true but the dataset contains no image pairs")
    if tool_pairs and model_cfg.get("tool_calling") is not True:
        errors.append(
            f"dataset contains {tool_pairs} tool-policy pairs; explicitly set "
            "model.tool_calling=true after verifying the checkpoint chat template"
        )

    split_stats = {
        split: _read_split_stats(dataset_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    allow_small = bool(data_cfg.get("allow_unsafe_small_dataset", False))
    min_train_pairs = int(data_cfg.get("minimum_train_pairs", 16) or 0)
    min_groups = int(data_cfg.get("minimum_unique_train_groups", 8) or 0)
    if split_stats["train"]["pair_count"] < min_train_pairs:
        message = (
            f"training split has {split_stats['train']['pair_count']} pairs; "
            f"at least {min_train_pairs} are required"
        )
        (warnings if allow_small else errors).append(message)
    if split_stats["train"]["unique_leakage_groups"] < min_groups:
        message = (
            "training split has "
            f"{split_stats['train']['unique_leakage_groups']} independent scenario "
            f"groups; at least {min_groups} are required"
        )
        (warnings if allow_small else errors).append(message)
    min_stages = int(data_cfg.get("minimum_unique_train_stages", 1) or 0)
    observed_stages = len(split_stats["train"]["pairs_by_stage"])
    if observed_stages < min_stages:
        message = (
            f"training split covers {observed_stages} stages; at least "
            f"{min_stages} are required"
        )
        (warnings if allow_small else errors).append(message)
    required_task_types = [
        str(value) for value in (data_cfg.get("required_task_types") or [])
    ]
    min_per_type = int(data_cfg.get("minimum_pairs_per_required_task_type", 1) or 0)
    for task_type in required_task_types:
        observed = int(
            split_stats["train"]["pairs_by_task_type"].get(task_type, 0) or 0
        )
        if observed < min_per_type:
            message = (
                f"training split has {observed} {task_type!r} pairs; "
                f"at least {min_per_type} are required"
            )
            (warnings if allow_small else errors).append(message)
    task_counts = split_stats["train"]["pairs_by_task_type"]
    train_pair_count = int(split_stats["train"]["pair_count"] or 0)
    maximum_task_ratio = float(data_cfg.get("maximum_single_task_type_ratio", 1.0))
    dominant_ratio = (
        max(task_counts.values(), default=0) / train_pair_count
        if train_pair_count
        else 0.0
    )
    if dominant_ratio > maximum_task_ratio:
        message = (
            f"one task type occupies {dominant_ratio:.1%} of the training split; "
            f"maximum allowed is {maximum_task_ratio:.1%}"
        )
        (warnings if allow_small else errors).append(message)
    if (
        gate_cfg.get("require_validation", True)
        and split_stats["validation"]["pair_count"] == 0
    ):
        errors.append("quality gate requires a non-empty validation split")
    if gate_cfg.get("require_test", True) and split_stats["test"]["pair_count"] == 0:
        errors.append("quality gate requires a non-empty held-out test split")

    manifest_path = dataset_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if not manifest_path.exists():
        errors.append("dataset manifest.json is missing")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"dataset manifest.json is invalid: {exc}")
        else:
            if manifest.get("schema_version") != "sceneexpert.dpo_dataset_manifest.v2":
                errors.append(
                    "training requires a v2 DPO package; re-export legacy trajectories "
                    "so tool, media, spatial, and causal gates are enforced"
                )
            policy = manifest.get("pairing_policy") or {}
            required_policy = {
                "synthetic_rejected_responses_allowed": False,
                "same_context_required": True,
                "authoritative_evidence_required": True,
                "same_tool_schema_required": True,
                "same_visual_input_required": True,
                "same_spatial_context_required": True,
                "critic_causal_evidence_required": True,
            }
            for key, expected in required_policy.items():
                if policy.get(key) is not expected:
                    errors.append(f"dataset pairing policy does not enforce {key}")

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
        "multimodal": multimodal,
        "dataset_validation": validation,
        "split_stats": split_stats,
        "config_fingerprint": config_fingerprint(config),
    }


def evaluate_training_promotion(
    config: dict[str, Any],
    *,
    evaluation_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Decide whether an adapter is safe to expose to SceneExpert experiments."""

    gate = (
        config.get("quality_gate")
        if isinstance(config.get("quality_gate"), dict)
        else {}
    )
    accuracy_keys = (
        "eval_rewards/accuracies",
        "eval_rewards_accuracies",
        "eval_reward_accuracy",
    )
    accuracy = next(
        (
            float(evaluation_metrics[key])
            for key in accuracy_keys
            if isinstance(evaluation_metrics.get(key), (int, float))
        ),
        None,
    )
    eval_loss = evaluation_metrics.get("eval_loss")
    eval_loss = float(eval_loss) if isinstance(eval_loss, (int, float)) else None
    reasons: list[str] = []
    if gate.get("require_validation", True) and accuracy is None:
        reasons.append("TRL validation preference accuracy was not reported")
    minimum_accuracy = float(gate.get("minimum_preference_accuracy", 0.55))
    if accuracy is not None and (
        not math.isfinite(accuracy) or accuracy < minimum_accuracy
    ):
        reasons.append(
            f"validation preference accuracy {accuracy:.4f} < {minimum_accuracy:.4f}"
        )
    if eval_loss is not None and not math.isfinite(eval_loss):
        reasons.append("validation loss is not finite")
    return {
        "promotable": not reasons,
        "status": "candidate_accepted" if not reasons else "candidate_rejected",
        "reasons": reasons,
        "observed_preference_accuracy": accuracy,
        "minimum_preference_accuracy": minimum_accuracy,
        "observed_eval_loss": eval_loss,
        "scene_level_gate_required": True,
        "scene_level_gate_note": (
            "Offline DPO metrics are necessary but insufficient. Promote the served "
            "adapter only after paired SceneEval runs improve hard-pass and relation "
            "metrics without regressing Main critic success."
        ),
    }
