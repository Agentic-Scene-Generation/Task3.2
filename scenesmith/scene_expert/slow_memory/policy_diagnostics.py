"""Read-only diagnosis of the exact policy targets in a completed DPO pilot."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from scenesmith.scene_expert.slow_memory.training_lifecycle import (
    check_training_completion,
)

# These tools observe or maintain a plan; they do not execute a scene design.
# Unknown tools are reported separately rather than assumed to change a scene.
READ_ONLY_OR_PLANNING_TOOLS = frozenset({
    "observe_scene", "get_current_scene_state", "list_available_assets",
    "designer_todo_manager", "get_asset_info", "get_asset_details",
})
SCENE_DESIGN_TOOLS = frozenset({
    "generate_assets", "place_furniture", "place_furniture_batch",
    "place_object", "move_object", "remove_object", "modify_object",
})


def completion_scope(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe the visible target without assigning new preference labels."""
    names = [
        str(call.get("function", {}).get("name", ""))
        for message in messages for call in message.get("tool_calls", [])
    ]
    substantive = sorted(set(names) & SCENE_DESIGN_TOOLS)
    unknown = sorted(set(names) - READ_ONLY_OR_PLANNING_TOOLS - SCENE_DESIGN_TOOLS)
    return {
        "tool_names": names,
        "scene_design_tools": substantive,
        "unknown_tools": unknown,
        "scope": (
            "scene_design" if substantive else
            "unknown_tool" if unknown else
            "observation_or_plan" if names else "text_only"
        ),
    }


def load_diagnostic_source(training_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Require completed training and its immutable dataset before model loading."""
    training_run = training_run.resolve()
    supervisor = json.loads((training_run / "training_supervisor.json").read_text())
    if supervisor.get("state") != "completed" or supervisor.get("exit_code") != 0:
        raise ValueError("source training supervisor did not complete successfully")
    completion = check_training_completion(
        training_run, str(supervisor.get("execution_id", ""))
    )
    if not completion["completed"]:
        raise ValueError("source training is incomplete: " + "; ".join(completion["errors"]))
    manifest = json.loads((training_run / "training_manifest.json").read_text())
    if manifest.get("training_profile") != "furniture_initial_pilot":
        raise ValueError("this diagnostic supports the furniture initial pilot only")
    if manifest.get("dataset_features", {}).get("has_images"):
        raise ValueError("this diagnostic currently supports the text/tool pilot only")
    dataset_dir = Path(manifest["dataset_manifest"]).parent
    for name, expected in manifest["dataset_snapshot"].items():
        actual = hashlib.sha256((dataset_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"source dataset changed after training: {name}")
    validation_scores = manifest.get("evaluation_pair_scores")
    if validation_scores:
        path = (training_run / validation_scores["path"]).resolve()
        if not path.is_relative_to(training_run) or hashlib.sha256(path.read_bytes()).hexdigest() != validation_scores["sha256"]:
            raise ValueError("source validation pair-score evidence changed after training")
    rows = []
    for split in ("train", "validation"):
        for line in (dataset_dir / f"{split}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["split"] = split
            row["target_scope"] = {
                side: completion_scope(row[side]) for side in ("chosen", "rejected")
            }
            rows.append(row)
    if not rows:
        raise ValueError("source dataset has no policy pairs")
    return manifest, rows


def pair_score(metrics: dict[str, float], *, beta: float, token_counts: list[int]) -> dict[str, Any]:
    """Recover reference log probabilities from the pinned fused DPO outputs."""
    if beta <= 0 or len(token_counts) != 2 or min(token_counts) < 1:
        raise ValueError("positive beta and two nonempty completions are required")
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("nonfinite policy diagnostic metric")
    score: dict[str, Any] = dict(metrics)
    for side, count in zip(("chosen", "rejected"), token_counts, strict=True):
        policy = metrics[f"logps/{side}"]
        reference = policy - metrics[f"rewards/{side}"] / beta
        score[f"completion_tokens/{side}"] = count
        score[f"reference_logps/{side}"] = reference
        score[f"policy_mean_logp/{side}"] = policy / count
        score[f"reference_mean_logp/{side}"] = reference / count
    margin = metrics["rewards/chosen"] - metrics["rewards/rejected"]
    expected_loss = max(-margin, 0.0) + math.log1p(math.exp(-abs(margin)))
    if abs(metrics["loss"] - expected_loss) > 0.002:
        raise ValueError("diagnostic loss does not match the sigmoid DPO objective")
    score["shift_margin"] = margin
    score["shift_correct"] = margin > 0
    score["interpretation"] = "relative policy shift; not a scene success measurement"
    return score


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep training and validation separate and count auxiliary supervision."""
    scopes = Counter(
        scope["scope"] for row in rows for scope in row["target_scope"].values()
    )
    splits = {}
    for split in ("train", "validation"):
        selected = [row for row in rows if row["split"] == split and "score" in row]
        if selected:
            splits[split] = {
                "pair_count": len(selected),
                "shift_correct_count": sum(row["score"]["shift_correct"] for row in selected),
                "shift_accuracy": sum(row["score"]["shift_correct"] for row in selected) / len(selected),
                "mean_shift_margin": sum(row["score"]["shift_margin"] for row in selected) / len(selected),
                "mean_dpo_loss": sum(row["score"]["loss"] for row in selected) / len(selected),
            }
    return {"pair_count": len(rows), "target_scope_counts": dict(scopes), "splits": splits}


def evaluate_policy_pairs(
    trainer: Any,
    rows: list[dict[str, Any]],
    *,
    dataset: Any = None,
    beta: float,
    on_score: Callable[[dict[str, Any]], None],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Observe native evaluation without bypassing preparation or metric logging.

    The pinned trainer exposes batch means, so this observer requires one pair
    per batch and one process. It never clears or modifies the trainer's metrics.
    """
    if trainer.args.per_device_eval_batch_size != 1 or trainer.args.world_size != 1:
        raise ValueError("pair diagnostics require batch size one and one process")
    if not rows or len({row["split"] for row in rows}) != 1:
        raise ValueError("native evaluation must receive one nonempty dataset split")
    scored: list[dict[str, Any]] = []
    original = trainer.prediction_step
    names = ("logps/chosen", "logps/rejected", "rewards/chosen", "rewards/rejected")

    def observe(model: Any, inputs: Any, prediction_loss_only: bool, ignore_keys: Any = None) -> Any:
        if len(scored) >= len(rows) or inputs["completion_mask"].shape[0] != 2:
            raise ValueError("evaluation batches do not match the selected policy pairs")
        before = {name: len(trainer._metrics["eval"].get(name, [])) for name in names}
        result = original(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
        if any(len(trainer._metrics["eval"].get(name, [])) != before[name] + 1 for name in names):
            raise ValueError("native DPO evaluation did not emit one score per pair")
        counts = inputs["completion_mask"][:, 1:].sum(dim=1).tolist()
        metrics = {name: float(trainer._metrics["eval"][name][-1]) for name in names}
        row = rows[len(scored)]
        record = {key: row[key] for key in ("pair_id", "task_id", "split", "target_scope")}
        record["score"] = pair_score(dict(metrics, loss=float(result[0])), beta=beta, token_counts=counts)
        scored.append(record)
        on_score(record)
        return result

    trainer.prediction_step = observe
    try:
        metrics = dict(trainer.evaluate(eval_dataset=dataset))
    finally:
        trainer.prediction_step = original
    if len(scored) != len(rows):
        raise ValueError("native evaluation returned before scoring all selected pairs")
    aggregate = summarize_scores(scored)["splits"][rows[0]["split"]]
    for key, expected in (
        ("eval_loss", aggregate["mean_dpo_loss"]),
        ("eval_rewards/accuracies", aggregate["shift_accuracy"]),
        ("eval_rewards/margins", aggregate["mean_shift_margin"]),
    ):
        if key not in metrics or not math.isfinite(float(metrics[key])) or abs(metrics[key] - expected) > 1e-6:
            raise ValueError(f"per-pair scores disagree with native aggregate: {key}")
    return metrics, scored


def validation_reproduction(
    summary: dict[str, Any], source_metrics: dict[str, float], source_count: int,
) -> dict[str, Any]:
    """Keep execution completeness separate from historical numeric fidelity."""
    validation = summary["splits"].get("validation", {})
    checked = source_count > 0 and validation.get("pair_count") == source_count
    result: dict[str, Any] = {"checked": checked}
    if checked:
        loss_difference = validation["mean_dpo_loss"] - source_metrics["eval_loss"]
        accuracy_difference = validation["shift_accuracy"] - source_metrics["eval_rewards/accuracies"]
        result.update(
            loss_difference=loss_difference, accuracy_difference=accuracy_difference,
            loss_tolerance=0.002, accuracy_tolerance=1e-8,
            passed=abs(loss_difference) <= 0.002 and abs(accuracy_difference) < 1e-8,
        )
    return result


def diagnostic_outcome(*, scored_count: int, expected_count: int, fidelity: dict[str, Any]) -> dict[str, Any]:
    """An audit completes when it measures all pairs, including adverse results.

    Historical reproduction is a reported finding, not a policy promotion gate.
    Dataset/adapter integrity, finite loss and agreement with native evaluation
    are mandatory checks performed before this outcome can be constructed.
    """
    complete = expected_count > 0 and scored_count == expected_count
    warnings = (
        ["historical_validation_reproduction_mismatch"]
        if fidelity.get("checked") and not fidelity.get("passed") else []
    )
    return {
        "completed": complete, "execution_completed": complete,
        "status": "incomplete" if not complete else "completed_with_warnings" if warnings else "completed",
        "failure_reasons": [] if complete else ["selected_pairs_not_fully_evaluated"],
        "warnings": warnings, "historical_reproduction_blocks_completion": False,
    }


def prepare_training_kernels_for_evaluation(trainer: Any) -> dict[str, Any]:
    """Replay the model-kernel setup that pinned Trainer.train() performs.

    DPOTrainer's fused loss and the model's patched normalization/MLP kernels
    are separate. Standalone evaluate() initializes the loss but does not replay
    the model patch applied at the train entry point in Transformers 5.15.
    """
    from transformers.integrations.liger import apply_liger_kernel

    def inventory() -> dict[str, int]:
        return dict(Counter(
            module.forward.__module__ for _, module in trainer.model.named_modules()
            if callable(getattr(module, "forward", None))
            and str(getattr(module.forward, "__module__", "")).startswith("liger_kernel")
        ))

    if not trainer.args.use_liger_kernel:
        raise ValueError("this evaluation requires the source pilot's Liger model kernels")
    before = inventory()
    apply_liger_kernel(trainer.model, trainer.args.liger_kernel_config)
    after = inventory()
    return {
        "api": "transformers.integrations.liger.apply_liger_kernel",
        "kernel_config": trainer.args.liger_kernel_config,
        "source_train_entrypoint_setup_replayed": True,
        "patched_forward_modules_before": before,
        "patched_forward_modules_after": after,
    }
