"""Read-only diagnosis of the exact policy targets in a completed DPO pilot."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

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
