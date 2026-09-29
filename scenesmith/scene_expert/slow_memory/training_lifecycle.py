"""Durable training progress and fail-closed completion checks (no CUDA imports)."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a small status record, including after a signal."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def check_training_completion(output: Path, execution_id: str) -> dict[str, Any]:
    """Require this invocation's final artifacts, independently of process exit."""
    errors: list[str] = []
    try:
        manifest = json.loads((output / "training_manifest.json").read_text())
        if not execution_id or manifest.get("execution_id") != execution_id:
            errors.append("training manifest belongs to another invocation")
        completion = manifest.get("completion", {})
        steps = int(completion.get("optimizer_steps", 0))
        expected = int(completion.get("expected_optimizer_steps", 0))
        if completion.get("completed") is not True or expected < 1 or steps != expected:
            errors.append("optimizer schedule is incomplete")
        train = manifest.get("train_metrics", {})
        if not math.isfinite(float(train.get("train_loss", "nan"))):
            errors.append("finite final training loss is missing")
        if int(train.get("nonzero_lora_b_tensors", 0)) < 1:
            errors.append("adapter update evidence is missing")
        preflight = json.loads((output / "preflight.json").read_text())
        val_count = preflight["dataset_validation"]["split_counts"]["validation"]
        if val_count and manifest.get("capacity_probe") is None:
            if not math.isfinite(float(manifest.get("evaluation_metrics", {}).get("eval_loss", "nan"))):
                errors.append("finite final validation loss is missing")
            if not (output / "eval_results.json").is_file():
                errors.append("eval_results.json is missing")
        for name in ("train_results.json", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
            path = output / name
            if not path.is_file() or not path.stat().st_size:
                errors.append(f"missing or empty artifact: {name}")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"cannot verify final artifacts: {exc}")
    return {"completed": not errors, "errors": errors}


def checkpoint_complete(checkpoint: Path, *, require_marker: bool = True) -> None:
    """Reject an interrupted checkpoint before invoking model loading."""
    names = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json", "adapter_model.safetensors"]
    if require_marker:
        names.append("checkpoint_complete.json")
    missing = [name for name in names if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size]
    if not list(checkpoint.glob("rng_state*.pth")):
        missing.append("rng_state*.pth")
    if missing:
        raise ValueError(f"incomplete resume checkpoint {checkpoint}: {missing}")
    if require_marker:
        marker = json.loads((checkpoint / "checkpoint_complete.json").read_text())
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if marker.get("global_step") != state.get("global_step") or not marker.get("complete"):
            raise ValueError("resume checkpoint completion marker does not match trainer state")


def make_progress_callback(output: Path) -> Any:
    """Construct an HF callback only inside the managed training environment."""
    from transformers import TrainerCallback

    class DurableProgress(TrainerCallback):
        def record(self, event: str, state: Any, **extra: Any) -> None:
            if not getattr(state, "is_world_process_zero", True):
                return
            payload = {
                "event": event, "time_unix": time.time(),
                "execution_id": os.environ.get("SCENEEXPERT_TRAIN_EXECUTION_ID", ""),
                "global_step": state.global_step, "max_steps": state.max_steps,
                "epoch": state.epoch, **extra,
            }
            write_json(output / "training_progress.json", payload)
            with (output / "training_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, default=str) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.record("train_begin", state)

        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
            self.record("log", state, metrics=logs or {})

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if not getattr(state, "is_world_process_zero", True):
                return
            checkpoint = output / f"checkpoint-{state.global_step}"
            checkpoint_complete(checkpoint, require_marker=False)
            write_json(checkpoint / "checkpoint_complete.json", {"complete": True, "global_step": state.global_step})
            self.record("checkpoint_saved", state, checkpoint=str(checkpoint))

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.record("train_end", state)

    return DurableProgress()
