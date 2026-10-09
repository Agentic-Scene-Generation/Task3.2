"""Diagnostic evidence must remain attached to the frozen source and objective."""

import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from scenesmith.scene_expert.slow_memory.policy_diagnostics import (
    completion_scope, diagnostic_outcome, evaluate_policy_pairs, load_diagnostic_source, pair_score,
    summarize_scores, validation_reproduction,
)


def _json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def source(tmp_path):
    dataset = tmp_path / "dataset"
    _json(dataset / "manifest.json", {})
    calls = [{"role": "assistant", "tool_calls": [{"function": {"name": "observe_scene", "arguments": {}}}]}]
    row = {"pair_id": "pair1", "task_id": "task1", "chosen": calls, "rejected": calls}
    (dataset / "train.jsonl").write_text(json.dumps(row) + "\n")
    (dataset / "validation.jsonl").write_text("")
    (dataset / "test.jsonl").write_text("")
    run = tmp_path / "training"
    _json(run / "training_supervisor.json", {"state": "completed", "exit_code": 0, "execution_id": "own"})
    _json(run / "training_manifest.json", {
        "execution_id": "own", "training_profile": "furniture_initial_pilot",
        "completion": {"completed": True, "optimizer_steps": 8, "expected_optimizer_steps": 8},
        "train_metrics": {"train_loss": 0.67, "nonzero_lora_b_tensors": 256},
        "dataset_features": {"has_images": False},
        "dataset_manifest": str(dataset / "manifest.json"),
        "adapter_dir": str(run / "adapter"),
        "dataset_snapshot": {name: hashlib.sha256((dataset/name).read_bytes()).hexdigest()
                             for name in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl")},
    })
    _json(run / "preflight.json", {"dataset_validation": {"split_counts": {"validation": 0}}})
    _json(run / "effective_config.json", {"training": {"loss_type": "sigmoid"}})
    _json(run / "train_results.json", {})
    _json(run / "adapter/adapter_config.json", {})
    (run / "adapter/adapter_model.safetensors").write_bytes(b"nonempty")
    return run, dataset


def test_changed_dataset_or_incomplete_source_is_rejected(source):
    run, dataset = source
    manifest, rows = load_diagnostic_source(run)
    assert manifest["execution_id"] == "own" and len(rows) == 1
    (dataset / "train.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="dataset changed"):
        load_diagnostic_source(run)
    _json(run / "training_supervisor.json", {"state": "failed", "exit_code": 4})
    with pytest.raises(ValueError, match="supervisor"):
        load_diagnostic_source(run)


def test_source_without_actual_completion_is_rejected(source):
    run, _ = source
    (run / "adapter/adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="missing or empty artifact"):
        load_diagnostic_source(run)


def test_validation_pair_evidence_is_frozen_with_the_training_manifest(source):
    run, _ = source
    path = run / "validation_pair_scores.jsonl"
    path.write_text('{"pair_id":"saved"}\n')
    manifest_path = run / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation_pair_scores"] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    _json(manifest_path, manifest)
    load_diagnostic_source(run)
    path.write_text('{"pair_id":"tampered"}\n')
    with pytest.raises(ValueError, match="pair-score evidence"):
        load_diagnostic_source(run)


def test_dry_run_requires_no_torch_and_never_overwrites_source(source, tmp_path):
    run, _ = source
    before = {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    script = Path(__file__).resolve().parents[2] / "scripts/diagnose_sceneexpert_dpo.py"
    cmd = [sys.executable, str(script), "--training-run", str(run),
           "--output-dir", str(tmp_path / "diagnostic"), "--dry-run"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()} == before
    again = subprocess.run(cmd, capture_output=True, text=True)
    assert again.returncode != 0 and "immutable" in again.stderr


def test_planning_and_unknown_tools_are_not_counted_as_scene_actions():
    def messages(names):
        return [{"tool_calls": [{"function": {"name": name}} for name in names]}]
    assert completion_scope(messages(["observe_scene", "designer_todo_manager"]))["scope"] == "observation_or_plan"
    assert completion_scope(messages(["generate_assets"]))["scope"] == "scene_design"
    assert completion_scope(messages(["new_tool"]))["scope"] == "unknown_tool"


def test_reference_recovery_and_relative_shift_are_distinct_from_absolute_likelihood():
    margin = 0.2
    metrics = {"logps/chosen": -9.0, "logps/rejected": -7.0,
               "rewards/chosen": 0.1, "rewards/rejected": -0.1,
               "loss": math.log1p(math.exp(-margin))}
    score = pair_score(metrics, beta=0.1, token_counts=[90, 70])
    assert score["reference_logps/chosen"] == -10
    assert score["reference_logps/rejected"] == -6
    assert score["shift_correct"] is True  # Chosen likelihood is still below rejected.
    assert score["policy_mean_logp/chosen"] == -0.1
    with pytest.raises(ValueError, match="objective"):
        pair_score(dict(metrics, loss=0.1), beta=0.1, token_counts=[90, 70])
    with pytest.raises(ValueError, match="nonfinite"):
        pair_score(dict(metrics, loss=float("nan")), beta=0.1, token_counts=[90, 70])


def test_summary_does_not_mix_training_and_validation():
    rows = [
        {"split": split, "target_scope": {"chosen": {"scope": "observation_or_plan"},
         "rejected": {"scope": "observation_or_plan"}},
         "score": {"shift_correct": correct, "shift_margin": margin, "loss": loss}}
        for split, correct, margin, loss in [("train", True, 0.2, 0.59), ("validation", False, -0.1, 0.74)]
    ]
    summary = summarize_scores(rows)
    assert summary["splits"]["train"]["shift_accuracy"] == 1
    assert summary["splits"]["validation"]["shift_accuracy"] == 0
    assert summary["target_scope_counts"] == {"observation_or_plan": 4}


def test_observer_uses_native_preparation_and_preserves_aggregate_metrics(tmp_path):
    class Mask:
        shape = (2, 4)

        def __getitem__(self, key):
            return self

        def sum(self, dim):
            return self

        def tolist(self):
            return [3, 3]

    class Trainer:
        args = SimpleNamespace(per_device_eval_batch_size=1, world_size=1, beta=.1)

        def __init__(self):
            self._metrics = {"eval": defaultdict(list)}
            self.prepared = False

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            assert self.prepared, "evaluation must prepare the model before prediction"
            for name, value in {
                "logps/chosen": -1., "logps/rejected": -2.,
                "rewards/chosen": .1, "rewards/rejected": -.1,
            }.items():
                self._metrics["eval"][name].append(value)
            return math.log1p(math.exp(-.2)), None, None

        def evaluate(self, eval_dataset=None):
            self.prepared = True
            losses = [self.prediction_step(None, {"completion_mask": Mask()}, True)[0]
                      for _ in (eval_dataset if eval_dataset is not None else range(2))]
            assert len(self._metrics["eval"]["rewards/chosen"]) == 2
            self._metrics["eval"].clear()
            return {"eval_loss": sum(losses)/len(losses),
                    "eval_rewards/accuracies": 1., "eval_rewards/margins": .2}

    trainer = Trainer()
    original = trainer.prediction_step
    rows = [{"pair_id": str(i), "task_id": str(i), "split": "validation",
             "target_scope": {"chosen": {"scope": "text_only"},
                              "rejected": {"scope": "text_only"}}} for i in range(2)]
    recorded = []
    metrics, scores = evaluate_policy_pairs(
        trainer, rows, dataset=[0, 1], beta=.1, on_score=recorded.append,
    )
    assert scores == recorded and len(scores) == 2
    assert metrics["eval_rewards/accuracies"] == 1
    assert trainer.prediction_step == original

    from scripts.train_sceneexpert_dpo import _evaluate_with_pair_audit

    dataset = tmp_path / "dataset"
    output = tmp_path / "training"
    dataset.mkdir()
    output.mkdir()
    payload = [dict(row, chosen=[{"role": "assistant", "content": "chosen"}],
                    rejected=[{"role": "assistant", "content": "rejected"}]) for row in rows]
    (dataset / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in payload))
    native_metrics, evidence = _evaluate_with_pair_audit(
        trainer, dataset_dir=dataset, output_dir=output, capture_pairs=True,
    )
    saved = output / evidence["path"]
    assert native_metrics == metrics and evidence["pair_count"] == 2
    assert evidence["sha256"] == hashlib.sha256(saved.read_bytes()).hexdigest()
    assert [json.loads(line)["pair_id"] for line in saved.read_text().splitlines()] == ["0", "1"]

    trainer.args = SimpleNamespace(per_device_eval_batch_size=2, world_size=1)
    with pytest.raises(ValueError, match="batch size"):
        evaluate_policy_pairs(trainer, rows, dataset=[], beta=.1, on_score=recorded.append)


def test_reproduction_mismatch_is_distinct_from_a_complete_evaluation():
    summary = {"splits": {"validation": {"pair_count": 7, "mean_dpo_loss": .71188,
                                        "shift_accuracy": 2/7}}}
    original = {"eval_loss": .71043, "eval_rewards/accuracies": 3/7}
    result = validation_reproduction(summary, original, 7)
    assert result["checked"] and not result["passed"]
    assert result["loss_difference"] == pytest.approx(.00145)
    assert result["accuracy_difference"] == pytest.approx(-1/7)
    outcome = diagnostic_outcome(scored_count=22, expected_count=22, fidelity=result)
    assert outcome["completed"] and outcome["status"] == "completed_with_warnings"
    assert outcome["warnings"] == ["historical_validation_reproduction_mismatch"]
    assert outcome["failure_reasons"] == [] and not result["passed"]
    assert not diagnostic_outcome(scored_count=21, expected_count=22, fidelity=result)["completed"]
    assert not diagnostic_outcome(scored_count=0, expected_count=0, fidelity=result)["completed"]
    assert not validation_reproduction(summary, original, 8)["checked"]
    summary["splits"]["validation"]["shift_accuracy"] = 3/7
    assert validation_reproduction(summary, original, 7)["passed"]
