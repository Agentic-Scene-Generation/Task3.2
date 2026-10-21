"""The approved profile cannot train a raw, mixed, or under-floor dataset."""

import hashlib
import json

import pytest

from scenesmith.scene_expert.slow_memory import pairlocal_qualification as gate
from scenesmith.scene_expert.slow_memory.training import (
    PAIR_LOCAL_PILOT,
    apply_training_profile,
    evaluate_training_promotion,
    validate_training_request,
)


def test_new_profile_keeps_recipe_floor_and_is_not_promotable(tmp_path):
    config = apply_training_profile({}, PAIR_LOCAL_PILOT)
    assert config["data"]["minimum_train_pairs"] == 8
    assert config["data"]["minimum_unique_validation_groups"] == 4
    assert config["data"]["allow_unsafe_small_dataset"] is False
    assert {
        key: config["training"][key]
        for key in (
            "seed",
            "num_train_epochs",
            "learning_rate",
            "beta",
            "gradient_accumulation_steps",
        )
    } == {
        "seed": 42,
        "num_train_epochs": 2,
        "learning_rate": 1e-5,
        "beta": 0.1,
        "gradient_accumulation_steps": 4,
    }
    promotion = evaluate_training_promotion(
        config, evaluation_metrics={"eval_loss": 0.6, "eval_reward_accuracy": 1.0}
    )
    assert promotion["status"] == "pilot_only" and not promotion["promotable"]
    preflight = validate_training_request(config, dataset_dir=tmp_path)
    assert not preflight["valid"] and any(
        "qualification gate" in error for error in preflight["errors"]
    )


def qualified_fixture(root, monkeypatch, *, full=True):
    """Software fixture, never a native observation or a qualification decision."""
    train = [368, 371, 238, 205, 212, 297, 120, 419] if full else [368, 371, 238]
    val = [479, 302, 266, 188] if full else [479, 302, 266]

    def put(name, value):
        path = root / name
        path.write_bytes(json.dumps(value).encode())
        return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    originals = [
        {
            "sceneeval_id": i,
            "task_id": f"task-{i}",
            "split": "train" if i in train else "validation",
        }
        for i in [*train, 479, 302, 266]
    ]
    partition = put("original.json", {"tasks": originals})
    selection = put(
        "selection.json",
        {
            "supplemental_validation": [
                {"sceneeval_id": 188, "task_id": "task-188", "split": "validation"}
            ]
        },
    )
    monkeypatch.setattr(gate, "ORIGINAL_CAMPAIGN_SHA256", partition["sha256"])
    monkeypatch.setattr(gate, "APPROVED_SELECTION_SHA256", selection["sha256"])
    proof = put("fixture-proof.json", {"software_fixture_only": True})
    bridge = put(
        "bridge.json",
        {
            "passed": True,
            "old_source_fingerprint": gate.SOURCE_605,
            "new_source_fingerprint": "fixture-new-source",
            "certified_original_ids": sorted(gate.CERTIFIED_605_IDS),
            "evidence_bindings": [proof],
        },
    )
    entries = []
    for split, ids in (("train", train), ("validation", val)):
        lines = []
        for i in ids:
            row = json.dumps({"pair_id": str(i), "task_id": f"task-{i}"}).encode()
            lines.append(row)
            entries.append(
                {
                    "sceneeval_id": i,
                    "pair_id": str(i),
                    "split": split,
                    "dataset_row_sha256": hashlib.sha256(row).hexdigest(),
                    "semantic_qualified": True,
                    "cohort": "supplemental_v2" if i == 188 else "original184",
                    "source_fingerprint": gate.SOURCE_605
                    if i in gate.CERTIFIED_605_IDS
                    else "fixture-new-source",
                    "semantic_decision": proof,
                    "native_pair": proof,
                    "full_review_delivery": proof,
                }
            )
        (root / f"{split}.jsonl").write_bytes(b"\n".join(lines) + b"\n")
    put(
        "qualification.json",
        {
            "schema_version": "sceneexpert.pairlocal_qualified_dataset.v1",
            "profile": gate.PROFILE,
            "source_bridge": bridge,
            "original_campaign": partition,
            "approved_selection": selection,
            "pairs": entries,
        },
    )


def test_three_T_three_V_never_passes_new_profile(tmp_path, monkeypatch):
    qualified_fixture(tmp_path, monkeypatch, full=False)
    assert any(
        "actual8originalT4independentV" in error
        for error in gate.validate_qualified_dataset(tmp_path)
    )


@pytest.mark.parametrize(
    "change",
    [None, "row", "bridge", "partition", "duplicate", "unknown", "unapproved_old"],
)
def test_byte_bound_qualification_and_conditional_bridge(tmp_path, monkeypatch, change):
    qualified_fixture(tmp_path, monkeypatch)
    path = tmp_path / "qualification.json"
    d = json.loads(path.read_bytes())
    if change == "row":
        with (tmp_path / "train.jsonl").open("ab") as stream:
            stream.write(b'{"pair_id":"raw","task_id":"unqualified"}\n')
    elif change == "bridge":
        (tmp_path / "bridge.json").write_text('{"passed":false}')
    elif change == "partition":
        d["pairs"][0]["split"] = "validation"
    elif change == "duplicate":
        d["pairs"].append(d["pairs"][0])
    elif change == "unknown":
        d["pairs"][0]["semantic_qualified"] = False
    elif change == "unapproved_old":
        d["pairs"][3]["source_fingerprint"] = gate.SOURCE_605
    path.write_text(json.dumps(d), encoding="utf-8")
    errors = gate.validate_qualified_dataset(tmp_path)
    assert bool(errors) is (change is not None)
