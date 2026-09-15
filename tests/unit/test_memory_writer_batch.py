"""Bounded collection guards; no live LLM, scene generation or old-bank writes."""

import copy
import json

import pytest

from scripts import replay_sceneexpert_memory_batch as batch
from scenesmith.scene_expert.schemas import FullVerifyReport


def plan(monkeypatch, tmp_path, count=2):
    payloads = {}
    cases = []
    for i in range(count):
        payload = {
            "evidence": {"prompt": f"source {i}"},
            "full_report": FullVerifyReport().model_dump(),
        }
        source = tmp_path / f"scene{i}"
        source.mkdir()
        payloads[source] = payload
        cases.append(
            {
                "case_id": f"case{i}",
                "scene_expert_dir": source.name,
                "prompt_sha256": batch.fingerprint(payload["evidence"]["prompt"]),
                "input_sha256": batch.fingerprint(payload),
            }
        )
    monkeypatch.setattr(batch, "load_input", lambda p: copy.deepcopy(payloads[p]))
    return {
        "schema_version": "bounded-writer-replay.v1",
        "cases": cases,
        "holdout_cases": [
            {"case_id": "heldout", "prompt_sha256": batch.fingerprint("heldout")}
        ],
    }


def test_preflight_accepts_complete_plan(monkeypatch, tmp_path):
    manifest = plan(monkeypatch, tmp_path)
    assert len(batch.prepare(manifest, tmp_path)) == 2
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "change", ["overlap", "duplicate_id", "input_change", "traversal", "too_many"]
)
def test_plan_guardrails(monkeypatch, tmp_path, change):
    manifest = plan(monkeypatch, tmp_path)
    if change == "overlap":
        manifest["holdout_cases"][0]["prompt_sha256"] = manifest["cases"][0][
            "prompt_sha256"
        ]
    elif change == "duplicate_id":
        manifest["holdout_cases"][0]["case_id"] = "case0"
    elif change == "input_change":
        manifest["cases"][1]["input_sha256"] = "0" * 64
    elif change == "traversal":
        manifest["cases"][0]["scene_expert_dir"] = "../other"
    else:
        manifest["cases"] *= 5
    with pytest.raises(ValueError):
        batch.prepare(manifest, tmp_path)


def test_no_overwrite_and_dry_run_forwarded(monkeypatch, tmp_path):
    manifest = plan(monkeypatch, tmp_path)
    prepared = batch.prepare(manifest, tmp_path)
    calls = []

    def execute(payload, target, model, base_url, **kwargs):
        calls.append(kwargs)
        return {"phase": "projected"}, 0

    monkeypatch.setattr(batch, "execute_replay", execute)
    target = tmp_path / "output"
    assert (
        batch.run_collection(
            manifest, prepared, target, model="test", base_url="unused", dry_run=True
        )
        == 0
    )
    assert calls == [{"dry_run": True}] * 2
    assert not list(target.rglob("bank"))
    with pytest.raises(FileExistsError):
        batch.run_collection(
            manifest, prepared, target, model="test", base_url="unused"
        )


def test_failure_stops_and_checkpoints_unattempted(monkeypatch, tmp_path):
    manifest = plan(monkeypatch, tmp_path)
    prepared = batch.prepare(manifest, tmp_path)
    monkeypatch.setattr(
        batch, "execute_replay", lambda *a, **k: ({"phase": "preflight_failed"}, 2)
    )
    target = tmp_path / "output"
    assert (
        batch.run_collection(
            manifest, prepared, target, model="test", base_url="unused"
        )
        == 1
    )
    result = json.loads((target / "replay_summary.json").read_text())
    assert result["not_attempted"] == ["case1"]
    assert not (target / "case1").exists()


def test_refuses_output_inside_source(monkeypatch, tmp_path):
    manifest = plan(monkeypatch, tmp_path)
    prepared = batch.prepare(manifest, tmp_path)
    with pytest.raises(ValueError, match="outside"):
        batch.run_collection(
            manifest,
            prepared,
            tmp_path / "scene0/output",
            model="test",
            base_url="unused",
        )


def test_review_preserves_full_record_without_seed_promotion(tmp_path):
    folder = tmp_path / "bank"
    folder.mkdir()
    record = {
        "case_id": "id",
        "status": "active",
        "placement_experience": {"procedure": ["step"]},
    }
    (folder / "success_cases.jsonl").write_text(json.dumps(record) + "\n")
    rows = batch.review_records(tmp_path, "source")
    assert rows[0]["record"] == record
    assert rows[0]["decision"] == "pending_human_review"
