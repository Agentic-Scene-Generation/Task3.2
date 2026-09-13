"""Offline collection gates and archive-copy regressions, without a simulator."""

from __future__ import annotations

import hashlib
import json
import shutil

from pathlib import Path

import pytest

from scenesmith.scene_expert.slow_memory.collection import audit_collection
from scenesmith.scene_expert.slow_memory.dpo import load_trajectories
from scenesmith.scene_expert.slow_memory.schemas import (
    PreferenceEvidence,
    TrajectoryRecord,
)

MODEL = "unsloth/Qwen3.8-27B-GGUF"
STAGES = ["floor_plan", "furniture", "wall_mounted", "ceiling_mounted", "manipuland"]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _collection(root: Path) -> tuple[Path, Path]:
    """Model the four equivalent final traces in a returned ACP archive."""
    hydra = root / "critic_on/batch_041/hydra"
    scene = hydra / "scene_040/scene_expert"
    trajectories = scene / "slow_memory/trajectories.jsonl"
    trajectories.parent.mkdir(parents=True)
    media = trajectories.parent / "media/image.png"
    media.parent.mkdir()
    media.write_bytes(b"test-only image content")
    image_ref = {
        "path": "media/image.png",
        "sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
    }
    record = TrajectoryRecord(
        trajectory_id="candidate_1",
        created_at="2026-09-13T00:00:00Z",
        run_id="critic_on/batch_041/hydra",
        scene_id="scene_040",
        task_id="bedroom",
        model_id=MODEL,
        stage="furniture",
        agent_role="designer",
        event="request_initial_design",
        task_type="designer_initial",
        context_hash="initial-context",
        prompt="Place a bed.",
        response="Placed a bed.",
        response_hash="response-1",
        evidence=PreferenceEvidence(
            evidence_id="report-1",
            kind="critic",
            verdict="rejected",
            authoritative=True,
            report_ref="report.json",
        ),
        source_refs=["report.json"],
        image_refs=[image_ref],
        provenance={"tool_media_refs": [image_ref]},
    )
    trajectories.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    trace = {
        "scene_id": "scene_040",
        "status": "completed",
        "final_report": {
            "completed_stages": STAGES,
            "missing_stages": [],
            "generation_status": "complete",
            "pass_scene": False,
            "deterministic_pass": False,
        },
        "stages": [],
    }
    final_trace = scene / "trace/trace_000040.json"
    _write_json(final_trace, trace)
    _write_json(hydra / "traces/trace_000040.json", trace)
    _write_json(scene / "trace/trace_000040_partial.json", {})
    shutil.copytree(hydra, hydra.parent / "latest-run")
    # Partial shared-base generation is not the final full-scene result.
    _write_json(root / "shared_base/batch_041/hydra/traces/trace_000040.json", {})
    _write_json(
        root / "metrics/run_metrics.json",
        {
            "run_id": "fixture",
            "summary": {
                "expected_scenes": 1,
                "completed_scenes": 1,
                "missing_or_nonterminal_scenes": 0,
                "failed_scenes": 0,
                "degraded_scenes": 1,
            },
        },
    )
    return trajectories, final_trace


def test_identical_archive_copies_deduplicate_before_media_rebasing(
    tmp_path: Path,
) -> None:
    source, _ = _collection(tmp_path)
    records, diagnostics = load_trajectories([tmp_path, source])
    assert diagnostics == []
    assert len(records) == 1
    assert Path(records[0].image_refs[0]["path"]).is_absolute()
    assert Path(records[0].image_refs[0]["path"]).is_file()


def test_real_id_conflict_quarantines_all_versions_even_after_original_reappears(
    tmp_path: Path,
) -> None:
    source, _ = _collection(tmp_path)
    original = source.read_text(encoding="utf-8")
    conflicting = json.loads(original)
    conflicting["response"] = "A different decision under the same ID."
    source.write_text(
        original + json.dumps(conflicting) + "\n" + original, encoding="utf-8"
    )
    records, diagnostics = load_trajectories([tmp_path])
    assert records == []
    assert {row["reason"] for row in diagnostics} == {"trajectory_id_collision"}


def test_complete_quality_failed_scene_is_raw_data_but_never_an_empty_training_pass(
    tmp_path: Path,
) -> None:
    _collection(tmp_path)
    result = audit_collection(tmp_path, tmp_path / "collection", expected_model=MODEL)
    assert result["errors"] == []
    assert result["observer_collection_ready"] is True
    assert result["gate_passed"] is True
    assert result["dpo_export_ready"] is False
    assert result["training_preflight_status"] == "not_run"
    assert result["contexts_with_multiple_candidates"] == 0
    assert len(result["final_traces"]) == 1
    assert result["final_traces"][0]["pass_scene"] is False
    assert result["checked_media_count"] == 2  # One file in each archive copy.
    strict = audit_collection(
        tmp_path, tmp_path / "collection", expected_model=MODEL, min_pairs=1
    )
    assert strict["observer_collection_ready"] is True
    assert strict["gate_passed"] is False
    assert strict["errors"] == ["dpo_pair_gate_failed"]
    assert (tmp_path / "collection/dpo_probe/manifest.json").is_file()


@pytest.mark.parametrize("remove", [False, True])
def test_archive_media_tampering_or_loss_blocks_collection(
    tmp_path: Path, remove: bool
) -> None:
    source, _ = _collection(tmp_path)
    media = source.parent / "media/image.png"
    if remove:
        media.unlink()
    else:
        media.write_bytes(b"corrupted")
    result = audit_collection(tmp_path, tmp_path / "collection", expected_model=MODEL)
    assert result["gate_passed"] is False
    assert "media_integrity_failed" in result["errors"]
    assert result["media_errors"][0]["reason"] == (
        "missing_media" if remove else "media_hash_mismatch"
    )


def test_mixed_models_are_rejected(tmp_path: Path) -> None:
    _collection(tmp_path)
    result = audit_collection(
        tmp_path, tmp_path / "collection", expected_model="wrong-model"
    )
    assert "unexpected_model" in result["errors"]
    assert result["gate_passed"] is False


def test_conflicting_trace_copies_are_not_silently_collapsed(tmp_path: Path) -> None:
    _, trace = _collection(tmp_path)
    changed = json.loads(trace.read_text(encoding="utf-8"))
    changed["final_report"]["pass_scene"] = True
    _write_json(trace, changed)
    result = audit_collection(tmp_path, tmp_path / "collection", expected_model=MODEL)
    assert "conflicting_final_trace" in result["errors"]


def test_missing_generation_artifacts_produce_diagnostics(tmp_path: Path) -> None:
    result = audit_collection(tmp_path, tmp_path / "collection", expected_model=MODEL)
    assert result["generation_complete"] is False
    assert result["gate_passed"] is False
    assert "no_trajectory_files" in result["errors"]
    assert "missing_or_invalid_run_metrics" in result["errors"]
    assert (tmp_path / "collection/collection_audit.json").is_file()


def test_real_two_candidate_fixture_exports_one_pair_not_duplicate_archive_pairs(
    tmp_path: Path,
) -> None:
    source, _ = _collection(tmp_path)
    rejected = json.loads(source.read_text(encoding="utf-8"))
    rejected["evidence"]["quality_score"] = 0.3
    accepted = json.loads(json.dumps(rejected))
    accepted.update(
        trajectory_id="candidate_2",
        response="Placed a better bed.",
        response_hash="response-2",
    )
    accepted["evidence"].update(
        evidence_id="report-2", verdict="accepted", quality_score=0.9
    )
    for path in tmp_path.rglob("trajectories.jsonl"):
        path.write_text(
            json.dumps(rejected) + "\n" + json.dumps(accepted) + "\n", encoding="utf-8"
        )
    result = audit_collection(
        tmp_path, tmp_path / "collection", expected_model=MODEL, min_pairs=1
    )
    assert result["errors"] == []
    assert result["dpo_export_ready"] is True
    assert result["dpo_stats"]["eligible_pair_count"] == 1
    assert result["trajectory_count"] == 2
    assert result["contexts_with_multiple_candidates"] == 1
