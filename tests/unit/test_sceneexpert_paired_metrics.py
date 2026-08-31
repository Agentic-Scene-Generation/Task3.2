from __future__ import annotations

from scenesmith.scene_expert.paired_metrics import compare_run_metrics


def _run(run_id: str, *, ready: bool, time_sec: float, critic: float) -> dict:
    return {
        "run_id": run_id,
        "quality_comparison_ready": ready,
        "experiment_identity": {
            "experiment_names": ["ablation_4c"],
            "experiment_signatures": ["stable-4c-signature"],
            "config_hashes": ["config-1"],
            "models": ["qwen-test"],
            "code_revisions": ["abc123"],
        },
        "code_provenance": {
            "git_revision": "abc123",
            "git_status_hash": "clean-hash",
            "source_hashes": {"hooks.py": "source-hash"},
        },
        "memory_identity": {
            "memory_bank_ids": ["bank-1"],
            "memory_dirs": ["/outputs/memory/ablation_4c"],
        },
        "summary": {"completion_rate": 1.0 if ready else 0.5},
        "scenes": [
            {
                "case_id": "bedroom_a",
                "prompt": "A bedroom.",
                "status": "completed" if ready else "failed",
                "generation_status": "complete" if ready else "failed",
                "requirement_status": "satisfied" if ready else "partial",
                "quality_status": "passed" if ready else "degraded",
                "required_coverage": critic,
                "trace_time_sec": time_sec,
                "critic_score": critic,
                "sceneexpert_overall_score": critic - 0.1,
                "memory_retrieved_stages": ["furniture"],
                "memory_injection_verified_stages": ["furniture"],
                "memory_cross_task_verified_stages": ["furniture"],
            }
        ],
    }


def test_ready_pair_reports_signed_deltas() -> None:
    result = compare_run_metrics(
        _run("cold", ready=True, time_sec=100.0, critic=0.8),
        _run("warm", ready=True, time_sec=80.0, critic=0.9),
    )

    assert result["comparison_ready"] is True
    assert result["claim_status"] == "paired_quality_and_outcomes_ready"
    assert result["quality_delta_ready"] is True
    assert result["summary"]["mean_time_delta_sec"] == -20.0
    assert result["summary"]["median_speedup_ratio"] == 1.25
    assert result["summary"]["mean_critic_score_delta"] == 0.1
    assert result["summary"]["mean_required_coverage_delta"] == 0.1
    assert result["summary"]["requirement_coverage_wins"] == 1
    assert result["pairs"][0]["baseline_generation_status"] == "complete"
    assert result["pairs"][0]["treatment_requirement_status"] == "satisfied"


def test_source_bundle_allows_controlled_pair_without_git_metadata() -> None:
    baseline = _run("cold", ready=True, time_sec=100.0, critic=0.8)
    treatment = _run("warm", ready=True, time_sec=90.0, critic=0.9)
    for metrics in (baseline, treatment):
        metrics["experiment_identity"]["code_revisions"] = []
        metrics["code_provenance"] = {"source_bundle_hash": "stable-source-bundle"}

    result = compare_run_metrics(baseline, treatment)

    assert result["comparison_ready"] is True
    assert result["identity_checks"]["code_provenance.source_bundle_hash"] is True
    assert not any(
        "git" in warning.casefold() for warning in result["data_quality_warnings"]
    )


def test_incomplete_run_keeps_outcome_comparison_but_guards_quality_claim() -> None:
    result = compare_run_metrics(
        _run("cold", ready=True, time_sec=100.0, critic=0.8),
        _run("warm", ready=False, time_sec=80.0, critic=0.9),
    )

    assert result["comparison_ready"] is True
    assert result["quality_delta_ready"] is False
    assert result["claim_status"] == "paired_outcomes_ready_with_partial_quality"
    assert result["summary"]["regressed_cases"] == 1
    assert "treatment_not_quality_ready" in result["data_quality_warnings"]
