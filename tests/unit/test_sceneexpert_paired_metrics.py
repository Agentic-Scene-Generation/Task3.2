from __future__ import annotations

from scenesmith.scene_expert.paired_metrics import compare_run_metrics


def _run(run_id: str, *, ready: bool, time_sec: float, critic: float) -> dict:
    return {
        "run_id": run_id,
        "quality_comparison_ready": ready,
        "experiment_identity": {
            "experiment_names": ["ablation_4c"],
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
    assert result["claim_status"] == "paired_deltas_ready"
    assert result["summary"]["mean_time_delta_sec"] == -20.0
    assert result["summary"]["median_speedup_ratio"] == 1.25
    assert result["summary"]["mean_critic_score_delta"] == 0.1


def test_incomplete_run_refuses_comparison_claim() -> None:
    result = compare_run_metrics(
        _run("cold", ready=True, time_sec=100.0, critic=0.8),
        _run("warm", ready=False, time_sec=80.0, critic=0.9),
    )

    assert result["comparison_ready"] is False
    assert result["claim_status"] == "not_ready"
    assert "treatment_not_quality_ready" in result["data_quality_warnings"]
