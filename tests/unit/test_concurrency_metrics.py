from scripts.evaluate_concurrency_metrics import evaluate_rows


def test_resource_headroom_passes_at_or_below_limit() -> None:
    report = evaluate_rows(
        [
            {
                "gpu_memory_used_mib": "900",
                "gpu_memory_total_mib": "1000",
                "cgroup_memory_current": "850",
                "cgroup_memory_limit": "1000",
            }
        ],
        0.10,
    )

    assert report["passed"] is True
    assert report["peak_gpu_used_fraction"] == 0.9


def test_resource_headroom_fails_when_either_resource_exceeds_limit() -> None:
    report = evaluate_rows(
        [
            {
                "gpu_memory_used_mib": "901",
                "gpu_memory_total_mib": "1000",
                "cgroup_memory_current": "500",
                "cgroup_memory_limit": "1000",
            }
        ],
        0.10,
    )

    assert report["passed"] is False
