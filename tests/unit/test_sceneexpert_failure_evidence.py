from scenesmith.scene_expert.failure_evidence import main_hard_failure_report


def test_recognizes_only_main_deterministic_hard_gate() -> None:
    report = main_hard_failure_report(
        "Furniture stage failed with unresolved core relations: "
        "paired_with:student_desk",
        "furniture",
    )

    assert report is not None
    assert report.pass_stage is False
    assert report.score_source == "scenesmith_main_hard_gate"
    assert report.hard_check_report["hard_valid"] is False
    assert main_hard_failure_report("CUDA connection reset", "furniture") is None
    assert (
        main_hard_failure_report(
            "Furniture stage failed with unresolved core relations: missing", "unknown"
        )
        is None
    )
