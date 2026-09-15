"""Rescoring completion and preference readiness have separate exit policies."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "integrity,gate,require_pairs,expected_exit",
    [
        (True, False, False, 0),
        (True, False, True, 2),
        (False, False, False, 2),
        (False, False, True, 2),
        (True, True, False, 0),
        (True, True, True, 0),
    ],
)
def test_rescore_exit_policy_keeps_dataset_gate_explicit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    integrity: bool,
    gate: bool,
    require_pairs: bool,
    expected_exit: int,
) -> None:
    from scenesmith.scene_expert.slow_memory import paired_rescore

    result = {
        "status": (
            "execution_failed"
            if not integrity
            else "completed_with_pairs" if gate else "completed_no_pairs"
        ),
        "gate_passed": gate,
        "execution_integrity_passed": integrity,
        "preference_gate_passed": gate,
        "candidate_count": 2 if integrity else 0,
        "candidate_verdict_counts": {"accepted": 2} if integrity else {},
        "eligible_pair_count": 1 if gate else 0,
    }
    monkeypatch.setattr(paired_rescore, "rescore_pairs", lambda source, output: result)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collector",
            "--rescore-source",
            "original",
            "--rescore-output",
            "new",
            *(["--require-pairs"] if require_pairs else []),
        ],
    )
    main = runpy.run_path(str(ROOT / "scripts/collect_sceneexpert_initial_pairs.py"))[
        "main"
    ]
    assert main() == expected_exit
    output = capsys.readouterr().out
    assert f'"preference_gate_passed": {str(gate).lower()}' in output
    if integrity and not gate:
        assert '"status": "completed_no_pairs"' in output
        assert "preference dataset is not ready" in output
    assert result["gate_passed"] is gate


def test_rescore_runtime_exception_cannot_be_reported_as_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scenesmith.scene_expert.slow_memory import paired_rescore

    def fail(source: Path, output: Path) -> None:
        raise RuntimeError("physics failed")

    monkeypatch.setattr(paired_rescore, "rescore_pairs", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        ["collector", "--rescore-source", "original", "--rescore-output", "new"],
    )
    main = runpy.run_path(str(ROOT / "scripts/collect_sceneexpert_initial_pairs.py"))[
        "main"
    ]
    with pytest.raises(RuntimeError, match="physics failed"):
        main()


def test_normal_collection_audit_keeps_strict_pair_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scenesmith.scene_expert.slow_memory import paired

    monkeypatch.setattr(
        paired,
        "audit_pairs",
        lambda *a, **kw: {
            "status": "completed_no_pairs",
            "gate_passed": False,
            "execution_integrity_passed": True,
            "preference_gate_passed": False,
            "candidate_count": 2,
            "candidate_verdict_counts": {"accepted": 2},
            "eligible_pair_count": 0,
            "errors": ["minimum_pair_gate_failed"],
        },
    )
    monkeypatch.setattr(sys, "argv", ["collector", "--audit-root", "original"])
    main = runpy.run_path(str(ROOT / "scripts/collect_sceneexpert_initial_pairs.py"))[
        "main"
    ]
    assert main() == 2
