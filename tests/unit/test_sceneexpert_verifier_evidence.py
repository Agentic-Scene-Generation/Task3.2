"""Evidence-integrity checks for stage-specific SceneBenchmark scores."""

from __future__ import annotations

from pathlib import Path

from scenesmith.scene_expert.verifier import (
    _critique_is_final,
    _deterministic_hard_check_report,
    _find_scores_yaml,
)


def test_stage_score_lookup_never_borrows_another_stage(tmp_path: Path) -> None:
    furniture = tmp_path / "scene_states" / "furniture"
    furniture.mkdir(parents=True)
    score_path = furniture / "scores.yaml"
    score_path.write_text("Summary: complete\nFunctionality: 9\n", encoding="utf-8")

    assert _find_scores_yaml(str(tmp_path), stage="furniture") == score_path
    assert _find_scores_yaml(str(tmp_path), stage="wall_mounted") is None


def test_incomplete_critic_preamble_is_not_authoritative_evidence() -> None:
    assert not _critique_is_final(
        "I'll validate all orientation-dependent furniture before drawing conclusions."
    )
    assert _critique_is_final("No collision or functional issue remains.")


def test_relation_satisfaction_uses_only_evaluated_deterministic_relations() -> None:
    payload = {
        "results": [
            {
                "check_id": "chair_faces_desk",
                "label": "pass",
                "scoring_tier": "core",
                "evidence": {
                    "intent_constraint": {
                        "constraint_id": "faces_1",
                        "relation": "faces",
                        "strength": "hard",
                        "stage": "furniture",
                    }
                },
            },
            {
                "check_id": "desk_near_wall",
                "label": "fail",
                "scoring_tier": "core",
                "evidence": {
                    "intent_constraint": {
                        "constraint_id": "near_1",
                        "relation": "near",
                        "strength": "hard",
                        "stage": "furniture",
                    }
                },
            },
            {
                "check_id": "future_manipuland_check",
                "label": "fail",
                "scoring_tier": "core",
                "evidence": {
                    "intent_constraint": {
                        "constraint_id": "future_1",
                        "relation": "on",
                        "strength": "hard",
                        "stage": "manipuland",
                    }
                },
            },
        ]
    }

    report = _deterministic_hard_check_report(payload, "furniture")

    assert report["hard_passed"] is False
    assert report["relation_satisfaction"] == 0.5
    assert report["evaluated_relation_check_ids"] == [
        "chair_faces_desk",
        "desk_near_wall",
    ]
