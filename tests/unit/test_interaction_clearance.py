from __future__ import annotations

from types import SimpleNamespace

from scenesmith.scenebenchmark_critic.metrics.interaction_clearance.evaluator import (
    build_clearance_checks,
    build_door_clearance_checks,
    build_window_clearance_checks,
    evaluate_clearance,
)
from scenesmith.scenebenchmark_critic.config import CriticConfig
from scenesmith.scenebenchmark_critic.evaluator import evaluate_case_pack


def _object(
    object_id: str,
    *,
    category: str,
    object_type: str,
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    yaw_deg: float = 0.0,
    clearance: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if clearance is not None:
        metadata["clearance"] = clearance
    return {
        "id": object_id,
        "name": object_id,
        "category": category,
        "object_type": object_type,
        "bbox_world": {"min": bbox_min, "max": bbox_max},
        "yaw_deg": yaw_deg,
        "metadata": metadata,
    }


def _monitor() -> dict[str, object]:
    return _object(
        "monitor_0",
        category="monitor",
        object_type="manipuland",
        bbox_min=(-1.1, -1.6, 0.75),
        bbox_max=(-0.9, -1.4, 1.1),
        yaw_deg=90.0,
        clearance={
            "clearance_type": "approach",
            "direction": "front",
            "depth_m": 0.45,
            "height_m": 1.8,
            "confidence": "high",
        },
    )


def _clearance_label(*objects: dict[str, object]) -> str:
    checks = build_clearance_checks({str(obj["id"]): obj for obj in objects})
    return str(checks[0]["clearance_result"]["label"])


def test_high_ceiling_fixture_does_not_block_horizontal_clearance() -> None:
    fixture = _object(
        "ceiling_light_0",
        category="ceiling_light",
        object_type="ceiling_mounted",
        bbox_min=(-1.2, -1.6, 2.36),
        bbox_max=(-0.9, -1.4, 2.7),
    )

    assert _clearance_label(_monitor(), fixture) == "pass"


def test_low_ceiling_fixture_still_blocks_horizontal_clearance() -> None:
    fixture = _object(
        "pendant_0",
        category="pendant_light",
        object_type="ceiling_mounted",
        bbox_min=(-1.2, -1.6, 1.85),
        bbox_max=(-0.9, -1.4, 2.1),
    )

    assert _clearance_label(_monitor(), fixture) == "fail"


def test_door_sweep_uses_door_width_beyond_legacy_clearance_box() -> None:
    geometry = {
        "scene_shell": {
            "doors": [
                {
                    "id": "door_0",
                    "opening_type": "door",
                    "center": (-2.5, 0.0, 1.05),
                    "width": 0.9,
                    "height": 2.1,
                    "sill_height": 0.0,
                    "wall_direction": "west",
                    # Deliberately shallower than the physical door leaf.  The
                    # interaction rule must derive the sweep from door width.
                    "bbox": {
                        "min": (-2.5, -0.45, 0.0),
                        "max": (-2.2, 0.45, 2.1),
                    },
                }
            ]
        }
    }
    sofa = _object(
        "sofa_0",
        category="sofa",
        object_type="furniture",
        bbox_min=(-1.75, -0.10, 0.0),
        bbox_max=(-1.65, 0.10, 0.8),
    )

    checks = build_door_clearance_checks(geometry, {"sofa_0": sofa})

    assert len(checks) == 1
    assert checks[0]["clearance_result"]["label"] == "fail"
    assert checks[0]["clearance_result"]["blocking_objects"] == ["sofa_0"]
    assert checks[0]["clearance_result"]["door_width_m"] == 0.9


def test_door_sweep_ignores_floor_coverings_and_clear_furniture() -> None:
    geometry = {
        "scene_shell": {
            "doors": [
                {
                    "id": "door_0",
                    "opening_type": "door",
                    "center": (0.0, -2.0, 1.05),
                    "width": 0.9,
                    "height": 2.1,
                    "wall_direction": "south",
                }
            ]
        }
    }
    rug = _object(
        "rug_0",
        category="rug",
        object_type="thin_covering",
        bbox_min=(-0.4, -1.9, 0.0),
        bbox_max=(0.4, -1.2, 0.02),
    )
    sofa = _object(
        "sofa_0",
        category="sofa",
        object_type="furniture",
        bbox_min=(1.0, 0.5, 0.0),
        bbox_max=(2.0, 1.2, 0.8),
    )

    check = build_door_clearance_checks(geometry, {"rug_0": rug, "sofa_0": sofa})[0]

    assert check["clearance_result"]["label"] == "pass"
    assert check["clearance_result"]["blocking_objects"] == []


def test_door_clearance_stays_core_while_window_clearance_is_auxiliary() -> None:
    geometry = {
        "scene_shell": {
            "doors": [
                {
                    "id": "door_0",
                    "center": (0.0, -2.0, 1.05),
                    "width": 0.9,
                    "height": 2.1,
                    "wall_direction": "south",
                }
            ],
            "windows": [
                {
                    "id": "window_0",
                    "sill_height": 0.9,
                    "wall_direction": "north",
                    "bbox": {
                        "min": (-0.75, 1.9, 0.9),
                        "max": (0.75, 2.1, 2.1),
                    },
                }
            ],
        }
    }
    cabinet = _object(
        "cabinet_0",
        category="cabinet",
        object_type="furniture",
        bbox_min=(-0.4, 1.8, 0.0),
        bbox_max=(0.4, 2.0, 1.8),
    )

    door_check = build_door_clearance_checks(geometry, {"cabinet_0": cabinet})[0]
    window_check = build_window_clearance_checks(geometry, {"cabinet_0": cabinet})[0]

    assert door_check["scoring_tier"] == "core"
    assert window_check["scoring_tier"] == "auxiliary"


def _window_geometry() -> dict[str, object]:
    return {
        "scene_shell": {
            "windows": [
                {
                    "id": "window_0",
                    "sill_height": 0.9,
                    "wall_direction": "north",
                    "bbox": {
                        "min": (-0.75, 1.9, 0.9),
                        "max": (0.75, 2.1, 2.1),
                    },
                }
            ]
        }
    }


def _evaluate_window(*objects: dict[str, object], core_ratio: float = 0.15) -> dict:
    check = build_window_clearance_checks(
        _window_geometry(), {str(obj["id"]): obj for obj in objects}
    )[0]
    config = SimpleNamespace(
        run=SimpleNamespace(
            window_clearance_advisory_occlusion_ratio=0.02,
            window_clearance_core_occlusion_ratio=core_ratio,
        )
    )
    return evaluate_clearance(check, config=config)


def test_window_clearance_ignores_furniture_below_sill() -> None:
    bench = _object(
        "bench_0",
        category="bench",
        object_type="furniture",
        bbox_min=(-0.5, 1.8, 0.0),
        bbox_max=(0.5, 2.0, 0.8),
    )

    result = _evaluate_window(bench)

    assert result["label"] == "pass"
    assert result["scoring_tier"] == "auxiliary"
    assert result["diagnostics"]["occlusion_ratio"] == 0.0


def test_window_clearance_marks_minor_projected_overlap_advisory() -> None:
    narrow_decor = _object(
        "narrow_decor_0",
        category="decor",
        object_type="furniture",
        bbox_min=(-0.15, 1.8, 0.0),
        bbox_max=(0.15, 2.0, 1.2),
    )

    result = _evaluate_window(narrow_decor)

    assert result["label"] == "degraded"
    assert result["scoring_tier"] == "auxiliary"
    assert result["blocking_objects"] == []
    assert result["diagnostics"]["advisory_blocking_objects"] == ["narrow_decor_0"]
    assert result["diagnostics"]["occlusion_ratio"] == 0.05


def test_window_clearance_upgrades_substantial_opaque_blockage_to_core() -> None:
    cabinet = _object(
        "cabinet_0",
        category="cabinet",
        object_type="furniture",
        bbox_min=(-0.4, 1.8, 0.0),
        bbox_max=(0.4, 2.0, 1.8),
    )

    result = _evaluate_window(cabinet)

    assert result["label"] == "fail"
    assert result["scoring_tier"] == "core"
    assert result["blocking_objects"] == ["cabinet_0"]
    assert result["diagnostics"]["occlusion_ratio"] == 0.4
    evidence = result["diagnostics"]["occlusion_evidence"][0]
    assert evidence["lateral_overlap_ratio"] > 0.5
    assert evidence["vertical_overlap_ratio"] == 0.75


def test_window_clearance_detects_wall_mounted_projected_blockage() -> None:
    television = _object(
        "television_0",
        category="television",
        object_type="wall_mounted",
        bbox_min=(-0.7, 2.02, 1.0),
        bbox_max=(0.7, 2.08, 2.0),
    )

    result = _evaluate_window(television)

    assert result["label"] == "fail"
    assert result["scoring_tier"] == "core"
    assert result["diagnostics"]["wall_mounted_blocking_objects"] == ["television_0"]
    assert result["diagnostics"]["occlusion_evidence"][0]["wall_mounted"] is True


def test_window_clearance_thresholds_come_from_rule_config() -> None:
    cabinet = _object(
        "cabinet_0",
        category="cabinet",
        object_type="furniture",
        bbox_min=(-0.4, 1.8, 0.0),
        bbox_max=(0.4, 2.0, 1.8),
    )

    result = _evaluate_window(cabinet, core_ratio=0.5)

    assert result["label"] == "degraded"
    assert result["scoring_tier"] == "auxiliary"
    assert result["diagnostics"]["core_occlusion_ratio_threshold"] == 0.5


def test_window_thresholds_flow_through_full_critic_evaluation() -> None:
    cabinet = _object(
        "cabinet_0",
        category="cabinet",
        object_type="furniture",
        bbox_min=(-0.4, 1.8, 0.0),
        bbox_max=(0.4, 2.0, 1.8),
    )
    geometry = _window_geometry()
    check = build_window_clearance_checks(geometry, {"cabinet_0": cabinet})[0]
    case_pack = {
        "stage": "final",
        "scene_geometry": geometry,
        "checks": [check],
    }
    config = CriticConfig(
        enabled=True,
        metrics=("interaction_clearance",),
        extra={"window_clearance_core_occlusion_ratio": 0.5},
    )

    payload = evaluate_case_pack(case_pack, config=config)
    result = next(
        row
        for row in payload["results"]
        if row["check_id"] == "window_clearance__window_0"
    )

    assert result["label"] == "degraded"
    assert result["scoring_tier"] == "auxiliary"
    assert result["diagnostics"]["core_occlusion_ratio_threshold"] == 0.5
